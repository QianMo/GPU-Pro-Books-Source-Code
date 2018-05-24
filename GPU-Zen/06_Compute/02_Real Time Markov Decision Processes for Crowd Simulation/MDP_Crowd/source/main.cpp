/*

Copyright 2013,2014 Sergio Ruiz, Benjamin Hernandez

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

In case you, or any of your employees or students, publish any article or
other material resulting from the use of this  software, that publication
must cite the following references:

Sergio Ruiz, Benjamin Hernandez, Adriana Alvarado, and Isaac Rudomin. 2013.
Reducing Memory Requirements for Diverse Animated Crowds. In Proceedings of
Motion on Games (MIG '13). ACM, New York, NY, USA, , Article 55 , 10 pages.
DOI: http://dx.doi.org/10.1145/2522628.2522901

Sergio Ruiz and Benjamin Hernandez. 2015. A Parallel Solver for Markov Decision Process
in Crowd Simulations. Fourteenth Mexican International Conference on Artificial
Intelligence (MICAI), Cuernavaca, 2015, pp. 107-116.
DOI: 10.1109/MICAI.2015.23

*/
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <sstream>

#ifdef __unix
	#include <unistd.h>
#elif defined _WIN32
	#include <direct.h>
#endif

#include <vector>

#include "cMacros.h"
#include "cVertex.h"
#include "cTimer.h"

#include "cGlErrorManager.h"
#include "cLogManager.h"
#include "cTextureManager.h"

#include "cGlslManager.h"
#include "cVboManager.h"
#include "cFboManager.h"
#include "cCrowdManager.h"

#include "cCamera.h"
#include "cObstacleManager.h"
#include "cAxes.h"
#include "cSkyboxManager.h"
#include "cModel3D.h"
#include "cScenario.h"

using namespace std;

//=======================================================================================

// GLOBAL_VARIABLES
#include "cGlobals.h"

//=======================================================================================

void cleanup( void );

//=======================================================================================

// INIT_FUNCTIONS
#include "cInits.h"

//=======================================================================================
//
void draw( void )
{
    glGetFloatv( GL_MODELVIEW_MATRIX,	view_mat );
    glGetFloatv( GL_PROJECTION_MATRIX, 	proj_mat );

#ifdef DRAW_SKYBOX
	glPushAttrib( GL_DEPTH_BUFFER_BIT );
    {
        glDisable( GL_DEPTH_TEST );
		skybox_manager->draw( false, false );
    }
    glPopAttrib();
#endif

	if( !hideCharacters )
	{
#ifdef DEMO_SHADER
		crowd_manager->draw(	camera,
								view_mat,
								proj_mat,
								shadow_mat,
								wireframe,
								drawShadows,
								doHeightAndDisplacement,
								doPatterns,
								doColor,
								doFacial			);
#else
		crowd_manager->draw(	camera,
								view_mat,
								proj_mat,
								shadow_mat,
								wireframe,
								drawShadows	);
#endif
	}

#ifdef DRAW_SCENARIO
	glPushMatrix();
	{
		if( spring_scenario )
		{
			glTranslatef( 0.0f, 5.0f, 0.0f );
		}
		else
		{
			glTranslatef( 0.0f, -100.0f, 0.0f );
		}
		scenario->draw();
	}
	glPopMatrix();
#endif

#ifdef DRAW_OBSTACLES
    obstacle_manager->draw( view_mat );
#endif


	if( showstats )
	{
		axes->draw();
		camera1->draw();
		camera2->draw();
	}
}
//
//=======================================================================================
//
void display( void )
{
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	fbo_manager->fbos[crowd_manager->getCrowds()[0]->getFboLodName()].fbo->Bind();
	{
		fbo_manager->setTarget( crowd_manager->getCrowds()[0]->getFboPosTexName(), true, true );
		fbo_manager->proj_manager->setOrthoProjection(
			crowd_manager->getCrowds()[0]->getWidth(),
			crowd_manager->getCrowds()[0]->getHeight(),
			true										);
		glsl_manager->activate( "tbo" );
		{
			glActiveTexture( GL_TEXTURE0 );
			glBindTexture( GL_TEXTURE_BUFFER, crowd_manager->getCrowds()[0]->getCudaTBO() );
			glBindBuffer( GL_ARRAY_BUFFER, crowd_manager->getCrowds()[0]->getPosTexCoords() );
			{
				glEnableClientState			( GL_VERTEX_ARRAY												);
				glVertexPointer				( 4, GL_FLOAT, sizeof(Vertex), LOCATION_OFFSET					);
				glDrawArrays				( GL_POINTS, 0, crowd_manager->getCrowds()[0]->getPosTexSize()	);
				glDisableClientState		( GL_VERTEX_ARRAY												);
			}
			glBindBuffer( GL_ARRAY_BUFFER, 0 );
			glActiveTexture( GL_TEXTURE0 );
			glBindTexture( GL_TEXTURE_BUFFER, 0 );
		}
		glsl_manager->deactivate( "tbo" );
		fbo_manager->proj_manager->restoreProjection();
	}
	fbo_manager->fbos[crowd_manager->getCrowds()[0]->getFboLodName()].fbo->Disable();

	fbo_manager->fbos["mdp_fbo"].fbo->Bind();
	{
		fbo_manager->setTarget( "mdp_tex", true, true );
		fbo_manager->proj_manager->setOrthoProjection(	fbo_manager->fbos["mdp_fbo"].fbo_width,
														fbo_manager->fbos["mdp_fbo"].fbo_height,
														true									);
		glsl_manager->activate( "mdp_floor" );
		{
			glActiveTexture( GL_TEXTURE0 );
			glBindTexture( GL_TEXTURE_RECTANGLE, obstacle_manager->getPolicyTextureId( 0 ) );
			glActiveTexture( GL_TEXTURE1 );
			glBindTexture( GL_TEXTURE_RECTANGLE, obstacle_manager->getDensityTextureId( 0 ) );
			glActiveTexture( GL_TEXTURE2 );
			glBindTexture( GL_TEXTURE_2D_ARRAY, obstacle_manager->getArrowsTextureId() );
			glActiveTexture( GL_TEXTURE3 );
			glBindTexture( GL_TEXTURE_2D, obstacle_manager->getLayer0TextureId() );
			glActiveTexture( GL_TEXTURE4 );
			glBindTexture( GL_TEXTURE_2D, obstacle_manager->getLayer1TextureId() );
			glBindBuffer( GL_ARRAY_BUFFER, obstacle_manager->getMdpTexCoordsId() );
			{
				glEnableClientState			( GL_VERTEX_ARRAY										);
				glVertexPointer				( 4, GL_FLOAT, sizeof(Vertex), LOCATION_OFFSET			);
				glEnableClientState			( GL_TEXTURE_COORD_ARRAY								);
				glTexCoordPointer			( 2, GL_FLOAT, sizeof(Vertex), TEXTURE_OFFSET			);
				glDrawArrays				( GL_POINTS, 0, obstacle_manager->getMdpTexCoordsSize()	);
				glDisableClientState		( GL_TEXTURE_COORD_ARRAY								);
				glDisableClientState		( GL_VERTEX_ARRAY										);
			}
			glBindBuffer( GL_ARRAY_BUFFER, 0 );
			glActiveTexture( GL_TEXTURE0 );
			glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
			glActiveTexture( GL_TEXTURE1 );
			glBindTexture( GL_TEXTURE_RECTANGLE, 0 );
			glActiveTexture( GL_TEXTURE2 );
			glBindTexture( GL_TEXTURE_2D_ARRAY, 0 );
			glActiveTexture( GL_TEXTURE3 );
			glBindTexture( GL_TEXTURE_2D, 0 );
			glActiveTexture( GL_TEXTURE4 );
			glBindTexture( GL_TEXTURE_2D, 0 );
		}
		glsl_manager->deactivate( "mdp_floor" );
		fbo_manager->proj_manager->restoreProjection();
	}
	fbo_manager->fbos["mdp_fbo"].fbo->Disable();

	fbo_manager->fbos["display_fbo"].fbo->Bind();
	{
		camera->setView();
		fbo_manager->setTarget( "display_tex", true, true );
		draw();

		glPushAttrib( GL_LIGHTING_BIT );
		{
			glDisable(GL_LIGHTING);
			if( showstats )
			{
				GLint viewp[4];
				glGetIntegerv( GL_VIEWPORT, &viewp[0] );
				int wScene = viewp[2];
				int hScene = viewp[3];

				fbo_manager->proj_manager->setOrthoProjection(	0,
																0,
																hScene/4,
																hScene/4,
																0,
																crowd_manager->getCrowds()[0]->getWidth(),
																0,
																crowd_manager->getCrowds()[0]->getHeight(),
																true										);
				fbo_manager->pushActiveBind( crowd_manager->getCrowds()[0]->getFboPosTexName(), 0 );
				fbo_manager->renderQuad(	crowd_manager->getCrowds()[0]->getFboLodName(),
											str_pass_rect,
											crowd_manager->getCrowds()[0]->getWidth(),
											crowd_manager->getCrowds()[0]->getHeight() );
				fbo_manager->popActiveBind();
				fbo_manager->proj_manager->restoreProjection();
			}

			fbo_manager->pushActiveBind( "mdp_tex", 0 );
			glsl_manager->activate( "pass_rect" );
			float tw = (float)fbo_manager->fbos["mdp_fbo"].fbo_width;
			float th = (float)fbo_manager->fbos["mdp_fbo"].fbo_height;
			glBegin( GL_QUADS );
			{
				glTexCoord2f( 0.0f,	0.0f	);	glVertex3f( -PLANE_SCALE,	0.0f,	 PLANE_SCALE	);
				glTexCoord2f( tw,	0.0f	);	glVertex3f(  PLANE_SCALE,	0.0f,	 PLANE_SCALE	);
				glTexCoord2f( tw,	th		);	glVertex3f(	 PLANE_SCALE,	0.0f,	-PLANE_SCALE	);
				glTexCoord2f( 0.0f,	th		);	glVertex3f( -PLANE_SCALE,	0.0f,	-PLANE_SCALE	);
			}
			glEnd();
			glsl_manager->deactivate( "pass_rect" );
			fbo_manager->popActiveBind();
		}
		glPopAttrib();

		if( showstats )
		{
			glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
			{
				glDisable( GL_LIGHTING );
				glDisable( GL_TEXTURE_2D );
				ProjectionManager::displayText( 10,  20, (char*)str_fps.c_str()				);
				ProjectionManager::displayText( 10,  40, (char*)str_delta_time.c_str()		);
				ProjectionManager::displayText( 10,  60, (char*)str_culled.c_str()			);
				ProjectionManager::displayText( 10, 100, (char*)str_lod1.c_str()			);
				ProjectionManager::displayText( 10, 120, (char*)str_lod2.c_str()			);
				ProjectionManager::displayText( 10, 140, (char*)str_lod3.c_str()			);
				ProjectionManager::displayText( 10, 160, (char*)str_racing_qw.c_str()		);
				ProjectionManager::displayText( 10, 180, (char*)str_scatter_gather.c_str()	);
			}
			glPopAttrib();
		}
	}
	fbo_manager->fbos["display_fbo"].fbo->Disable();

	fbo_manager->displayTexture( "pass_rect",
								 "display_tex",
								 fbo_manager->fbos["display_fbo"].fbo_width,
								 fbo_manager->fbos["display_fbo"].fbo_height );
	glFlush();
	glutSwapBuffers();
}
//
//=======================================================================================
//
void reshape( int w, int h )
{
	glViewport( 0, 0, (GLsizei) w, (GLsizei) h );
	if( !fbo_manager->updateFboDims( str_display_fbo, (unsigned int) w, (unsigned int) h ) )
	{
		log_manager->log( LogManager::LERROR, "FBO update failed!" );
		cleanup();
		exit( 1 );
	}
	else
	{
		log_manager->log( LogManager::FBO_MANAGER, "Reshaped to %ix%i", w, h );
	}
	glsl_manager->activate( "vfc_lod_assignment" );
	{
		glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_tang.c_str(),      	camera->getFrustum()->TANG			);
		glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_farPlane.c_str(),  	camera->getFrustum()->getFarD()		);
		glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_nearPlane.c_str(), 	camera->getFrustum()->getNearD()	);
		glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_ratio.c_str(),		camera->getFrustum()->getRatio()	);
	}
	glsl_manager->deactivate( "vfc_lod_assignment" );

	camera1->setView();
    camera2->setView();
    camera->setView();
}
//
//=======================================================================================
//
void idle( void )
{
	Timer::getInstance()->update();
	fps = Timer::getInstance()->getFps();
	delta_time = (fps <= 1.0f ? one_thirtieth : 1.0f / fps);
	frame_counter++;
	time_counter += delta_time;

	if( time_counter > one_sixtieth )
	{
		if( runpaths )
		{
			crowd_manager->runPaths();
		}
		if( animating )
		{
			crowd_manager->nextFrame();
		}
		time_counter = 0.0f;
	}

	// EVERY 40 FRAMES, UPDATE STATS
	if( frame_counter % 40 == 0 )
	{
		str_fps = string( "FPS:        " );
		stringstream ss1;
		ss1 << fps;
		str_fps.append( ss1.str() );

		str_delta_time = string( "DELTA TIME: " );
		stringstream ss6;
		ss6 << delta_time;
		str_delta_time.append( ss6.str() );

		str_racing_qw		= string( "RQW:        " );
		str_scatter_gather	= string( "SG:         " );
		stringstream ss_rqw;
		stringstream ss_sg;
		ss_rqw << crowd_manager->getAvgRacingQW();
		ss_sg << crowd_manager->getAvgScatterGather();
		str_racing_qw.append( ss_rqw.str() );
		str_scatter_gather.append( ss_sg.str() );

		d_lod1 	= 0;
		d_lod2 	= 0;
		d_lod3 	= 0;
		d_total	= 0;

		vector<Crowd*> d_crowds = crowd_manager->getCrowds();
		for( unsigned int c = 0; c < d_crowds.size(); c++ )
		{
			Crowd* d_crowd = d_crowds[c];
			d_lod1 += d_crowd->models_drawn[0];
			d_lod2 += d_crowd->models_drawn[1];
			d_lod3 += d_crowd->models_drawn[2];
			d_total += ( d_crowd->getWidth() * d_crowd->getHeight() );
		}
		d_culled = d_total - d_lod1 - d_lod2 - d_lod3;

		str_culled = string( "CULLED:     " );
		stringstream ss_culled;
		ss_culled << d_culled;
		str_culled.append( ss_culled.str() );

		str_lod1   = string( "LOD1:       " );
		stringstream ss_lod1;

		ss_lod1 << d_lod1;
		str_lod1.append( ss_lod1.str() );

		str_lod2   = string( "LOD2:       " );
		stringstream ss_lod2;
		ss_lod2 << d_lod2;
		str_lod2.append( ss_lod2.str() );

		str_lod3   = string( "LOD3:       " );
		stringstream ss_lod3;
		ss_lod3 << d_lod3;
		str_lod3.append( ss_lod3.str() );
	}

	// EVERY 30 FRAMES, UPDATE DENSITY
	if( frame_counter % 30 == 0 )
	{
		vector<float> a_density;
		crowd_manager->getDensity( a_density, 0 );
		obstacle_manager->updateDensity( a_density, 0 );
	}

//->MDP_ITERATION
#ifdef _WIN32
	DWORD init_tickAtStart = GetTickCount();
	DWORD init_elapsedTicks;
#elif defined __unix
	timeval start, stop, result_cpu;
	gettimeofday( &start, NULL );
#endif
	float time_cpu;
	switch( obstacle_manager->getState() )
	{
		case MDPS_IDLE:
			break;
		case MDPS_READY:
			if( mdp_iterations > 0 )
			{
				float fiter = (float)mdp_iterations;
				printf( "\n--------------------------------------------------\n"								);
				printf( "MDP VALUE ITERATIONS:                 %i\n",			mdp_iterations					);
				printf( "REGULAR (NO MDP) FRAME AVG. TIME:     %011.7f ms.\n",	delta_time_avg*1000				);
				printf( "WHILE ITERATING MDP FRAME AVG. TIME:  %011.7f ms.\n",	delta_time_mdp_avg*1000/fiter	);
				float diff = delta_time_mdp_avg*1000/fiter - delta_time_avg*1000;
				printf( "MDP PROCESS OVERHEAD:                 %011.7f ms.\n",	diff							);
				printf( "--------------------------------------------------\n"									);
				printf( "INIT MDP STRUCTURES AT HOST TIME:     %011.7f ms.\n",	init_mdp_structures				);
				printf( "INIT MDP PERMUTATIONS VECTOR TIME:    %011.7f ms.\n",	init_mdp_perms					);
				printf( "UPLOAD TO DEVICE TIME:                %011.7f ms.\n",	uploading_mdp					);
				printf( "TOTAL ITERATING TIME:                 %011.7f ms.\n",	total_mdp_iteration				);
				printf( "MDP ITERATION AVG. TIME:              %011.7f ms.\n",	iterating_mdp_avg/fiter			);
				printf( "DOWNLOAD TO HOST TIME:                %011.7f ms.\n",	downloading_mdp					);
				printf( "POLICY UPDATE TIME:                   %011.7f ms.\n",	update_mdp_policy				);
				printf( "--------------------------------------------------\n"									);
				printf( "TOTAL MDP PROCESS TIME:               %011.7f ms.\n",	total_mdp_process				);
				mdp_iterations		= 0;
				delta_time_avg		= 0.0f;
				delta_time_mdp_avg	= 0.0f;
				init_mdp_structures = 0.0f;
				init_mdp_perms		= 0.0f;
				uploading_mdp		= 0.0f;
				iterating_mdp_avg	= 0.0f;
				downloading_mdp		= 0.0f;
				update_mdp_policy	= 0.0f;
				total_mdp_process	= 0.0f;
				total_mdp_iteration = 0.0f;
			}
			else
			{
				delta_time_avg = (mdp_iterations == 0 ? delta_time : (delta_time_avg + delta_time) / 2.0f);
			}
			break;
		case MDPS_ERROR:
			break;
		case MDPS_INIT_STRUCTURES_ON_HOST:
			obstacle_manager->initStructuresOnHost();
#ifdef _WIN32
			init_elapsedTicks = GetTickCount() - init_tickAtStart;
			time_cpu = (float)init_elapsedTicks;
#elif defined __unix
			gettimeofday( &stop, NULL );
			timersub( &stop, &start, &result_cpu );
			time_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
			delta_time_mdp_avg += delta_time;
			init_mdp_structures = time_cpu;
			total_mdp_process += time_cpu;
			//printf( "HOST_INITED_STRUCTURES_IN: %010.7fms.\n", time_cpu );
			break;
		case MDPS_INIT_PERMS_ON_DEVICE:
			obstacle_manager->initPermsOnDevice();
#ifdef _WIN32
			init_elapsedTicks = GetTickCount() - init_tickAtStart;
			time_cpu = (float)init_elapsedTicks;
#elif defined __unix
			gettimeofday( &stop, NULL );
			timersub( &stop, &start, &result_cpu );
			time_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
			delta_time_mdp_avg += delta_time;
			init_mdp_perms = time_cpu;
			total_mdp_process += time_cpu;
			//printf( "DEVICE_INITED_PERMS_IN:    %010.7fms.\n", time_cpu );
			break;
		case MDPS_UPLOADING_TO_DEVICE:
			obstacle_manager->uploadToDevice();
#ifdef _WIN32
			init_elapsedTicks = GetTickCount() - init_tickAtStart;
			time_cpu = (float)init_elapsedTicks;
#elif defined __unix
			gettimeofday( &stop, NULL );
			timersub( &stop, &start, &result_cpu );
			time_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
			delta_time_mdp_avg += delta_time;
			uploading_mdp = time_cpu;
			total_mdp_process += time_cpu;
			//printf( "UPLOADED_TO_DEVICE_IN:     %010.7fms.\n", time_cpu );
			break;
		case MDPS_ITERATING_ON_DEVICE:
			obstacle_manager->iterateOnDevice();
#ifdef _WIN32
			init_elapsedTicks = GetTickCount() - init_tickAtStart;
			time_cpu = (float)init_elapsedTicks;
#elif defined __unix
			gettimeofday( &stop, NULL );
			timersub( &stop, &start, &result_cpu );
			time_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
			delta_time_mdp_avg += delta_time;
			iterating_mdp_avg += time_cpu;
			mdp_iterations++;
			total_mdp_iteration += time_cpu;
			total_mdp_process += time_cpu;
			//printf( "DEVICE_ITERATED_IN:        %010.7fms.\n", time_cpu );
			break;
		case MDPS_DOWNLOADING_TO_HOST:
			obstacle_manager->downloadToHost();
#ifdef _WIN32
			init_elapsedTicks = GetTickCount() - init_tickAtStart;
			time_cpu = (float)init_elapsedTicks;
#elif defined __unix
			gettimeofday( &stop, NULL );
			timersub( &stop, &start, &result_cpu );
			time_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
			delta_time_mdp_avg += delta_time;
			downloading_mdp = time_cpu;
			total_mdp_process += time_cpu;
			//printf( "DOWNLOADED_TO_HOST_IN:     %010.7fms.\n", time_cpu );
			break;
		case MDPS_UPDATING_POLICY:
			obstacle_manager->updatePolicy();
			if( obstacle_manager->getActiveMDPLayer() == MDP_CHANNELS )
			{
				crowd_manager->updatePolicies( obstacle_manager->getPolicies() );
			}
#ifdef _WIN32
			init_elapsedTicks = GetTickCount() - init_tickAtStart;
			time_cpu = (float)init_elapsedTicks;
#elif defined __unix
			gettimeofday( &stop, NULL );
			timersub( &stop, &start, &result_cpu );
			time_cpu = (float)result_cpu.tv_sec*1000.0f+(float)result_cpu.tv_usec/1000.0f;
#endif
			delta_time_mdp_avg += delta_time;
			update_mdp_policy = time_cpu;
			total_mdp_process += time_cpu;
			//printf( "POLICY_UPDATED_IN:         %010.7fms.\n", time_cpu );
			break;
	}
//<-MDP_ITERATION

    glutPostRedisplay();
}
//
//=======================================================================================
//
void cleanup( void )
{
	log_manager->separator			(												);
	log_manager->log				( LogManager::EXIT, "CASIM shutting down..." 	);
    glutIdleFunc					( NULL 											);
    glFlush							(												);
    Timer::freeInstance				(												);
    log_manager->log				( LogManager::EXIT, "Clearing TEXTURES..." 		);
    TextureManager::freeInstance	(												);
    log_manager->log				( LogManager::EXIT, "Clearing MANAGERS..." 		);
	FREE_INSTANCE					( err_manager  									);
	FREE_INSTANCE					( vbo_manager									);
	FREE_INSTANCE					( fbo_manager									);
	FREE_INSTANCE					( glsl_manager									);
	FREE_INSTANCE					( crowd_manager									);
	FREE_INSTANCE					( skybox_manager								);
	log_manager->log				( LogManager::EXIT, "Clearing MODELS..." 		);
	FREE_INSTANCE					( camera1      									);
	FREE_INSTANCE					( camera2      									);
	FREE_INSTANCE					( axes         									);
	log_manager->log				( LogManager::EXIT, "Memory free. Exiting." 	);
	log_manager->console_separator	(												);
	FREE_INSTANCE					( log_manager  									);
}
//
//=======================================================================================
//
//->KEYBOARD_AND_MOUSE_FUNCTIONS
#include "cPeripherals.h"
//<-KEYBOARD_AND_MOUSE_FUNCTIONS
//
//=======================================================================================
//
int main( int argc, char *argv[] )
{
#ifdef _WIN32
	srand						(	(unsigned int)time( NULL )	);
#elif defined __unix
    srandom						( 	getpid() 					);
#endif
    string log_file				( 	"CASIM_LOG.html" 			);
    log_manager = new LogManager( 	log_file 					);

    init						( 	argc,
    								argv 						);

    err_manager->getError		( "BEFORE_RENDERING::" 			);
    glutReshapeFunc 			( reshape   					);
    glutDisplayFunc 			( display   					);
    glutKeyboardFunc			( keyboard  					);
    glutSpecialFunc 			( special   					);
	glutMouseFunc				( mouse							);
	glutMotionFunc				( motion						);
    glutIdleFunc    			( idle      					);
    glutCloseFunc   			( cleanup   					);
    glutMainLoop    			(           					);

    return 0;
}
//
//=======================================================================================
