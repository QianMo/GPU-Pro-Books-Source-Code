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

//=======================================================================================
//
void keyboard( unsigned char key, int x, int y )
{
	//printf( "KEY=%d\n", key );
	int pdi     = 0;
	vec3 camPos = vec3( 0.0f );
	vec3 camDir = vec3( 0.0f );
	vec3 camUp  = vec3( 0.0f );
	switch( key )
	{
#ifdef DEMO_SHADER
	case 49:
		doHeightAndDisplacement = !doHeightAndDisplacement;
		break;
	case 50:
		doColor = !doColor;
		break;
	case 51:
		doPatterns = !doPatterns;
		break;
	case 52:
		doFacial = !doFacial;
		break;
#endif
	case 43:	//'+'
		predef_cam_index = (predef_cam_index + 1) % predef_cam_pos.size();
		camPos = vec3(	predef_cam_pos[predef_cam_index].x,
                        predef_cam_pos[predef_cam_index].y,
                        predef_cam_pos[predef_cam_index].z );
		camDir = vec3(  predef_cam_dir[predef_cam_index].x,
                        predef_cam_dir[predef_cam_index].y,
                        predef_cam_dir[predef_cam_index].z );
		camUp  = vec3(  predef_cam_up[predef_cam_index].x,
                        predef_cam_up[predef_cam_index].y,
                        predef_cam_up[predef_cam_index].z );
        camera->setPosition( camPos );
        camera->setDirection( camDir );
        camera->setUpVec( camUp );
        camera->setView();
		break;
	case 45:	//'-'
		pdi = (int)predef_cam_index;
		if( (pdi-1) >= 0  )
		{
			predef_cam_index = predef_cam_index - 1;
		}
		else
		{
			predef_cam_index = predef_cam_pos.size() - 1;
		}
		camPos = vec3(	predef_cam_pos[predef_cam_index].x,
                        predef_cam_pos[predef_cam_index].y,
                        predef_cam_pos[predef_cam_index].z );
		camDir = vec3(  predef_cam_dir[predef_cam_index].x,
                        predef_cam_dir[predef_cam_index].y,
                        predef_cam_dir[predef_cam_index].z );
		camUp  = vec3(  predef_cam_up[predef_cam_index].x,
                        predef_cam_up[predef_cam_index].y,
                        predef_cam_up[predef_cam_index].z );
        camera->setPosition( camPos );
        camera->setDirection( camDir );
        camera->setUpVec( camUp );
        camera->setView();
		break;
	case 19:	//'CTRL+s'
		camera->moveDown( camAccel );
		camera->setView();
		break;
	case 23:	//'CTRL+w'
		camera->moveUp( camAccel );
		camera->setView();
		break;
	case 27:
		glutExit();
		exit( 0 );
		break;
	case 'a':
		camera->moveLeft( camAccel );
		camera->setView();
		break;
	case 'c':
		if( camera == camera1 )
		{
			camera = camera2;
		}
		else
		{
			camera = camera1;
		}
		camNear = camera->getFrustum()->getNearD();
		break;
	case 'd':
		camera->moveRight( camAccel );
		camera->setView();
		break;
	case 'w':
		camera->moveForward( camAccel );
		camera->setView();
		break;
	case 's':
		camera->moveBackward( camAccel );
		camera->setView();
		break;
	case 'H':
		hideCharacters = !hideCharacters;
		break;
	case ' ':
		animating = !animating;
		break;
	case 'o':
		obstacle_manager->toggleObstacle();
		break;
	case 'e':
		obstacle_manager->toggleExit();
		break;
	case 'r':
		wireframe = !wireframe;
		break;
	case 'R':
		drawShadows = !drawShadows;
		break;
	case 'm':
		drawShadows				= !drawShadows;
		animating				= !animating;
		doHeightAndDisplacement = !doHeightAndDisplacement;
		doColor					= !doColor;
		doPatterns				= !doPatterns;
		doFacial				= !doFacial;
		runpaths				= !runpaths;

/*
		predef_cam_index = predef_cam_pos.size() - 1;
		camPos = vec3(	predef_cam_pos[predef_cam_index].x,
                        predef_cam_pos[predef_cam_index].y,
                        predef_cam_pos[predef_cam_index].z );
		camDir = vec3(  predef_cam_dir[predef_cam_index].x,
                        predef_cam_dir[predef_cam_index].y,
                        predef_cam_dir[predef_cam_index].z );
		camUp  = vec3(  predef_cam_up[predef_cam_index].x,
                        predef_cam_up[predef_cam_index].y,
                        predef_cam_up[predef_cam_index].z );
		camera->setPosition( camPos );
		camera->setDirection( camDir );
		camera->setUpVec( camUp );
		camera->setView();
*/
		break;
	case '.':
		policy_floor = !policy_floor;
		glsl_manager->activate( "mdp_floor" );
		{
			if( policy_floor )
			{
				glsl_manager->setUniformf( "mdp_floor", (char*)str_policy_on.c_str(), 1.0f );
			}
			else
			{
				glsl_manager->setUniformf( "mdp_floor", (char*)str_policy_on.c_str(), 0.0f );
			}
		}
		glsl_manager->deactivate( "mdp_floor" );
		break;
	case ',':
		density_floor = !density_floor;
		glsl_manager->activate( "mdp_floor" );
		{
			if( density_floor )
			{
				glsl_manager->setUniformf( "mdp_floor", (char*)str_density_on.c_str(), 1.0f );
			}
			else
			{
				glsl_manager->setUniformf( "mdp_floor", (char*)str_density_on.c_str(), 0.0f );
			}
		}
		glsl_manager->deactivate( "mdp_floor" );
		break;
	case 'S':
		showstats = !showstats;
		break;
	case 'z':
		spring_scenario = !spring_scenario;
		break;
	}
}
//
//=======================================================================================
//
void special( int key, int x, int y )
{
	//printf( "SKEY=%d\n", key );
	glut_mod = glutGetModifiers();
	switch( key )
	{
		case GLUT_KEY_LEFT:
			if( glut_mod == GLUT_ACTIVE_CTRL )
			{
				obstacle_manager->moveCursorLeft();
			}
			else
			{
				camera->moveLeft( camAccel * 5.0f );
			}
			break;
		case GLUT_KEY_RIGHT:
			if( glut_mod == GLUT_ACTIVE_CTRL )
			{
				obstacle_manager->moveCursorRight();
			}
			else
			{
				camera->moveRight( camAccel * 5.0f );
			}
			break;
		case GLUT_KEY_UP:
			if( glut_mod == GLUT_ACTIVE_CTRL )
			{
				obstacle_manager->moveCursorUp();
			}
			else
			{
				camera->moveForward( camAccel * 5.0f );
			}
			break;
		case GLUT_KEY_DOWN:
			if( glut_mod == GLUT_ACTIVE_CTRL )
			{
				obstacle_manager->moveCursorDown();
			}
			else
			{
				camera->moveBackward( camAccel * 5.0f );
			}
			break;
	}
}
//
//=======================================================================================
//
void mouse( int button, int state, int x, int y )
{
	if( state == GLUT_DOWN )
	{
		if( button == GLUT_LEFT_BUTTON )
		{
			lMouseDown = true;
			rMouseDown = false;
			mMouseDown = false;
		}
		else if( button == GLUT_RIGHT_BUTTON )
		{
			lMouseDown = false;
			rMouseDown = true;
			mMouseDown = false;
		}
		else if( button ==  GLUT_MIDDLE_BUTTON  )
		{
			lMouseDown = false;
			rMouseDown = false;
			mMouseDown = true;
		}
	}
}
//
//=======================================================================================
//
void motion( int x, int y )
{
    vec3 up( 0.0f, 1.0f, 0.0f );
    vec3 ri( 1.0f, 0.0f, 0.0f );

	int modifiers = glutGetModifiers();
	if( modifiers & GLUT_ACTIVE_CTRL )
	{
		//printf( "GLUT_ACTIVE_CTRL\n" );
		ctrlDown = true;
	}
	else
	{
		ctrlDown = false;
	}

	static int xlast = -1, ylast = -1;
	int dx, dy;

	dx = x - xlast;
	dy = y - ylast;
	if( lMouseDown )
	{
		if( ctrlDown == false )
		{
			if( dx > 0 )
			{
				camera->rotateAngle(  camAccel/4.0f, up );
			}
			else if( dx < 0 )
			{
				camera->rotateAngle( -camAccel/4.0f, up );
			}
		}
		else
		{
			if( dy > 0 )
			{
				camera->rotateAngle( -camAccel/4.0f, ri );
			}
			else if( dy < 0 )
			{
				camera->rotateAngle(  camAccel/4.0f, ri );
			}
		}
	}
	else if( rMouseDown )
	{
		if( dy > 0 )
		{
			camera->rotateAngle( -camAccel/4.0f, ri );
		}
		else if( dy < 0 )
		{
			camera->rotateAngle(  camAccel/4.0f, ri );
		}
	}
	xlast = x;
	ylast = y;
}
//
//=======================================================================================
