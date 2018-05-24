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
#include "cStaticLod.h"

//=======================================================================================
//
StaticLod::StaticLod(	GlslManager*	glsl_manager_,
						VboManager*		vbo_manager_,
						GlErrorManager* err_manager_,
						string			owner_name_		)
{
	glsl_manager			= glsl_manager_;
	vbo_manager				= vbo_manager_;
	err_manager				= err_manager_;
	owner_name				= string( owner_name_ );
	posVBOLODs				= 0;
	posTexBufferId			= 0;
	numElements				= 0;
	numAgentsWidth			= 0;
	numAgentsHeight			= 0;
	primitivesWritten[0]	= 0;
	primitivesWritten[1]	= 0;
	primitivesWritten[2]	= 0;
	primitivesWritten[3]	= 0;
	primitivesWritten[4]	= 0;
	primitivesGenerated[0]	= 0;
	primitivesGenerated[1]	= 0;
	primitivesGenerated[2]	= 0;
	primitivesGenerated[3]	= 0;
	primitivesGenerated[4]	= 0;
	lodAid					= glsl_manager->getId( "vfc_lod_assignment" );
	lodSid					= glsl_manager->getId( "vfc_lod_selection"  );
	locAid[0]				= glGetVaryingLocationNV( lodAid, "gl_Position" );
	locSid[0]				= glGetVaryingLocationNV( lodSid, "gl_Position" );
	tc_index				= 0;
	tc_frame				= 0;
	tc_size					= 0;
	texCoords				= 0;
	query_written			= 0;
	query_generated			= 0;
	numLODs					= 0;

	str_tang				= string( "tang"		);
	str_AGENTS_NPOT			= string( "AGENTS_NPOT"	);
	str_nearPlane			= string( "nearPlane"	);
	str_farPlane			= string( "farPlane"	);
	str_ratio				= string( "ratio"		);
	str_X					= string( "X"			);
	str_Y					= string( "Y"			);
	str_Z					= string( "Z"			);
	str_camPos				= string( "camPos"		);
	str_lod					= string( "lod"			);
	str_groupId				= string( "groupId"		);
}
//
//=======================================================================================
//
StaticLod::~StaticLod( void )
{
	glDeleteQueries( 1, &query_generated );
	glDeleteQueries( 1, &query_written );
}
//
//=======================================================================================
//
void StaticLod::init( unsigned int numAgentsWidth_, unsigned int numAgentsHeight_ )
{
	unsigned int i;
	numAgentsWidth  = numAgentsWidth_;
	numAgentsHeight = numAgentsHeight_;
	numLODs			= NUM_LOD;
	texCoords		= 0;
	numElements		= numLODs + 1;
	posVBOLODs		= new unsigned int[numElements];
	posTexBufferId	= new unsigned int[numElements];
	for( i = 0; i < numElements; i++ )
	{
		posVBOLODs[ i ]			= 0;
		posTexBufferId[ i ]		= 0;
		err_manager->getError( "BEGIN : StaticLod::StaticLod VBO" );

		glGenBuffers	(	1, &posVBOLODs[i]											);
		glBindBuffer	(	GL_ARRAY_BUFFER, posVBOLODs[i]								);
		glBufferData	(	GL_ARRAY_BUFFER,
							sizeof(float) * numAgentsWidth * numAgentsHeight * 4,
							NULL,
							GL_STATIC_DRAW												);
		glBindBuffer	(	GL_ARRAY_BUFFER, 0											);

		glGenTextures	( 1, &posTexBufferId[i]											);
		glActiveTexture ( GL_TEXTURE3													);
		glBindTexture	( GL_TEXTURE_BUFFER, posTexBufferId[i]							);
		glTexBuffer		( GL_TEXTURE_BUFFER, GL_RGBA32F, posVBOLODs[i]					);
		glBindTexture	( GL_TEXTURE_BUFFER, 0											);


		err_manager->getError( "END : StaticLod::StaticLod VBO" );
	}
	//glGenQueries( 1, &query_generated );
	glGenQueries( 1, &query_written   );
}
//
//=======================================================================================
//
unsigned int StaticLod::runAssignmentAndSelection(	unsigned int		target,
													unsigned int		posID,
													struct sVBOLod*		vboLOD,
													Camera*				cam		)
{
	unsigned int not_culled = 0;
	unsigned int i = 0;


	for( i = 0; i < numElements; i++ )
		primitivesWritten[i] = 0;

	err_manager->getError( "StaticLod VFC & LOD Assignment :BEGIN" );
	glTransformFeedbackVaryingsNV( lodAid, 1, locAid, GL_SEPARATE_ATTRIBS_NV );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( target, posID );
	{
		glsl_manager->activate( "vfc_lod_assignment" );
		{
			if ( !texCoords )
			{
				initTexCoords( target );
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_tang.c_str(),		cam->getFrustum()->TANG			);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_farPlane.c_str(),	cam->getFrustum()->getFarD()	);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_nearPlane.c_str(),	cam->getFrustum()->getNearD()	);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_ratio.c_str(),		cam->getFrustum()->getRatio()	);
			}

			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_X.c_str(),			&cam->getFrustum()->X[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Y.c_str(),			&cam->getFrustum()->Y[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Z.c_str(),			&cam->getFrustum()->Z[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_camPos.c_str(),	&cam->getPosition()[0],   3 );


			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[VBO_CULLED]);
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );
				vbo_manager->render_vbo( texCoords, vbo_manager->vbos[tc_index][tc_frame].vertices.size(), GL_POINTS );
				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[VBO_CULLED]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[VBO_CULLED] );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_assignment" );
	}
	glBindTexture( target, 0 );
	err_manager->getError( "StaticLod VFC & LOD Assignment :END" );

	// numLODs passes, waiting for multiple transform feedback targets extension
	glTransformFeedbackVaryingsNV( lodSid, 1, locSid, GL_SEPARATE_ATTRIBS_NV );
	for( i = 0; i < numLODs; i++ )
	{
		//err_manager->getError( "StaticLod LOD  Selection :BEGIN" );
		glsl_manager->activate( "vfc_lod_selection" );
		{
			glsl_manager->setUniformf( "vfc_lod_selection", (char*)str_lod.c_str(), i + 1.0f );

			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[i+1]);
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );

				glBindBuffer( GL_ARRAY_BUFFER, posVBOLODs[VBO_CULLED] );
				{
					glEnableClientState			( GL_VERTEX_ARRAY									);
					glVertexPointer				( 4, GL_FLOAT, 0, 0									);
					glDrawArrays				( GL_POINTS, 0, primitivesWritten[VBO_CULLED]		);
					glDisableClientState		( GL_VERTEX_ARRAY									);
				}
				glBindBuffer( GL_ARRAY_BUFFER, 0 );

				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[i+1]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[i+1] );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_selection" );
		//err_manager->getError( "StaticLod LOD  Selection :END" );
		vboLOD[i].id = posTexBufferId[i+1];
		vboLOD[i].primitivesWritten   = primitivesWritten[i+1];
		vboLOD[i].primitivesGenerated = primitivesGenerated[i+1];
		not_culled += primitivesWritten[i+1];
	}
	//printf ("primitivesWritten[VBO_CULLED]=%d vboLOD[0] = %d, vboLOD[1] = %d, vboLOD[2] = %d \n",primitivesWritten[VBO_CULLED], vboLOD[0].primitivesWritten, vboLOD[1].primitivesWritten, vboLOD[2].primitivesWritten);
	return not_culled;
}
//
//=======================================================================================
//
unsigned int StaticLod::runAssignmentAndSelection(	unsigned int		target,
													unsigned int		posID,
													unsigned int		agentsIdsTexture,
													struct sVBOLod*		vboLOD,
													Camera*				cam				)
{
	unsigned int not_culled = 0;
	unsigned int i = 0;

	for( i = 0; i < numElements; i++ )
		primitivesWritten[i] = 0;

	err_manager->getError( "StaticLod VFC & LOD Assignment :BEGIN" );
	glTransformFeedbackVaryingsNV( lodAid, 1, locAid, GL_SEPARATE_ATTRIBS_NV );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( target, posID );
	glActiveTexture( GL_TEXTURE2 );
	glBindTexture( target, agentsIdsTexture );
	{
		glsl_manager->activate( "vfc_lod_assignment" );
		{
			if ( !texCoords )
			{
				initTexCoords( target );
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_tang.c_str(),      	cam->getFrustum()->TANG			);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_farPlane.c_str(),  	cam->getFrustum()->getFarD()	);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_nearPlane.c_str(), 	cam->getFrustum()->getNearD()	);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_ratio.c_str(),		cam->getFrustum()->getRatio()	);
			}

			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_X.c_str(),			&cam->getFrustum()->X[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Y.c_str(),			&cam->getFrustum()->Y[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Z.c_str(),			&cam->getFrustum()->Z[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_camPos.c_str(),	&cam->getPosition()[0],   3 );

			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[VBO_CULLED]);
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );
				vbo_manager->render_vbo( texCoords, vbo_manager->vbos[tc_index][tc_frame].vertices.size(), GL_POINTS );
				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[VBO_CULLED]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[VBO_CULLED] );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_assignment" );
	}
	glActiveTexture( GL_TEXTURE2 );
	glBindTexture( target, 0 );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( target, 0 );
	err_manager->getError( "StaticLod VFC & LOD Assignment :END" );

	// numLODs passes, waiting for multiple transform feedback targets extension
	glTransformFeedbackVaryingsNV( lodSid, 1, locSid, GL_SEPARATE_ATTRIBS_NV );
	for( i = 0; i < numLODs; i++ )
	{
		//err_manager->getError( "StaticLod LOD  Selection :BEGIN" );
		glsl_manager->activate( "vfc_lod_selection" );
		{
			glsl_manager->setUniformf( "vfc_lod_selection", (char*)str_lod.c_str(), i + 1.0f );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[i+1]);
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );

				glBindBuffer( GL_ARRAY_BUFFER, posVBOLODs[VBO_CULLED] );
				{
					glEnableClientState			( GL_VERTEX_ARRAY									);
					glVertexPointer				( 4, GL_FLOAT, 0, 0									);
					glDrawArrays				( GL_POINTS, 0, primitivesWritten[VBO_CULLED]		);
					glDisableClientState		( GL_VERTEX_ARRAY									);
				}
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[i+1]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[i+1] );

			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_selection" );
		//err_manager->getError( "StaticLod LOD  Selection :END" );
		vboLOD[i].id = posTexBufferId[i+1];
		vboLOD[i].primitivesWritten   = primitivesWritten[i+1];
		vboLOD[i].primitivesGenerated = primitivesGenerated[i+1];
		not_culled += primitivesWritten[i+1];
	}
	//printf ("primitivesWritten[VBO_CULLED]=%d vboLOD[0] = %d, vboLOD[1] = %d, vboLOD[2] = %d \n",primitivesWritten[VBO_CULLED], vboLOD[0].primitivesWritten, vboLOD[1].primitivesWritten, vboLOD[2].primitivesWritten);
	return not_culled;
}
//
//=======================================================================================
//
void StaticLod::runAssignment(	unsigned int		target,
								unsigned int		posID,
								unsigned int		agentsIdsTexture,
								struct sVBOLod*		vboLOD,
								Camera*				cam				)
{
	for( unsigned int i = 0; i < numElements; i++ )
	{
		primitivesWritten[i] = 0;
	}

	err_manager->getError( "StaticLod VFC & LOD Assignment :BEGIN" );
	glTransformFeedbackVaryingsNV( lodAid, 1, locAid, GL_SEPARATE_ATTRIBS_NV );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( target, posID );
	glActiveTexture( GL_TEXTURE2 );
	glBindTexture( target, agentsIdsTexture );
	{
		glsl_manager->activate( "vfc_lod_assignment" );
		{
			if ( !texCoords )
			{
				initTexCoords( target );
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_tang.c_str(),      	cam->getFrustum()->TANG			);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_farPlane.c_str(),  	cam->getFrustum()->getFarD()	);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_nearPlane.c_str(), 	cam->getFrustum()->getNearD()	);
				glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_ratio.c_str(),		cam->getFrustum()->getRatio()	);
			}

			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_X.c_str(),			&cam->getFrustum()->X[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Y.c_str(),			&cam->getFrustum()->Y[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Z.c_str(),			&cam->getFrustum()->Z[0], 3 );
			glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_camPos.c_str(),	&cam->getPosition()[0],   3 );


			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[VBO_CULLED]);
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );
				vbo_manager->render_vbo( texCoords, vbo_manager->vbos[tc_index][tc_frame].vertices.size(), GL_POINTS );
				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[VBO_CULLED]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[VBO_CULLED] );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_assignment" );
	}
	glActiveTexture( GL_TEXTURE2 );
	glBindTexture( target, 0 );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( target, 0 );
	err_manager->getError( "StaticLod VFC & LOD Assignment :END" );
}
//
//=======================================================================================
//
void StaticLod::runAssignmentCuda(	unsigned int		target,
									unsigned int		posBufferId,
									unsigned int		agentsIdsTexture,
									struct sVBOLod*		vboLOD,
									Camera*				cam				)
{
	for( unsigned int i = 0; i < numElements; i++ )
	{
		primitivesWritten[i] = 0;
	}

	err_manager->getError( "StaticLod VFC & LOD Assignment :BEGIN" );
	glTransformFeedbackVaryingsNV( lodAid, 1, locAid, GL_SEPARATE_ATTRIBS_NV );
	glActiveTexture( GL_TEXTURE7 );
	glBindTexture( GL_TEXTURE_BUFFER, posBufferId );
	glActiveTexture( GL_TEXTURE2 );
	glBindTexture( target, agentsIdsTexture );

	glsl_manager->activate( "vfc_lod_assignment" );
	{
		if ( !texCoords )
		{
			initTexCoordsCuda( target );
#ifdef CUDA_PATHS
			glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_AGENTS_NPOT.c_str(),	(float)numAgentsWidth		);
#endif
			glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_tang.c_str(),		cam->getFrustum()->TANG			);
			glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_farPlane.c_str(),	cam->getFrustum()->getFarD()	);
			glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_nearPlane.c_str(),	cam->getFrustum()->getNearD()	);
			glsl_manager->setUniformf( "vfc_lod_assignment", (char*)str_ratio.c_str(),		cam->getFrustum()->getRatio()	);
		}

		glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_X.c_str(),			&cam->getFrustum()->X[0], 3 );
		glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Y.c_str(),			&cam->getFrustum()->Y[0], 3 );
		glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_Z.c_str(),			&cam->getFrustum()->Z[0], 3 );
		glsl_manager->setUniformfv( "vfc_lod_assignment", (char*)str_camPos.c_str(),	&cam->getPosition()[0],   3 );

		glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[VBO_CULLED]);
		//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
		glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
		//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

		glBeginTransformFeedbackNV( GL_POINTS );
		{
			glEnable( GL_RASTERIZER_DISCARD_NV );
			glBindBuffer( GL_ARRAY_BUFFER, texCoords );
			{
				glEnableClientState			( GL_VERTEX_ARRAY									);
				glVertexPointer				( 4, GL_FLOAT, 0, 0									);
				glDrawArrays				( GL_POINTS, 0, tc_size								);
				glDisableClientState		( GL_VERTEX_ARRAY									);
			}
			glBindBuffer( GL_ARRAY_BUFFER, 0 );
			glDisable( GL_RASTERIZER_DISCARD_NV );
		}
		glEndTransformFeedbackNV();

		glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
		//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
		glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[VBO_CULLED]   );
		//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[VBO_CULLED] );
		glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
	}
	glsl_manager->deactivate( "vfc_lod_assignment" );
	glActiveTexture( GL_TEXTURE7 );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );
	glActiveTexture( GL_TEXTURE2 );
	glBindTexture( target, 0 );
	err_manager->getError( "StaticLod VFC & LOD Assignment :END" );
}
//
//=======================================================================================
//
unsigned int StaticLod::runSelection(	unsigned int		target,
										float				groupID,
										struct sVBOLod*		vboLOD,
										Camera*				cam				)
{
	unsigned int not_culled = 0;

	// numLODs passes, waiting for multiple transform feedback targets extension
	glTransformFeedbackVaryingsNV( lodSid, 1, locSid, GL_SEPARATE_ATTRIBS_NV );
	for( unsigned int i = 0; i < numLODs; i++ )
	{
		//err_manager->getError( "StaticLod LOD  Selection :BEGIN" );
		glsl_manager->activate( "vfc_lod_selection" );
		{
			glsl_manager->setUniformf( "vfc_lod_selection", (char*)str_lod.c_str(), i + 1.0f );
			glsl_manager->setUniformf( "vfc_lod_selection", (char*)str_groupId.c_str(), groupID );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[i+1] );
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );
				glBindBuffer( GL_ARRAY_BUFFER, posVBOLODs[VBO_CULLED] );
				{
					glEnableClientState			( GL_VERTEX_ARRAY									);
					glVertexPointer				( 4, GL_FLOAT, 0, 0									);
					glDrawArrays				( GL_POINTS, 0, primitivesWritten[VBO_CULLED]		);
					glDisableClientState		( GL_VERTEX_ARRAY									);
				}
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[i+1]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[i+1] );

			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_selection" );
		//err_manager->getError( "StaticLod LOD  Selection :END" );
		vboLOD[i].id = posTexBufferId[i+1]; // ----
		vboLOD[i].primitivesWritten   = primitivesWritten[i+1];
		vboLOD[i].primitivesGenerated = primitivesGenerated[i+1];
		not_culled += primitivesWritten[i+1];
	}
	//printf ("primitivesWritten[VBO_CULLED]=%d vboLOD[0] = %d, vboLOD[1] = %d, vboLOD[2] = %d \n",primitivesWritten[VBO_CULLED], vboLOD[0].primitivesWritten, vboLOD[1].primitivesWritten, vboLOD[2].primitivesWritten);
	return not_culled;
}
//
//=======================================================================================
//
unsigned int StaticLod::runSelectionCuda(	unsigned int		target,
											float				groupID,
											struct sVBOLod*		vboLOD,
											Camera*				cam		)
{
	unsigned int not_culled = 0;

	// numLODs passes, waiting for multiple transform feedback targets extension
	glTransformFeedbackVaryingsNV( lodSid, 1, locSid, GL_SEPARATE_ATTRIBS_NV );
	for( unsigned int i = 0; i < numLODs; i++ )
	{
		//err_manager->getError( "StaticLod LOD  Selection :BEGIN" );
		glsl_manager->activate( "vfc_lod_selection" );
		{
			glsl_manager->setUniformf( "vfc_lod_selection", (char*)str_lod.c_str(), i + 1.0f );
			glsl_manager->setUniformf( "vfc_lod_selection", (char*)str_groupId.c_str(), groupID );
			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, posVBOLODs[i+1] );
			//http://developer.download.nvidia.com/opengl/specs/GL_NV_transform_feedback.txt
			glBeginQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV, query_written );
			//glBeginQuery( GL_PRIMITIVES_GENERATED_NV, query_generated );

			glBeginTransformFeedbackNV( GL_POINTS );
			{
				glEnable( GL_RASTERIZER_DISCARD_NV );
				glBindBuffer( GL_ARRAY_BUFFER, posVBOLODs[VBO_CULLED] );
				{
					glEnableClientState			( GL_VERTEX_ARRAY									);
					glVertexPointer				( 4, GL_FLOAT, 0, 0									);
					glDrawArrays				( GL_POINTS, 0, primitivesWritten[VBO_CULLED]		);
					glDisableClientState		( GL_VERTEX_ARRAY									);
				}
				glBindBuffer( GL_ARRAY_BUFFER, 0 );
				glDisable( GL_RASTERIZER_DISCARD_NV );
			}
			glEndTransformFeedbackNV();

			glEndQuery( GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN_NV );
			//glEndQuery( GL_PRIMITIVES_GENERATED_NV );
			glGetQueryObjectuiv( query_written,   GL_QUERY_RESULT, &primitivesWritten[i+1]   );
			//glGetQueryObjectuiv( query_generated, GL_QUERY_RESULT, &primitivesGenerated[i+1] );

			glBindBufferBaseNV( GL_TRANSFORM_FEEDBACK_BUFFER_NV, 0, 0 );
		}
		glsl_manager->deactivate( "vfc_lod_selection" );
		//err_manager->getError( "StaticLod LOD  Selection :END" );
		vboLOD[i].id = posTexBufferId[i+1]; // ----
		vboLOD[i].primitivesWritten   = primitivesWritten[i+1];
		vboLOD[i].primitivesGenerated = primitivesGenerated[i+1];
		not_culled += primitivesWritten[i+1];
	}
	//printf ("primitivesWritten[VBO_CULLED]=%d vboLOD[0] = %d, vboLOD[1] = %d, vboLOD[2] = %d \n",primitivesWritten[VBO_CULLED], vboLOD[0].primitivesWritten, vboLOD[1].primitivesWritten, vboLOD[2].primitivesWritten);
	return not_culled;
}
//
//=======================================================================================
//
void StaticLod::initTexCoords( unsigned int target )
{
	tc_index = 0;
	tc_frame = 0;
	string vbo_name = owner_name;
	vbo_name.append( "_TEX_COORDS" );
	tc_index = vbo_manager->createVBOContainer( vbo_name, tc_frame );

	unsigned int i, j;
	for( i = 0; i < numAgentsWidth; i++ )
	{
		for( j = 0; j < numAgentsHeight; j++ )
		{
			Vertex v;
			INITVERTEX( v );
			if( target == GL_TEXTURE_2D )
			{
				v.location[0] = (float)i / (float)numAgentsWidth;
				v.location[1] = (float)j / (float)numAgentsHeight;
				v.location[2] = 0.0f;
				v.location[3] = 1.0f;
			}
			else if( target == GL_TEXTURE_RECTANGLE )
			{
				v.location[0] = (float)i;
				v.location[1] = (float)j;
				v.location[2] = 0.0f;
				v.location[3] = 1.0f;
			}
			else
			{
				v.location[0] = 0.0f;
				v.location[1] = 0.0f;
				v.location[2] = 0.0f;
				v.location[3] = 1.0f;
			}
			v.texture[0] = (float)i / (float)numAgentsWidth;
			v.texture[1] = (float)j / (float)numAgentsHeight;
			vbo_manager->vbos[tc_index][tc_frame].vertices.push_back( v );
		}
	}
	vbo_manager->gen_vbo( texCoords, tc_index, tc_frame );
	tc_size = vbo_manager->vbos[tc_index][tc_frame].vertices.size();
}
//
//=======================================================================================
//
void StaticLod::initTexCoordsCuda( unsigned int target )
{
	tc_index = 0;
	tc_frame = 0;
	string vbo_name = owner_name;
	vbo_name.append( "_TEX_COORDS_CUDA" );
	tc_index = vbo_manager->createVBOContainer( vbo_name, tc_frame );

	unsigned int i;
	unsigned int numAgents = numAgentsWidth * numAgentsHeight;
	for( i = 0; i < numAgents; i++ )
	{
		Vertex v;
		INITVERTEX( v );
		if( target == GL_TEXTURE_2D )
		{
			v.location[0] = (float)i / (float)numAgents;
			v.location[1] = 0.0f;
			v.location[2] = 0.0f;
			v.location[3] = 1.0f;
		}
		else if( target == GL_TEXTURE_RECTANGLE )
		{
			v.location[0] = (float)i;
			v.location[1] = 0.0f;
			v.location[2] = 0.0f;
			v.location[3] = 1.0f;
		}
		else
		{
			v.location[0] = 0.0f;
			v.location[1] = 0.0f;
			v.location[2] = 0.0f;
			v.location[3] = 1.0f;
		}
		v.texture[0] = (float)i / (float)numAgents;
		v.texture[1] = 0.0f;
		vbo_manager->vbos[tc_index][tc_frame].vertices.push_back( v );
	}
	vbo_manager->gen_vbo( texCoords, tc_index, tc_frame );
	tc_size = vbo_manager->vbos[tc_index][tc_frame].vertices.size();
}
//
//=======================================================================================
