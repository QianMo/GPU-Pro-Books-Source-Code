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
#include "cSkyboxManager.h"

//=======================================================================================
//
SkyboxManager::SkyboxManager( 	unsigned int	id_,
								GlslManager*	glsl_manager_,
								VboManager*		vbo_manager_,
								LogManager*		log_manager_,
								GlErrorManager*	err_manager_,
								vec3&			center_,
								vec3&			extents,
								vector<bool>&	bump,
								vector<float>&	tile,
								bool			instancing_,
								bool			lighting_		)
{
	center					= center_;
	id						= id_;
	instancing				= instancing_;
	lighting				= lighting_;
	glsl_manager   			= glsl_manager_;
	vbo_manager      		= vbo_manager_;
	log_manager				= log_manager_;
	err_manager				= err_manager_;
	WIDTH					= extents.x;
	HEIGHT					= extents.y;
	LENGTH					= extents.z;
	HWIDTH					= WIDTH  / 2.0f;
	HHEIGHT					= HEIGHT / 2.0f;
	HLENGTH					= LENGTH / 2.0f;
	bumped.push_back( bump[FRONT]  );
	bumped.push_back( bump[BACK]   );
	bumped.push_back( bump[LEFT]   );
	bumped.push_back( bump[RIGHT]  );
	bumped.push_back( bump[TOP]	   );
	bumped.push_back( bump[BOTTOM] );
	tiling.push_back( tile[FRONT]  );
	tiling.push_back( tile[BACK]   );
	tiling.push_back( tile[LEFT]   );
	tiling.push_back( tile[RIGHT]  );
	tiling.push_back( tile[TOP]	   );
	tiling.push_back( tile[BOTTOM] );

	string str_front 	= string( "SKYBOX_FRONT" );
	string str_back 	= string( "SKYBOX_BACK" );
	string str_left 	= string( "SKYBOX_LEFT" );
	string str_right 	= string( "SKYBOX_RIGHT" );
	string str_top 		= string( "SKYBOX_TOP" );
	string str_bottom 	= string( "SKYBOX_BOTTOM" );

	unsigned int front_i	= vbo_manager->createVBOContainer( str_front, 	0 );
	unsigned int back_i		= vbo_manager->createVBOContainer( str_back, 	0 );
	unsigned int left_i		= vbo_manager->createVBOContainer( str_left, 	0 );
	unsigned int right_i	= vbo_manager->createVBOContainer( str_right, 	0 );
	unsigned int top_i		= vbo_manager->createVBOContainer( str_top, 	0 );
	unsigned int bottom_i	= vbo_manager->createVBOContainer( str_bottom,	0 );
	vbo_indices.push_back( front_i  );
	vbo_indices.push_back( back_i   );
	vbo_indices.push_back( left_i   );
	vbo_indices.push_back( right_i  );
	vbo_indices.push_back( top_i    );
	vbo_indices.push_back( bottom_i );
	init();
}
//
//=======================================================================================
//
void SkyboxManager::init( void )
{
	for( unsigned int i = 0; i < 6; i++ )
	{
		vertexCounts[i]		= 0;
		texture_weights[i]	= 0;
		DIFFUSE_IDS[i]		= 0;
		BUMPMAP_IDS[i]		= 0;
		SPECMAP_IDS[i]		= 0;
	}

	vboIds.push_back( 0 );	// FRONT
	vboIds.push_back( 0 );	// BACK
	vboIds.push_back( 0 );	// LEFT
	vboIds.push_back( 0 );	// RIGHT
	vboIds.push_back( 0 );	// TOP
	vboIds.push_back( 0 );	// BOTTOM

	ivboIds.push_back( 0 );	// FRONT
	ivboIds.push_back( 0 );	// BACK
	ivboIds.push_back( 0 );	// LEFT
	ivboIds.push_back( 0 );	// RIGHT
	ivboIds.push_back( 0 );	// TOP
	ivboIds.push_back( 0 );	// BOTTOM

	gen_vbos();

	front_tile_width  = WIDTH  / tiling[FRONT];
	front_tile_height = HEIGHT / tiling[FRONT];
	for( unsigned int f = 0; f < front_positions.size() / 4; f++ )
	{
		unsigned int index = f * 4;
		WallBox tile;
		tile.center.x	= front_positions[index    ] + front_tile_width  / 2.0f;
		tile.center.y	= front_positions[index + 1] + front_tile_height / 2.0f;
		tile.center.z	= front_positions[index + 2];
		tile.halfdiag.x = front_tile_width  / 2.0f;
		tile.halfdiag.y	= front_tile_height / 2.0f;
		tile.halfdiag.z	= 2.0f;
		tile.bR			= (float)(rand() % 256) / 255.0f;
		tile.bG			= (float)(rand() % 256) / 255.0f;
		tile.bB			= (float)(rand() % 256) / 255.0f;
		front_boxes.push_back( tile );
	}

	back_tile_width  = WIDTH  / tiling[BACK];
	back_tile_height = HEIGHT / tiling[BACK];
	for( unsigned int b = 0; b < back_positions.size() / 4; b++ )
	{
		unsigned int index = b * 4;
		WallBox tile;
		tile.center.x	= back_positions[index    ] - back_tile_width  / 2.0f;
		tile.center.y	= back_positions[index + 1] + back_tile_height / 2.0f;
		tile.center.z	= back_positions[index + 2];
		tile.halfdiag.x = back_tile_width  / 2.0f;
		tile.halfdiag.y	= back_tile_height / 2.0f;
		tile.halfdiag.z	= 2.0f;
		tile.bR			= (float)(rand() % 256) / 255.0f;
		tile.bG			= (float)(rand() % 256) / 255.0f;
		tile.bB			= (float)(rand() % 256) / 255.0f;
		back_boxes.push_back( tile );
	}

	left_tile_width  = LENGTH / tiling[LEFT];
	left_tile_height = HEIGHT / tiling[LEFT];
	for( unsigned int l = 0; l < left_positions.size() / 4; l++ )
	{
		unsigned int index = l * 4;
		WallBox tile;
		tile.center.x	= left_positions[index    ];
		tile.center.y	= left_positions[index + 1] + left_tile_height / 2.0f;
		tile.center.z	= left_positions[index + 2] - left_tile_width  / 2.0f;
		tile.halfdiag.x = 2.0f;
		tile.halfdiag.y	= left_tile_height / 2.0f;
		tile.halfdiag.z	= left_tile_width  / 2.0f;
		tile.bR			= (float)(rand() % 256) / 255.0f;
		tile.bG			= (float)(rand() % 256) / 255.0f;
		tile.bB			= (float)(rand() % 256) / 255.0f;
		left_boxes.push_back( tile );
	}

	right_tile_width  = LENGTH / tiling[RIGHT];
	right_tile_height = HEIGHT / tiling[RIGHT];
	for( unsigned int r = 0; r < right_positions.size() / 4; r++ )
	{
		unsigned int index = r * 4;
		WallBox tile;
		tile.center.x	= right_positions[index    ];
		tile.center.y	= right_positions[index + 1] + right_tile_height / 2.0f;
		tile.center.z	= right_positions[index + 2] + right_tile_width  / 2.0f;
		tile.halfdiag.x = 2.0f;
		tile.halfdiag.y	= right_tile_height / 2.0f;
		tile.halfdiag.z	= right_tile_width  / 2.0f;
		tile.bR			= (float)(rand() % 256) / 255.0f;
		tile.bG			= (float)(rand() % 256) / 255.0f;
		tile.bB			= (float)(rand() % 256) / 255.0f;
		right_boxes.push_back( tile );
	}

	top_tile_width  = WIDTH  / tiling[TOP];
	top_tile_height = LENGTH / tiling[TOP];
	for( unsigned int t = 0; t < top_positions.size() / 4; t++ )
	{
		unsigned int index = t * 4;
		WallBox tile;
		tile.center.x	= top_positions[index    ] + top_tile_width  / 2.0f;
		tile.center.y	= top_positions[index + 1];
		tile.center.z	= top_positions[index + 2] + top_tile_height / 2.0f;
		tile.halfdiag.x = top_tile_width  / 2.0f;
		tile.halfdiag.y	= 2.0f;
		tile.halfdiag.z	= top_tile_height / 2.0f;
		tile.bR			= (float)(rand() % 256) / 255.0f;
		tile.bG			= (float)(rand() % 256) / 255.0f;
		tile.bB			= (float)(rand() % 256) / 255.0f;
		top_boxes.push_back( tile );
	}

	bottom_tile_width  = WIDTH  / tiling[BOTTOM];
	bottom_tile_height = LENGTH / tiling[BOTTOM];
	for( unsigned int b = 0; b < bottom_positions.size() / 4; b++ )
	{
		unsigned int index = b * 4;
		WallBox tile;
		tile.center.x	= bottom_positions[index    ] + top_tile_width  / 2.0f;
		tile.center.y	= bottom_positions[index + 1];
		tile.center.z	= bottom_positions[index + 2] - top_tile_height / 2.0f;
		tile.halfdiag.x = bottom_tile_width  / 2.0f;
		tile.halfdiag.y	= 2.0f;
		tile.halfdiag.z	= bottom_tile_height / 2.0f;
		tile.bR			= (float)(rand() % 256) / 255.0f;
		tile.bG			= (float)(rand() % 256) / 255.0f;
		tile.bB			= (float)(rand() % 256) / 255.0f;
		bottom_boxes.push_back( tile );
	}
}
//
//=======================================================================================
//
SkyboxManager::~SkyboxManager( void )
{
	v_front.clear();
	v_back.clear();
	v_left.clear();
	v_right.clear();
	v_top.clear();
	v_bottom.clear();
	front_positions.clear();
	front_rotations.clear();
	back_positions.clear();
	back_rotations.clear();
	left_positions.clear();
	left_rotations.clear();
	right_positions.clear();
	right_rotations.clear();
	top_positions.clear();
	top_rotations.clear();
	bottom_positions.clear();
	bottom_rotations.clear();
	front_boxes.clear();
	back_boxes.clear();
	left_boxes.clear();
	right_boxes.clear();
	top_boxes.clear();
	bottom_boxes.clear();
	vbo_indices.clear();
	bumped.clear();
	tiling.clear();
}
//
//=======================================================================================
//
bool SkyboxManager::LoadSkyboxTextures(	string& 		FRONT_Filename,
										unsigned int 	front_env,
										string& 		BACK_Filename,
										unsigned int 	back_env,
										string& 		LEFT_Filename,
										unsigned int 	left_env,
										string& 		RIGHT_Filename,
										unsigned int 	right_env,
										string& 		TOP_Filename,
										unsigned int 	top_env,
										string& 		BOTTOM_Filename,
										unsigned int 	bottom_env		)
{
	DIFFUSE_IDS[FRONT]		= TextureManager::getInstance()->loadTexture( FRONT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  front_env );
	texture_weights[FRONT]	= TextureManager::getInstance()->getCurrWeight();
	if( !DIFFUSE_IDS[FRONT] ) return false;

	DIFFUSE_IDS[BACK]		= TextureManager::getInstance()->loadTexture( BACK_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  back_env );
	texture_weights[BACK]	= TextureManager::getInstance()->getCurrWeight();
	if( !DIFFUSE_IDS[BACK] ) return false;

	DIFFUSE_IDS[LEFT]		= TextureManager::getInstance()->loadTexture( LEFT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  left_env );
	texture_weights[LEFT]	= TextureManager::getInstance()->getCurrWeight();
	if( !DIFFUSE_IDS[LEFT] ) return false;

	DIFFUSE_IDS[RIGHT]		= TextureManager::getInstance()->loadTexture( RIGHT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  right_env );
	texture_weights[RIGHT]	= TextureManager::getInstance()->getCurrWeight();
	if( !DIFFUSE_IDS[RIGHT] ) return false;

	DIFFUSE_IDS[TOP]		= TextureManager::getInstance()->loadTexture( TOP_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  top_env );
	texture_weights[TOP]	= TextureManager::getInstance()->getCurrWeight();
	if( !DIFFUSE_IDS[TOP] ) return false;

	DIFFUSE_IDS[BOTTOM]		= TextureManager::getInstance()->loadTexture( BOTTOM_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  bottom_env );
	texture_weights[BOTTOM] = TextureManager::getInstance()->getCurrWeight();
	if( !DIFFUSE_IDS[BOTTOM] ) return false;

	return true;
}
//
//=======================================================================================
//
bool SkyboxManager::LoadSkyboxBumpTextures( string& 		FRONT_Filename,
											unsigned int 	front_env,
											string& 		BACK_Filename,
											unsigned int 	back_env,
											string& 		LEFT_Filename,
											unsigned int 	left_env,
											string& 		RIGHT_Filename,
											unsigned int 	right_env,
											string& 		TOP_Filename,
											unsigned int 	top_env,
											string& 		BOTTOM_Filename,
											unsigned int 	bottom_env		)
{
	if( bumped[FRONT] )
	{
		BUMPMAP_IDS[FRONT]  = TextureManager::getInstance()->loadTexture( FRONT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  front_env );
		texture_bump_weights[0] = TextureManager::getInstance()->getCurrWeight();
		if( !BUMPMAP_IDS[FRONT] ) return false;
	}
	if( bumped[BACK] )
	{
		BUMPMAP_IDS[BACK]   = TextureManager::getInstance()->loadTexture( BACK_Filename,
																	      false,
																		  GL_TEXTURE_2D,
																		  back_env );
		texture_bump_weights[1] = TextureManager::getInstance()->getCurrWeight();
		if( !BUMPMAP_IDS[BACK] ) return false;
	}
	if( bumped[LEFT] )
	{
		BUMPMAP_IDS[LEFT]   = TextureManager::getInstance()->loadTexture( LEFT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  left_env );
		texture_bump_weights[2] = TextureManager::getInstance()->getCurrWeight();
		if( !BUMPMAP_IDS[LEFT] ) return false;
	}
	if( bumped[RIGHT] )
	{
		BUMPMAP_IDS[RIGHT]  = TextureManager::getInstance()->loadTexture( RIGHT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  right_env );
		texture_bump_weights[3] = TextureManager::getInstance()->getCurrWeight();
		if( !BUMPMAP_IDS[RIGHT] ) return false;
	}
	if( bumped[TOP] )
	{
		BUMPMAP_IDS[TOP]    = TextureManager::getInstance()->loadTexture( TOP_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  top_env );
		texture_bump_weights[4] = TextureManager::getInstance()->getCurrWeight();
		if( !BUMPMAP_IDS[TOP] ) return false;
	}
	if( bumped[BOTTOM] )
	{
		BUMPMAP_IDS[BOTTOM] = TextureManager::getInstance()->loadTexture( BOTTOM_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  bottom_env );
		texture_bump_weights[5] = TextureManager::getInstance()->getCurrWeight();
		if( !BUMPMAP_IDS[BOTTOM] ) return false;
	}
	return true;
}
//
//=======================================================================================
//
bool SkyboxManager::LoadSkyboxSpecTextures( string& 		FRONT_Filename,
											unsigned int 	front_env,
											string& 		BACK_Filename,
											unsigned int 	back_env,
											string& 		LEFT_Filename,
											unsigned int 	left_env,
											string& 		RIGHT_Filename,
											unsigned int 	right_env,
											string& 		TOP_Filename,
											unsigned int 	top_env,
											string& 		BOTTOM_Filename,
											unsigned int 	bottom_env		)
{
	if( bumped[FRONT] )
	{
		SPECMAP_IDS[FRONT]  = TextureManager::getInstance()->loadTexture( FRONT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  front_env );
		texture_spec_weights[0] = TextureManager::getInstance()->getCurrWeight();
		if( !SPECMAP_IDS[FRONT] ) return false;
	}
	if( bumped[BACK] )
	{
		SPECMAP_IDS[BACK]   = TextureManager::getInstance()->loadTexture( BACK_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  back_env );
		texture_spec_weights[1] = TextureManager::getInstance()->getCurrWeight();
		if( !SPECMAP_IDS[BACK] ) return false;
	}
	if( bumped[LEFT] )
	{
		SPECMAP_IDS[LEFT]   = TextureManager::getInstance()->loadTexture( LEFT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  left_env );
		texture_spec_weights[2] = TextureManager::getInstance()->getCurrWeight();
		if( !SPECMAP_IDS[LEFT] ) return false;
	}
	if( bumped[RIGHT] )
	{
		SPECMAP_IDS[RIGHT]  = TextureManager::getInstance()->loadTexture( RIGHT_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  right_env );
		texture_spec_weights[3] = TextureManager::getInstance()->getCurrWeight();
		if( !SPECMAP_IDS[RIGHT] ) return false;
	}
	if( bumped[TOP] )
	{
		SPECMAP_IDS[TOP]    = TextureManager::getInstance()->loadTexture( TOP_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  top_env );
		texture_spec_weights[4] = TextureManager::getInstance()->getCurrWeight();
		if( !SPECMAP_IDS[TOP] ) return false;
	}
	if( bumped[BOTTOM] )
	{
		SPECMAP_IDS[BOTTOM] = TextureManager::getInstance()->loadTexture( BOTTOM_Filename,
																		  false,
																		  GL_TEXTURE_2D,
																		  bottom_env );
		texture_spec_weights[5] = TextureManager::getInstance()->getCurrWeight();
		if( !SPECMAP_IDS[BOTTOM] ) return false;
	}
	return true;
}
//
//=======================================================================================
//
void SkyboxManager::draw( Frustum* frustum, bool bump_enabled, bool draw_bv )
{
    glPushAttrib( GL_LIGHTING_BIT );
	{
		if( !lighting )
		{
			glDisable( GL_LIGHTING );
		}

		if( !instancing )
		{
			if( tilesFrustumCheck( frustum, front_boxes ) )
			{
				glActiveTexture( GL_TEXTURE0 );
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
															DIFFUSE_IDS[FRONT] );
				if( bumped[FRONT] && bump_enabled )
				{
					glsl_manager->activate( "bump" );
					{
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	BUMPMAP_IDS[FRONT] );
						glActiveTexture( GL_TEXTURE2 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	SPECMAP_IDS[FRONT] );

						vbo_manager->render_bumped_vbo( vboIds[0], v_front.size() );

						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
					}
					glsl_manager->deactivate( "bump" );
				}
				else
				{
					vbo_manager->render_vbo( vboIds[0], v_front.size() );
				}
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
			}
			else
			{
				frustum->incCulledCount();
			}

			if( tilesFrustumCheck( frustum, back_boxes ) )
			{
				glActiveTexture( GL_TEXTURE0 );
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
															DIFFUSE_IDS[BACK] );
				if( bumped[BACK] && bump_enabled )
				{
					glsl_manager->activate( "bump" );
					{
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	BUMPMAP_IDS[BACK] );
						glActiveTexture( GL_TEXTURE2 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	SPECMAP_IDS[BACK] );

						vbo_manager->render_bumped_vbo( vboIds[1], v_back.size() );

						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
					}
					glsl_manager->deactivate( "bump" );
				}
				else
				{
					vbo_manager->render_vbo( vboIds[1], v_back.size() );
				}
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
			}
			else
			{
				frustum->incCulledCount();
			}

			if( tilesFrustumCheck( frustum, left_boxes ) )
			{
				glActiveTexture( GL_TEXTURE0 );
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
															DIFFUSE_IDS[LEFT] );
				if( bumped[LEFT] && bump_enabled )
				{
					glsl_manager->activate( "bump" );
					{
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	BUMPMAP_IDS[LEFT] );
						glActiveTexture( GL_TEXTURE2 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	SPECMAP_IDS[LEFT] );

						vbo_manager->render_bumped_vbo( vboIds[2], v_left.size() );

						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
					}
					glsl_manager->deactivate( "bump" );
				}
				else
				{
					vbo_manager->render_vbo( vboIds[2], v_left.size() );
				}
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
			}
			else
			{
				frustum->incCulledCount();
			}

			if( tilesFrustumCheck( frustum, right_boxes ) )
			{
				glActiveTexture( GL_TEXTURE0 );
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
															DIFFUSE_IDS[RIGHT] );
				if( bumped[RIGHT] && bump_enabled )
				{
					glsl_manager->activate( "bump" );
					{
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	BUMPMAP_IDS[RIGHT] );
						glActiveTexture( GL_TEXTURE2 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	SPECMAP_IDS[RIGHT] );

						vbo_manager->render_bumped_vbo( vboIds[3], v_right.size() );

						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
					}
					glsl_manager->deactivate( "bump" );
				}
				else{
					vbo_manager->render_vbo( vboIds[3], v_right.size() );
				}
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
			}
			else
			{
				frustum->incCulledCount();
			}

			if( tilesFrustumCheck( frustum, top_boxes ) )
			{
				glActiveTexture( GL_TEXTURE0 );
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
															DIFFUSE_IDS[TOP] );
				if( bumped[TOP] && bump_enabled )
				{
					glsl_manager->activate( "bump" );
					{
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	BUMPMAP_IDS[TOP] );
						glActiveTexture( GL_TEXTURE2 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	SPECMAP_IDS[TOP] );

						vbo_manager->render_bumped_vbo( vboIds[4], v_top.size() );

						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
					}
					glsl_manager->deactivate( "bump" );
				}
				else{
					vbo_manager->render_vbo( vboIds[4], v_top.size() );
				}
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
			}
			else
			{
				frustum->incCulledCount();
			}

			if( tilesFrustumCheck( frustum, bottom_boxes ) )
			{
				glActiveTexture( GL_TEXTURE0 );
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
															DIFFUSE_IDS[BOTTOM] );
				if( bumped[BOTTOM] && bump_enabled )
				{
					glsl_manager->activate( "bump" );
					{
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	BUMPMAP_IDS[BOTTOM] );
						glActiveTexture( GL_TEXTURE2 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																	SPECMAP_IDS[BOTTOM] );

						vbo_manager->render_bumped_vbo( vboIds[5], v_bottom.size() );

						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE1 );
						TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
						glActiveTexture( GL_TEXTURE0 );
					}
					glsl_manager->deactivate( "bump" );
				}
				else{
					vbo_manager->render_vbo( vboIds[5], v_bottom.size() );
				}
				TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
			}
			else
			{
				frustum->incCulledCount();
			}
		}
		else
		{
			float view_mat[16];
			glGetFloatv( GL_MODELVIEW_MATRIX, view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[0],
											   DIFFUSE_IDS[FRONT],
											   vbo_manager->vbos[
												   vbo_indices[0]][0].vertices.size() *
													   sizeof( Vertex ),
											   front_positions.size() / 4,
											   front_positions.size() * sizeof( float ),
											   &front_positions[0],
											   front_rotations.size() * sizeof( float ),
											   &front_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[1],
											   DIFFUSE_IDS[BACK],
											   vbo_manager->vbos[
												   vbo_indices[1]][0].vertices.size() *
													   sizeof( Vertex ),
											   back_positions.size() / 4,
											   back_positions.size() * sizeof( float ),
											   &back_positions[0],
											   back_rotations.size() * sizeof( float ),
											   &back_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[2],
											   DIFFUSE_IDS[LEFT],
											   vbo_manager->vbos[
												   vbo_indices[2]][0].vertices.size() *
													   sizeof( Vertex ),
											   left_positions.size() / 4,
											   left_positions.size() * sizeof( float ),
											   &left_positions[0],
											   left_rotations.size() * sizeof( float ),
											   &left_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[3],
											   DIFFUSE_IDS[RIGHT],
											   vbo_manager->vbos[
												   vbo_indices[3]][0].vertices.size() *
													   sizeof( Vertex ),
											   right_positions.size() / 4,
											   right_positions.size() * sizeof( float ),
											   &right_positions[0],
											   right_rotations.size() * sizeof( float ),
											   &right_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[4],
											   DIFFUSE_IDS[TOP],
											   vbo_manager->vbos[
												   vbo_indices[4]][0].vertices.size() *
													   sizeof( Vertex ),
											   top_positions.size() / 4,
											   top_positions.size() * sizeof( float ),
											   &top_positions[0],
											   top_rotations.size() * sizeof( float ),
											   &top_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[5],
											   DIFFUSE_IDS[BOTTOM],
											   vbo_manager->vbos[
												   vbo_indices[5]][0].vertices.size() *
													   sizeof( Vertex ),
											   bottom_positions.size() / 4,
											   bottom_positions.size() * sizeof( float ),
											   &bottom_positions[0],
											   bottom_rotations.size() * sizeof( float ),
											   &bottom_rotations[0],
											   view_mat );

		}
		if( draw_bv )
		{
			drawBVs();
		}
	}
	glPopAttrib();
}
//
//=======================================================================================
//
void SkyboxManager::draw( bool bump_enabled, bool draw_bv )
{
    glPushAttrib( GL_LIGHTING_BIT );
	{
		if( !lighting )
		{
			glDisable( GL_LIGHTING );
		}

		if( !instancing )
		{
			glActiveTexture( GL_TEXTURE0 );
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
														DIFFUSE_IDS[FRONT] );
			if( bumped[FRONT] && bump_enabled )
			{
				glsl_manager->activate( "bump" );
				{
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																BUMPMAP_IDS[FRONT] );
					glActiveTexture( GL_TEXTURE2 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																SPECMAP_IDS[FRONT] );

					vbo_manager->render_bumped_vbo( vboIds[0], v_front.size() );

					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE0 );
				}
				glsl_manager->deactivate( "bump" );
			}
			else
			{
				vbo_manager->render_vbo( vboIds[0], v_front.size() );
			}
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );

			glActiveTexture( GL_TEXTURE0 );
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
														DIFFUSE_IDS[BACK] );
			if( bumped[BACK] && bump_enabled )
			{
				glsl_manager->activate( "bump" );
				{
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																BUMPMAP_IDS[BACK] );
					glActiveTexture( GL_TEXTURE2 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																SPECMAP_IDS[BACK] );

					vbo_manager->render_bumped_vbo( vboIds[1], v_back.size() );

					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE0 );
				}
				glsl_manager->deactivate( "bump" );
			}
			else
			{
				vbo_manager->render_vbo( vboIds[1], v_back.size() );
			}
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );

			glActiveTexture( GL_TEXTURE0 );
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
														DIFFUSE_IDS[LEFT] );
			if( bumped[LEFT] && bump_enabled )
			{
				glsl_manager->activate( "bump" );
				{
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																BUMPMAP_IDS[LEFT] );
					glActiveTexture( GL_TEXTURE2 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																SPECMAP_IDS[LEFT] );

					vbo_manager->render_bumped_vbo( vboIds[2], v_left.size() );

					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE0 );
				}
				glsl_manager->deactivate( "bump" );
			}
			else
			{
				vbo_manager->render_vbo( vboIds[2], v_left.size() );
			}
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );

			glActiveTexture( GL_TEXTURE0 );
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
														DIFFUSE_IDS[RIGHT] );
			if( bumped[RIGHT] && bump_enabled )
			{
				glsl_manager->activate( "bump" );
				{
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																BUMPMAP_IDS[RIGHT] );
					glActiveTexture( GL_TEXTURE2 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																SPECMAP_IDS[RIGHT] );

					vbo_manager->render_bumped_vbo( vboIds[3], v_right.size() );

					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE0 );
				}
				glsl_manager->deactivate( "bump" );
			}
			else{
				vbo_manager->render_vbo( vboIds[3], v_right.size() );
			}
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );

			glActiveTexture( GL_TEXTURE0 );
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
														DIFFUSE_IDS[TOP] );
			if( bumped[TOP] && bump_enabled )
			{
				glsl_manager->activate( "bump" );
				{
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																BUMPMAP_IDS[TOP] );
					glActiveTexture( GL_TEXTURE2 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																SPECMAP_IDS[TOP] );

					vbo_manager->render_bumped_vbo( vboIds[4], v_top.size() );

					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE0 );
				}
				glsl_manager->deactivate( "bump" );
			}
			else{
				vbo_manager->render_vbo( vboIds[4], v_top.size() );
			}
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );

			glActiveTexture( GL_TEXTURE0 );
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
														DIFFUSE_IDS[BOTTOM] );
			if( bumped[BOTTOM] && bump_enabled )
			{
				glsl_manager->activate( "bump" );
				{
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																BUMPMAP_IDS[BOTTOM] );
					glActiveTexture( GL_TEXTURE2 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D,
																SPECMAP_IDS[BOTTOM] );

					vbo_manager->render_bumped_vbo( vboIds[5], v_bottom.size() );

					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE1 );
					TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
					glActiveTexture( GL_TEXTURE0 );
				}
				glsl_manager->deactivate( "bump" );
			}
			else{
				vbo_manager->render_vbo( vboIds[5], v_bottom.size() );
			}
			TextureManager::getInstance()->bindTexture( GL_TEXTURE_2D, 0 );
		}
		else
		{
			float view_mat[16];
			glGetFloatv( GL_MODELVIEW_MATRIX, view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[0],
											   DIFFUSE_IDS[FRONT],
											   vbo_manager->vbos[
												   vbo_indices[0]][0].vertices.size() *
													   sizeof( Vertex ),
											   front_positions.size() / 4,
											   front_positions.size() * sizeof( float ),
											   &front_positions[0],
											   front_rotations.size() * sizeof( float ),
											   &front_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[1],
											   DIFFUSE_IDS[BACK],
											   vbo_manager->vbos[
												   vbo_indices[1]][0].vertices.size() *
													   sizeof( Vertex ),
											   back_positions.size() / 4,
											   back_positions.size() * sizeof( float ),
											   &back_positions[0],
											   back_rotations.size() * sizeof( float ),
											   &back_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[2],
											   DIFFUSE_IDS[LEFT],
											   vbo_manager->vbos[
												   vbo_indices[2]][0].vertices.size() *
													   sizeof( Vertex ),
											   left_positions.size() / 4,
											   left_positions.size() * sizeof( float ),
											   &left_positions[0],
											   left_rotations.size() * sizeof( float ),
											   &left_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[3],
											   DIFFUSE_IDS[RIGHT],
											   vbo_manager->vbos[
												   vbo_indices[3]][0].vertices.size() *
													   sizeof( Vertex ),
											   right_positions.size() / 4,
											   right_positions.size() * sizeof( float ),
											   &right_positions[0],
											   right_rotations.size() * sizeof( float ),
											   &right_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[4],
											   DIFFUSE_IDS[TOP],
											   vbo_manager->vbos[
												   vbo_indices[4]][0].vertices.size() *
													   sizeof( Vertex ),
											   top_positions.size() / 4,
											   top_positions.size() * sizeof( float ),
											   &top_positions[0],
											   top_rotations.size() * sizeof( float ),
											   &top_rotations[0],
											   view_mat );

			vbo_manager->render_instanced_vbo( ivboIds[5],
											   DIFFUSE_IDS[BOTTOM],
											   vbo_manager->vbos[
												   vbo_indices[5]][0].vertices.size() *
													   sizeof( Vertex ),
											   bottom_positions.size() / 4,
											   bottom_positions.size() * sizeof( float ),
											   &bottom_positions[0],
											   bottom_rotations.size() * sizeof( float ),
											   &bottom_rotations[0],
											   view_mat );

		}
		if( draw_bv )
		{
			drawBVs();
		}
	}
	glPopAttrib();
}
//
//=======================================================================================
//
void SkyboxManager::gen_vbos( void )
{
	// Center the SkyboxManager around the given x,y,z position
	float x = center.x - WIDTH  / 2;
	float y = center.y - HEIGHT / 2;
	float z = center.z - LENGTH / 2;

	// Set Front side CCW
	log_manager->log( LogManager::SKYBOX, "Generating FRONT VBO... " );
	if( tiling[FRONT] >= 1.0f )
	{
		float sw = WIDTH;
		float sh = HEIGHT;
		float dw = sw / tiling[FRONT];
		float dh = sh / tiling[FRONT];

		if( instancing )
		{
			Vertex v1, v2, v3, v4;
			fill_vertex( v1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
			fill_vertex( v2, dw,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
			fill_vertex( v3, dw,   dh,   0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
			fill_vertex( v4, 0.0f, dh,   0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
			vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.push_back( v2 );
			vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.push_back( v4 );
		}

		for( float j = y; j < (y + sh); j += dh )
		{
			for( float i = x; i < (x + sw); i += dw )
			{
				Vertex vert1, vert2, vert3, vert4;
				fill_vertex( vert1, i,    j+dh, z, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
				fill_vertex( vert2, i+dw, j+dh, z, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
				fill_vertex( vert3, i+dw, j,    z, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
				fill_vertex( vert4, i,    j,    z, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
				fill_tangents( vert1, vert2, vert3, vert4 );
				v_front.push_back( vert1 );
				v_front.push_back( vert2 );
				v_front.push_back( vert3 );
				v_front.push_back( vert1 );
				v_front.push_back( vert3 );
				v_front.push_back( vert4 );

				float posx = i;
				float posy = j;
				float posz = -LENGTH / 2.0f;
				front_positions.push_back( posx );
				front_positions.push_back( posy );
				front_positions.push_back( posz );
				front_positions.push_back( 0.0f );

				float cenital = 0.0f;
				float azimuth = 0.0f;
				front_rotations.push_back( cenital );
				front_rotations.push_back( azimuth );
				front_rotations.push_back( 0.0f );
				front_rotations.push_back( 0.0f );
			}
		}
	}

	if( instancing )
	{
		vbo_manager->gen_vbo( ivboIds[FRONT], vbo_indices[FRONT], 0 );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.size(),
												(vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.size() *
												sizeof( Vertex ) / 1024)								);
		vertexCounts[0] = vbo_manager->vbos[vbo_indices[FRONT]][0].vertices.size();
	}
	else
	{
		glGenBuffers( 1, &vboIds[FRONT] );
		glBindBuffer( GL_ARRAY_BUFFER, vboIds[FRONT] );
		glBufferData( GL_ARRAY_BUFFER,
					  sizeof( Vertex ) * v_front.size(),
					  NULL,
					  GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER,
						 0,
						 sizeof( Vertex ) * v_front.size(),
						 &v_front[0] );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												v_front.size(),
												(v_front.size() * sizeof( Vertex ) / 1024) );
		vertexCounts[FRONT] = v_front.size();
	}

	// Set Back side CCW
	log_manager->log( LogManager::SKYBOX, "Generating BACK VBO... " );
	if( tiling[BACK] >= 1.0f )
	{
		float sw = WIDTH;
		float sh = HEIGHT;
		float dw = sw / tiling[BACK];
		float dh = sh / tiling[BACK];

		if( instancing )
		{
			Vertex v1, v2, v3, v4;
			fill_vertex( v1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
			fill_vertex( v2, dw,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
			fill_vertex( v3, dw,   dh,   0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
			fill_vertex( v4, 0.0f, dh,   0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
			vbo_manager->vbos[vbo_indices[BACK]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[BACK]][0].vertices.push_back( v2 );
			vbo_manager->vbos[vbo_indices[BACK]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[BACK]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[BACK]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[BACK]][0].vertices.push_back( v4 );
		}

		for( float j = y; j < (y + sh); j += dh )
		{
			for( float i = x + sw - dw; i >= x; i -= dw )
			{
				Vertex vert1, vert2, vert3, vert4;
				fill_vertex( vert1,
							 i+dw,
							 j,
							 z+LENGTH,
							 0.0f,
							 0.0f,
							 -1.0f,
							 0.0f,
							 0.0f );
				fill_vertex( vert2,
							 i,
							 j,
							 z+LENGTH,
							 0.0f,
							 0.0f,
							 -1.0f,
							 1.0f,
							 0.0f );
				fill_vertex( vert3,
							 i,
							 j+dh,
							 z+LENGTH,
							 0.0f,
							 0.0f,
							 -1.0f,
							 1.0f,
							 1.0f );
				fill_vertex( vert4,
							 i+dw,
							 j+dh,
							 z+LENGTH,
							 0.0f,
							 0.0f,
							 -1.0f,
							 0.0f,
							 1.0f );
				fill_tangents( vert1, vert2, vert3, vert4 );
				v_back.push_back( vert1 );
				v_back.push_back( vert2 );
				v_back.push_back( vert3 );
				v_back.push_back( vert1 );
				v_back.push_back( vert3 );
				v_back.push_back( vert4 );

				float posx = i + dw;
				float posy = j;
				float posz = LENGTH / 2.0f;
				back_positions.push_back( posx );
				back_positions.push_back( posy );
				back_positions.push_back( posz );
				back_positions.push_back( 0.0f );

				float cenital = 0.0f;
				float azimuth = 180.0f * (float)M_PI / 180.0f;
				back_rotations.push_back( cenital );
				back_rotations.push_back( azimuth );
				back_rotations.push_back( 0.0f );
				back_rotations.push_back( 0.0f );
			}
		}
	}

	if( instancing )
	{
		vbo_manager->gen_vbo( ivboIds[BACK], vbo_indices[BACK], 0 );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												vbo_manager->vbos[vbo_indices[BACK]][0].vertices.size(),
												(vbo_manager->vbos[vbo_indices[BACK]][0].vertices.size() *
												sizeof( Vertex ) / 1024)								);
		vertexCounts[BACK] = vbo_manager->vbos[vbo_indices[BACK]][0].vertices.size();
	}
	else
	{
		glGenBuffers( 1, &vboIds[BACK] );
		glBindBuffer( GL_ARRAY_BUFFER, vboIds[BACK] );
		glBufferData( GL_ARRAY_BUFFER,
					  sizeof( Vertex ) * v_back.size(),
					  NULL,
					  GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER,
						 0,
						 sizeof( Vertex ) * v_back.size(),
						 &v_back[0] );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												v_back.size(),
												(v_back.size() * sizeof( Vertex ) / 1024) );
		vertexCounts[BACK] = v_back.size();
	}

	// Set Left side CCW
	log_manager->log( LogManager::SKYBOX, "Generating LEFT VBO... " );
	if( tiling[LEFT] >= 1.0f )
	{
		float sw = LENGTH;
		float sh = HEIGHT;
		float dw = sw / tiling[LEFT];
		float dh = sh / tiling[LEFT];

		if( instancing )
		{
			Vertex v1, v2, v3, v4;
			fill_vertex( v1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
			fill_vertex( v2, dw,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
			fill_vertex( v3, dw,   dh,   0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
			fill_vertex( v4, 0.0f, dh,   0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
			vbo_manager->vbos[vbo_indices[2]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[2]][0].vertices.push_back( v2 );
			vbo_manager->vbos[vbo_indices[2]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[2]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[2]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[2]][0].vertices.push_back( v4 );
		}

		for( float j = y; j < (y + sh); j += dh )
		{
			for( float i = z; i < (z + sw); i += dw )
			{
				Vertex vert1, vert2, vert3, vert4;
				fill_vertex( vert1, x, j,    i+dw, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f );
				fill_vertex( vert2, x, j,	 i,	   1.0f, 0.0f, 0.0f, 1.0f, 0.0f );
				fill_vertex( vert3, x, j+dh, i,    1.0f, 0.0f, 0.0f, 1.0f, 1.0f );
				fill_vertex( vert4, x, j+dh, i+dw, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f );
				fill_tangents( vert1, vert2, vert3, vert4 );
				v_left.push_back( vert1 );
				v_left.push_back( vert2 );
				v_left.push_back( vert3 );
				v_left.push_back( vert1 );
				v_left.push_back( vert3 );
				v_left.push_back( vert4 );

				float posx = -WIDTH / 2.0f;
				float posy = j;
				float posz = i + dw;
				left_positions.push_back( posx );
				left_positions.push_back( posy );
				left_positions.push_back( posz );
				left_positions.push_back( 0.0f );

				float cenital = 0.0f;
				float azimuth = 90.0f * (float)M_PI / 180.0f;
				left_rotations.push_back( cenital );
				left_rotations.push_back( azimuth );
				left_rotations.push_back( 0.0f );
				left_rotations.push_back( 0.0f );
			}
		}
	}

	if( instancing )
	{
		vbo_manager->gen_vbo( ivboIds[2], vbo_indices[2], 0 );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												vbo_manager->vbos[vbo_indices[2]][0].vertices.size(),
												(vbo_manager->vbos[vbo_indices[2]][0].vertices.size() *
												sizeof( Vertex ) / 1024)								);
		vertexCounts[2] = vbo_manager->vbos[vbo_indices[2]][0].vertices.size();
	}
	else
	{
		glGenBuffers( 1, &vboIds[2] );
		glBindBuffer( GL_ARRAY_BUFFER, vboIds[2] );
		glBufferData( GL_ARRAY_BUFFER,
					  sizeof( Vertex ) * v_left.size(),
					  NULL, GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER,
						 0,
						 sizeof( Vertex ) * v_left.size(),
						 &v_left[0] );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												v_left.size(),
												(v_left.size() * sizeof( Vertex ) / 1024) );
		vertexCounts[2] = v_left.size();
	}

	// Set Right side CCW
	log_manager->log( LogManager::SKYBOX, "Generating RIGHT VBO... " );
	if( tiling[RIGHT] >= 1.0f )
	{
		float sw = LENGTH;
		float sh = HEIGHT;
		float dw = sw / tiling[RIGHT];
		float dh = sh / tiling[RIGHT];

		if( instancing )
		{
			Vertex v1, v2, v3, v4;
			fill_vertex( v1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
			fill_vertex( v2, dw,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
			fill_vertex( v3, dw,   dh,   0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
			fill_vertex( v4, 0.0f, dh,   0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
			vbo_manager->vbos[vbo_indices[3]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[3]][0].vertices.push_back( v2 );
			vbo_manager->vbos[vbo_indices[3]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[3]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[3]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[3]][0].vertices.push_back( v4 );
		}

		for( float j = y; j < (y + sh); j += dh )
		{
			for( float i = z + sw - dw; i >= z; i -= dw )
			{
				Vertex vert1, vert2, vert3, vert4;
				fill_vertex( vert1, x+WIDTH, j,    i,    -1.0f, 0.0f, 0.0f, 0.0f, 0.0f );
				fill_vertex( vert2, x+WIDTH, j,    i+dw, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f );
				fill_vertex( vert3, x+WIDTH, j+dh, i+dw, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f );
				fill_vertex( vert4, x+WIDTH, j+dh, i,    -1.0f, 0.0f, 0.0f, 0.0f, 1.0f );
				fill_tangents( vert1, vert2, vert3, vert4 );
				v_right.push_back( vert1 );
				v_right.push_back( vert2 );
				v_right.push_back( vert3 );
				v_right.push_back( vert1 );
				v_right.push_back( vert3 );
				v_right.push_back( vert4 );

				float posx = WIDTH / 2.0f;
				float posy = j;
				float posz = i;
				right_positions.push_back( posx );
				right_positions.push_back( posy );
				right_positions.push_back( posz );
				right_positions.push_back( 0.0f );

				float cenital = 0.0f;
				float azimuth = -90.0f * (float)M_PI / 180.0f;
				right_rotations.push_back( cenital );
				right_rotations.push_back( azimuth );
				right_rotations.push_back( 0.0f );
				right_rotations.push_back( 0.0f );
			}
		}
	}

	if( instancing )
	{
		vbo_manager->gen_vbo( ivboIds[3], vbo_indices[3], 0 );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												vbo_manager->vbos[vbo_indices[3]][0].vertices.size(),
												(vbo_manager->vbos[vbo_indices[3]][0].vertices.size() *
												sizeof( Vertex ) / 1024)								);
		vertexCounts[3] = vbo_manager->vbos[vbo_indices[3]][0].vertices.size();
	}
	else
	{
		glGenBuffers( 1, &vboIds[3] );
		glBindBuffer( GL_ARRAY_BUFFER, vboIds[3] );
		glBufferData( GL_ARRAY_BUFFER,
					  sizeof( Vertex ) * v_right.size(),
					  NULL,
					  GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER,
						 0,
						 sizeof( Vertex ) * v_right.size(),
						 &v_right[0] );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												v_right.size(),
												(v_right.size() * sizeof( Vertex ) / 1024) );
		vertexCounts[3] = v_right.size();
	}

	// Set Top side CCW
	log_manager->log( LogManager::SKYBOX, "Generating TOP VBO... " );
	if( tiling[TOP] >= 1.0f )
	{
		float sw = WIDTH;
		float sh = LENGTH;
		float dw = sw / tiling[TOP];
		float dh = sh / tiling[TOP];

		if( instancing )
		{
			Vertex v1, v2, v3, v4;
			fill_vertex( v1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
			fill_vertex( v2, dw,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
			fill_vertex( v3, dw,   dh,   0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
			fill_vertex( v4, 0.0f, dh,   0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
			vbo_manager->vbos[vbo_indices[4]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[4]][0].vertices.push_back( v2 );
			vbo_manager->vbos[vbo_indices[4]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[4]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[4]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[4]][0].vertices.push_back( v4 );
		}

		for( float j = z + sh - dh; j >= z; j-= dh )
		{
			for( float i = x; i < (x + sw); i+= dw )
			{
				Vertex vert1, vert2, vert3, vert4;
				fill_vertex( vert1,
							 i+dw,
							 y+HEIGHT,
							 j+dh,
							 0.0f,
							 -1.0f,
							 0.0f,
							 0.0f,
							 0.0f );
				fill_vertex( vert2,
							 i,
							 y+HEIGHT,
							 j+dh,
							 0.0f,
							 -1.0f,
							 0.0f,
							 1.0f,
							 0.0f );
				fill_vertex( vert3,
							 i,
							 y+HEIGHT,
							 j,
							 0.0f,
							 -1.0f,
							 0.0f,
							 1.0f,
							 1.0f );
				fill_vertex( vert4,
							 i+dw,
							 y+HEIGHT,
							 j,
							 0.0f,
							 -1.0f,
							 0.0f,
							 0.0f,
							 1.0f );
				fill_tangents( vert1, vert2, vert3, vert4 );
				v_top.push_back( vert1 );
				v_top.push_back( vert2 );
				v_top.push_back( vert3 );
				v_top.push_back( vert1 );
				v_top.push_back( vert3 );
				v_top.push_back( vert4 );

				float posx = i;
				float posy = HEIGHT / 2.0f;
				float posz = j;
				top_positions.push_back( posx );
				top_positions.push_back( posy );
				top_positions.push_back( posz );
				top_positions.push_back( 0.0f );

				float cenital = 90.0f * (float)M_PI / 180.0f;
				float azimuth = 0.0f;
				top_rotations.push_back( cenital );
				top_rotations.push_back( azimuth );
				top_rotations.push_back( 0.0f );
				top_rotations.push_back( 0.0f );
			}
		}
	}

	if( instancing )
	{
		vbo_manager->gen_vbo( ivboIds[4], vbo_indices[4], 0 );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).\n",
												vbo_manager->vbos[vbo_indices[4]][0].vertices.size(),
												(vbo_manager->vbos[vbo_indices[4]][0].vertices.size() *
												sizeof( Vertex ) / 1024)								);
		vertexCounts[4] = vbo_manager->vbos[vbo_indices[4]][0].vertices.size();
	}
	else
	{
		glGenBuffers( 1, &vboIds[4] );
		glBindBuffer( GL_ARRAY_BUFFER, vboIds[4] );
		glBufferData( GL_ARRAY_BUFFER,
					  sizeof( Vertex ) * v_top.size(),
					  NULL,
					  GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER,
						 0,
						 sizeof( Vertex ) * v_top.size(),
						 &v_top[0] );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												v_top.size(),
												(v_top.size() * sizeof( Vertex ) / 1024) );
		vertexCounts[4] = v_top.size();
	}

	// Set Bottom side CCW
	log_manager->log( LogManager::SKYBOX, "Generating BOTTOM VBO... " );
	if( tiling[BOTTOM] >= 1.0f )
	{
		float sw = WIDTH;
		float sh = LENGTH;
		float dw = sw / tiling[BOTTOM];
		float dh = sh / tiling[BOTTOM];

		if( instancing )
		{
			Vertex v1, v2, v3, v4;
			fill_vertex( v1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f );
			fill_vertex( v2, dw,   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f );
			fill_vertex( v3, dw,   dh,   0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f );
			fill_vertex( v4, 0.0f, dh,   0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f );
			vbo_manager->vbos[vbo_indices[5]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[5]][0].vertices.push_back( v2 );
			vbo_manager->vbos[vbo_indices[5]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[5]][0].vertices.push_back( v1 );
			vbo_manager->vbos[vbo_indices[5]][0].vertices.push_back( v3 );
			vbo_manager->vbos[vbo_indices[5]][0].vertices.push_back( v4 );
		}

		for( float j = z; j < (z + sh); j+= dh )
		{
			for( float i = x; i < (x + sw); i+= dw )
			{
				Vertex vert1, vert2, vert3, vert4;
				fill_vertex( vert1, i,    y, j+dh, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f );
				fill_vertex( vert2, i+dw, y, j+dh, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f );
				fill_vertex( vert3, i+dw, y, j,    0.0f, 1.0f, 0.0f, 1.0f, 1.0f );
				fill_vertex( vert4, i,    y, j,    0.0f, 1.0f, 0.0f, 0.0f, 1.0f );
				fill_tangents( vert1, vert2, vert3, vert4 );
				v_bottom.push_back( vert1 );
				v_bottom.push_back( vert2 );
				v_bottom.push_back( vert3 );
				v_bottom.push_back( vert1 );
				v_bottom.push_back( vert3 );
				v_bottom.push_back( vert4 );

				float posx = i;
				float posy = -HEIGHT / 2.0f;
				float posz = j + dh;
				bottom_positions.push_back( posx );
				bottom_positions.push_back( posy );
				bottom_positions.push_back( posz );
				bottom_positions.push_back( 0.0f );

				float cenital = -90.0f * (float)M_PI / 180.0f;
				float azimuth = 0.0f;
				bottom_rotations.push_back( cenital );
				bottom_rotations.push_back( azimuth );
				bottom_rotations.push_back( 0.0f );
				bottom_rotations.push_back( 0.0f );
			}
		}
	}

	if( instancing )
	{
		vbo_manager->gen_vbo( ivboIds[5], vbo_indices[5], 0 );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												vbo_manager->vbos[vbo_indices[5]][0].vertices.size(),
												(vbo_manager->vbos[vbo_indices[5]][0].vertices.size() *
												sizeof( Vertex ) / 1024)								);
		vertexCounts[5] = vbo_manager->vbos[vbo_indices[5]][0].vertices.size();
	}
	else
	{
		glGenBuffers( 1, &vboIds[5] );
		glBindBuffer( GL_ARRAY_BUFFER, vboIds[5] );
		glBufferData( GL_ARRAY_BUFFER,
					  sizeof( Vertex ) * v_bottom.size(),
					  NULL,
					  GL_STATIC_DRAW );
		glBufferSubData( GL_ARRAY_BUFFER,
						 0,
						 sizeof( Vertex ) * v_bottom.size(),
						 &v_bottom[0] );
		log_manager->log( LogManager::SKYBOX, 	"Done. Vertices: %lu (%lu KB).",
												v_bottom.size(),
												(v_bottom.size() * sizeof( Vertex ) / 1024) );
		vertexCounts[5] = v_bottom.size();
	}
}
//
//=======================================================================================
//
unsigned int SkyboxManager::getVertexCount( int face )
{
	return vertexCounts[face];
}
//
//=======================================================================================
//
unsigned int SkyboxManager::getTextureWeight( int face )
{
	return texture_weights[face];
}
//
//=======================================================================================
//
vec3 SkyboxManager::findTangent( Vertex& vert1, Vertex& vert2, Vertex& vert3 )
{
	vec3 v1( 	vert2.location[0] - vert1.location[0],
				vert2.location[1] - vert1.location[1],
				vert2.location[2] - vert1.location[2] );
	vec3 v2( 	vert3.location[0] - vert1.location[0],
				vert3.location[1] - vert1.location[1],
				vert3.location[2] - vert1.location[2] );

	float coef = 1.0f / (vert1.texture[0] * vert3.texture[1] -
						 vert3.texture[0] * vert1.texture[1]);
	vec3 T;
	T.x = coef * ((v1.x * vert3.texture[1])  + (v2.x * -vert1.texture[1]));
	T.y = coef * ((v1.y * vert3.texture[1])  + (v2.y * -vert1.texture[1]));
	T.z = coef * ((v1.z * vert3.texture[1])  + (v2.z * -vert1.texture[1]));
	T = normalize( T );
	return T;
}
//
//=======================================================================================
//
void SkyboxManager::drawBVs( void )
{
	glPushAttrib( GL_LIGHTING_BIT | GL_TEXTURE_BIT );
	{
		glDisable( GL_LIGHTING );
		glDisable( GL_TEXTURE_2D );
		renderSkyboxBVs();
	}
	glPopAttrib();
}
//
//=======================================================================================
//
void SkyboxManager::renderSkyboxBVs( void )
{
	for( unsigned int f = 0; f < front_boxes.size(); f++ )
	{
		const WallBox front_box = front_boxes[f];
		glPushMatrix();
		{
			glColor3f( front_box.bR, front_box.bG, front_box.bB );
			glTranslatef( front_box.center.x, front_box.center.y, front_box.center.z );
			glutWireSphere( 2.0, 10, 10 );
			glScalef( front_box.halfdiag.x * 2.0f,
					  front_box.halfdiag.y * 2.0f,
					  front_box.halfdiag.z * 2.0f );
			glutWireCube( 1.0f );
		}
		glPopMatrix();
	}
	for( unsigned int b = 0; b < back_boxes.size(); b++ )
	{
		const WallBox back_box = back_boxes[b];
		glPushMatrix();
		{
			glColor3f( back_box.bR, back_box.bG, back_box.bB );
			glTranslatef( back_box.center.x, back_box.center.y, back_box.center.z );
			glutWireSphere( 2.0, 10, 10 );
			glScalef( back_box.halfdiag.x * 2.0f,
					  back_box.halfdiag.y * 2.0f,
					  back_box.halfdiag.z * 2.0f );
			glutWireCube( 1.0f );
		}
		glPopMatrix();
	}
	for( unsigned int l = 0; l < left_boxes.size(); l++ )
	{
		const WallBox left_box = left_boxes[l];
		glPushMatrix();
		{
			glColor3f( left_box.bR, left_box.bG, left_box.bB );
			glTranslatef( left_box.center.x, left_box.center.y, left_box.center.z );
			glutWireSphere( 2.0, 10, 10 );
			glScalef( left_box.halfdiag.x * 2.0f,
					  left_box.halfdiag.y * 2.0f,
					  left_box.halfdiag.z * 2.0f );
			glutWireCube( 1.0f );
		}
		glPopMatrix();
	}
	for( unsigned int r = 0; r < right_boxes.size(); r++ )
	{
		const WallBox right_box = right_boxes[r];
		glPushMatrix();
		{
			glColor3f( right_box.bR, right_box.bG, right_box.bB );
			glTranslatef( right_box.center.x, right_box.center.y, right_box.center.z );
			glutWireSphere( 2.0, 10, 10 );
			glScalef( right_box.halfdiag.x * 2.0f,
					  right_box.halfdiag.y * 2.0f,
					  right_box.halfdiag.z * 2.0f );
			glutWireCube( 1.0f );
		}
		glPopMatrix();
	}
	for( unsigned int t = 0; t < top_boxes.size(); t++ )
	{
		const WallBox top_box = top_boxes[t];
		glPushMatrix();
		{
			glColor3f( top_box.bR, top_box.bG, top_box.bB );
			glTranslatef( top_box.center.x, top_box.center.y, top_box.center.z );
			glutWireSphere( 2.0, 10, 10 );
			glScalef( top_box.halfdiag.x * 2.0f,
					  top_box.halfdiag.y * 2.0f,
					  top_box.halfdiag.z * 2.0f );
			glutWireCube( 1.0f );
		}
		glPopMatrix();
	}
	for( unsigned int b = 0; b < bottom_boxes.size(); b++ )
	{
		const WallBox bottom_box = bottom_boxes[b];
		glPushMatrix();
		{
			glColor3f( bottom_box.bR, bottom_box.bG, bottom_box.bB );
			glTranslatef( bottom_box.center.x,
						  bottom_box.center.y,
						  bottom_box.center.z );
			glutWireSphere( 2.0, 10, 10 );
			glScalef( bottom_box.halfdiag.x * 2.0f,
					  bottom_box.halfdiag.y * 2.0f,
					  bottom_box.halfdiag.z * 2.0f );
			glutWireCube( 1.0f );
		}
		glPopMatrix();
	}
}
//
//=======================================================================================
//
bool SkyboxManager::tilesFrustumCheck( Frustum* frustum, vector<WallBox>& boxes )
{
	for( unsigned int b = 0; b < boxes.size(); b++ )
	{
		if( frustum->boxInFrustum( boxes[b].center,
								   boxes[b].halfdiag ) != Frustum::OUTSIDE )
		{
			return true;
		}
	}
	return false;
}
//
//=======================================================================================
//
void SkyboxManager::fill_vertex(	Vertex&		v,
									float		loc0,
									float		loc1,
									float		loc2,
									float		nor0,
									float		nor1,
									float		nor2,
									float		tex0,
									float		tex1	)
{
	INITVERTEX( v );
	v.color[0]		= 1.0f;
	v.color[1]		= 1.0f;
	v.color[2]		= 1.0f;
	v.color[3]		= 1.0f;
	v.location[0]	= loc0;
	v.location[1]	= loc1;
	v.location[2]	= loc2;
	v.normal[0]		= nor0;
	v.normal[1]		= nor1;
	v.normal[2]		= nor2;
	v.texture[0]	= tex0;
	v.texture[1]	= tex1;
	v.tangent[0]	= 0.0f;
	v.tangent[1]	= 0.0f;
	v.tangent[2]	= 0.0f;
}
//
//=======================================================================================
//
void SkyboxManager::fill_tangents( Vertex& vert1, Vertex& vert2, Vertex& vert3, Vertex& vert4 )
{
	vec3 T1 = findTangent( vert1, vert2, vert3 );
	vec3 T2 = findTangent( vert1, vert3, vert4 );
	vec3 T( (T1.x + T2.x) / 2.0f, (T1.y + T2.y) / 2.0f, (T1.z + T2.z) / 2.0f );
	vert1.tangent[0] = T.x; vert1.tangent[1] = T.y; vert1.tangent[2] = T.z;
	vert2.tangent[0] = T.x; vert2.tangent[1] = T.y; vert2.tangent[2] = T.z;
	vert3.tangent[0] = T.x; vert3.tangent[1] = T.y; vert3.tangent[2] = T.z;
	vert4.tangent[0] = T.x; vert4.tangent[1] = T.y; vert4.tangent[2] = T.z;
}
//
//=======================================================================================
