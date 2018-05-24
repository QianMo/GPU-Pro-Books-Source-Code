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
#include "cCrowd.h"

//=======================================================================================
//
Crowd::Crowd(	LogManager*		log_manager_,
				GlErrorManager*	err_manager_,
				FboManager*		fbo_manager_,
				VboManager*		vbo_manager_,
				GlslManager*	glsl_manager_,
				string&			name_,
				GLuint			width_,
				GLuint			height_			)
{
	log_manager				= log_manager_;
	err_manager				= err_manager_;
	fbo_manager				= fbo_manager_;
	vbo_manager				= vbo_manager_;
	glsl_manager			= glsl_manager_;

	name					= string( name_ );
	width					= width_;
	height					= height_;
	fbo_lod_name			= string( "FBO_LOD_" );
	fbo_lod_name.append( name );
	fbo_pos_tex_name		= string( "FBO_POS_TEX_" );
	fbo_pos_tex_name.append( name );
	fbo_ids_tex_name		= string( "FBO_IDS_TEX_" );
	fbo_ids_tex_name.append( name );
	path_param				= 0.0f;

	static_lod				= new StaticLod( glsl_manager, vbo_manager, err_manager, name );
	static_lod->init( width, height );
	cuda_path_manager	= NULL;
	for( unsigned int i = 0; i < NUM_LOD; i++ )
	{
		vbo_lod[i].id					= 0;
		vbo_lod[i].primitivesGenerated	= 0;
		vbo_lod[i].primitivesWritten	= 0;
		models_drawn.push_back( 0 );
	}

	pos_tc_index			= 0;
	pos_tc_frame			= 0;
	pos_tc_size				= 0;
	posTexCoords			= 0;
	initTexCoords();

	scene_width				= 0.0f;
	scene_height			= 0.0f;
	scene_width_in_tiles	= 0.0f;
	scene_height_in_tiles	= 0.0f;
	tile_width				= 0.0f;
	tile_height				= 0.0f;
	tile_side				= 0.0f;

	pos_crashes				= 0;
	//personal_space		= 35.0f;
	personal_space			= 10.0f;
}
//
//=======================================================================================
//
Crowd::~Crowd( void )
{

}
//
//=======================================================================================
//
void Crowd::addGroup(	CharacterGroup*		_group,
						StaticLod*			_static_lod,
						string&				_animation,
						string&				_fbo_lod_name,
						string&				_fbo_pos_tex_name,
						string&				_fbo_ids_tex_name,
						float				_percentage,
						GLuint				_frames,
						GLuint				_duration		)
{

	unsigned int total_agents = width * height;
	unsigned int group_agents_width = (unsigned int)sqrtf((float)total_agents * _percentage / 100.0f);
	//http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	group_agents_width--;
	group_agents_width |= group_agents_width >> 1;
	group_agents_width |= group_agents_width >> 2;
	group_agents_width |= group_agents_width >> 4;
	group_agents_width |= group_agents_width >> 8;
	group_agents_width |= group_agents_width >> 16;
	group_agents_width++;

	CrowdGroup* cg		= new CrowdGroup(	_group,
											_static_lod,
											_animation,
											_fbo_lod_name,
											_fbo_pos_tex_name,
											_fbo_ids_tex_name,
											_percentage,
											0.0f,
											_frames,
											_duration,
											0,
											group_agents_width,
											group_agents_width	);
	groups.push_back( cg );
}
//
//=======================================================================================
//
void Crowd::addGroup(	CharacterGroup*		_group,
						string&				_animation,
						float				_percentage,
						GLuint				_frames,
						GLuint				_duration		)
{
	CrowdGroup* cg		= new CrowdGroup(	_group,
											_animation,
											_percentage,
											0.0f,
											_frames,
											_duration,
											0			);
	groups.push_back( cg );
}
//
//=======================================================================================
//
void Crowd::initFboInputs( vector<InputFbo*>& fbo_inputs )
{
#ifdef GLOBAL_POS_TEXTURE
	vector<InputFboTexture*> crowd_input_fbo_textures;
	InputFboTexture* texA = new InputFboTexture( fbo_pos_tex_name,   InputFbo::GPGPU	);
	InputFboTexture* texB = new InputFboTexture( fbo_ids_tex_name,   InputFbo::GPGPU	);
	string depth_name = fbo_lod_name;
	depth_name.append( "_depth" );
	InputFboTexture* texC = new InputFboTexture( depth_name, InputFbo::DEPTH_NO_COMPARE	);
	crowd_input_fbo_textures.push_back( texA );
	crowd_input_fbo_textures.push_back( texB );
	crowd_input_fbo_textures.push_back( texC );
	InputFbo* ci_fbo = new InputFbo(	fbo_lod_name,
										GL_TEXTURE_RECTANGLE,
										crowd_input_fbo_textures,
										width,
										height					);
	fbo_inputs.push_back( ci_fbo );
#elif defined LOCAL_POS_TEXTURE
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		CrowdGroup* cg = groups[g];
		vector<InputFboTexture*> group_input_fbo_textures;
		InputFboTexture* texA = new InputFboTexture( cg->fbo_pos_tex_name,   InputFbo::GPGPU	);
		InputFboTexture* texB = new InputFboTexture( cg->fbo_ids_tex_name,   InputFbo::GPGPU	);
		string depth_name = cg->fbo_lod_name;
		depth_name.append( "_depth" );
		InputFboTexture* texC = new InputFboTexture( depth_name, InputFbo::DEPTH_NO_COMPARE		);
		group_input_fbo_textures.push_back( texA );
		group_input_fbo_textures.push_back( texB );
		group_input_fbo_textures.push_back( texC );
		InputFbo* gi_fbo = new InputFbo(	cg->fbo_lod_name,
											GL_TEXTURE_RECTANGLE,
											group_input_fbo_textures,
											cg->getWidth(),
											cg->getHeight()		);
		fbo_inputs.push_back( gi_fbo );
	}
#endif
}
//
//=======================================================================================
//
string& Crowd::getName(	void )
{
	return name;
}
//
//=======================================================================================
//
string& Crowd::getFboLodName( void )
{
	return fbo_lod_name;
}
//
//=======================================================================================
//
string& Crowd::getFboPosTexName( void )
{
	return fbo_pos_tex_name;
}
//
//=======================================================================================
//
GLuint&	Crowd::getCudaTBO( void )
{
	return cuda_path_manager->getPosTboId();
}
//
//=======================================================================================
//
GLuint Crowd::getWidth(	void )
{
	return width;
}
//
//=======================================================================================
//
GLuint Crowd::getHeight( void )
{
	return height;
}
//
//=======================================================================================
//
vector<CrowdGroup*>& Crowd::getGroups( void )
{
	return groups;
}
//
//=======================================================================================
//
bool Crowd::init_paths(	CROWD_POSITION&				crowd_position,
						SCENARIO_TYPE&				scenario_type,
						vector<float>&				scenario_speeds,
						float						plane_scale,
					    vector<vector<float>>&		_policies,
						unsigned int				grid_width,
						unsigned int				grid_height,
						vector<glm::vec2>&			occupation,
						vector<GROUP_FORMATION>&	formations			)
{
	pos_crashes				= 0;
	scene_width				= plane_scale * 2.0f;
	scene_height			= plane_scale * 2.0f;
	instance_speed.clear();

	if( scenario_type == ST_O1P2 )
	{
		personal_space	= 300.0f;
	}
	else
	{
		personal_space	= 35.0f;
	}

#ifdef GLOBAL_POS_TEXTURE

#ifdef CUDA_PATHS
	log_manager->log( LogManager::CROWD_MANAGER,	"CROWD::\"%s\" personal character space: %07.3f.",
													(char*)name.c_str(),
													personal_space 									);
	log_manager->log( LogManager::CROWD_MANAGER, 	"CROWD::\"%s\" initializing positions...",
													(char*)name.c_str() 							);


#if defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
	init_paths_hx(	crowd_position,
					scenario_type,
					plane_scale,
					_policies,
					grid_width,
					grid_height,
					occupation,
					formations			);
#elif defined MDPS_SQUARE_NOLCA || defined MDPS_SQUARE_LCA
#ifdef MDPS_SQUARE_LCA
	init_paths_sq(	crowd_position,
					scenario_type,
					plane_scale,
					_policies,
					grid_width,
					grid_height,
					occupation,
					formations			);
#elif defined MDPS_SQUARE_NOLCA
	init_paths_hx(	crowd_position,
					scenario_type,
					plane_scale,
					_policies,
					grid_width,
					grid_height,
					occupation,
					formations			);
#endif
#endif

	log_manager->log( LogManager::CROWD_MANAGER, "CROWD::\"%s\" positions initialized.", (char*)name.c_str() );
	log_manager->log( LogManager::CROWD_MANAGER, "CROWD::\"%s\" position crashes: %u.", (char*)name.c_str(), pos_crashes );

#else
	for( unsigned int s = 0; s < width * height; s++ )
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		float w = 0.0f;

#ifndef DEMO_ALONE

		x = (float)(rand() % (int)(plane_scale));
		z = (float)(rand() % (int)(plane_scale));
		int sign = rand() % 100;
		if( sign < 50 )
		{
			x = -x;
		}
		sign = rand() % 100;
		if( sign < 50 )
		{
			z = -z;
		}
#endif
		instance_positions_flat.push_back( x     );
		instance_positions_flat.push_back( y     );
		instance_positions_flat.push_back( z     );
		instance_positions_flat.push_back( w	 );

		instance_rotations_flat.push_back( 90.0f );		// CENITAL
		instance_rotations_flat.push_back( 0.0f  );		// AZIMUTH
		instance_rotations_flat.push_back( 0.0f  );
		instance_rotations_flat.push_back( 0.0f  );

		instance_control_points.push_back( x	 );
		instance_control_points.push_back( y	 );
		instance_control_points.push_back( z     );
		instance_control_points.push_back( w     );

		instance_control_points.push_back( x - 500.0f	);
		instance_control_points.push_back( y			);
		instance_control_points.push_back( z + 500.0f	);
		instance_control_points.push_back( w			);

		instance_control_points.push_back( x + 500.0f	);
		instance_control_points.push_back( y			);
		instance_control_points.push_back( z + 500.0f   );
		instance_control_points.push_back( w			);

		instance_control_points.push_back( x			);
		instance_control_points.push_back( y			);
		instance_control_points.push_back( z			);
		instance_control_points.push_back( w			);
	}
#endif

	cuda_path_manager = new MDPCudaPathManager( log_manager,
												vbo_manager,
												fbo_manager,
												fbo_lod_name,
												fbo_pos_tex_name );
#if defined MDPS_HEXAGON_NOLCA || defined MDPS_SQUARE_NOLCA
	return cuda_path_manager->init_nolca(	scene_width,
											scene_height,
											tile_side,
											tile_height,
											scene_width_in_tiles,
											scene_height_in_tiles,
											instance_positions_flat,
											instance_speed,
											policies[0],
											density							);
#elif defined MDPS_HEXAGON_LCA || defined MDPS_SQUARE_LCA
#if defined MDPS_HEXAGON_LCA
	return cuda_path_manager->init_lca	(	scene_width,
											scene_height,
											tile_side,
											tile_height,
											scene_width_in_tiles,
											scene_height_in_tiles,
											scene_width_in_tiles * LCA_RATIO,
											scene_height_in_tiles * LCA_RATIO,
											instance_positions_flat,
											instance_speed,
											policies,
											density,
											scenario_speeds					);
#elif defined MDPS_SQUARE_LCA
	return cuda_path_manager->init_lca	(	scene_width,
											scene_height,
											tile_width,
											tile_height,
											(unsigned int)scene_width_in_tiles,
											(unsigned int)scene_height_in_tiles,
											(unsigned int)scene_width_in_tiles * LCA_RATIO,
											(unsigned int)scene_height_in_tiles * LCA_RATIO,
											instance_positions_flat,
											instance_speed,
											policies,
											density,
											scenario_speeds									);
#endif
#endif


#elif defined LOCAL_POS_TEXTURE
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		CrowdGroup* group = groups[g];
		if( group->init_paths(	log_manager,
								fbo_manager,
								vbo_manager,
								plane_scale	) == false )
		{
			return false;
		}
	}
	return true;
#endif
}
//
//=======================================================================================
//
void Crowd::init_paths_hx(	CROWD_POSITION&				crowd_position,
							SCENARIO_TYPE&				scenario_type,
							float						plane_scale,
							vector<vector<float>>&		_policies,
							unsigned int				grid_width,
							unsigned int				grid_height,
							vector<glm::vec2>&			occupation,
							vector<GROUP_FORMATION>&	formations		)
{
	float cellWidth			= scene_width  / (float)grid_width;
	float cellHeight		= scene_height / (float)grid_height;

	scene_width_in_tiles	= (float)grid_width;
	scene_height_in_tiles	= (float)grid_height;
	tile_width				= cellWidth;
	tile_height				= cellHeight;
#if defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
	float R					= tile_height / sqrtf( 3.0f );
	tile_width				= R * 2.0f;
	tile_side				= 3.0f * R / 2.0f;
#endif

	//0...DIR=POLICY, DIR+1=WALL, DIR+2=EXIT
	policies				= _policies;
	for( unsigned int i = 0; i < policies[0].size(); i++ )
	{
		density.push_back( 0.0f );
	}

	vector<float> xSectors;
	vector<float> zSectors;
	vector<float> fSectors;

	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		float xSector			= (float)(rand() % grid_width);
		float zSector			= (float)(rand() % grid_height);
		float fSector			= zSector * (float)grid_width + xSector;
		unsigned int iSector	= (unsigned int)fSector;
		bool sUsed				= false;
		for( unsigned int f = 0; f < fSectors.size(); f++ )
		{
			if( fSector == fSectors[f] )
			{
				sUsed = true;
				break;
			}
		}
		float NQ = (float)DIRECTIONS;
		while( !is_policy_open( iSector ) || sUsed )
		{
			xSector		= (float)(rand() % grid_width);
			zSector		= (float)(rand() % grid_height);
			fSector		= zSector * (float)grid_height + xSector;
			iSector		= (unsigned int)fSector;
			sUsed		= false;
			for( unsigned int f = 0; f < fSectors.size(); f++ )
			{
				if( fSector == fSectors[f] )
				{
					sUsed = true;
					break;
				}
			}
		}
		xSectors.push_back( xSector );
		zSectors.push_back( zSector );
		fSectors.push_back( fSector );
	}

	float zDisp = 0.0f;
	float xDisp = 0.0f;

	unsigned int aCounter = 0;
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		GROUP_FORMATION form = formations[g];
		unsigned int gSize = (unsigned int)(groups[g]->percentage / 100.0f * (float)width * (float)height);
		for( unsigned int a = 0; a < gSize; a++ )
		{
			float xSector			= xSectors[aCounter%groups.size()];
			float zSector			= zSectors[aCounter%groups.size()];
			float fSector			= zSector * (float)grid_height + xSector;
			unsigned int iSector	= (unsigned int)fSector;

			float x = 0.0f;
			int fY = (int)rand_between( 15.0f, 20.0f );
			float y = (float)fY;
			float z = 0.0f;
			float w = 0.0f;
			glm::vec2 pos( x, z );
			do
			{
				switch( form )
				{
				case GFRM_TRIANGLE:
				case GFRM_SQUARE:
				case GFRM_HEXAGON:
					x = xSector * cellWidth  + (float)(rand() % (int)cellWidth);
					z = zSector * cellHeight + (float)(rand() % (int)cellHeight);
#if defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
					x = xSector * tile_side  + rand_between( R / 2.0f, tile_side );
					z = zSector * cellHeight + rand_between( cellHeight / 6.0f, 5.0f * cellHeight / 6.0f );
					if( (int)xSector % 2 == 1 )
					{
						z += cellHeight / 2.0f;
					}
#endif
					break;
				case GFRM_CIRCLE:
					//(x-h)^2 + (y-k)^2 = r^2
					// x = a + r cos(t)
					// y = b + r sin(t)
					float r = rand_between( 0.0001f, cellWidth / 2.0f );
					if( cellHeight < cellWidth )
					{
						r = rand_between( 0.0001f, cellHeight / 2.0f );
					}
					float rNum = rand_between( 0.0f, 2.0f * (float)M_PI );
					x = (xSector * cellWidth  + cellWidth/2.0f)  + (r * cos(rNum));
					z = (zSector * cellHeight + cellHeight/2.0f) + (r * sin(rNum));
#if defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
					r = rand_between( 0.0001f, 2.0f * R / 3.0f );
					x = (xSector * tile_side  + tile_width / 2.0f)  + (r * cos(rNum));
					z = (zSector * cellHeight  + cellHeight / 2.0f) + (r * sin(rNum));
					if( (int)xSector % 2 == 1 )
					{
						z += cellHeight / 2.0f;
					}
#endif
					break;
				}
				x -= plane_scale;
				z -= plane_scale;
				pos.x = x;
				pos.y = z;
			}
			while( pos_occupied( occupation, pos, personal_space ) );


			float accel = 1.0f;
			float test = rand_between( 0.0f, 10.0f );
			if( test < 10.0f / 3.0f )
			{
				accel = 2.0f;
			}
			else if( test < 20.0f / 3.0f )
			{
				accel = 5.0f;
			}
			else
			{
				accel = 10.0f;
			}

			w = accel / 100.0f;


			density[iSector] += 1.0f;

			instance_positions_flat.push_back	( x				);
			instance_positions_flat.push_back	( y				);
			instance_positions_flat.push_back	( z				);
			instance_positions_flat.push_back	( w				);

			instance_speed.push_back			( w				);

			instance_rotations_flat.push_back	( 90.0f			);		// CENITAL
			instance_rotations_flat.push_back	( 0.0f			);		// AZIMUTH
			instance_rotations_flat.push_back	( 0.0f			);
			instance_rotations_flat.push_back	( 0.0f			);

			instance_control_points.push_back	( x				);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z				);
			instance_control_points.push_back	( w				);

			instance_control_points.push_back	( x - 500.0f	);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z + 500.0f	);
			instance_control_points.push_back	( w				);

			instance_control_points.push_back	( x + 500.0f	);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z + 500.0f	);
			instance_control_points.push_back	( w				);

			instance_control_points.push_back	( x				);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z				);
			instance_control_points.push_back	( w				);

			aCounter++;
		}
	}
}
//
//=======================================================================================
//
void Crowd::init_paths_sq(	CROWD_POSITION&				crowd_position,
							SCENARIO_TYPE&				scenario_type,
							float						plane_scale,
							vector<vector<float>>&		_policies,
							unsigned int				grid_width,
							unsigned int				grid_height,
							vector<glm::vec2>&			occupation,
							vector<GROUP_FORMATION>&	formations			)
{
	float cellWidth			= scene_width  / (float)grid_width;
	float cellHeight		= scene_height / (float)grid_height;

	scene_width_in_tiles	= (float)grid_width;
	scene_height_in_tiles	= (float)grid_height;
	tile_width				= cellWidth;
	tile_height				= cellHeight;

	//0...DIR=POLICY, DIR+1=WALL, DIR+2=EXIT
	policies				= _policies;
	for( unsigned int i = 0; i < policies[0].size(); i++ )
	{
		density.push_back( 0.0f );
	}

	unsigned int aCounter	= 0;
	float xSector			= 0.0f;
	float zSector			= 0.0f;
	float fSector			= 0.0f;
	unsigned int iSector	= 0;
	float one_prc			= plane_scale / 50.0f;

	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		GROUP_FORMATION form = formations[g];
		unsigned int gSize = (unsigned int)(groups[g]->percentage / 100.0f * (float)width * (float)height);
		for( unsigned int a = 0; a < gSize; a++ )
		{
			float x = 0.0f;
			int fY = (int)rand_between( 15.0f, 20.0f );
			float y = (float)fY;
			float z = 0.0f;
			float w = 0.0f;
			glm::vec2 pos( x, z );
			do
			{
				switch(crowd_position)
				{
				case CPOS_NONE:
					x = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					z = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					break;
				case CPOS_TOP:
					x = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					z = rand_between( -plane_scale + one_prc,			one_prc							);
					break;
				case CPOS_BOTTOM:
					x = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					z = rand_between( -one_prc,							plane_scale - one_prc			);
					break;
				case CPOS_LEFT:
					x = rand_between( -plane_scale + one_prc,			one_prc							);
					z = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					break;
				case CPOS_RIGHT:
					x = rand_between( -one_prc,							plane_scale - one_prc			);
					z = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					break;
				case CPOS_CENTER:
					x = rand_between( -(plane_scale/2.0f) + one_prc,	(plane_scale/2.0f) - one_prc	);
					z = rand_between( -(plane_scale/2.0f) + one_prc,	(plane_scale/2.0f) - one_prc	);
					break;
				default:
					x = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
					z = rand_between( -plane_scale + one_prc,			plane_scale - one_prc			);
				}

				xSector = floor( (x + plane_scale) / cellWidth );
				zSector = floor( (z + plane_scale) / cellHeight );
				fSector	= zSector * (float)grid_height + xSector;
				iSector	= (unsigned int)fSector;
				pos.x = x;
				pos.y = z;
			}
			while( pos_occupied( occupation, pos, personal_space ) || !is_policy_open( iSector ) );

			
			float accel = 1.0f;
			float test = rand_between( 0.0f, 10.0f );
			

			if( scenario_type == ST_O1P2 )
			{
				/*
				if( test < 5.0f )
				{
					accel = 5.0f;
				}
				else
				{
					accel = 10.0f;
				}
				w = accel / 50.0f;
				*/
				w = rand_between( 0.04f, 0.15f );

			}
			else if( scenario_type == ST_EIFFEL )
			{
				/*
				if( test < 10.0f / 3.0f )
				{
					accel = 0.00333f;
				}
				else if( test < 20.0f / 3.0f )
				{
					accel = 0.005f;
				}
				else
				{
					accel = 0.01f;
				}
				w = accel;
				*/
				w = rand_between( 0.004f, 0.01f );
			}
			else
			{
				/*
				if( test < 10.0f / 3.0f )
				{
					accel = 2.0f;
				}
				else if( test < 20.0f / 3.0f )
				{
					accel = 5.0f;
				}
				else
				{
					accel = 10.0f;
				}
				w = accel / 100.0f;
				*/
				w = rand_between( 0.005f, 0.08f );
			}


			density[iSector] += 1.0f;

			instance_positions_flat.push_back	( x				);
			instance_positions_flat.push_back	( y				);
			instance_positions_flat.push_back	( z				);
			instance_positions_flat.push_back	( w				);

			instance_rotations_flat.push_back	( 90.0f			);		// CENITAL
			instance_rotations_flat.push_back	( 0.0f			);		// AZIMUTH
			instance_rotations_flat.push_back	( 0.0f			);
			instance_rotations_flat.push_back	( 0.0f			);

			instance_control_points.push_back	( x				);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z				);
			instance_control_points.push_back	( w				);

			instance_control_points.push_back	( x - 500.0f	);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z + 500.0f	);
			instance_control_points.push_back	( w				);

			instance_control_points.push_back	( x + 500.0f	);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z + 500.0f	);
			instance_control_points.push_back	( w				);

			instance_control_points.push_back	( x				);
			instance_control_points.push_back	( y				);
			instance_control_points.push_back	( z				);
			instance_control_points.push_back	( w				);

			instance_speed.push_back			( w				);

			aCounter++;
		}
	}
}
//
//=======================================================================================
//
void Crowd::updatePolicy( vector<float>& _policy, unsigned int channel )
{
	if( channel < MDP_CHANNELS )
	{
		policies[channel] = _policy;
		cuda_path_manager->updateMDPPolicy( _policy, channel );
	}
}
//
//=======================================================================================
//
void Crowd::updatePolicies( vector<vector<float>>& _policies )
{
	policies = _policies;
	cuda_path_manager->updateMDPPolicies( _policies );
}
//
//=======================================================================================
//
void Crowd::getDensity( vector<float>& _density )
{
	cuda_path_manager->getDensity( _density );
}
//
//=======================================================================================
//
void Crowd::nextFrame( void )
{
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		groups[g]->nextFrame();
	}
}
//
//=======================================================================================
//
void Crowd::run_paths( void )
{
#ifdef GLOBAL_POS_TEXTURE
	cuda_path_manager->runCuda( width,
								height,
								path_param			);
	path_param += 0.01f;
	if( path_param > 1.0f )
	{
		// Reset param:
		path_param = 0.0f;
	}
#elif defined LOCAL_POS_TEXTURE
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		groups[g]->run_paths();
	}
#endif
}
//
//=======================================================================================
//
void Crowd::setFboManager( FboManager* _fbo_manager )
{
	fbo_manager = _fbo_manager;
}
//
//=======================================================================================
//
void Crowd::init_ids(	GLuint&	groupOffset,
						GLuint&	agentOffset	)
{
#ifdef GLOBAL_POS_TEXTURE
	unsigned int groupId = groupOffset;
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		unsigned int group_num_agents = (unsigned int)(groups[g]->percentage/100.0f * (float)(width * height));
		GLuint u_id = 0;
		for( unsigned int n = 0; n < group_num_agents; n++ )
		{
			u_id = agentOffset + n;
			instance_ids_flat.push_back( (float)groupId	);	//GROUP_ID
			instance_ids_flat.push_back( (float)u_id	);	//UNIQUE_ID
			instance_ids_flat.push_back( (float)0.0f	);	//FREE
			instance_ids_flat.push_back( (float)0.0f	);	//FREE
		}
		agentOffset = agentOffset + group_num_agents;
		groups[g]->id = groupId;
		groupId++;
	}
	groupOffset = groupId;

	float* data = new float[width*height*4];
	for( unsigned int i = 0; i < width*height*4; i++ )
	{
		data[i] = 0.0f;
	}
	for( unsigned int i = 0; i < instance_ids_flat.size(); i++ )
	{
		data[i] = instance_ids_flat[i];
	}

	glBindTexture( fbo_manager->fbos[fbo_lod_name].fbo_tex_target, fbo_manager->texture_ids[fbo_ids_tex_name] );
	{
		glTexSubImage2D( fbo_manager->fbos[fbo_lod_name].fbo_tex_target,
						 0,
						 0,
						 0,
						 fbo_manager->fbos[fbo_lod_name].fbo_width,
						 fbo_manager->fbos[fbo_lod_name].fbo_height,
						 GL_RGBA,
						 GL_FLOAT,
						 data										);
	}
	glBindTexture( fbo_manager->fbos[fbo_lod_name].fbo_tex_target, 0 );
	delete data;
	return;
#elif defined LOCAL_POS_TEXTURE
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
		CrowdGroup* cg	= groups[g];
		GLuint gId		= groupOffset;
		agentOffset		= cg->init_ids( fbo_manager, gId, agentOffset );
		groupOffset		= gId;
	}
#endif
}
//
//=======================================================================================
//
#ifdef DEMO_SHADER
void Crowd::draw(	Camera*			camera,
					float*			viewMat,
					float*			projMat,
					float*			shadowMat,
					bool			wireframe,
					bool			shadows,
					bool			doHandD,
					bool			doPatterns,
					bool			doColor,
					bool			doFacial	)
{
#ifdef GLOBAL_POS_TEXTURE
	GLuint fbo_target		= fbo_manager->fbos[fbo_lod_name].fbo_tex_target;
	GLuint ids_texture		= fbo_manager->texture_ids[fbo_ids_tex_name];

#ifdef CUDA_PATHS
	static_lod->runAssignmentCuda(	fbo_target,
									cuda_path_manager->getPosTboId(),
									ids_texture,
									vbo_lod,
									camera								);
#else
	GLuint fbo_texture		= fbo_manager->texture_ids[fbo_pos_tex_name];
	static_lod->runAssignment(	fbo_target,
								fbo_texture,
								ids_texture,
								vbo_lod,
								camera		);
#endif
	//printf( "CROWD primitivesWritten=%d\n", static_lod->primitivesWritten[0] );
	for( unsigned int lod = 0; lod < NUM_LOD; lod++ )
	{
		models_drawn[lod] = 0;
	}

	for( unsigned int g = 0; g < groups.size(); g++ )
	{
#ifdef CUDA_PATHS
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							cuda_path_manager,
							static_lod,
							vbo_lod,
							fbo_lod_name,
							fbo_pos_tex_name,
							width,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows,
							doHandD,
							doPatterns,
							doColor,
							doFacial		);
#else
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							static_lod,
							vbo_lod,
							fbo_lod_name,
							fbo_pos_tex_name,
							width,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows,
							doHandD,
							doPatterns,
							doColor,
							doFacial		);
#endif
		for( unsigned int lod = 0; lod < NUM_LOD; lod++ )
		{
			models_drawn[lod] += groups[g]->vbo_lod[lod].primitivesWritten;
		}
	}
#elif defined LOCAL_POS_TEXTURE
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
#ifdef CUDA_PATHS
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							cuda_path_manager,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows,
							doHandD,
							doPatterns,
							doColor,
							doFacial	);
#else
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows,
							doHandD,
							doPatterns,
							doColor,
							doFacial	);
#endif
	}
#endif
//
//=======================================================================================
//
#else
void Crowd::draw(	Camera*			camera,
					float*			viewMat,
					float*			projMat,
					float*			shadowMat,
					bool			wireframe,
					bool			shadows		)
{
#ifdef GLOBAL_POS_TEXTURE
	GLuint fbo_target		= fbo_manager->fbos[fbo_lod_name].fbo_tex_target;
	GLuint fbo_texture		= fbo_manager->texture_ids[fbo_pos_tex_name];
	GLuint ids_texture		= fbo_manager->texture_ids[fbo_ids_tex_name];
	static_lod->runAssignment(	fbo_target,
								fbo_texture,
								ids_texture,
								vbo_lod,
								camera		);
	//printf( "CROWD primitivesWritten=%d\n", static_lod->primitivesWritten[0] );
	for( unsigned int lod = 0; lod < NUM_LOD; lod++ )
	{
		models_drawn[lod] = 0;
	}

	for( unsigned int g = 0; g < groups.size(); g++ )
	{
#ifdef CUDA_PATHS
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							cuda_path_manager,
							static_lod,
							vbo_lod,
							fbo_lod_name,
							fbo_pos_tex_name,
							width,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows			);
#else
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							static_lod,
							vbo_lod,
							fbo_lod_name,
							fbo_pos_tex_name,
							width,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows			);
#endif
		for( unsigned int lod = 0; lod < NUM_LOD; lod++ )
		{
			models_drawn[lod] += groups[g]->vbo_lod[lod].primitivesWritten;
		}
	}
#elif defined LOCAL_POS_TEXTURE
	for( unsigned int g = 0; g < groups.size(); g++ )
	{
#ifdef CUDA_PATHS
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							cuda_path_manager,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows		);
#else
		groups[g]->draw(	fbo_manager,
							vbo_manager,
							glsl_manager,
							camera,
							viewMat,
							projMat,
							shadowMat,
							wireframe,
							shadows		);
#endif
	}
#endif
#endif
}
//
//=======================================================================================
//
void Crowd::initTexCoords( void )
{
	pos_tc_index = 0;
	pos_tc_frame = 0;
	string vbo_name = "POS_TEX_COORDS_CUDA_";
	vbo_name.append( name );
	pos_tc_index = vbo_manager->createVBOContainer( vbo_name, pos_tc_frame );

	unsigned int i, j;
	for( i = 0; i < width; i++ )
	{
		for( j = 0; j < height; j++ )
		{
			Vertex v;
			INITVERTEX( v );
			v.location[0] = (float)i;
			v.location[1] = (float)j;
			v.location[2] = 0.0f;
			v.location[3] = 1.0f;
			vbo_manager->vbos[pos_tc_index][pos_tc_frame].vertices.push_back( v );
		}
	}

	vbo_manager->gen_vbo( posTexCoords, pos_tc_index, pos_tc_frame );
	pos_tc_size = vbo_manager->vbos[pos_tc_index][pos_tc_frame].vertices.size();
}
//
//=======================================================================================
//
GLuint& Crowd::getPosTexCoords( void )
{
	return posTexCoords;
}
//
//=======================================================================================
//
GLuint Crowd::getPosTexSize( void )
{
	return pos_tc_size;
}
//
//=======================================================================================
//
float Crowd::rand_between(	float minf,
							float maxf )
{
	return minf + ((float)rand() / (float)RAND_MAX) * (maxf - minf);
}
//
//=======================================================================================
//
bool Crowd::pos_occupied(	vector<glm::vec2>&	occupation,
							glm::vec2			position,
							float				radius		)
{
	float rA = radius;
	float rB = radius;
	float r2 = rA + rB;
	r2 *= r2;
	vec2  cB = position;

	for( unsigned int o = 0; o < occupation.size(); o++ )
	{
		vec2  cA = occupation[o];
		float dx = cB.x - cA.x;
		dx *= dx;
		float dy = cB.y - cA.y;
		dy *= dy;
		float d2 = dx + dy;

		if( d2 <= r2 )
		{
			pos_crashes++;
			//printf( "POS_CRASH[%04d]::(%09.3f,%09.3f)::(%09.3f,%09.3f)\n", pos_crashes, cA.x, cA.y, cB.x, cB.y );
			return true;
		}
	}
	occupation.push_back( position );
	return false;
}
//
//=======================================================================================
//
bool Crowd::is_policy_open( unsigned int index )
{
	float NQ = (float)DIRECTIONS;
	for( unsigned int p = 0; p < policies.size(); p++ )
	{
		if( policies[p][index] >= NQ ) return false;
	}
	return true;
}
//
//=======================================================================================
//
float Crowd::getAvgRacingQW( void )
{
	return cuda_path_manager->getAvgRacingQW();
}
//
//=======================================================================================
//
float Crowd::getAvgScatterGather( void )
{
	return cuda_path_manager->getAvgScatterGather();
}
//
//=======================================================================================
