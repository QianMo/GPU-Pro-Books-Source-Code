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
#include "cMDPCudaPathManager.h"

//=======================================================================================
//
//ccMDP_square_paths.cu
extern "C" void launch_mdp_square_kernel(	float*			current_positions,
											float*			original_positions,
											float*			policy,
											float*			density,
											float			scene_width_in_tiles,
											float			scene_height_in_tiles,
											float			tile_width,
											float			tile_height,
											unsigned int	mesh_width,
											unsigned int	mesh_height,
											float			time,
											bool			policy_reset		);
//
//=======================================================================================
//
//ccMDP_hexagon_paths.cu
extern "C" void launch_mdp_hexagon_kernel(	float*			current_positions,
											float*			original_positions,
											float*			policy,
											float*			density,
											float			scene_width,
											float			scene_height,
											float			scene_width_in_tiles,
											float			scene_height_in_tiles,
											float			tile_side,
											float			tile_height,
											unsigned int	mesh_width,
											unsigned int	mesh_height,
											float			time,
											bool			policy_reset		);
//
//=======================================================================================
//
//ccMDP_LCA_paths.cu
extern "C" void update_mdp_lca			(	std::vector<std::vector<float>>&	lca,
											std::vector<std::vector<float>>&	policy,
											float								u_time		);
//
//=======================================================================================
//
//ccMDP_LCA_paths.cu
extern "C" void launch_lca_kernel		(	std::vector<float>&					pos,
											std::vector<std::vector<float>>&	lca,
											std::vector<float>&					density,
											float&								avg_racing_qw,
											float&								avg_scatter_gather	);

//
//=======================================================================================
//
//ccMDP_LCA_paths.cu
extern "C" void init_cells_and_agents	(	unsigned int						mdp_channels,
											std::vector<float>&					pos,
											std::vector<float>&					speed,
											std::vector<float>&					scene_speeds,
											std::vector<std::vector<float>>&	lca,
											std::vector<std::vector<float>>&	policy,
											float								scene_width,
											float								scene_depth,
											unsigned int						lca_width,
											unsigned int						lca_depth,
											unsigned int						mdp_width,
											unsigned int						mdp_depth,
											unsigned int						num_agents,
											bool&								result		);
//
//=======================================================================================
//
//ccMDP_LCA_paths.cu
extern "C" void cleanup_lca				(	void								);
//
//=======================================================================================
//
MDPCudaPathManager::MDPCudaPathManager( LogManager*		log_manager,
										VboManager*		vbo_manager,
										FboManager*		fbo_manager,
										string			_fbo_pos_name,
										string			_fbo_pos_tex_name	)
{
	this->log_manager	= log_manager;
	this->vbo_manager	= vbo_manager;
	this->fbo_manager	= fbo_manager;
	proj_manager		= new ProjectionManager();
	fbo_pos_name		= string( _fbo_pos_name );
	fbo_pos_tex_name	= string( _fbo_pos_tex_name );
	string vbo_name;
	avg_racing_qw		= 0.0f;
	avg_scatter_gather	= 0.0f;


#if defined MDPS_SQUARE_NOLCA || defined MDPS_HEXAGON_NOLCA
	vbo_name = string( "cuda_init_positions" );
	cuda_ipos_vbo_frame	= 0;
	cuda_ipos_vbo_index	=
		vbo_manager->createGlCudaVboContainer( vbo_name,
											   cuda_ipos_vbo_frame					);
	log_manager->log( LogManager::CUDA, "Created GL-CUDA init positions container."	);
	cuda_ipos_vbo_id	= 0;
	cuda_ipos_vbo_size	= 0;
	cuda_ipos_vbo_res	= NULL;

	vbo_name = string( "cuda_curr_positions" );
	cuda_cpos_vbo_frame	= 0;
	cuda_cpos_vbo_index	=
		vbo_manager->createGlCudaVboContainer( vbo_name,
											   cuda_cpos_vbo_frame					);
	log_manager->log( LogManager::CUDA, "Created GL-CUDA curr positions container." );
	cuda_cpos_vbo_id	= 0;
	cuda_cpos_vbo_size	= 0;
	cuda_cpos_vbo_res	= NULL;

	vbo_name = string( "cuda_policy" );
	cuda_poli_vbo_frame	= 0;
	cuda_poli_vbo_index	=
		vbo_manager->createGlCudaVboContainer( vbo_name,
											   cuda_poli_vbo_frame					);
	log_manager->log( LogManager::CUDA, "Created GL-CUDA policy container."			);
	cuda_poli_vbo_id	= 0;
	cuda_poli_vbo_size	= 0;
	cuda_poli_vbo_res	= NULL;

	vbo_name = string( "cuda_density" );
	cuda_dens_vbo_frame	= 0;
	cuda_dens_vbo_index	=
		vbo_manager->createGlCudaVboContainer( vbo_name,
											   cuda_dens_vbo_frame					);
	log_manager->log( LogManager::CUDA, "Created GL-CUDA density container."		);
	cuda_dens_vbo_id	= 0;
	cuda_dens_vbo_size	= 0;
	cuda_dens_vbo_res	= NULL;
#elif defined MDPS_SQUARE_LCA || defined MDPS_HEXAGON_LCA
	vbo_name				= string( "cuda_curr_positions" );
	cuda_cpos_vbo_frame		= 0;
	cuda_cpos_vbo_index		=
		vbo_manager->createVBOContainer	(	vbo_name,
											cuda_cpos_vbo_frame						);
	log_manager->log( LogManager::CUDA, "Created GL-CUDA curr positions container."	);
	cuda_cpos_vbo_id		= 0;
	cuda_cpos_vbo_size		= 0;

	vbo_name				= string( "cuda_goals" );
	cuda_goals_vbo_frame	= 0;
	cuda_goals_vbo_index	=
		vbo_manager->createVBOContainer	(	vbo_name,
											cuda_goals_vbo_frame					);
	log_manager->log( LogManager::CUDA, "Created GL goals container."				);
	cuda_goals_vbo_id		= 0;
	cuda_goals_vbo_size		= 0;

	vbo_name = string( "cuda_density" );
	cuda_dens_vbo_frame		= 0;
	cuda_dens_vbo_index		=
		vbo_manager->createVBOContainer	(	vbo_name,
											cuda_dens_vbo_frame						);
	log_manager->log( LogManager::CUDA, "Created GL density container."				);
	cuda_dens_vbo_id		= 0;
	cuda_dens_vbo_size		= 0;
#endif

	mdp_policy_reset		= false;

	pos_tbo_id				= 0;

	scene_width				= 0.0f;
	scene_depth				= 0.0f;
	tile_width_or_side		= 0.0f;
	tile_depth				= 0.0f;

	mdp_width				= 0;
	mdp_depth				= 0;

	lca_width				= 0;
	lca_depth				= 0;

	lca_cells_in_mdp_cell	= 0;
	lca_in_mdp_width		= 0;
	lca_in_mdp_depth		= 0;

	//cudaSetDevice( getMaxGflopsDeviceId() );
	//cudaGLSetGLDevice( getMaxGflopsDeviceId() );
}
//
//=======================================================================================
//
MDPCudaPathManager::~MDPCudaPathManager( void )
{
	render_textures.clear();
	init_positions.clear();
	curr_positions.clear();
	init_speeds.clear();
	mdp_policies.clear();
	lca_exits.clear();
	density.clear();
	FREE_INSTANCE( proj_manager );
	cleanup_lca();
}
//
//=======================================================================================
//
bool MDPCudaPathManager::init_nolca(	float				_scene_width,
										float				_scene_depth,
										float				_tile_width_or_side,
										float				_tile_depth,
										unsigned int		_mdp_width,
										unsigned int		_mdp_depth,
										vector<float>&		_init_pos,
										vector<float>&		_init_speed,
										vector<float>&		_mdp_policy,
										vector<float>&		_density			)
{
	scene_width			= _scene_width;
	scene_depth			= _scene_depth;
	tile_width_or_side	= _tile_width_or_side;
	tile_depth			= _tile_depth;
	mdp_width			= _mdp_width;
	mdp_depth			= _mdp_depth;
	init_positions		= _init_pos;
	curr_positions		= _init_pos;
	init_speeds			= _init_speed;
	vector<float> mdp_policy = _mdp_policy;
	mdp_policies.push_back( mdp_policy );
	density				= _density;

	bool result = init_nolca();
	//cudaSetDevice(getMaxGflopsDeviceId());
	//cudaGLSetGLDevice(getMaxGflopsDeviceId());
	return result;
}
//
//=======================================================================================
//
bool MDPCudaPathManager::init_lca(	float					_scene_width,
									float					_scene_depth,
									float					_tile_width_or_side,
									float					_tile_depth,
									unsigned int			_mdp_width,
									unsigned int			_mdp_depth,
									unsigned int			_lca_width,
									unsigned int			_lca_depth,
									vector<float>&			_init_pos,
									vector<float>&			_init_speed,
									vector<vector<float>>&	_mdp_policies,
									vector<float>&			_density,
									vector<float>&			_scenario_speeds	)
{
	scene_width			= _scene_width;
	scene_depth			= _scene_depth;
	tile_width_or_side	= _tile_width_or_side;
	tile_depth			= _tile_depth;
	mdp_width			= _mdp_width;
	mdp_depth			= _mdp_depth;
	lca_width			= _lca_width;
	lca_depth			= _lca_depth;
	init_positions		= _init_pos;
	curr_positions		= _init_pos;
	init_speeds			= _init_speed;
	mdp_policies		= _mdp_policies;
	density				= _density;
	scenario_speeds		= _scenario_speeds;

	if( setLCAExits() )
	{
		bool result = init_lca();
		cudaSetDevice(getMaxGflopsDeviceId());
		cudaGLSetGLDevice(getMaxGflopsDeviceId());
		return result;
	}
	else
	{
		log_manager->log( LogManager::LERROR, "While setting LCA Exits." );
		return false;
	}
}
//
//=======================================================================================
//
bool MDPCudaPathManager::init_lca( void )
{
	if( init_positions.size() % 4 == 0 )
	{
		for( unsigned int i = 0; i < init_positions.size(); i += 4 )
		{
			Vertex v;
			INITVERTEX( v );
			v.location[0]		= init_positions[i+0];
			v.location[1]		= init_positions[i+1];
			v.location[2]		= init_positions[i+2];
			v.location[3]		= init_positions[i+3];
			vbo_manager->vbos[cuda_cpos_vbo_index][cuda_cpos_vbo_frame].vertices.push_back( v );
		}

		cuda_cpos_vbo_size		= vbo_manager->gen_vbo(		cuda_cpos_vbo_id,
															cuda_cpos_vbo_index,
															cuda_cpos_vbo_frame	);

		log_manager->log( LogManager::CUDA, "Generated GL positions VBO. Vertices: %u (%uKB).",
										    cuda_cpos_vbo_size,
										    cuda_cpos_vbo_size * sizeof( float ) / 1024			);

		// ALSO_ATTACH_A_TEXTURE_BUFFER_OBJECT:
		glGenTextures	( 1, &pos_tbo_id									);
		glActiveTexture ( GL_TEXTURE7										);
		glBindTexture	( GL_TEXTURE_BUFFER, pos_tbo_id						);
		glTexBuffer		( GL_TEXTURE_BUFFER, GL_RGBA32F, cuda_cpos_vbo_id	);
		glBindTexture	( GL_TEXTURE_BUFFER, (GLuint)0						);


		cuda_dens_vbo_size		= vbo_manager->gen_vbo3(	cuda_dens_vbo_id,
															density,
															cuda_dens_vbo_index,
															cuda_dens_vbo_frame				);
		log_manager->log( LogManager::CUDA, "Generated GL density VBO. Floats: %u (%uKB).",
										    cuda_dens_vbo_size,
										    cuda_dens_vbo_size * sizeof( float ) / 1024		);


		glBindBuffer					(	GL_ARRAY_BUFFER, 
											cuda_cpos_vbo_id						);
		glBufferSubData					(	GL_ARRAY_BUFFER,
											0,
											sizeof(float) * curr_positions.size(), 
											&curr_positions[0]						);
		glBindBuffer					(	GL_ARRAY_BUFFER,
											0										);

		bool result = false;
		init_cells_and_agents			(	MDP_CHANNELS,
											curr_positions,
											init_speeds,
											scenario_speeds,
											lca_exits,
											mdp_policies,
											scene_width,
											scene_depth,
											lca_width,
											lca_depth,
											mdp_width,
											mdp_depth,
											curr_positions.size() / 4,
											result									);

		if( result )
		{
			log_manager->log( LogManager::CUDA, "LCA Kernel initialized successfully." );
		}
		else
		{
			log_manager->log( LogManager::LERROR, "While initializing LCA Kernel." );
		}
		return result;
	}
	else
	{
		log_manager->log( LogManager::LERROR, "Wrong CUDA paths inputs size." );
		return false;
	}
}
//
//=======================================================================================
//
bool MDPCudaPathManager::init_nolca( void )
{
	if( init_positions.size() % 4 == 0 )
	{
		for( unsigned int i = 0; i < init_positions.size(); i += 4 )
		{
			Vertex vi;
			INITVERTEX( vi );
			vi.location[0]		= init_positions[i+0];
			vi.location[1]		= init_positions[i+1];
			vi.location[2]		= init_positions[i+2];
			vi.location[3]		= init_positions[i+3];
			Vertex vc;
			INITVERTEX( vc );
			vc.location[0]		= init_positions[i+0];
			vc.location[1]		= -1.0f;
			vc.location[2]		= init_positions[i+2];
			vc.location[3]		= init_positions[i+3];
			vbo_manager->gl_cuda_vbos[cuda_ipos_vbo_index][cuda_ipos_vbo_frame].vertices.push_back( vi );
			vbo_manager->gl_cuda_vbos[cuda_cpos_vbo_index][cuda_cpos_vbo_frame].vertices.push_back( vc );
		}

		cuda_ipos_vbo_size  = vbo_manager->gen_gl_cuda_vbo2(	cuda_ipos_vbo_id,
																cuda_ipos_vbo_res,
																cuda_ipos_vbo_index,
																cuda_ipos_vbo_frame	);
		log_manager->log( LogManager::CUDA, "Generated GL-CUDA init positions VBO. Vertices: %u (%uKB).",
										    cuda_ipos_vbo_size/4,
										    cuda_ipos_vbo_size * sizeof( float ) / 1024					);


		cuda_cpos_vbo_size  = vbo_manager->gen_gl_cuda_vbo2(	cuda_cpos_vbo_id,
																cuda_cpos_vbo_res,
																cuda_cpos_vbo_index,
																cuda_cpos_vbo_frame	);
		log_manager->log( LogManager::CUDA, "Generated GL-CUDA curr positions VBO. Vertices: %u (%uKB).",
										    cuda_cpos_vbo_size/4,
										    cuda_cpos_vbo_size * sizeof( float ) / 1024					);


		cuda_poli_vbo_size  = vbo_manager->gen_gl_cuda_vbo3(	cuda_poli_vbo_id,
																mdp_policies[0],
																cuda_poli_vbo_res,
																cuda_poli_vbo_index,
																cuda_poli_vbo_frame	);
		log_manager->log( LogManager::CUDA, "Generated GL-CUDA policy VBO. Floats: %u (%uKB).",
										    cuda_poli_vbo_size,
										    cuda_poli_vbo_size * sizeof( float ) / 1024					);

		cuda_dens_vbo_size  = vbo_manager->gen_gl_cuda_vbo3(	cuda_dens_vbo_id,
																density,
																cuda_dens_vbo_res,
																cuda_dens_vbo_index,
																cuda_dens_vbo_frame	);
		log_manager->log( LogManager::CUDA, "Generated GL-CUDA density VBO. Floats: %u (%uKB).",
										    cuda_dens_vbo_size,
										    cuda_dens_vbo_size * sizeof( float ) / 1024					);


		// ALSO_ATTACH_A_TEXTURE_BUFFER_OBJECT:
		glGenTextures	( 1, &pos_tbo_id									);
		glActiveTexture ( GL_TEXTURE7										);
		glBindTexture	( GL_TEXTURE_BUFFER, pos_tbo_id						);
		glTexBuffer		( GL_TEXTURE_BUFFER, GL_RGBA32F, cuda_cpos_vbo_id	);
		glBindTexture	( GL_TEXTURE_BUFFER, (GLuint)0						);

		VboRenderTexture texture_buffer;
		texture_buffer.id		= pos_tbo_id;
		texture_buffer.target	= GL_TEXTURE_BUFFER;
		texture_buffer.offset	= GL_TEXTURE7;
		render_textures.push_back( texture_buffer );

		return true;
	}
	else
	{
		log_manager->log( LogManager::CUDA, "ERROR::Wrong CUDA paths inputs size." );
		return false;
	}
}
//
//=======================================================================================
//
void MDPCudaPathManager::updateMDPPolicies( vector<vector<float>>& _policies )
{
	mdp_policies		= _policies;
	glBindBuffer					(	GL_ARRAY_BUFFER,
										cuda_poli_vbo_id						);
	glBufferSubData					(	GL_ARRAY_BUFFER,
										0,
										sizeof(float) * mdp_policies[0].size(),
										&mdp_policies[0]						);
	glBindBuffer					(	GL_ARRAY_BUFFER,
										0										);
	mdp_policy_reset	= true;
	if( setLCAExits() )
	{
		float u_time = 0.0f;
		update_mdp_lca( lca_exits, mdp_policies, u_time );
		log_manager->log( LogManager::INFORMATION, "Policies and LCA data updated in %.5fs.", u_time );
	}
	else
	{
		log_manager->log( LogManager::LERROR, "While setting LCA Exits." );
	}
}
//
//=======================================================================================
//
void MDPCudaPathManager::updateMDPPolicy(	vector<float>&	_policy,
											unsigned int	channel	)
{
//->UPDATE_POLICY
	if( channel < MDP_CHANNELS )
	{
		mdp_policies[channel]	= _policy;
		glBindBuffer					(	GL_ARRAY_BUFFER,
											cuda_poli_vbo_id						);
		glBufferSubData					(	GL_ARRAY_BUFFER,
											0,
											sizeof(float) * mdp_policies[0].size(),
											&mdp_policies[0]						);
		glBindBuffer					(	GL_ARRAY_BUFFER,
											0										);
		mdp_policy_reset		= true;
	}
//<-UPDATE_POLICY
/*
//->COPY_CURRENT_TO_INITIAL_POSITION_(RESET_POSITIONS)
    glBindBuffer					(	GL_ARRAY_BUFFER,
										cuda_cpos_vbo_id					);
	GLfloat* data;
	data = (GLfloat*)glMapBuffer	(	GL_ARRAY_BUFFER,
										GL_READ_WRITE						);
	if( data != (GLfloat*)NULL )
	{
		for( unsigned int i = 0; i < cuda_cpos_vbo_size; i += 4 )
		{
			data[i+1] = 0.0f;  // Set Y values (prev_tile in Kernel) to 0
		}
		glUnmapBuffer				(	GL_ARRAY_BUFFER						);

		glBindBuffer				(	GL_COPY_WRITE_BUFFER,
										cuda_ipos_vbo_id					);
		glCopyBufferSubData			(	GL_ARRAY_BUFFER,
										GL_COPY_WRITE_BUFFER,
										0,
										0,
										sizeof(float) * cuda_cpos_vbo_size	);
		glBindBuffer				(	GL_COPY_WRITE_BUFFER,
										0									);
		glBindBuffer				(	GL_ARRAY_BUFFER,
										0									);
	}
	else
	{
		glBindBuffer				(	GL_ARRAY_BUFFER,
										0									);
	}
//<-COPY_CURRENT_TO_INITIAL_POSITION_(RESET_POSITIONS)
*/
}
//
//=======================================================================================
//
void MDPCudaPathManager::runCuda(	unsigned int	texture_width,
									unsigned int	texture_height,
									float			parameter				)
{

	
	
#if defined MDPS_SQUARE_LCA || defined MDPS_HEXAGON_LCA
	launch_lca_kernel(	curr_positions,
						lca_exits,
						density,
						avg_racing_qw,
						avg_scatter_gather	);

	glBindBuffer					(	GL_ARRAY_BUFFER, 
										cuda_cpos_vbo_id						);
	glBufferSubData					(	GL_ARRAY_BUFFER,
										0,
										sizeof(float) * curr_positions.size(), 
										&curr_positions[0]						);
	glBindBuffer					(	GL_ARRAY_BUFFER,
										0										);

	glBindBuffer					(	GL_ARRAY_BUFFER, 
										cuda_dens_vbo_id						);
	glBufferSubData					(	GL_ARRAY_BUFFER,
										0,
										sizeof(float) * density.size(), 
										&density[0]								);
	glBindBuffer					(	GL_ARRAY_BUFFER,
										0										);
#elif defined MDPS_SQUARE_NOLCA || defined MDPS_HEXAGON_NOLCA
	// Map OpenGL buffer object for writing from CUDA:
    float*	dptr1;
	float*	dptr2;
	float*	dptr3;
	float*	dptr4;
	struct cudaGraphicsResource* res[4] = {
		vbo_manager->gl_cuda_vbos[cuda_cpos_vbo_index][cuda_cpos_vbo_frame].cuda_vbo_res,
		vbo_manager->gl_cuda_vbos[cuda_ipos_vbo_index][cuda_ipos_vbo_frame].cuda_vbo_res,
		vbo_manager->gl_cuda_vbos[cuda_poli_vbo_index][cuda_poli_vbo_frame].cuda_vbo_res,
		vbo_manager->gl_cuda_vbos[cuda_dens_vbo_index][cuda_dens_vbo_frame].cuda_vbo_res
	};
	cudaGraphicsMapResources( 4, res , 0 );
    size_t num_bytes1;
	size_t num_bytes2;
	size_t num_bytes3;
	size_t num_bytes4;
    cudaGraphicsResourceGetMappedPointer( (void **)&dptr1, &num_bytes1, res[0] );
#ifdef CASIM_CUDA_PATH_DEBUG
	printf( "CUDA mapped VBO1: May access %ld bytes\n", num_bytes1 );
#endif
	cudaGraphicsResourceGetMappedPointer( (void **)&dptr2, &num_bytes2, res[1] );
#ifdef CASIM_CUDA_PATH_DEBUG
    printf( "CUDA mapped VBO2: May access %ld bytes\n", num_bytes2 );
#endif
	cudaGraphicsResourceGetMappedPointer( (void **)&dptr3, &num_bytes3, res[2] );
#ifdef CASIM_CUDA_PATH_DEBUG
    printf( "CUDA mapped VBO3: May access %ld bytes\n", num_bytes3);
#endif
	cudaGraphicsResourceGetMappedPointer( (void **)&dptr4, &num_bytes4, res[3] );
#ifdef CASIM_CUDA_PATH_DEBUG
    printf( "CUDA mapped VBO4: May access %ld bytes\n", num_bytes4 );
#endif

#if defined MDPS_SQUARE_NOLCA
	launch_mdp_square_kernel(	dptr1,
								dptr2,
								dptr3,
								dptr4,
								mdp_width,
								mdp_depth,
								tile_width_or_side,
								tile_depth,
								texture_width,
								texture_height,
								parameter,
								mdp_policy_reset		);
#elif defined MDPS_HEXAGON_NOLCA
	launch_mdp_hexagon_kernel(	dptr1,
								dptr2,
								dptr3,
								dptr4,
								scene_width,
								scene_depth,
								mdp_width,
								mdp_depth,
								tile_width_or_side,
								tile_depth,
								texture_width,
								texture_height,
								parameter,
								mdp_policy_reset		);
#endif
    // Unmap buffer object:
    cudaGraphicsUnmapResources( 4, res, 0 );
#endif
	if( mdp_policy_reset )
	{
		mdp_policy_reset = false;
	}
}
//
//=======================================================================================
//
void MDPCudaPathManager::getDensity( vector<float>& _density )
{
	_density.clear();
    glBindBuffer					(	GL_ARRAY_BUFFER,
										cuda_dens_vbo_id					);
	GLfloat* data;
	data = (GLfloat*)glMapBuffer	(	GL_ARRAY_BUFFER,
										GL_READ_ONLY						);
	if( data != (GLfloat*)NULL )
	{
		for( unsigned int i = 0; i < cuda_dens_vbo_size; i++ )
		{
			_density.push_back( data[i] );
		}
		density = _density;
	}
	glUnmapBuffer					(	GL_ARRAY_BUFFER						);
    glBindBuffer					(	GL_ARRAY_BUFFER,
										0									);
}
//
//=======================================================================================
//
bool MDPCudaPathManager::setLCAExits( void )
{
	if( fmod( (float)lca_width, (float)mdp_width ) > 0.0f )
	{
		log_manager->log( LogManager::LERROR, "MDP and LCA dimensions not compatible." );
		return false;
	}

	lca_exits.clear();

	for( unsigned int mc = 0; mc < MDP_CHANNELS; mc++ )
	{

		log_manager->log( LogManager::INFORMATION, "Preparing LCA[%i] topology data...", mc );

		lca_in_mdp_width		= lca_width / mdp_width;
		lca_in_mdp_depth		= lca_depth / mdp_depth;
		lca_cells_in_mdp_cell	= lca_in_mdp_width * lca_in_mdp_depth;

		vector<float> lca_data;
		for( unsigned int i = 0; i < lca_width * lca_depth; i++ )
		{
			lca_data.push_back( 10.0f );
		}

		for( unsigned int lx = 0; lx < lca_width; lx++ )
		{
			for( unsigned int lz = 0; lz < lca_depth; lz++ )
			{
				unsigned int mx = lx / lca_in_mdp_width;
				unsigned int mz = lz / lca_in_mdp_depth;
				unsigned int mi = mz * mdp_width + mx;
				if( mdp_policies[ mc ][ mi ] > 7.0f )
				{
					unsigned int li = lz * lca_width + lx;
					lca_data[ li ] = mdp_policies[ mc ][ mi ];
				}
			}
		}

		for( unsigned int mdp_x = 0; mdp_x < mdp_width; mdp_x++ )
		{
			for( unsigned int mdp_z = 0; mdp_z < mdp_depth; mdp_z++ )
			{
				unsigned int mdp_i		= mdp_z * mdp_width + mdp_x;
				// POLICY: 0=UP_LEFT 1=LEFT 2=DOWN_LEFT 3=DOWN 4=DOWN_RIGHT 5=RIGHT 6=UP_RIGHT 7=UP 8=WALL 9=EXIT
				float f_policy			= mdp_policies[ mc ][ mdp_i ];
				unsigned int ui_policy	= (unsigned int)f_policy;
				vector<float> mdp_neighbor_walls;
				for( unsigned int i = 0; i < 8; i++ )
				{
					mdp_neighbor_walls.push_back( 0.0f );
				}
				if( mdp_x > 0 )
				{
					if( mdp_z > 0 )
					{
						unsigned int mdp_ul = (mdp_z - 1) * mdp_width + (mdp_x - 1);
						if( mdp_policies[ mc ][ mdp_ul ] == 8.0f )
						{
							mdp_neighbor_walls[0] = 1.0f;
						}
					}

					unsigned int mdp_l = mdp_z * mdp_width + (mdp_x - 1);
					if( mdp_policies[ mc ][ mdp_l ] == 8.0f )
					{
						mdp_neighbor_walls[1] = 1.0f;
					}

					if( mdp_z + 1 < mdp_depth )
					{
						unsigned int mdp_dl = (mdp_z + 1) * mdp_width + (mdp_x - 1);
						if( mdp_policies[ mc ][ mdp_dl ] == 8.0f )
						{
							mdp_neighbor_walls[2] = 1.0f;
						}
					}
				}

				if( mdp_z + 1 < mdp_depth )
				{
					unsigned int mdp_d = (mdp_z + 1) * mdp_width + mdp_x;
					if( mdp_policies[ mc ][ mdp_d ] == 8.0f )
					{
						mdp_neighbor_walls[3] = 1.0f;
					}
				}

				if( mdp_x + 1 < mdp_width )
				{
					if( mdp_z + 1 < mdp_depth )
					{
						unsigned int mdp_dr = (mdp_z + 1) * mdp_width + (mdp_x + 1);
						if( mdp_policies[ mc ][ mdp_dr ] == 8.0f )
						{
							mdp_neighbor_walls[4] = 1.0f;
						}
					}

					unsigned int mdp_r = mdp_z * mdp_width + (mdp_x + 1);
					if( mdp_policies[ mc ][ mdp_r ] == 8.0f )
					{
						mdp_neighbor_walls[5] = 1.0f;
					}

					if( mdp_z > 0 )
					{
						unsigned int mdp_ur = (mdp_z - 1) * mdp_width + (mdp_x + 1);
						if( mdp_policies[ mc ][ mdp_ur ] == 8.0f )
						{
							mdp_neighbor_walls[6] = 1.0f;
						}
					}
				}

				if( mdp_z > 0 )
				{
					unsigned int mdp_u = (mdp_z - 1) * mdp_width + mdp_x;
					if( mdp_policies[ mc ][ mdp_u ] == 8.0f )
					{
						mdp_neighbor_walls[7] = 1.0f;
					}
				}

				unsigned int lca_x		= mdp_x * lca_in_mdp_width;
				unsigned int lca_z		= mdp_z * lca_in_mdp_depth;
				unsigned int lca_i		= 0;

				switch( ui_policy )
				{
				case 0:	//UP_LEFT
					if( mdp_neighbor_walls[7] == 0.0f )
					{
						for( unsigned int x = lca_x + 1; x < lca_x + lca_in_mdp_width; x++ )	//UP
						{
							lca_i				= lca_z * lca_width + x;
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[1] == 0.0f )
					{
						for( unsigned int z = lca_z + 1; z < lca_z + lca_in_mdp_depth; z++ )	//LEFT
						{
							lca_i				= z * lca_width + lca_x;
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[7] == 0.0f && mdp_neighbor_walls[1] == 0.0f )
					{
						lca_i				= lca_z * lca_width + lca_x;
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 1:	//LEFT
					for( unsigned int z = lca_z; z < lca_z + lca_in_mdp_depth; z++ )	//LEFT
					{
						lca_i				= z * lca_width + lca_x;
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 2:	//DOWN_LEFT
					if( mdp_neighbor_walls[3] == 0.0f )
					{
						for( unsigned int x = lca_x + 1; x < lca_x + lca_in_mdp_width; x++ )	//DOWN
						{
							lca_i				= (lca_z + lca_in_mdp_depth - 1) * lca_width + x;
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[1] == 0.0f )
					{
						for( unsigned int z = lca_z; z < lca_z + lca_in_mdp_depth - 1; z++ )	//LEFT
						{
							lca_i				= z * lca_width + lca_x;
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[3] == 0.0f && mdp_neighbor_walls[1] == 0.0f )
					{
						lca_i				= (lca_z + lca_in_mdp_depth - 1) * lca_width + lca_x;
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 3:	//DOWN
					for( unsigned int x = lca_x; x < lca_x + lca_in_mdp_width; x++ )	//DOWN
					{
						lca_i				= (lca_z + lca_in_mdp_depth - 1) * lca_width + x;
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 4:	//DOWN_RIGHT
					if( mdp_neighbor_walls[3] == 0.0f )
					{
						for( unsigned int x = lca_x; x < lca_x + lca_in_mdp_width - 1; x++ )	//DOWN
						{
							lca_i				= (lca_z + lca_in_mdp_depth - 1) * lca_width + x;
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[5] == 0.0f )
					{
						for( unsigned int z = lca_z; z < lca_z + lca_in_mdp_depth - 1; z++ )	//RIGHT
						{
							lca_i				= z * lca_width + (lca_x + lca_in_mdp_width - 1);
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[3] == 0.0f && mdp_neighbor_walls[5] == 0.0f )
					{
						lca_i				= (lca_z + lca_in_mdp_depth - 1) * lca_width + (lca_x + lca_in_mdp_width - 1);
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 5:	//RIGHT
					for( unsigned int z = lca_z; z < lca_z + lca_in_mdp_depth; z++ )	//RIGHT
					{
						lca_i				= z * lca_width + (lca_x + lca_in_mdp_width - 1);
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 6:	//UP_RIGHT
					if( mdp_neighbor_walls[7] == 0.0f )
					{
						for( unsigned int x = lca_x; x < lca_x + lca_in_mdp_width - 1; x++ )	//UP
						{
							lca_i				= lca_z * lca_width + x;
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[5] == 0.0f )
					{
						for( unsigned int z = lca_z + 1; z < lca_z + lca_in_mdp_depth; z++ )	//RIGHT
						{
							lca_i				= z * lca_width + (lca_x + lca_in_mdp_width - 1);
							lca_data[lca_i]		= f_policy;
						}
					}
					if( mdp_neighbor_walls[0] == 0.0f && mdp_neighbor_walls[5] == 0.0f )
					{
						lca_i				= lca_z * lca_width + (lca_x + lca_in_mdp_width - 1);
						lca_data[lca_i]		= f_policy;
					}
					break;
				case 7:	//UP
					for( unsigned int x = lca_x; x < lca_x + lca_in_mdp_width; x++ )	//UP
					{
						lca_i				= lca_z * lca_width + x;
						lca_data[lca_i]		= f_policy;
					}
					break;
				}
			}
		}
		lca_exits.push_back( lca_data );

#ifdef PRINT_LCA_DATA
		printf( "\n\nLCA_TOPOLOGY[%i]:\n", mc );
		for( unsigned int z = 0; z < lca_depth; z++ )
		{
			for( unsigned int x = 0; x < lca_width; x++ )
			{
				unsigned int lca_i = z * lca_width + x;
				if( lca_exits[ mc ][ lca_i ] == 8.0f )			//WALL
				{
					printf( "%c ", 219 );
				}
				else if( lca_exits[ mc ][ lca_i ] == 9.0f )		//GLOBAL_EXIT
				{
					printf( "# " );
				}
				else if( lca_exits[ mc ][ lca_i ] == 10.0f )	//FREE
				{
					printf( "%c ", 176 );
				}
				else											//LOCAL_EXIT
				{
					switch ( (unsigned int)lca_exits[ mc ][lca_i] )
					{
						case 0:		// UL
							printf( "%c ", 218 );
							break;
						case 1:		// LL
							printf( "%c ", 60 );
							break;
						case 2:		// DL
							printf( "%c ", 192 );
							break;
						case 3:		// DD
							printf( "%c ", 118 );
							break;
						case 4:		// DR
							printf( "%c ", 217 );
							break;
						case 5:		// RR
							printf( "%c ", 62 );
							break;
						case 6:		// UR
							printf( "%c ", 191 );
							break;
						case 7:		// UU
							printf( "%c ", 94 );
							break;
						default:	// ??
							printf( "%i ", (unsigned int)lca_exits[ mc ][lca_i] );
							break;
					}
				}
			}
			printf( "\n" );
		}
		printf( "\n" );
#endif

		log_manager->log( LogManager::INFORMATION, "Done. LCA[%i] topology data is ready.", mc );
	}
	return true;
}
//
//=======================================================================================
//
void MDPCudaPathManager::getMDPPolicy(	vector<float>&	_policy,
										unsigned int	channel	)
{
	if( channel < MDP_CHANNELS )
	{
		_policy = mdp_policies[channel];
	}
}
//
//=======================================================================================
//
GLuint& MDPCudaPathManager::getPosTboId( void )
{
	return pos_tbo_id;
}
//
//=======================================================================================
//
inline int MDPCudaPathManager::getMaxGflopsDeviceId( void )
{
	// This function returns the best GPU (with maximum GFLOPS)
	int current_device   = 0;
	int sm_per_multiproc = 0;
	int max_compute_perf = 0;
	int max_perf_device  = 0;
	int device_count     = 0;
	int best_SM_arch     = 0;
	cudaDeviceProp deviceProp;

	cudaGetDeviceCount( &device_count );
	// Find the best major SM Architecture GPU device
	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &deviceProp, current_device );
		if( deviceProp.major > 0 && deviceProp.major < 9999 )
		{
			best_SM_arch = std::max( best_SM_arch, deviceProp.major );
		}
		current_device++;
	}

    // Find the best CUDA capable GPU device
	current_device = 0;
	while( current_device < device_count )
	{
		cudaGetDeviceProperties( &deviceProp, current_device );
		if( deviceProp.major == 9999 && deviceProp.minor == 9999 )
		{
		    sm_per_multiproc = 1;
		}
		else
		{
			sm_per_multiproc = _ConvertSMVer2Cores( deviceProp.major, deviceProp.minor );
		}

		int compute_perf  =
			deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
		if( compute_perf  > max_compute_perf )
		{
            // If we find GPU with SM major > 2, search only these
			if( best_SM_arch > 2 )
			{
				// If our device==dest_SM_arch, choose this, or else pass
				if( deviceProp.major == best_SM_arch )
				{
					max_compute_perf  = compute_perf;
					max_perf_device   = current_device;
				}
			}
			else
			{
				max_compute_perf  = compute_perf;
				max_perf_device   = current_device;
			}
		}
		++current_device;
	}
	return max_perf_device;
}
//
//=======================================================================================
//
inline int MDPCudaPathManager::_ConvertSMVer2Cores( int major, int minor )
{
	// Defines for GPU Architecture types
	// (using the SM version to determine the # of cores per SM)
	typedef struct {
		int SM; // 0xMm (hexidecimal notation),
				// M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{ 
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192 }, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192 }, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192 }, // Kepler Generation (SM 3.5) GK11x class
		{ 0x37, 192 }, // Kepler Generation (SM 3.7) GK21x class
		{ 0x50, 128 }, // Maxwell Generation (SM 5.0) GM10x class
		{ 0x52, 128 }, // Maxwell Generation (SM 5.2) GM20x class
		{ -1, -1 }
	};

	int index = 0;
	while( nGpuArchCoresPerSM[index].SM != -1 )
	{
		if( nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) )
		{
			return nGpuArchCoresPerSM[index].Cores;
		}
		index++;
	}
	log_manager->log( LogManager::CUDA,
					  "MapSMtoCores undefined SMversion: %d.%d!",
					  major,
					  minor										);
	return -1;
}
//
//=======================================================================================
//
float MDPCudaPathManager::getAvgRacingQW( void )
{
	return avg_racing_qw;
}
//
//=======================================================================================
//
float MDPCudaPathManager::getAvgScatterGather( void )
{
	return avg_scatter_gather;
}
//
//=======================================================================================
