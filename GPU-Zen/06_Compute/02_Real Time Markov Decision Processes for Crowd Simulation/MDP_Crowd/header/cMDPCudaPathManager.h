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

#pragma once

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <math.h>

#include "cMacros.h"
#include "cVertex.h"
#include "cLogManager.h"
#include "cVboManager.h"
#include "cFboManager.h"
#include "cStaticLod.h"
#include "cProjectionManager.h"
#include "cGlErrorManager.h"

//=======================================================================================

#ifndef __MDP_CUDA_PATH_MANAGER
#define __MDP_CUDA_PATH_MANAGER

class MDPCudaPathManager
{
public:
								MDPCudaPathManager		(	LogManager*				log_manager,
															VboManager*				vbo_manager,
															FboManager*				fbo_manager,
															string					_fbo_pos_name,
															string					_fbo_pos_tex_name		);
								~MDPCudaPathManager		(	void											);

	bool						init_nolca				(	float					_scene_width,
															float					_scene_depth,
															float					_tile_width_or_side,
															float					_tile_height,
															unsigned int			_mdp_width,
															unsigned int			_mdp_depth,
															vector<float>&			_init_pos,
															vector<float>&			_init_speed,
															vector<float>&			_mdp_policy,
															vector<float>&			_density				);
	bool						init_lca				(	float					_scene_width,
															float					_scene_depth,
															float					_tile_width_or_side,
															float					_tile_height,
															unsigned int			_mdp_width,
															unsigned int			_mdp_depth,
															unsigned int			_lca_width,
															unsigned int			_lca_depth,
															vector<float>&			_init_pos,
															vector<float>&			_init_speed,
															vector<vector<float>>&	_mdp_policies,
															vector<float>&			_density,
															vector<float>&			_scenario_speeds		);
	void						runCuda					(	unsigned int			texture_width,
															unsigned int			texture_height,
															float					parameter				);
	void						updateMDPPolicy			(	vector<float>&			_policy,
															unsigned int			channel					);
	void						updateMDPPolicies		(	vector<vector<float>>&	_policies				);
	void						getDensity				(	vector<float>&			_density				);
	void						getMDPPolicy			(	vector<float>&			_policy,
															unsigned int			channel					);
	GLuint&						getPosTboId				(	void											);
	float						getAvgRacingQW			(	void											);
	float						getAvgScatterGather		(	void											);

private:
	bool						init_nolca				(	void											);
	bool						init_lca				(	void											);
	bool						setLCAExits				(	void											);
	int							getMaxGflopsDeviceId	(	void											);
	int							_ConvertSMVer2Cores		(	int						major,
															int						minor					);

	LogManager*					log_manager;
	VboManager*					vbo_manager;
	FboManager*					fbo_manager;
	cudaGraphicsResource*		cuda_ipos_vbo_res;
	cudaGraphicsResource*		cuda_cpos_vbo_res;
	cudaGraphicsResource*		cuda_poli_vbo_res;
	cudaGraphicsResource*		cuda_dens_vbo_res;
	ProjectionManager*			proj_manager;

	string						fbo_pos_name;
	string						fbo_pos_tex_name;

	bool						mdp_policy_reset;

	float						scene_width;
	float						scene_depth;
	float						tile_width_or_side;
	float						tile_depth;
	float						avg_racing_qw;
	float						avg_scatter_gather;

	unsigned int				lca_cells_in_mdp_cell;
	unsigned int				lca_in_mdp_width;
	unsigned int				lca_in_mdp_depth;

	unsigned int				mdp_width;
	unsigned int				mdp_depth;

	unsigned int				lca_width;
	unsigned int				lca_depth;

	unsigned int				cuda_ipos_vbo_id;
	unsigned int				cuda_ipos_vbo_size;
	unsigned int				cuda_ipos_vbo_index;
	unsigned int				cuda_ipos_vbo_frame;

	unsigned int				cuda_cpos_vbo_id;
	unsigned int				cuda_cpos_vbo_size;
	unsigned int				cuda_cpos_vbo_index;
	unsigned int				cuda_cpos_vbo_frame;

	unsigned int				cuda_poli_vbo_id;
	unsigned int				cuda_poli_vbo_size;
	unsigned int				cuda_poli_vbo_index;
	unsigned int				cuda_poli_vbo_frame;

	unsigned int				cuda_dens_vbo_id;
	unsigned int				cuda_dens_vbo_size;
	unsigned int				cuda_dens_vbo_index;
	unsigned int				cuda_dens_vbo_frame;

	unsigned int				cuda_goals_vbo_id;
	unsigned int				cuda_goals_vbo_size;
	unsigned int				cuda_goals_vbo_index;
	unsigned int				cuda_goals_vbo_frame;

	GLuint						pos_tbo_id;
	vector<vector<float>>		mdp_policies;
	vector<vector<float>>		lca_exits;
	vector<float>				init_positions;
	vector<float>				curr_positions;
	vector<float>				init_speeds;
	vector<float>				density;
	vector<float>				scenario_speeds;
	vector<VboRenderTexture>	render_textures;
};

#endif

//=======================================================================================
