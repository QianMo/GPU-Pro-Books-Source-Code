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
#include "cMacros.h"
#include "cModel3D.h"
#include "cGlslManager.h"
#include "cVboManager.h"
#include "cFboManager.h"
#include "cMDPSquareManager.h"
#include "cMDPHexagonManager.h"
#include "cLogManager.h"

#include <glm/glm.hpp>

#include <vector>
#include <string>

using namespace std;

//=======================================================================================

#ifndef __OBSTACLE_MANAGER
#define __OBSTACLE_MANAGER

class ObstacleManager
{
public:
								ObstacleManager			(	GlslManager*	_glsl_manager,
															VboManager*		_vbo_manager,
															LogManager*		_log_manager,
															float			_scene_width,
															float			_scene_depth			);
								~ObstacleManager		(	void									);

	void						init					(	OBSTACLE_TYPE&	_obstacle_type,
															vector<string>&	_mdp_csv_files,
															FboManager*		fbo_manager				);
	void						draw					(	float*			view_mat				);
	void						moveCursorUp			(	void									);
	void						moveCursorDown			(	void									);
	void						moveCursorLeft			(	void									);
	void						moveCursorRight			(	void									);
	void						toggleObstacle			(	void									);
	void						toggleExit				(	void									);
	void						initStructuresOnHost	(	void									);
	void						initPermsOnDevice		(	void									);
	void						uploadToDevice			(	void									);
	void						iterateOnDevice			(	void									);
	void						downloadToHost			(	void									);
	void						updatePolicy			(	void									);
	void						updateDensity			(	vector<float>&	density,
															unsigned int	channel					);

	unsigned int				getSceneWidthInTiles	(	void									);
	unsigned int				getSceneDepthInTiles	(	void									);
	unsigned int				getPolicyTextureId		(	unsigned int	channel					);
	unsigned int				getDensityTextureId		(	unsigned int	channel					);
	unsigned int				getArrowsTextureId		(	void									);
	unsigned int				getLayer0TextureId		(	void									);
	unsigned int				getLayer1TextureId		(	void									);
	unsigned int				getMdpTexCoordsId		(	void									);
	unsigned int				getMdpTexCoordsSize		(	void									);
	unsigned int				getActiveMDPLayer		(	void									);
	vector<float>&				getPolicy				(	unsigned int	channel					);
	vector<vector<float>>&		getPolicies				(	void									);
	MDP_MACHINE_STATE&			getState				(	void									);

private:
	void						initObstacles			(	void									);
	void						initCursor				(	void									);
	void						initMDP					(	void									);
	void						initMdpTexCoords		(	FboManager*		fbo_manager				);
	void						drawCursor				(	void									);

	GlslManager*				glsl_manager;
	VboManager*					vbo_manager;
	LogManager*					log_manager;
//->CURSOR
	Model3D*					cursor_obj;
	glm::vec3					cursor_scale;
	glm::vec3					cursor_position;
	unsigned int				scene_width_in_tiles;
	unsigned int				scene_depth_in_tiles;
	float						scene_width;
	float						scene_depth;
	float						tile_width;
	float						tile_depth;
	float*						cursor_tint;
	float*						cursor_amb;
	float						R;
	float						S;
	float						W;
	float						H;
	unsigned int				cursor_row;
	unsigned int				cursor_col;
	unsigned int				cursor_index;
//<-CURSOR
//->OBSTACLE
	Model3D*					obstacle_obj;
	unsigned int				obstacle_tex;
	vector<float>				obstacle_pos;
	vector<float>				obstacle_rot;
	vector<float>				obstacle_scale;
//<-OBSTACLE
//->MDP
	unsigned int				NQ;
#if defined MDPS_SQUARE_NOLCA || defined MDPS_SQUARE_LCA
	vector<MDPSquareManager*>	mdp_managers;
#elif defined MDPS_HEXAGON_NOLCA || defined MDPS_HEXAGON_LCA
	vector<MDPHexagonManager*>	mdp_managers;
#endif
	OBSTACLE_TYPE				obstacle_type;
	vector<string>				mdp_csv_files;
	vector<vector<float>>		policies;
	vector<unsigned int>		policy_tex;
	vector<unsigned int>		density_tex;
	unsigned int				layer0_tex;
	unsigned int				layer1_tex;
	unsigned int				mdp_tc_index;
	unsigned int				mdp_tc_frame;
	unsigned int				mdp_tc_size;
	unsigned int				mdpTexCoords;
	unsigned int				arrows_tex;
	unsigned int				ACTIVE_MDP_LAYER;
	MDP_MACHINE_STATE			status;
	vector<vector<float>>		mdp_topologies;
//<-MDP
	string						str_amb;
	string						str_textured;
	string						str_tint;
};

#endif

//=======================================================================================
