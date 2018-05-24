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
#include "cCrowdGroup.h"
#include "cCharacterGroup.h"
#include "cLogManager.h"
#include "cGlErrorManager.h"
#include "cFboManager.h"
#include "cCamera.h"
#include "cMDPCudaPathManager.h"

#include <string>
#include <glm/glm.hpp>

using namespace std;

//=======================================================================================

#ifndef __CROWD
#define __CROWD

class Crowd
{
public:
							Crowd			(	LogManager*					log_manager_,
												GlErrorManager*				err_manager_,
												FboManager*					fbo_manager_,
												VboManager*					vbo_manager_,
												GlslManager*				glsl_manager_,
												string&						name_,
												GLuint						width_,
												GLuint						height_				);
							~Crowd			(	void											);

	void					addGroup		(	CharacterGroup*				_group,
												StaticLod*					_static_lod,
												string&						_animation,
												string&						_fbo_lod_name,
												string&						_fbo_pos_tex_name,
												string&						_fbo_ids_tex_name,
												float						_percentage,
												GLuint						_frames,
												GLuint						_duration			);
	void					addGroup		(	CharacterGroup*				_group,
												string&						_animation,
												float						_percentage,
												GLuint						_frames,
												GLuint						_duration			);
	string&					getName			(	void											);
	string&					getFboLodName	(	void											);
	string&					getFboPosTexName(	void											);
	GLuint					getWidth		(	void											);
	GLuint					getHeight		(	void											);
	vector<CrowdGroup*>&	getGroups		(	void											);
	void					initFboInputs	(	vector<InputFbo*>&  		fbo_inputs			);
	void					setFboManager	(	FboManager*					_fbo_manager		);
	bool					init_paths		(	CROWD_POSITION&				crowd_position,
												SCENARIO_TYPE&				scenario_type,
												vector<float>&				scenario_speeds,
												float						plane_scale,
												vector<vector<float>>&		_policies,
												unsigned int				grid_width,
												unsigned int				grid_height,
												vector<glm::vec2>&			occupation,
												vector<GROUP_FORMATION>&	formations			);
	void					updatePolicy	(	vector<float>&				_policy,
												unsigned int				channel				);
	void					updatePolicies	(	vector<vector<float>>&		_policies			);
	void					getDensity		(	vector<float>&				_density			);
	void					init_ids		(	GLuint&						groupOffset,
												GLuint&						agentOffset			);
	void					run_paths		(	void											);
	void					nextFrame		(	void											);
	GLuint&					getCudaTBO		(	void											);
	void					initTexCoords	(	void											);
	GLuint&					getPosTexCoords	(	void											);
	GLuint					getPosTexSize	(	void											);
	float					getAvgRacingQW	(	void											);
	float					getAvgScatterGather(	void										);

#ifdef DEMO_SHADER
	void					draw			(	Camera*						camera,
												float*						viewMat,
												float*						projMat,
												float*						shadowMat,
												bool						wireframe,
												bool						shadows,
												bool						doHandD,
												bool						doPatterns,
												bool						doColor,
												bool						doFacial			);
#else
	void					draw			(	Camera*						camera,
												float*						viewMat,
												float*						projMat,
												float*						shadowMat,
												bool						wireframe,
												bool						shadows				);
#endif

	vector<GLuint>			models_drawn;

private:
	float					rand_between	(	float						minf,
												float						maxf				);
	bool					pos_occupied	(	vector<glm::vec2>&			occupation,
												glm::vec2					position,
												float						radius				);
	void					init_paths_sq	(	CROWD_POSITION&				crowd_position,
												SCENARIO_TYPE&				scenario_type,
												float						plane_scale,
												vector<vector<float>>&		_policies,
												unsigned int				grid_width,
												unsigned int				grid_height,
												vector<glm::vec2>&			occupation,
												vector<GROUP_FORMATION>&	formations			);
	void					init_paths_hx	(	CROWD_POSITION&				crowd_position,
												SCENARIO_TYPE&				scenario_type,
												float						plane_scale,
												vector<vector<float>>&		_policies,
												unsigned int				grid_width,
												unsigned int				grid_height,
												vector<glm::vec2>&			occupation,
												vector<GROUP_FORMATION>&	formations			);
	bool					is_policy_open	(	unsigned int				index				);

	LogManager*				log_manager;
	GlErrorManager*			err_manager;
	FboManager*				fbo_manager;
	VboManager*				vbo_manager;
	GlslManager*			glsl_manager;
	MDPCudaPathManager*		cuda_path_manager;

	string					fbo_lod_name;
	string					fbo_pos_tex_name;
	string					fbo_ids_tex_name;
	float					path_param;
	vector<float>			instance_positions_flat;
	vector<float>			instance_rotations_flat;
	vector<float>			instance_control_points;
	vector<float>			instance_ids_flat;
	vector<float>			instance_speed;

	StaticLod*				static_lod;
	sVBOLod					vbo_lod[NUM_LOD];

	string					name;
	GLuint					width;
	GLuint					height;
	vector<CrowdGroup*>		groups;

	float					scene_width;
	float					scene_height;
	float					scene_width_in_tiles;
	float					scene_height_in_tiles;
	float					tile_width;
	float					tile_height;
	float					tile_side;
	float					personal_space;
	vector<vector<float>>	policies;
	vector<float>			density;

	unsigned int			pos_tc_index;
	unsigned int			pos_tc_frame;
	unsigned int			pos_tc_size;
	unsigned int			posTexCoords;

	unsigned int			pos_crashes;
};

#endif

//=======================================================================================
