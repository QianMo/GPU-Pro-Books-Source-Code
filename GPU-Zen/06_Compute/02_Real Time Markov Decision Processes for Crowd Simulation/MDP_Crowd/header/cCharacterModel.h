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

#include <iostream>
#include <fstream>

#include <unordered_map>
#include <vector>
#include <string>

#include "cMacros.h"

#include "cVertex.h"
#include "cModel3D.h"
#include "cCamera.h"

using namespace std;

//=======================================================================================

#ifndef __CHARACTER_MODEL
#define __CHARACTER_MODEL

class CharacterModel
{
public:
											CharacterModel				(	void										);
											CharacterModel				(	LOD_TYPE				_LOD,
																			string					_name,
																			VboManager*				_vbo_manager,
																			GlslManager*			_glsl_manager,
																			Model3D*				_head,
																			Model3D*				_hair,
																			Model3D*				_torso,
																			Model3D*				_legs				);
											CharacterModel				(	LOD_TYPE				_LOD,
																			string					_name,
																			VboManager*				_vbo_manager,
																			GlslManager*			_glsl_manager,
																			Model3D*				_model				);

											~CharacterModel				(	void										);

	bool									stitch_parts				(	void										);
	bool									save_obj					(	string&					filename			);

#ifdef CUDA_PATHS
#ifdef DEMO_SHADER
	void									draw_instanced_culled_rigged(	Camera*					cam,
																			unsigned int			frame,
																			unsigned int			instances,
																			unsigned int			_AGENTS_NPOT,
																			unsigned int			_ANIMATION_LENGTH,
																			unsigned int			_STEP,
																			GLuint&					posTextureId,
																			GLuint&					pos_vbo_id,
																			GLuint&					clothing_color_table_tex_id,
																			GLuint&					pattern_color_table_tex_id,
																			GLuint&					global_mt_tex_id,
																			GLuint&					torso_mt_tex_id,
																			GLuint&					legs_mt_tex_id,
																			GLuint&					rigging_mt_tex_id,
																			GLuint&					animation_mt_tex_id,
																			GLuint&					facial_mt_tex_id,
																			float					lod,
																			float					gender,
																			float*					viewMat,
																			bool					wireframe,
																			float					doHandD,
																			float					doPatterns,
																			float					doColor,
																			float					doFacial					);
#else
	void									draw_instanced_culled_rigged(	Camera*					cam,
																			unsigned int			frame,
																			unsigned int			instances,
																			unsigned int			_AGENTS_NPOT,
																			unsigned int			_ANIMATION_LENGTH,
																			unsigned int			_STEP,
																			GLuint&					posTextureId,
																			GLuint&					pos_vbo_id,
																			GLuint&					clothing_color_table_tex_id,
																			GLuint&					pattern_color_table_tex_id,
																			GLuint&					global_mt_tex_id,
																			GLuint&					torso_mt_tex_id,
																			GLuint&					legs_mt_tex_id,
																			GLuint&					rigging_mt_tex_id,
																			GLuint&					animation_mt_tex_id,
																			GLuint&					facial_mt_tex_id,
																			float					lod,
																			float					gender,
																			float*					viewMat,
																			bool					wireframe			);
#endif
#else
#ifdef DEMO_SHADER
	void									draw_instanced_culled_rigged(	Camera*					cam,
																			unsigned int			frame,
																			unsigned int			instances,
																			unsigned int			_AGENTS_NPOT,
																			unsigned int			_ANIMATION_LENGTH,
																			unsigned int			_STEP,
																			GLuint&					posTextureTarget,
																			GLuint&					posTextureId,
																			GLuint&					pos_vbo_id,
																			GLuint&					clothing_color_table_tex_id,
																			GLuint&					pattern_color_table_tex_id,
																			GLuint&					global_mt_tex_id,
																			GLuint&					torso_mt_tex_id,
																			GLuint&					legs_mt_tex_id,
																			GLuint&					rigging_mt_tex_id,
																			GLuint&					animation_mt_tex_id,
																			GLuint&					facial_mt_tex_id,
																			float					lod,
																			float					gender,
																			float*					viewMat,
																			bool					wireframe,
																			float					doHandD,
																			float					doPatterns,
																			float					doColor,
																			float					doFacial					);
#else
	void									draw_instanced_culled_rigged(	Camera*					cam,
																			unsigned int			frame,
																			unsigned int			instances,
																			unsigned int			_AGENTS_NPOT,
																			unsigned int			_ANIMATION_LENGTH,
																			unsigned int			_STEP,
																			GLuint&					posTextureTarget,
																			GLuint&					posTextureId,
																			GLuint&					pos_vbo_id,
																			GLuint&					clothing_color_table_tex_id,
																			GLuint&					pattern_color_table_tex_id,
																			GLuint&					global_mt_tex_id,
																			GLuint&					torso_mt_tex_id,
																			GLuint&					legs_mt_tex_id,
																			GLuint&					rigging_mt_tex_id,
																			GLuint&					animation_mt_tex_id,
																			GLuint&					facial_mt_tex_id,
																			float					lod,
																			float					gender,
																			float*					viewMat,
																			bool					wireframe			);
#endif
#endif

#ifdef CUDA_PATHS
	void									draw_instanced_culled_rigged_shadow(	Camera*					cam,
																					unsigned int			frame,
																					unsigned int			instances,
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					GLuint&					posTextureId,
																					GLuint&					pos_vbo_id,
																					GLuint&					rigging_mt_tex_id,
																					GLuint&					animation_mt_tex_id,
																					float					lod,
																					float					gender,
																					float*					viewMat,
																					float*					projMat,
																					float*					shadowMat,
																					bool					wireframe			);
#else
	void									draw_instanced_culled_rigged_shadow(	Camera*					cam,
																					unsigned int			frame,
																					unsigned int			instances,
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					GLuint&					posTextureTarget,
																					GLuint&					posTextureId,
																					GLuint&					pos_vbo_id,
																					GLuint&					rigging_mt_tex_id,
																					GLuint&					animation_mt_tex_id,
																					float					lod,
																					float					gender,
																					float*					viewMat,
																					float*					projMat,
																					float*					shadowMat,
																					bool					wireframe			);
#endif

private:

	LOD_TYPE								LOD;
	Model3D*								head;
	Model3D*								hair;
	Model3D*								torso;
	Model3D*								legs;
	Model3D*								model;
	VboManager*								vbo_manager;
	GlslManager*							glsl_manager;

	vector<Location4>						unique_locations;
	vector<Normal>							unique_normals;
	vector<Uv>								unique_uvs;
	vector<Face3>							faces;

	string									name;
	vector<GLuint>							sizes;
	vector<GLuint>							ids;
};

#endif
