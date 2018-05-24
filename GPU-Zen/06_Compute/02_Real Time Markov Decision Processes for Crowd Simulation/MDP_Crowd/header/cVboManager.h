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
#include "cVertex.h"
#include "cGlslManager.h"
#include "cLogManager.h"
#include "cGlErrorManager.h"
#include "cCamera.h"

#include <cuda_gl_interop.h>

#include <map>
#include <vector>
#include <string>

using namespace std;

//=======================================================================================

#ifndef __VBO_MANAGER
#define __VBO_MANAGER

//=======================================================================================

typedef struct
{
	vector<Vertex>					vertices;
	unsigned int					id;
	unsigned int					attached_tbo_id;
}									VBO;

#define INITVBO( vbo )				\
	vbo.id				= 0;		\
	vbo.attached_tbo_id = 0

//=======================================================================================

typedef struct
{
	vector<Vertex>					vertices;
	unsigned int					gl_id;
	unsigned int					attached_tbo_gl_id;
	struct cudaGraphicsResource*	cuda_vbo_res;
}									GL_CUDA_VBO;

#define INITGL_CUDA_VBO( vbo )		\
	vbo.gl_id				= 0;	\
	vbo.attached_tbo_gl_id	= 0;	\
	vbo.cuda_vbo_res		= 0

//=======================================================================================

typedef struct
{
	GLenum							offset;
	GLenum							target;
	GLuint							id;
}									VboRenderTexture;

//=======================================================================================

class VboManager
{
public:
									VboManager									(	LogManager*				log_manager_,
																					GlErrorManager*			err_manager_,
																					GlslManager*			glsl_manager_										);
									~VboManager									(	void																		);

	unsigned int					getReusedVBOs								(	void																		);
	bool							isFrameRegistered							(	string&					filename,
																					unsigned int			frame												);
	VBO								getVBO										(	string&					filename,
																					unsigned int			frame												);
	unsigned int					reuseFrame									(	GLuint&					vboId,
																					string&					filename,
																					unsigned int			frame												);
	unsigned int					createVBOContainer							(	string&					filename,
																					unsigned int			frame												);
	unsigned int					createGlCudaVboContainer					(	string&					filename,
																					unsigned int			frame												);
	unsigned int					gen_vbo										(	GLuint&					vboId,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame											);
	unsigned int					gen_vbo3									(	GLuint&					vboId,
																					vector<float>&			data,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame											);
	void							attach_tbo2vbo								(	GLuint&					tboId,
																					unsigned int			gl_tex,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame											);
	void							delete_vbo									(	unsigned int			vbo_index,
																					unsigned int			vbo_frame											);
	unsigned int					gen_gl_cuda_vbo								(	GLuint&					vboId,
																					cudaGraphicsResource*	cuda_vbo_res,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame,
																					unsigned int			vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard	);
	unsigned int					gen_gl_cuda_vbo2							(	GLuint&					vboId,
																					cudaGraphicsResource*	cuda_vbo_res,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame,
																					unsigned int			vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard	);
	unsigned int					gen_gl_cuda_vbo3							(	GLuint&					vboId,
																					vector<float>&			data,
																					cudaGraphicsResource*	cuda_vbo_res,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame,
																					unsigned int			vbo_res_flags = cudaGraphicsMapFlagsWriteDiscard	);
	void							delete_gl_cuda_vbo							(	unsigned int			vbo_index,
																					unsigned int			vbo_frame											);
	void							gen_empty_vbo								(	GLuint&					vboId,
																					unsigned int			vbo_index,
																					unsigned int			vbo_frame,
																					unsigned int			size												);
	void							update_vbo									(	unsigned int			vbo_index,
																					unsigned int			vbo_frame											);
	void							update_gl_cuda_vbo							(	unsigned int			gl_cuda_vbo_index,
																					unsigned int			gl_cuda_vbo_frame									);
	void							render_vbo									(	GLuint&					vboId,
																					unsigned int			size												);
	void							render_vbo									(	GLuint&					vboId,
																					unsigned int			size,
																					unsigned int			mode												);
	void							render_vbo									(	GLuint&					vboId,
																					GLint					data_size,
																					GLenum					data_type,
																					GLenum					draw_mode,
																					GLint					draw_size											);
	void							render_bumped_vbo							(	GLuint&					vboId,
																					unsigned int			size												);
	void							render_instanced_vbo						(	GLuint&					vboId,
																					GLuint&					texture,
																					unsigned int			size,
																					unsigned int			count,
																					unsigned int			positions_sizeInBytes,
																					float*					positions,
																					unsigned int			rotations_sizeInBytes,
																					float*					rotations,
																					float*					viewMat												);
	void							render_instanced_vbo						(	GLuint&					vboId,
																					GLuint&					texture,
																					unsigned int			size,
																					unsigned int			count,
																					unsigned int			positions_sizeInBytes,
																					float*					positions,
																					unsigned int			rotations_sizeInBytes,
																					float*					rotations,
																					unsigned int			scales_sizeInBytes,
																					float*					scales,
																					float*					viewMat												);
	void							render_textured_instanced_vbo				(	MODEL_MESH&				mesh,
																					unsigned int			count,
																					unsigned int			positions_sizeInBytes,
																					float*					positions,
																					unsigned int			rotations_sizeInBytes,
																					float*					rotations,
																					unsigned int			scales_sizeInBytes,
																					float*					scales,
																					float*					viewMat												);
	void							render_untextured_instanced_vbo				(	MODEL_MESH&				mesh,
																					unsigned int			count,
																					unsigned int			positions_sizeInBytes,
																					float*					positions,
																					unsigned int			rotations_sizeInBytes,
																					float*					rotations,
																					unsigned int			scales_sizeInBytes,
																					float*					scales,
																					float*					viewMat												);
	void							render_instanced_culled_vbo					(	GLuint&					vboId,
																					GLuint&					texture,
																					GLuint&					pos_vbo_id,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat												);
	void							render_instanced_culled_rigged_vbo			(	GLuint&					vboId,
																					GLuint&					texture,
																					GLuint&					zonesTexture,
																					GLuint&					weightsTexture,
																					GLuint&					displacementTexture,
																					GLuint&					pos_vbo_id,
																					float					dt,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);

	void							render_instanced_culled_rigged_vbo			(	GLuint&					vboId,
																					GLuint&					posTextureTarget,
																					GLuint&					posTextureId,
																					GLuint&					texture,
																					GLuint&					zonesTexture,
																					GLuint&					weightsTexture,
																					GLuint&					displacementTexture,
																					GLuint&					pos_vbo_id,
																					float					dt,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);

	void							render_instanced_culled_rigged_vbo			(	GLuint&					vboId,
																					GLuint&					posTextureTarget,
																					GLuint&					posTextureId,
																					GLuint&					skin_texture,
																					GLuint&					hair_texture,
																					GLuint&					cap_texture,
																					GLuint&					torso_texture,
																					GLuint&					legs_texture,
																					GLuint&					zonesTexture,
																					GLuint&					weightsTexture,
																					GLuint&					displacementTexture,
																					GLuint&					pos_vbo_id,
																					float					dt,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);

	void							render_instanced_culled_rigged_vbo			(	GLuint&					vboId,
																					GLuint&					posTextureTarget,
																					GLuint&					posTextureId,
																					GLuint&					skin_texture,
																					GLuint&					hair_texture,
																					GLuint&					cap_texture,
																					GLuint&					torso_texture,
																					GLuint&					legs_texture,
																					GLuint&					clothing_color_table_tex_id,
																					GLuint&					clothing_patterns_tex_id,
																					GLuint&					zonesTexture,
																					GLuint&					weightsTexture,
																					GLuint&					displacementTexture,
																					GLuint&					pos_vbo_id,
																					float					dt,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);

#ifdef DEMO_SHADER
	void							render_instanced_culled_rigged_vbo2			(	Camera*					cam,
																					GLuint&					vboId,
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
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe,

																					float					doHandD,
																					float					doPatterns,
																					float					doColor,
																					float					doFacial											);
#else
	void							render_instanced_culled_rigged_vbo2			(	Camera*					cam,
																					GLuint&					vboId,
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
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);
#endif

	void							render_instanced_culled_rigged_vbo3			(	Camera*					cam,
																					GLuint&					vboId,
																					GLuint&					posTextureTarget,
																					GLuint&					posTextureId,
																					GLuint&					pos_vbo_id,

																					GLuint&					rigging_mt_tex_id,
																					GLuint&					animation_mt_tex_id,

																					float					lod,
																					float					gender,
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					float*					projMat,
																					float*					shadowMat,
																					bool					wireframe											);

#ifdef DEMO_SHADER
	void							render_instanced_culled_rigged_vbo4			(	Camera*					cam,
																					GLuint&					vboId,
																					GLuint&					ids_buffer,
																					GLuint&					pos_buffer,

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
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe,

																					float					doHandD,
																					float					doPatterns,
																					float					doColor,
																					float					doFacial											);
#else
	void							render_instanced_culled_rigged_vbo4			(	Camera*					cam,
																					GLuint&					vboId,
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
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);
#endif

	void							render_instanced_culled_rigged_vbo5			(	Camera*					cam,
																					GLuint&					vboId,
																					GLuint&					posTextureId,
																					GLuint&					pos_vbo_id,

																					GLuint&					rigging_mt_tex_id,
																					GLuint&					animation_mt_tex_id,

																					float					lod,
																					float					gender,
																					unsigned int			_AGENTS_NPOT,
																					unsigned int			_ANIMATION_LENGTH,
																					unsigned int			_STEP,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					float*					projMat,
																					float*					shadowMat,
																					bool					wireframe											);

	void							render_instanced_culled_rigged_vbo			(	GLuint&					vboId,
																					GLuint&					texture1,
																					GLuint&					texture2,
																					GLuint&					texture3,
																					GLuint&					texture4,
																					GLuint&					zonesTexture,
																					GLuint&					weightsTexture,
																					GLuint&					displacementTexture,
																					GLuint&					pos_vbo_id,
																					float					dt,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					bool					wireframe											);
	void							render_instanced_vbo						(	GLuint&					vboId,
																					GLuint&					texture,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					GLuint					uPosBuffer,
																					GLuint					uRotBuffer,
																					GLuint					uScaBuffer											);
	void							render_instanced_bumped_vbo					(	GLuint&					vboId,
																					GLuint&					texture,
																					GLuint&					normal_texture,
																					GLuint&					specular_texture,
																					unsigned int			size,
																					unsigned int			count,
																					unsigned int			positions_sizeInBytes,
																					float*					positions,
																					unsigned int			rotations_sizeInBytes,
																					float*					rotations,
																					float*					viewMat												);
	void							render_instanced_bumped_vbo					(	GLuint&					vboId,
																					GLuint&					texture,
																					GLuint&					normal_texture,
																					GLuint&					specular_texture,
																					unsigned int			size,
																					unsigned int			count,
																					float*					viewMat,
																					GLuint					uPosBuffer,
																					GLuint					uRotBuffer											);
	void							setInstancingLocations						(	string&					instancing_textured_shader_name,
																					string&					instancing_textured_positions_name,
																					string&					instancing_textured_rotations_name,
																					string&					instancing_textured_scales_name,
																					string&					instancing_textured_normal_name,
																					string&					instancing_textured_texCoord0_name,
																					string&					instancing_textured_viewmat_name,
																					string&					instancing_untextured_shader_name,
																					string&					instancing_untextured_positions_name,
																					string&					instancing_untextured_rotations_name,
																					string&					instancing_untextured_scales_name,
																					string&					instancing_untextured_normal_name,
																					string&					instancing_untextured_viewmat_name					);
	void							setInstancingCulledLocations				(	string&					instancing_culled_shader_name,
																					string&					instancing_culled_normal_name,
																					string&					instancing_culled_texCoord0_name,
																					string&					instancing_culled_viewmat_name						);
	void							setInstancingCulledRiggedLocations			(	string&					instancing_culled_rigged_shader_name,
																					string&					instancing_culled_rigged_normal_name,
																					string&					instancing_culled_rigged_dt_name,
																					string&					instancing_culled_rigged_texCoord0_name,
																					string&					instancing_culled_rigged_viewmat_name				);
	void							setInstancingCulledRiggedShadowLocations	(	string&					instancing_culled_rigged_shadow_shader_name,
																					string&					instancing_culled_rigged_shadow_normal_name,
																					string&					instancing_culled_rigged_shadow_texCoord0_name,
																					string&					instancing_culled_rigged_shadow_dt_name,
																					string&					instancing_culled_rigged_shadow_viewmat_name,
																					string&					instancing_culled_rigged_shadow_projmat_name,
																					string&					instancing_culled_rigged_shadow_shadowmat_name				);
	void							setBumpedInstancingLocations				(	string&					bump_shader_name,
																					string&					bump_positions_name,
																					string&					bump_rotations_name,
																					string&					bump_normal_name,
																					string&					bump_tangent_name,
																					string&					bump_texCoord0_name,
																					string&					bump_viewmat_name									);

public:
	vector<vector<VBO> >			vbos;
	vector<vector<GL_CUDA_VBO>>		gl_cuda_vbos;
	string							instancing_culled_rigged_shader_name;
	string							instancing_culled_rigged_shadow_shader_name;
private:
	LogManager*						log_manager;
	GlErrorManager*					err_manager;
	GlslManager*					glsl_manager;
	unsigned int					reused;
	map<string, unsigned int>		filename_index_map;

	string							instancing_textured_shader_name;
	string							instancing_textured_positions_name;
	string							instancing_textured_rotations_name;
	string							instancing_textured_scales_name;
	string							instancing_textured_normal_name;
	string							instancing_textured_texCoord0_name;
	string							instancing_textured_viewmat_name;
	GLuint							instancing_textured;
	unsigned int					instancing_textured_normalLocation;
	unsigned int					instancing_textured_texCoord0Location;
	GLint							instancing_textured_uPosLocation;
	GLint							instancing_textured_uRotLocation;
	GLint							instancing_textured_uScaLocation;

	string							instancing_untextured_shader_name;
	string							instancing_untextured_positions_name;
	string							instancing_untextured_rotations_name;
	string							instancing_untextured_scales_name;
	string							instancing_untextured_normal_name;
	string							instancing_untextured_viewmat_name;
	GLuint							instancing_untextured;
	unsigned int					instancing_untextured_normalLocation;
	GLint							instancing_untextured_uPosLocation;
	GLint							instancing_untextured_uRotLocation;
	GLint							instancing_untextured_uScaLocation;

	string							instancing_culled_shader_name;
	string							instancing_culled_normal_name;
	string							instancing_culled_texCoord0_name;
	string							instancing_culled_viewmat_name;
	GLuint							instancing_culled;
	unsigned int					instancing_culled_normalLocation;
	unsigned int					instancing_culled_texCoord0Location;

	string							instancing_culled_rigged_normal_name;
	string							instancing_culled_rigged_dt_name;
	string							instancing_culled_rigged_texCoord0_name;
	string							instancing_culled_rigged_viewmat_name;
	GLuint							instancing_culled_rigged;
	unsigned int					instancing_culled_rigged_normalLocation;
	unsigned int					instancing_culled_rigged_dtLocation;
	unsigned int					instancing_culled_rigged_texCoord0Location;

	string							instancing_culled_rigged_shadow_normal_name;
	string							instancing_culled_rigged_shadow_texCoord0_name;
	string							instancing_culled_rigged_shadow_dt_name;
	string							instancing_culled_rigged_shadow_viewmat_name;
	string							instancing_culled_rigged_shadow_projmat_name;
	string							instancing_culled_rigged_shadow_shadowmat_name;
	GLuint							instancing_culled_rigged_shadow;
	unsigned int					instancing_culled_rigged_shadow_normalLocation;
	unsigned int					instancing_culled_rigged_shadow_texCoord0Location;
	unsigned int					instancing_culled_rigged_shadow_dtLocation;

	string							bump_shader_name;
	string							bump_positions_name;
	string							bump_rotations_name;
	string							bump_normal_name;
	string							bump_tangent_name;
	string							bump_texCoord0_name;
	string							bump_viewmat_name;
	GLuint							bumped_instancing;
	unsigned int					bumped_instancing_tangentLocation;
	unsigned int					bumped_instancing_normalLocation;
	unsigned int					bumped_instancing_texCoord0Location;
	GLint							bumped_instancing_uPosLocation;
	GLint							bumped_instancing_uRotLocation;

	unsigned int*					indexPtr;
	GLuint							ibHandle;

	string							str_amb;
	string							str_tint;
	string							str_opacity;
};

#endif

//=======================================================================================
