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
#include <string>

#include "cMacros.h"
#include "cCharacterGroup.h"
#include "cStaticLod.h"
#include "cLogManager.h"
#include "cFboManager.h"
#include "cVboManager.h"
#include "cCamera.h"
#include "cMDPCudaPathManager.h"

using namespace std;

//=======================================================================================

#ifndef __CROWD_GROUP
#define __CROWD_GROUP

class CrowdGroup
{
public:
						CrowdGroup	( 	CharacterGroup*	_cgroup,
										StaticLod*		_static_lod,
										string			_animation,
										string			_fbo_lod_name,
										string			_fbo_pos_tex_name,
										string			_fbo_ids_tex_name,
										float			_percentage,
										float			_dt,
										GLuint			_frames,
										GLuint			_duration,
										GLuint			_frame,
										GLuint			_width,
										GLuint			_height			);
						CrowdGroup	( 	CharacterGroup*	_cgroup,
										string			_animation,
										float			_percentage,
										float			_dt,
										GLuint			_frames,
										GLuint			_duration,
										GLuint			_frame			);
						~CrowdGroup	( 	void 							);

	GLuint				getWidth	(	void							);
	GLuint				getHeight	(	void							);
#ifdef LOCAL_POS_TEXTURE
	bool				init_paths	(	LogManager*		log_manager,
										FboManager*		fbo_manager,
										VboManager*		vbo_manager,
										float			plane_scale		);
	GLuint				init_ids	(	FboManager*		fbo_manager,
										GLuint&			_groupId,
										GLuint			offset			);
	void				run_paths	(	void							);
#endif
	void				nextFrame	(	void							);

#ifdef CUDA_PATHS
#ifdef DEMO_SHADER
	GLuint				draw		(	FboManager*			fbo_manager,
										VboManager*			vbo_manager,
										GlslManager*		glsl_manager,
										MDPCudaPathManager*	cuda_path_manager,
										StaticLod*			_static_lod,
										struct sVBOLod*		vboCulledLOD,
										string				_fbo_lod_name,
										string				_fbo_pos_tex_name,
										unsigned int		_AGENTS_NPOT,
										Camera*				camera,
										float*				viewMat,
										float*				projMat,
										float*				shadowMat,
										bool				wireframe,
										bool				shadows,
										bool				doHandD,
										bool				doPatterns,
										bool				doColor,
										bool				doFacial		);
	void				draw		(	FboManager*			fbo_manager,
										VboManager*			vbo_manager,
										GlslManager*		glsl_manager,
										MDPCudaPathManager*	cuda_path_manager,
										Camera*				camera,
										float*				viewMat,
										float*				projMat,
										float*				shadowMat,
										bool				wireframe,
										bool				shadows,
										bool				doHandD,
										bool				doPatterns,
										bool				doColor,
										bool				doFacial		);
#else
	GLuint				draw		(	FboManager*			fbo_manager,
										VboManager*			vbo_manager,
										GlslManager*		glsl_manager,
										CudaPathManager*	cuda_path_manager,
										StaticLod*			_static_lod,
										struct sVBOLod*		vboCulledLOD,
										string				_fbo_lod_name,
										string				_fbo_pos_tex_name,
										unsigned int		_AGENTS_NPOT,
										Camera*				camera,
										float*				viewMat,
										float*				projMat,
										float*				shadowMat,
										bool				wireframe,
										bool				shadows			);
	void				draw		(	FboManager*			fbo_manager,
										VboManager*			vbo_manager,
										GlslManager*		glsl_manager,
										CudaPathManager*	cuda_path_manager,
										Camera*				camera,
										float*				viewMat,
										float*				projMat,
										float*				shadowMat,
										bool				wireframe,
										bool				shadows			);
#endif
#else
#ifdef DEMO_SHADER
	GLuint				draw		(	FboManager*		fbo_manager,
										VboManager*		vbo_manager,
										GlslManager*	glsl_manager,
										StaticLod*		_static_lod,
										struct sVBOLod*	vboCulledLOD,
										string			_fbo_lod_name,
										string			_fbo_pos_tex_name,
										unsigned int	_AGENTS_NPOT,
										Camera*			camera,
										float*			viewMat,
										float*			projMat,
										float*			shadowMat,
										bool			wireframe,
										bool			shadows,
										bool			doHandD,
										bool			doPatterns,
										bool			doColor,
										bool			doFacial		);
	void				draw		(	FboManager*		fbo_manager,
										VboManager*		vbo_manager,
										GlslManager*	glsl_manager,
										Camera*			camera,
										float*			viewMat,
										float*			projMat,
										float*			shadowMat,
										bool			wireframe,
										bool			shadows,
										bool			doHandD,
										bool			doPatterns,
										bool			doColor,
										bool			doFacial		);
#else
	GLuint				draw		(	FboManager*		fbo_manager,
										VboManager*		vbo_manager,
										GlslManager*	glsl_manager,
										StaticLod*		_static_lod,
										struct sVBOLod*	vboCulledLOD,
										string			_fbo_lod_name,
										string			_fbo_pos_tex_name,
										unsigned int	_AGENTS_NPOT,
										Camera*			camera,
										float*			viewMat,
										float*			projMat,
										float*			shadowMat,
										bool			wireframe,
										bool			shadows			);
	void				draw		(	FboManager*		fbo_manager,
										VboManager*		vbo_manager,
										GlslManager*	glsl_manager,
										Camera*			camera,
										float*			viewMat,
										float*			projMat,
										float*			shadowMat,
										bool			wireframe,
										bool			shadows			);
#endif
#endif

	StaticLod*			static_lod;
#ifdef LOCAL_POS_TEXTURE
	CudaPathManager*	cuda_path_manager;
	float				dt;
	float				path_param;
	vector<float>		instance_positions_flat;
	vector<float>		instance_rotations_flat;
	vector<float>		instance_control_points;
	vector<float>		instance_ids_flat;
#endif
	sVBOLod				vbo_lod[NUM_LOD];
	CharacterGroup*		cgroup;
	string				animation;
	string				fbo_lod_name;
	string				fbo_pos_tex_name;
	string				fbo_ids_tex_name;
	float				percentage;
	GLuint				frames;
	GLuint				duration;
	GLuint				frame;
	GLuint				width;
	GLuint				height;
	GLuint				id;
};
#endif
