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
#include <vector>
#include <string>

#include "cMacros.h"
#include "cTextureManager.h"
#include "cLogManager.h"
#include "cModelProps.h"
#include "cCharacterModel.h"
#include "cAnimation.h"
#include "cCamera.h"
#include "cStringUtils.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace std;
using namespace glm;

//=======================================================================================

#ifndef __CHARACTER_GROUP
#define __CHARACTER_GROUP

class CharacterGroup
{
public:
								CharacterGroup					(	LogManager*				log_manager_,
																	GlErrorManager*			err_manager_,
																	string&					_name,
																	MODEL_GENDER			_gender,
																	MODEL_TYPE				_type,
																	ModelProps*				mProps,
																	ModelProps*				mFacialProps,
																	ModelProps*				mRiggingProps		);
								~CharacterGroup					(	void										);

	ModelProps*					getModelProps					(	void										);

	void						setOutfit						(	string					ref_head,
																	string					ref_hair,
																	string					ref_torso,
																	string					ref_legs,
																	string					ref_palette			);
	void						setWeight						(	float					fat,
																	float					average,
																	float					thin,
																	float					strong				);
	void						setHeight						(	float					min,
																	float					max					);
	void						setFace							(	string					rf_wrinkles,
																	string					rf_eye_sockets,
																	string					rf_spots,
																	string					rf_beard,
																	string					rf_moustache,
																	string					rf_makeup			);
	void						setRig							(	string					rr_zones,
																	string					rr_weights,
																	string					rr_displacement,
																	string					rr_animation		);
	void						addAnimation					(	string					_animation,
																	GLuint					_frames,
																	GLuint					_duration			);
	bool						genCharacterModel				(	VboManager*				vbo_manager,
																	GlslManager*			glsl_manager,
																	float					scale				);
	void						nextFrame						(	void										);
	string&						getName							(	void										);
#ifdef CUDA_PATHS
#ifdef DEMO_SHADER
	void						draw_instanced_culled_rigged	(	Camera*					cam,
																	unsigned int			frame,
																	unsigned int			_AGENTS_NPOT,
																	struct sVBOLod*			vbo_lods,
																	GLuint&					posTextureBuffer,
																	float					dt,
																	float*					viewMat,
																	float*					projMat,
																	float*					shadowMat,
																	bool					wireframe,
																	bool					shadows,
																	bool					doHandD,
																	bool					doPatterns,
																	bool					doColor,
																	bool					doFacial			);
#else
	void						draw_instanced_culled_rigged	(	Camera*					cam,
																	unsigned int			frame,
																	unsigned int			_AGENTS_NPOT,
																	struct sVBOLod*			vbo_lods,
																	GLuint&					posTextureId,
																	float					dt,
																	float*					viewMat,
																	float*					projMat,
																	float*					shadowMat,
																	bool					wireframe,
																	bool					shadows				);
#endif
#else
#ifdef DEMO_SHADER
	void						draw_instanced_culled_rigged	(	Camera*					cam,
																	unsigned int			frame,
																	unsigned int			_AGENTS_NPOT,
																	struct sVBOLod*			vbo_lods,
																	GLuint&					posTextureTarget,
																	GLuint&					posTextureId,
																	float					dt,
																	float*					viewMat,
																	float*					projMat,
																	float*					shadowMat,
																	bool					wireframe,
																	bool					shadows,
																	bool					doHandD,
																	bool					doPatterns,
																	bool					doColor,
																	bool					doFacial			);
#else
	void						draw_instanced_culled_rigged	(	Camera*					cam,
																	unsigned int			frame,
																	unsigned int			_AGENTS_NPOT,
																	struct sVBOLod*			vbo_lods,
																	GLuint&					posTextureTarget,
																	GLuint&					posTextureId,
																	float					dt,
																	float*					viewMat,
																	float*					projMat,
																	float*					shadowMat,
																	bool					wireframe,
																	bool					shadows				);
#endif
#endif

private:
	void						genClothingMultitextures		(	void										);
	void						genFacialMultitexture			(	void										);
	void						genRiggingMultitexture			(	void										);

	ModelProps*					model_props;
	ModelProps*					facial_props;
	ModelProps*					rigging_props;

	float						height_min;
	float						height_max;
	float						weight_average;
	float						weight_fat;
	float						weight_thin;
	float						weight_strong;

	string						ref_outfit_head;
	string						ref_outfit_hair;
	string						ref_outfit_legs;
	string						ref_outfit_torso;

	string						ref_facial_wrinkles;
	string						ref_facial_eye_sockets;
	string						ref_facial_spots;
	string						ref_facial_beard;
	string						ref_facial_moustache;
	string						ref_facial_makeup;

	string						ref_rigging_zones;
	string						ref_rigging_weights;
	string						ref_rigging_displacement;
	string						ref_rigging_animation;

	string						outfit_palette;
	string						name;

	string						animation;
	GLuint						animation_frames;
	GLuint						animation_duration;

	MODEL_GENDER				gender;
	MODEL_TYPE					type;
	CharacterModel*				character_model[NUM_LOD];
	vector<Animation*>			animations;

	GLuint						globalMtId;
	GLuint						torsoMtId;
	GLuint						legsMtId;
	GLuint						combinedMtId;

	GLuint						facialMtId;
	GLuint						riggingMtId;
	GLuint						clothingColorTableId;
	GLuint						patternColorTableId;

	vector<string>				globalMtNames;
	vector<GLint>				globalMtParameters;

	vector<string>				torsoMtNames;
	vector<GLint>				torsoMtParameters;

	vector<string>				legsMtNames;
	vector<GLint>				legsMtParameters;

	vector<string>				combinedMtNames;
	vector<GLint>				combinedMtParameters;

	vector<string>				facialMtNames;
	vector<GLint>				facialMtParameters;
	vector<string>				riggingMtNames;
	vector<GLint>				riggingMtParameters;

	GlErrorManager*				err_manager;
	LogManager*					log_manager;
};

#endif

//=======================================================================================
