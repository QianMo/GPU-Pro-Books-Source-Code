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
#include <vector>
#include <map>

#include "cVertex.h"
#include "cMacros.h"
#include "cVboManager.h"
#include "cGlslManager.h"

#include "cTextureManager.h"
#ifdef ASSIMP_PURE_C
	#include <assimp/cimport.h>
	#include <assimp/scene.h>
	#include <assimp/postprocess.h>
	#include <assimp/material.h>
	#include <assimp/Importer.hpp>
#else
	#include <assimp/assimp.hpp>
	#include <assimp/aiScene.h>
	#include <assimp/aiPostProcess.h>
	#include <assimp/aiMaterial.h>
#endif

using namespace std;

//=======================================================================================

#ifndef __SCENARIO
#define __SCENARIO

class Scenario
{
public:
							Scenario						(	string&					name_,
																string&					rel_path_,
																VboManager*				vbo_manager_,
																GlslManager*			shader_manager_,
																string&					fname_,
																float					scale_			);
							~Scenario						(	void									);

	void					draw							(	void									);

	string					name;
private:
	void					assimp_recursive_gather_data	(	const struct aiScene*	sc,
																const struct aiNode*	nd				);

	float					scale;
	VboManager*				vbo_manager;
#ifndef ASSIMP_PURE_C
	Assimp::Importer		importer;
#endif
	const struct aiScene*	scene;
	vector<GLuint>			sizes;
	vector<GLuint>			ids;
	int						numTextures;
	map<string, GLuint>		textureIdMap;
	vector<model_mesh>		meshes;
	string					rel_path;
	string					fname;
	unsigned int			vbo_frame;

	GlslManager*			glsl_manager;

	string					str_amb;
	string					str_textured;
	string					str_tint;
	string					str_opacity;
};

#endif

//=======================================================================================
