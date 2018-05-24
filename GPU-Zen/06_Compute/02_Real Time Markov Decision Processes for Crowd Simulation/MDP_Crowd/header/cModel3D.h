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

#include <unordered_map>
#include <vector>
#include <string>

#include "cMacros.h"
#include "cVertex.h"
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

#if defined _WIN32
	#include <unordered_map>
	using namespace std;
	using namespace std::tr1;
#else
	using namespace std;
#endif

typedef unordered_multimap<float,unsigned int> RefMap;

//=======================================================================================

#ifndef __MODEL3D
#define __MODEL3D

class Model3D
{
public:
											Model3D							(	void										);
											Model3D							(	string&					mname,
																				string&					rel_path,
																				VboManager*				vbo_manager,
																				GlslManager*			glsl_manager,
																				string&					fname,
																				float					scale				);
											Model3D							(	Model3D*				other				);
											~Model3D						(	void										);

	bool									init							(	bool					gen_vbos			);
	vector<GLuint>&							getSizes						(	void										);
	vector<GLuint>&							getIds							(	void										);
	vector<MODEL_MESH>&						getMeshes						(	void										);

	vector<unsigned int>&					getIndicesUvs					(	void										);
	vector<unsigned int>&					getIndicesNormals				(	void										);
	vector<unsigned int>&					getIndicesLocations				(	void										);
	vector<Location4>&						getUniqueLocations				(	void										);
	vector<Normal>&							getUniqueNormals				(	void										);
	vector<Uv>&								getUniqueUvs					(	void										);
	vector<Face3>&							getFaces						(	void										);

	string									name;
#ifndef ASSIMP_PURE_C
	Assimp::Importer						importer;
#endif
	const struct aiScene*					scene;
	unsigned int							ppsteps;
	string									rel_path;
	string									fileName;
	string									currFile;
	float									scale;
	bool									inited;

private:

	void									recursive_gather_data_A			(	const struct aiScene*	sc,
																				const struct aiNode*	nd					);
	void									recursive_gather_data_B			(	const struct aiScene*	sc,
																				const struct aiNode*	nd					);

	VboManager*								vbo_manager;
	GlslManager*							glsl_manager;

	vector<GLuint>							sizes;
	vector<GLuint>							ids;

	vector<unsigned int>					indices_uvs;
	vector<unsigned int>					indices_normals;
	vector<unsigned int>					indices_locations;

	vector<Location4>						unique_locations;
	vector<Normal>							unique_normals;
	vector<Uv>								unique_uvs;
	vector<Face3>							faces;

	unsigned int							vbo_frame;

	RefMap									l_map;
	RefMap									n_map;
	RefMap									u_map;

	vector<model_mesh>						meshes;
};

#endif

//=======================================================================================
