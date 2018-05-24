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

#include "cMacros.h"
#include "cLogManager.h"
#include "cCrowdParser.h"
#include "cModelProps.h"
#include "cCharacterGroup.h"
#include "cCrowd.h"
#include "cFboManager.h"
#include "cVboManager.h"
#include "cGlslManager.h"
#include "cGlErrorManager.h"
#include "cCamera.h"
#include "cTextureManager.h"
#include "cModel3D.h"

using namespace std;

//=======================================================================================

#ifndef __CROWD_MANAGER
#define __CROWD_MANAGER

class CrowdManager
{
public:
								CrowdManager					(	LogManager*							log_manager_,
																	GlErrorManager*						err_manager_,
																	FboManager*							fbo_manager_,
																	VboManager*							vbo_manager_,
																	GlslManager*						glsl_manager_,
																	string&								xml_file_			);
								~CrowdManager					(	void													);

	bool						init							(	void													);
	bool						loadAssets						(	void													);
	void						setFboManager					(	FboManager*							fbo_manager_		);
	void						initPaths						(	CROWD_POSITION&						crowd_position,
																	SCENARIO_TYPE&						scenario_type,
																	string&								str_scenario_speed,
																	float								plane_scale,
																	vector<vector<float>>&				policies,
																	unsigned int						grid_width,
																	unsigned int						grid_height,
																	vector<glm::vec2>&					occupation,
																	vector<vector<GROUP_FORMATION>>&	formations			);
	void						updatePolicy					(	vector<float>&						policy				);
	void						updatePolicies					(	vector<vector<float>>&				policies			);
	void						getDensity						(	vector<float>&						_density,
																	unsigned int						index				);
	void						runPaths						(	void													);
	void						setModelScale					(	float								_scale				);
	float						getModelScale					(	void													);
	float						getAvgRacingQW					(	void													);
	float						getAvgScatterGather				(	void													);

#ifdef DEMO_SHADER
	void						draw							(	Camera*								camera,
																	float*								viewMat,
																	float*								projMat,
																	float*								shadowMat,
																	bool								wireframe,
																	bool								shadows,
																	bool								doHandD,
																	bool								doPatterns,
																	bool								doColor,
																	bool								doFacial			);
#else
	void						draw							(	Camera*								camera,
																	float*								viewMat,
																	float*								projMat,
																	float*								shadowMat,
																	bool								wireframe,
																	bool								shadows				);
#endif
	void						nextFrame						(	void													);
	vector<Crowd*>&				getCrowds						(	void													);

private:

	void						split							(	const string&						str,
																	const string&						delimiters,
																	vector<string>&						tokens				);
	vector<float>				read_speeds						(	string&								file_name			);

	vector<Crowd*>				crowds;
	vector<ModelProps*>			clothing_model_props;
	vector<ModelProps*>			facial_model_props;
	vector<ModelProps*>			rigging_model_props;
	vector<CharacterGroup*>		char_groups;
	LogManager*					log_manager;
	FboManager*					fbo_manager;
	VboManager*					vbo_manager;
	GlslManager*				glsl_manager;
	GlErrorManager*				err_manager;
	CrowdParser*				crowd_parser;
	string						xml_file;
	float						model_scale;
	float						avg_racing_qw;
	float						avg_scatter_gather;

	map<string, unsigned int>	clothing_props_names_map;
	map<string, unsigned int>	facial_props_names_map;
	map<string, unsigned int>	rigging_props_names_map;
	map<string, unsigned int>	group_names_map;
};

#endif

//=======================================================================================
