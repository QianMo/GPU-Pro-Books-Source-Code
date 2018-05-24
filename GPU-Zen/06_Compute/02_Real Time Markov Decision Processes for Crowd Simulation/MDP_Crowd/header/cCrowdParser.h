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
#include "cXmlParser.h"
#include "cStringUtils.h"
#include <glm/glm.hpp>
#include "cLogManager.h"

#include <vector>
#include <map>
#include <sstream>

using namespace std;
using namespace glm;

//=======================================================================================

#ifndef __PARSED_PROP
#define __PARSED_PROP

class ParsedProp
{
public:
	ParsedProp( void )
	{
		type			= PP_HEAD;
		subtype			= PST_SKIN;
		name			= "";
		file			= "";
		ref_wrinkles	= "";
	};
	~ParsedProp( void )
	{

	};
	PROP_PART		type;
	PROP_SUBTYPE	subtype;
	string			name;
	string			file;
	string			ref_wrinkles;
	vector<string>	ref_pattern;
	vector<string>	ref_atlas;
};

#endif

//=======================================================================================

#ifndef __PARSED_MODEL_PROPS
#define __PARSED_MODEL_PROPS

class ParsedModelProps
{
public:
	ParsedModelProps( void )
	{

	};
	~ParsedModelProps( void )
	{

	}
	MODEL_PROPS_TYPE			type;
	string						name;
	MODEL_GENDER				sex;
	vector<ParsedProp*>			parsed_props;
	map<string, unsigned int>	props_names_map;
};

#endif

//=======================================================================================

#ifndef __PARSED_GROUP
#define __PARSED_GROUP

class ParsedGroup
{
public:
	ParsedGroup( void )
	{
		name				= "";
		ref_clothing_props	= "";
		ref_facial_props	= "";
		ref_rigging_props	= "";
		ref_head			= "";
		ref_hair			= "";
		ref_torso			= "";
		ref_legs			= "";
		ref_wrinkles		= "";
		ref_eye_sockets		= "";
		ref_spots			= "";
		ref_beard			= "";
		ref_moustache		= "";
		ref_makeup			= "";
		ref_zones			= "";
		ref_weights			= "";
		ref_displacement	= "";
		ref_animation		= "";
		fat					= 0.0f;
		average				= 0.0f;
		thin				= 0.0f;
		strong				= 0.0f;
		min					= 0.0f;
		max					= 0.0f;
		sex					= MG_MALE;
		type				= MT_HUMAN;
	};
	~ParsedGroup( void )
	{

	};
	string			name;
	string			ref_clothing_props;
	string			ref_facial_props;
	string			ref_rigging_props;
	string			ref_head;
	string			ref_hair;
	string			ref_torso;
	string			ref_legs;
	string			ref_palette;
	string			ref_wrinkles;
	string			ref_eye_sockets;
	string			ref_spots;
	string			ref_beard;
	string			ref_moustache;
	string			ref_makeup;
	string			ref_zones;
	string			ref_weights;
	string			ref_displacement;
	string			ref_animation;
	float			fat;
	float			average;
	float			thin;
	float			strong;
	float			min;
	float			max;
	MODEL_GENDER	sex;
	MODEL_TYPE		type;
};

#endif

//=======================================================================================

#ifndef __PARSED_CROWD_GROUP
#define __PARSED_CROWD_GROUP

class ParsedCrowdGroup
{
public:
	ParsedCrowdGroup( void )
	{
		ref					= "";
		percentage			= 0.0f;
		animation			= "";
		animation_frames	= 0;
		animation_duration	= 0;
	};
	~ParsedCrowdGroup( void )
	{

	};
	string			ref;
	float			percentage;
	string			animation;
	unsigned int	animation_frames;
	unsigned int	animation_duration;
};

#endif

//=======================================================================================

#ifndef __PARSED_CROWD
#define __PARSED_CROWD

class ParsedCrowd
{
public:
	ParsedCrowd( void )
	{

	};
	~ParsedCrowd( void )
	{

	};
	string						name;
	GLuint						width;
	GLuint						height;
	vector<ParsedCrowdGroup*>	groups;
};

#endif

//=======================================================================================

#ifndef __CROWD_PARSER
#define __CROWD_PARSER

class CrowdParser
{
public:
								CrowdParser				(	LogManager*		log_manager,
															char*			fname		);
								~CrowdParser			(	void						);

	bool						init					(	void						);
	unsigned int				getNumCrowds			(	void						);
	unsigned int				getNumGroups			(	void						);
	unsigned int				getNumModelProps		(	void						);
	vector<ParsedCrowd*>&		getParsed_Crowds		(	void						);
	vector<ParsedGroup*>&		getParsed_Groups		(	void						);
	vector<ParsedModelProps*>&	getParsed_ModelProps	(	void						);
	map<string, unsigned int>	parsed_model_props_names_map;
	map<string, unsigned int>	parsed_groups_names_map;

private:
	char*						filename;
	XmlParser*					parser;
	vector<ParsedCrowd*>		parsed_crowds;
	vector<ParsedGroup*>		parsed_groups;
	vector<ParsedModelProps*>	parsed_model_props;

	unsigned int				NUM_CROWDS;
	unsigned int				NUM_GROUPS;
	unsigned int				NUM_MODEL_PROPS;
	unsigned int				rand_min;
	unsigned int				rand_max;
	LogManager*					log_manager;
private:
	void						parse					(	void						);
	float						getRandom				(	void						);
	void						walk_tree				(	TiXmlNode*		pParent,
															unsigned int	curr_crowd,
															unsigned int	curr_model_props,
															unsigned int	curr_group,
															char*			curr_tag	);
};

#endif

//=======================================================================================
