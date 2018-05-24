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
#include <map>

#include "cMacros.h"
#include "cModel3D.h"

using namespace std;

//=======================================================================================

#ifndef __MODEL_PROP
#define __MODEL_PROP
class ModelProp
{
public:
	ModelProp( void )
	{
		type			= PP_HEAD;
		subtype			= PST_SKIN;
		name			= "";
		file			= "";
		ref_wrinkles	= NULL;
		model3D			= NULL;
		textureID		= 0;
		loaded			= false;
	};
	~ModelProp( void )
	{

	}

	PROP_PART				type;
	PROP_SUBTYPE			subtype;
	string					name;
	string					file;
	ModelProp*				ref_wrinkles;
	vector<ModelProp*>		ref_pattern;
	vector<ModelProp*>		ref_atlas;
	Model3D*				model3D;
	GLuint					textureID;
	bool					loaded;
};
#endif

//=======================================================================================

#ifndef __MODEL_PROPS
#define __MODEL_PROPS

class ModelProps
{
public:
								ModelProps	(	MODEL_PROPS_TYPE	_type,
												string				_name,
												MODEL_GENDER		_gender			);
								~ModelProps	(	void								);

	void						addProp		(	ModelProp*			prop			);
	void						addProp		(	PROP_PART			type,
												PROP_SUBTYPE		subtype,
												string				name,
												string				file,
												ModelProp*			ref_wrinkles,
												vector<ModelProp*>&	ref_pattern,
												vector<ModelProp*>&	ref_atlas,
												Model3D*			model3D,
												GLuint&				textureID,
												bool				loaded			);
	ModelProp*					getProp		(	string				prop_name		);
	vector<ModelProp*>&			getProps	(	void								);
	string&						getName		(	void								);
	MODEL_PROPS_TYPE&			getType		(	void								);

private:
	MODEL_PROPS_TYPE			type;
	string						name;
	MODEL_GENDER				gender;
	vector<ModelProp*>			props;
	map<string, unsigned int>	name_prop_map;
};

#endif
//=======================================================================================
