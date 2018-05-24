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
#include "cModelProps.h"

//=======================================================================================
//
ModelProps::ModelProps( MODEL_PROPS_TYPE _type, string _name, MODEL_GENDER _gender )
{
	type	= _type;
	name	= string( _name );
	gender	= _gender;
}
//
//=======================================================================================
//
ModelProps::~ModelProps( void )
{

}
//
//=======================================================================================
//
//
void ModelProps::addProp( ModelProp* prop )
{
	if( name_prop_map.find( prop->name ) == name_prop_map.end() )
	{
		name_prop_map[prop->name] = props.size();
		props.push_back( prop );
	}
}
//
//=======================================================================================
//
void ModelProps::addProp(	PROP_PART				type,
							PROP_SUBTYPE			subtype,
							string					name,
							string					file,
							ModelProp*				ref_wrinkles,
							vector<ModelProp*>&		ref_pattern,
							vector<ModelProp*>&		ref_atlas,
							Model3D*				model3D,
							GLuint&					textureID,
							bool					loaded		)
{
	if( name_prop_map.find( name ) == name_prop_map.end() )
	{
		ModelProp* mp				= new ModelProp();
		mp->type					= type;
		mp->subtype					= subtype;
		mp->name					= name;
		mp->file					= file;
		mp->ref_wrinkles			= ref_wrinkles;
		for( unsigned int p = 0; p < ref_pattern.size(); p++ )
		{
			mp->ref_pattern.push_back( ref_pattern[p] );
		}
		for( unsigned int a = 0; a < ref_atlas.size(); a++ )
		{
			mp->ref_atlas.push_back( ref_atlas[a] );
		}
		mp->model3D					= model3D;
		mp->textureID				= textureID;
		mp->loaded					= loaded;
		name_prop_map[mp->name]		= props.size();
		props.push_back( mp );
	}
}
//
//=======================================================================================
//
ModelProp* ModelProps::getProp( string prop_name )
{
	if( name_prop_map.find( prop_name ) != name_prop_map.end() )
	{
		return props[name_prop_map[prop_name]];
	}
	else
	{
		return NULL;
	}
}
//
//=======================================================================================
//
vector<ModelProp*>& ModelProps::getProps( void )
{
	return props;
}
//
//=======================================================================================
//
string& ModelProps::getName( void )
{
	return name;
}
//
//=======================================================================================
//
MODEL_PROPS_TYPE& ModelProps::getType( void )
{
	return type;
}
//
//=======================================================================================
