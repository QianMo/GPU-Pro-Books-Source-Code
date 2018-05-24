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

#include "cTextureManager.h"
//
//=======================================================================================
//
Texture3D::Texture3D
(
	unsigned int&		_id,
	string&				_name,
	vector<string>&		_file_names,
	unsigned int		_width,
	unsigned int		_height,
	unsigned int		_depth,
	unsigned int		_weight,
	unsigned int		_bpp,
	unsigned int		_texTarget,
	bool				_vert_flipped,
	bool				_horiz_flipped,
	bool				_mipmapped
)
{
	id					= _id;
	name				= string( _name );
	for( unsigned int n = 0; n < _file_names.size(); n++ )
	{
		file_names.push_back( _file_names[n] );
	}
	width				= _width;
	height				= _height;
	depth				= _depth;
	weight				= _weight;
	weight_kb			= BYTE2KB( weight );
	weight_mb			= BYTE2MB( weight );
	weight_gb			= BYTE2GB( weight );
	bpp					= _bpp;
	if( bpp == 4 )
	{
		transparent = true;
	}
	else
	{
		transparent = false;
	}
	texTarget			= _texTarget;
	vert_flipped		= _vert_flipped;
	horiz_flipped		= _horiz_flipped;
	mipmapped			= _mipmapped;
}
//
//=======================================================================================
//
Texture3D::Texture3D
(
	string&				_name,
	vector<string>&		_file_names,
	unsigned int		_width,
	unsigned int		_height,
	unsigned int		_depth,
	unsigned int		_weight,
	unsigned int		_bpp,
	unsigned int		_texTarget,
	bool				_vert_flipped,
	bool				_horiz_flipped,
	bool				_mipmapped
)
{
	id					= 0;
	name				= string( _name );
	for( unsigned int n = 0; n < _file_names.size(); n++ )
	{
		file_names.push_back( _file_names[n] );
	}
	width				= _width;
	height				= _height;
	depth				= _depth;
	weight				= _weight;
	weight_kb			= BYTE2KB( weight );
	weight_mb			= BYTE2MB( weight );
	weight_gb			= BYTE2GB( weight );
	bpp					= _bpp;
	texTarget			= _texTarget;
	vert_flipped		= _vert_flipped;
	horiz_flipped		= _horiz_flipped;
	mipmapped			= _mipmapped;
}
//
//=======================================================================================
//
Texture3D::~Texture3D( void )
{
	FREE_TEXTURE( id );
	name.erase();
}
//
//=======================================================================================
//
unsigned int Texture3D::getId( void )
{
	return id;
}
//
//=======================================================================================
//
string& Texture3D::getName( void )
{
	return name;
}
//
//=======================================================================================
//
unsigned int Texture3D::getWidth( void )
{
	return width;
}
//
//=======================================================================================
//
unsigned int Texture3D::getHeight( void )
{
	return height;
}
//
//=======================================================================================
//
unsigned int Texture3D::getDepth( void )
{
	return depth;
}
//
//=======================================================================================
//
unsigned int Texture3D::getWeightB( void )
{
	return weight;
}
//
//=======================================================================================
//
unsigned int Texture3D::getWeightKb( void )
{
	return weight_kb;
}
//
//=======================================================================================
//
unsigned int Texture3D::getWeightMb( void )
{
	return weight_mb;
}
//
//=======================================================================================
//
unsigned int Texture3D::getWeightGb( void )
{
	return weight_gb;
}
//
//=======================================================================================
//
unsigned int Texture3D::getBpp( void )
{
	return bpp;
}
//
//=======================================================================================
//
bool Texture3D::isHorizFlipped( void )
{
	return horiz_flipped;
}
//
//=======================================================================================
//
bool Texture3D::isVertFlipped( void )
{
	return vert_flipped;
}
//
//=======================================================================================
//
bool Texture3D::hasMipmaps( void )
{
	return mipmapped;
}
//
//=======================================================================================
//
bool Texture3D::isTransparent( void )
{
	return transparent;
}
//
//=======================================================================================
//
vector<string>& Texture3D::getFileNames( void )
{
	return file_names;
}
//
//=======================================================================================
