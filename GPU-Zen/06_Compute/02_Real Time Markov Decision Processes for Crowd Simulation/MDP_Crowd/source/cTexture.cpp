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
Texture::Texture
(
	unsigned int&		id,
	string&				name,
	unsigned int		width,
	unsigned int		height,
	unsigned int		weight,
	unsigned int		bpp,
	unsigned int		texTarget,
	bool				vert_flipped,
	bool				horiz_flipped,
	bool				mipmapped
)
{
	this->id			= id;
	this->name			= string( name );
	this->width			= width;
	this->height		= height;
	this->weight		= weight;
	weight_kb			= BYTE2KB( weight );
	weight_mb			= BYTE2MB( weight );
	weight_gb			= BYTE2GB( weight );
	this->bpp			= bpp;
	if( bpp == 4 )
	{
		transparent = true;
	}
	else
	{
		transparent = false;
	}
	this->texTarget		= texTarget;
	this->vert_flipped	= vert_flipped;
	this->horiz_flipped	= horiz_flipped;
	this->mipmapped		= mipmapped;
}
//
//=======================================================================================
//
Texture::Texture
(
	string&			name,
	unsigned int	width,
	unsigned int	height,
	unsigned int	weight,
	unsigned int	bpp,
	unsigned int	texTarget,
	bool			vert_flipped,
	bool			horiz_flipped,
	bool			mipmapped
)
{
	id					= 0;
	this->name			= string( name );
	this->width			= width;
	this->height		= height;
	this->weight		= weight;
	weight_kb			= BYTE2KB( weight );
	weight_mb			= BYTE2MB( weight );
	weight_gb			= BYTE2GB( weight );
	this->bpp			= bpp;
	this->texTarget		= texTarget;
	this->vert_flipped	= vert_flipped;
	this->horiz_flipped	= horiz_flipped;
	this->mipmapped		= mipmapped;
}
//
//=======================================================================================
//
Texture::~Texture( void )
{
	FREE_TEXTURE( id );
	name.erase();
}
//
//=======================================================================================
//
unsigned int Texture::getId( void )
{
	return id;
}
//
//=======================================================================================
//
string& Texture::getName( void )
{
	return name;
}
//
//=======================================================================================
//
unsigned int Texture::getWidth( void )
{
	return width;
}
//
//=======================================================================================
//
unsigned int Texture::getHeight( void )
{
	return height;
}
//
//=======================================================================================
//
unsigned int Texture::getWeightB( void )
{
	return weight;
}
//
//=======================================================================================
//
unsigned int Texture::getWeightKb( void )
{
	return weight_kb;
}
//
//=======================================================================================
//
unsigned int Texture::getWeightMb( void )
{
	return weight_mb;
}
//
//=======================================================================================
//
unsigned int Texture::getWeightGb( void )
{
	return weight_gb;
}
//
//=======================================================================================
//
unsigned int Texture::getBpp( void )
{
	return bpp;
}
//
//=======================================================================================
//
bool Texture::isHorizFlipped( void )
{
	return horiz_flipped;
}
//
//=======================================================================================
//
bool Texture::isVertFlipped( void )
{
	return vert_flipped;
}
//
//=======================================================================================
//
bool Texture::hasMipmaps( void )
{
	return mipmapped;
}
//
//=======================================================================================
//
bool Texture::isTransparent( void )
{
	return transparent;
}
//
//=======================================================================================
