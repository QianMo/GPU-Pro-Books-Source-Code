// ================================================================================ //
// Copyright (c) 2011 Arturo Garcia, Francisco Avila, Sergio Murguia and Leo Reyes	//
//																					//
// Permission is hereby granted, free of charge, to any person obtaining a copy of	//
// this software and associated documentation files (the "Software"), to deal in	//
// the Software without restriction, including without limitation the rights to		//
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies	//
// of the Software, and to permit persons to whom the Software is furnished to do	//
// so, subject to the following conditions:											//
//																					//
// The above copyright notice and this permission notice shall be included in all	//
// copies or substantial portions of the Software.									//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR		//
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,			//
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE		//
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER			//
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,	//
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE	//
// SOFTWARE.																		//
// ================================================================================ //

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "Geometry.h"
#include <string>

class Material
{
private:
	Color			m_Color; // float3 color for the material
	float			m_fReflection; // reflective rate
	float			m_fRefraction; // refraction rate
	float			m_fDiffuse; // diffuse rate
	std::string		m_sTextureName; // texture name
	unsigned int	m_uiIndex; // id of the material
public:
	Material( unsigned int uiIndex = 0 ) : 
		m_Color( Color ( 0.2f, 0.2f, 0.2f ) ), m_fReflection ( 0 ), m_fDiffuse( 0.2f ), 
		m_fRefraction( 0 ), m_sTextureName( "" ), m_uiIndex( uiIndex ) {}
		Material( unsigned int uiIndex, std::string sTextureName) : 
		m_sTextureName( sTextureName ), m_Color( Color ( 0.2f, 0.2f, 0.2f ) ), 
		m_fReflection ( 0 ), m_fDiffuse( 0.2f ), m_fRefraction( 0 ), m_uiIndex( uiIndex ) { }
	~Material( void );

	Color			GetColor( void ) { return m_Color; }
	float			GetReflection( void ) { return m_fReflection; }
	float			GetRefraction( void ) { return m_fRefraction; }
	float			GetDiffuse( void ) { return m_fDiffuse; }
	std::string		GetTextureName( void ) { return m_sTextureName; }
	std::string		GetSpecularMap( void );
	std::string		GetNormalMap( void );
	unsigned int	GetIndex( void ) { return m_uiIndex; }

	void			SetColor( Color &color ) { m_Color = color; }
	void			SetReflection( float fReflection ) { m_fReflection = fReflection; }
	void			SetRefraction( float fRefraction ) { m_fRefraction = fRefraction; }
	void			SetDiffuse( float fDiffuse ) { m_fDiffuse = fDiffuse; }
};

#endif