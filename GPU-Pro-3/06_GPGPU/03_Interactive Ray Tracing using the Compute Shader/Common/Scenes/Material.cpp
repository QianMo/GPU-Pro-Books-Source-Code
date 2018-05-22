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

#include "Material.h"

Material::~Material(void)
{
	m_uiIndex = 0;
}

//-------------------------------------------------------------------------
// Get the specular map file name
//-------------------------------------------------------------------------
std::string Material::GetSpecularMap( void ) 
{ 
	std::string extension = m_sTextureName.substr(m_sTextureName.size()-4, 4);
	std::string diff = m_sTextureName.substr(m_sTextureName.size()-8, 4);
	if(diff == "diff")
	{
		return m_sTextureName.substr(0, m_sTextureName.size()-8) + "spec" + extension;
	}
	else
	{
		return m_sTextureName.substr(0, m_sTextureName.size()-4) + "_spec" + extension;
	}
}

//-------------------------------------------------------------------------
// Get the normal map file name
//-------------------------------------------------------------------------
std::string Material::GetNormalMap( void ) 
{ 
	std::string extension = m_sTextureName.substr(m_sTextureName.size()-4, 4);
	std::string diff = m_sTextureName.substr(m_sTextureName.size()-8, 4);
	if(diff == "diff")
	{
		return m_sTextureName.substr(0, m_sTextureName.size()-8) + "ddn" + extension;
	}
	else
	{
		return m_sTextureName.substr(0, m_sTextureName.size()-4) + "_ddn" + extension;
	}
	
}