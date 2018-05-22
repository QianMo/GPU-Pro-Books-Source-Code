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

#ifndef __PRIMITIVE_H__
#define __PRIMITIVE_H__

#include "Material.h"
#include "Geometry.h"
#include "Vertex.h"

class Primitive
{
private:
	Material*				m_pMaterial;
	unsigned				int m_uiType;

	union
	{
		// Triangle
		struct
		{
			Vertex*			m_pVertex[3];
			//Vector		m_Normal;
		};
	};
public:
	enum
	{
		TRIANGLE = 1,
	};
	Primitive(Vertex* pVertex1, Vertex* pVertex2, Vertex* pVertex3);
	~Primitive(void);

	unsigned int			GetType() { return m_uiType; }
	
	int						IntersectP(Ray &a_Ray, float &a_T);
	Color					GetColor( void ) { }

	Material*				GetMaterial( void ) { return m_pMaterial; }
	Vertex*					GetVertex( unsigned int uiIdx ) { return m_pVertex[uiIdx]; }

	void					SetMaterial( Material* pMaterial ) { m_pMaterial = pMaterial; }
	void					SetVertex( unsigned int uiIdx, Vertex* pVertex ) { m_pVertex[uiIdx] = pVertex; }
};

#endif