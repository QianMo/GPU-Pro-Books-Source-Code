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

#ifndef _MODEL_H_
#define _MODEL_H_

#include "string.h"
#include "Common.h"
#include "Triangle.h"
#include "Vertex.h"

#include "NullShader.h"
#include "bvh.h"
//#include "CubicGrid.h"
//#include "Simple.h"
//#include "BIH.h"
//#include "LBVH.h"
//#include "KDTree.h"

#include <stdio.h>
#include "Loaders/model_obj.h"

class Model
{
private:
	Primitive					**m_ppPrimitives; // pointer to the list of primitives
	Material					**m_ppMaterials; // pointer to the list of materials
	AccelerationStructure		*m_pAccelStructure; // pointer to the current AS
	Vertex						*m_pVertices; // pointer to vertex array
	DWORD						*m_pIndices; // 3 indices = 1 triangle
	unsigned int				m_uiNumVertices; // total number of vertices
	unsigned int				m_uiNumPrimitives; // total number of primitives
	unsigned int				m_uiNumMaterials; // total number of materials
	string						m_sName; // name of the model
	ACCELERATION_STRUCTURE		m_eAccelStructureType;  // acceleration structure built type
	ModelOBJ					m_rModelObj; // *.obj model loader
public:
	Model( string sFileName = NULL );
	~Model(void);

	Primitive**					GetPrimitives( void ) { return m_ppPrimitives; }
	AccelerationStructure*		GetAccelStructure( void ) { return m_pAccelStructure; }
	Vertex*						GetVertices( void ) { return m_pVertices; }
	Material**					GetMaterials( void ) { return m_ppMaterials; }
	string						GetName( void ) { return m_sName; }
	unsigned int				GetNumPrimitives() { return m_uiNumPrimitives; }
	unsigned int				GetNumVertices() { return m_uiNumVertices; }
	unsigned int				GetNumMaterials() { return m_uiNumMaterials; }
	DWORD*						GetIndices() { return m_pIndices; }
	void						RebuildStructure();
	void						SetCurrentStructureType(ACCELERATION_STRUCTURE eStructure);
	void						LoadFile(string a_FileName);
	void						Scale();
};

#endif