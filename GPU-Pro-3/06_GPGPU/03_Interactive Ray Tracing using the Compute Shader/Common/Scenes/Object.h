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

// ------------------------------------------------
// Object.h
// ------------------------------------------------
// Object properties of a scene

#ifndef _OBJECT_H_
#define _OBJECT_H_

#include "Common.h"
#include "Geometry.h"
#include "Vertex.h"
#include "Matrix4.h"
#include "Model.h"

// Instancing models
class Object
{
private:
	Matrix4			m_World; // Position of the instance
	Model*			m_Model; // Pointer to the model
public:

	Object(Model* a_Model, Point &a_Position);
	~Object(void);

	Model*			GetModel() { return m_Model; }
	Vertex*			GetVertices() { return m_Model->GetVertices(); }
	unsigned int	GetNumPrimitives() { return m_Model->GetNumPrimitives(); }
	unsigned int	GetNumVertices() { return m_Model->GetNumVertices(); }
	Primitive**		GetPrimitives() { return m_Model->GetPrimitives(); }
	string			GetName()	{ return m_Model->GetName(); }
	Matrix4			GetWorldMatrix() { return m_World; }
};

#endif