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

#ifndef ACCELERATIONSTRUCTURE_H
#define ACCELERATIONSTRUCTURE_H

#include "Geometry.h"
#include "Primitive.h"

#ifndef ACCELERATION_STRUCTURE
enum ACCELERATION_STRUCTURE
{
    AS_NULLSTRUCT,
    AS_SIMPLE,
    AS_LBVH,
    AS_BIH,
    AS_KDTREE,
    AS_GRID,
    AS_BVH
};

inline ACCELERATION_STRUCTURE operator++( ACCELERATION_STRUCTURE &as, int ) 
{
   return as = (ACCELERATION_STRUCTURE)(as + 1);
}
#endif

class AccelerationStructure
{
public:
	AccelerationStructure(void);
	virtual ~AccelerationStructure(void) = 0;
	char* GetName() { return m_sName; }
	virtual TIntersection IntersectP(Ray &a_Ray) = 0;
	virtual void PrintOutput(float &totaltime) = 0;
	virtual void Build() = 0;
	virtual unsigned int GetNumberOfElements() = 0;

	Primitive**		GetPrimitives() { return m_pPrimitives; }
	unsigned int	GetNumPrimitives() { return m_uiNumPrimitives; }
	int				GetInitNode() { return m_iInitNode; }
	unsigned int	GetCandidates() { return m_uiCandidates; }
protected:
	Primitive**		m_pPrimitives;
	unsigned int	m_uiNumPrimitives;
	char*			m_sName;
	int				m_iInitNode;
	unsigned int	m_uiCandidates;
};

//Tests if a Ray Intersects a triangle
inline bool RayTriangleTest(Point &Start,Vector &Direction,TIntersection &intersection, unsigned int current, Primitive** a_Primitives)
{	
	Point A(a_Primitives[current]->GetVertex(0)->Pos);
	Point B(a_Primitives[current]->GetVertex(1)->Pos);
	Point C(a_Primitives[current]->GetVertex(2)->Pos);

	Vector E1(B-A);
	Vector E2(C-A);

	Vector P, Q;
	Cross(P, Direction, E2);
	float det;
	Dot(det, E1,P);
	det = 1.0f/det;

	Vector T(Start - A);
	float res;
	Dot(res,T,P);
	intersection.u = res*det;
	Cross(Q, T, E1);
	Dot(intersection.v, Direction,Q);
	intersection.v *= det;
	Dot(intersection.t, E2,Q);
	intersection.t *= det;
	
	//there is a hit
	return ((intersection.u>=0.0f)&&(intersection.u<=1.0f)&&(intersection.v>=0.0f)&&((intersection.u+intersection.v)<=1.0f)&&(intersection.t>=0.0f));
}

#endif