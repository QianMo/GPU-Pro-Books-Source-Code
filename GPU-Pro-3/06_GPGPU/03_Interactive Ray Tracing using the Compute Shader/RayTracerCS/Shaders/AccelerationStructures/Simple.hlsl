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

// -----------------------------------------------------------
// Simple ray intersection (No Acceleration Structure)
// -----------------------------------------------------------
Intersection IntersectP(Ray ray)
{
	Intersection cIntersection;
	Intersection bIntersection;
	bIntersection.iTriangleId = -1;
	bIntersection.fT = 10000.f;
	bIntersection.fU = -1;
	bIntersection.fV = -1;

	const int iNumPrimitives = 10;

	for(int i = 0; i < iNumPrimitives; ++i)
	{
		unsigned int offset = i*3;
		float3 A = g_sVertices[g_sIndices[offset]].vfPosition;
		float3 B = g_sVertices[g_sIndices[offset+1]].vfPosition;
		float3 C = g_sVertices[g_sIndices[offset+2]].vfPosition;
		
		cIntersection = getIntersection(ray,A,B,C);
		if(ray.iTriangleId != i && 
			RayTriangleTest(cIntersection) && 
			cIntersection.fT < bIntersection.fT)
		{
			bIntersection = cIntersection;
			bIntersection.iTriangleId = i;
		}
	}

	return bIntersection;
}