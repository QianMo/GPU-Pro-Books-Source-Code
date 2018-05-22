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

// ------------------------------------------
// Gets the current ray intersection
// ------------------------------------------
Intersection getIntersection(Ray ray, float3 A, float3 B, float3 C)
{
	Intersection intersection;
	intersection.iTriangleId = -1;
	
	float3 P, T, Q;
	float3 E1 = B-A;
	float3 E2 = C-A;
	P = cross(ray.vfDirection,E2);
	float det = 1.0f/dot(E1,P);
	T = ray.vfOrigin - A;
	intersection.fU = dot(T,P) * det;
	Q = cross(T,E1);
	intersection.fV = dot(ray.vfDirection,Q)*det;
	intersection.fT = dot(E2,Q)*det;
	
	return intersection;
}

// ------------------------------------------
// Ray-Triangle intersection test
// ------------------------------------------
bool RayTriangleTest(Intersection intersection)
{
	return (
		(intersection.fU >= 0.0f)
		//&& (intersection.fU <= 1.0f)
		&& (intersection.fV >= 0.0f)
		&& ((intersection.fU+intersection.fV) <= 1.0f)
		&& (intersection.fT >= 0.0f)
	);
}

// ------------------------------------------
// Calculate UVs for a given triangle
// ------------------------------------------
float2 computeUVs(int iTriangleId, Intersection intersection)
{
	int offset = iTriangleId*3;
	float2 A,B,C;

	A.xy = g_sVertices[g_sIndices[offset]].vfUvs;
	B.xy = g_sVertices[g_sIndices[offset+1]].vfUvs;
	C.xy = g_sVertices[g_sIndices[offset+2]].vfUvs;

	return intersection.fU* B.xy +intersection.fV * C.xy + (1.0f-(intersection.fU + intersection.fV))*A;
}

// ------------------------------------------
// Calculate barycentric coordinates
// ------------------------------------------
float3 Vec3BaryCentric(float3 V1, float3 V2, float3 V3, float2 UV)
{
	float3 fOut = 0.f;
	float t = 1.f - (UV.x+UV.y);

	fOut[0] = V1[0]*t + V2[0]*UV.x + V3[0]*UV.y;
	fOut[1] = V1[1]*t + V2[1]*UV.x + V3[1]*UV.y;
	fOut[2] = V1[2]*t + V2[2]*UV.x + V3[2]*UV.y;

	return fOut;
}
