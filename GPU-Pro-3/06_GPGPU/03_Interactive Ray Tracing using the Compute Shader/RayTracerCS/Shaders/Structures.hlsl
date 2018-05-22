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
// Structures
// ------------------------------------------
struct Node						// 32 bytes	
{
	// Bounds
	float3 vfMin;				// 12 bytes
	float3 vfMax;				// 12 bytes

	int nPrimitives;			// 4 bytes
	uint primitivesOffset;		// 4 bytes
};

struct LBVHNode					// 32 bytes
{
	// Bounds
	float3 vfMin;				// 12 bytes
	float3 vfMax;				// 12 bytes
	
	int iPrimCount;				// 4 bytes
	int iPrimPos;				// 4 bytes
};

struct Ray						// 48 bytes
{
	float3 vfOrigin;			// 12 bytes
	float3 vfDirection;			// 12 bytes
	float3 vfReflectiveFactor;	// 12 bytes
	float fMinT,fMaxT;			// 8 bytes
	int iTriangleId;			// 4 bytes
};

struct Vertex					// 32 bytes
{
	float3 vfPosition;			// 12 bytes
	float3 vfNormal;			// 12 bytes
	float2 vfUvs;				// 8 bytes
};

struct Intersection				// 24 bytes
{
	int iTriangleId;			// 4 bytes
	float fU, fV, fT;			// 12 bytes
	int iVisitedNodes;			// 4 bytes
	int iRoot;					// 4 bytes
};

struct Material					// 4 bytes
{
	int iMaterialId;			// 4 bytes
};

struct MortonCode				// 8 bytes
{
	uint iCode;					// 4 bytes
	int iId;					// 4 bytes
};

struct Primitive				// 64 bytes
{
	float3 vfCentroid;			// 12 bytes
	float3 vfMin;				// 12 bytes
	float3 vfMax;				// 12 bytes
	float3 g_vfPadding1;		// 12 bytes
	float3 g_vfPadding2;		// 12 bytes
	int iMaterial;				// 4 bytes
};