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

// Ray Tracing on the Compute Shader.

// Use or not global illumination
//#define GLOBAL_ILLUM

// Include constant buffers and user-defined structures
#include "ConstantBuffers.hlsl"
#include "Structures.hlsl"

// SRVs
StructuredBuffer<int>				g_sPrimitives		:	register( t0 );
StructuredBuffer<Node>				g_sNodes			:	register( t1 );
StructuredBuffer<Vertex>			g_sVertices			:	register( t2 );
StructuredBuffer<DWORD>				g_sIndices			:	register( t3 );
StructuredBuffer<LBVHNode>			g_sLBVHNodes		:	register( t10 );
Texture2DArray						g_sTextures			:	register( t4 );
Texture2DArray						g_sSpecularMaps		:	register( t5 );
Texture2DArray						g_sNormalMaps		:	register( t6 );
StructuredBuffer<Material>			g_sMaterials		:	register( t7 );
Texture2D							g_sRandomTx			:	register( t8 );
TextureCube							g_sEnvironmentTx	:	register( t9 );

// UAVs
RWTexture2D<float4>					g_uResultTexture	:	register( u0 );
RWStructuredBuffer<Ray>				g_uRays				:	register( u1 );
RWStructuredBuffer<Intersection>	g_uIntersections	:	register( u2 );
RWStructuredBuffer<float4>			g_uAccumulation		:	register( u3 );
RWStructuredBuffer<Primitive>		g_uPrimitives		:	register( u4 );
RWStructuredBuffer<LBVHNode>		g_uLVBH				:	register( u5 );
RWStructuredBuffer<MortonCode>		g_uMortonCode		:	register( u6 );

// Samplers
SamplerState g_ssSampler : register(s0);

// Include application files
#include "Core/Intersection.hlsl"
//#include "AccelerationStructures/LBVH.hlsl"
#include "AccelerationStructures/BVH.hlsl"
#include "AccelerationStructures/Simple.hlsl"
#include "CS/CSPrimaryRays.hlsl"
#include "CS/CSIntersections.hlsl"
#ifndef GLOBAL_ILLUM
#include "CS/CSComputeColorRT.hlsl"
#else // global illumination
#include "CS/CSComputeColorGI.hlsl"
#endif