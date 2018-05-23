#ifdef HOST_CODE
#pragma once
#endif

#ifndef __MATERIAL_PACKING__
#define __MATERIAL_PACKING__

#include "SharedTypes.h"


#if defined __cuda_cuda_h__ || defined __optix_optix_h__
#include <cmath>
using namespace std;
using namespace optix;
#endif


#if defined __cuda_cuda_h__ || defined __optix_optix_h__
	#if defined HOST_CODE
		// CPU
		float4 R8G8B8A8_UNORM_to_float4(unsigned int packedInput)
	#else
		// CUDA
		__device__ __host__  float4 R8G8B8A8_UNORM_to_float4(uint packedInput)
	#endif
#else
	// D3D
	float4 R8G8B8A8_UNORM_to_float4(uint packedInput)
#endif
{
	float4 unpackedOutput;
	unpackedOutput.x = (float)  (packedInput      & 0x000000ff) / 255;
	unpackedOutput.y = (float) ((packedInput>> 8) & 0x000000ff) / 255;
	unpackedOutput.z = (float) ((packedInput>>16) & 0x000000ff) / 255;
	unpackedOutput.w = (float) ((packedInput>>24) & 0x000000ff) / 255;
	return unpackedOutput;
}


// Quantize float4 to 8 bit per component
#if defined HOST_CODE
	// CPU
	unsigned int float4_to_R8G8B8A8_UNORM(float4 unpackedInput)
#elif defined __cuda_cuda_h__ || defined __optix_optix_h__	
	// CUDA
	__device__ __host__  uint float4_to_R8G8B8A8_UNORM(float4 unpackedInput)
#else
	// D3D
	uint float4_to_R8G8B8A8_UNORM(float4 unpackedInput)
#endif
{
	uint packedOutput;	
	unpackedInput.x = min(max(unpackedInput.x,0.0f), 1.0f);	// NaN gets set to 0.
	unpackedInput.y = min(max(unpackedInput.y,0.0f), 1.0f);	// NaN gets set to 0.
	unpackedInput.z = min(max(unpackedInput.z,0.0f), 1.0f);	// NaN gets set to 0.
	unpackedInput.w = min(max(unpackedInput.w,0.0f), 1.0f);	// NaN gets set to 0.
	unpackedInput.x = unpackedInput.x * 255.0f + 0.5f;
	unpackedInput.y = unpackedInput.y * 255.0f + 0.5f;
	unpackedInput.z = unpackedInput.z * 255.0f + 0.5f;
	unpackedInput.w = unpackedInput.w * 255.0f + 0.5f;
	
	unpackedInput = floor(unpackedInput);
	packedOutput =	(((uint)unpackedInput.x)      |
					(((uint)unpackedInput.y)<< 8) |
					(((uint)unpackedInput.z)<<16) |
					(((uint)unpackedInput.w)<<24) );
	return packedOutput;
}


// D3D
#ifndef __optix_optix_h__
float4 make_float4(float x, float y, float z, float w) {return float4(x, y, z, w); }
uint2 make_uint2(uint x, uint y) {return uint2(x, y); }
#endif

//-------------------------------------------------------------------------
// PackMaterial
//-------------------------------------------------------------------------

#if defined HOST_CODE
	// CPU
	uint2 packMaterial(const MaterialProperty& mat)
#elif defined __cuda_cuda_h__ || defined __optix_optix_h__	
	// CUDA
	__device__ __host__ uint2 packMaterial(const MaterialProperty& mat)
#else
	// D3D
	uint2 packMaterial(MaterialProperty mat)
#endif
{
	return make_uint2(  float4_to_R8G8B8A8_UNORM(make_float4(mat.Water, mat.Dirt, mat.Metal, mat.Wood)),
						float4_to_R8G8B8A8_UNORM(make_float4(mat.Organic, mat.Rust, mat.Stone, mat.Dummy)) );
}


//-------------------------------------------------------------------------
// UnpackMaterial
//-------------------------------------------------------------------------

#if defined __cuda_cuda_h__ || defined __optix_optix_h__
	#if defined HOST_CODE
		// CPU
		MaterialProperty unpackMaterial(uint2 packedMat)
	#else
		// CUDA
		__device__ __host__  MaterialProperty unpackMaterial(uint2 packedMat)
	#endif
#else
	// D3D
	MaterialProperty unpackMaterial(uint2 packedMat)
#endif
{
	float4 unpacked0 = R8G8B8A8_UNORM_to_float4(packedMat.x);
	float4 unpacked1 = R8G8B8A8_UNORM_to_float4(packedMat.y);
	MaterialProperty mat;
	mat.Water = unpacked0.x;
	mat.Dirt = unpacked0.y;
	mat.Metal = unpacked0.z;
	mat.Wood = unpacked0.w;
	mat.Organic = unpacked1.x;
	mat.Rust = unpacked1.y;
	mat.Stone = unpacked1.z;
	mat.Dummy = unpacked1.w;
	return mat;
}


//-------------------------------------------------------------------------
// ToFloat4
//-------------------------------------------------------------------------

#if defined __cuda_cuda_h__ || defined __optix_optix_h__
#if defined HOST_CODE
	// CPU
	MaterialProperty CreateMaterialPropertyFromFloat4(const float4& a, const float4& b)
#else
	// CUDA
	__device__ __host__  MaterialProperty CreateMaterialPropertyFromFloat4(const float4& a, const float4& b)	
#endif
#else
	// D3D
	MaterialProperty CreateMaterialPropertyFromFloat4(float4 a, float4 b)	
#endif
	{
		MaterialProperty prop;
		prop.Water = a.x;
		prop.Dirt = a.y;
		prop.Metal = a.z;
		prop.Wood = a.w;
		prop.Organic = b.x;
		prop.Rust = b.y;
		prop.Stone = b.z;
		prop.Dummy = b.w;
		return prop;
	}


#if defined __cuda_cuda_h__ || defined __optix_optix_h__
#if defined HOST_CODE
	// CPU
	MaterialProperty Substract(const MaterialProperty& a, const MaterialProperty& b)
#else
	// CUDA
	__device__ __host__  MaterialProperty Substract(const MaterialProperty& a, const MaterialProperty& b)	
#endif
#else
	// D3D
	MaterialProperty Substract(MaterialProperty a, MaterialProperty b)	
#endif
	{
		MaterialProperty prop;
		prop.Water = a.Water - b.Water;
		prop.Dirt = a.Dirt - b.Dirt;
		prop.Metal = a.Metal - b.Metal;
		prop.Wood = a.Wood - b.Wood;
		prop.Organic = a.Organic - b.Organic;
		prop.Rust = a.Rust - b.Rust;
		prop.Stone = a.Stone - b.Stone;
		prop.Dummy = a.Dummy - b.Dummy;
		return prop;
	}


#endif
