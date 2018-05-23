#ifndef BE_RAYTRACING_VOXEL_REP_H
#define BE_RAYTRACING_VOXEL_REP_H

#include "Engine/BindPoints.fx"

/// Voxel representation constants.
struct VoxelRepLayout
{
	uint3 Resolution;
	uint RastResolution;
	float3 VoxelWidth;
	float RastVoxelWidth;

	float3 Min;
	float _pad3;
	float3 Max;
	float _pad4;

	float3 Center;
	float _pad5;
	float3 Ext;
	float _pad6;
	float3 UnitScale;
	float _pad7;

	float3 VoxelSize;
	float _pad8;
	float3 VoxelScale;
	float _pad9;
};

#ifdef BE_VOXEL_REP_SETUP
	cbuffer VoxelRepConstants : register(b3)
#else
	cbuffer prebound(VoxelRepConstants) : register(b3)
#endif
{
	VoxelRepLayout VoxelRep;
}

/// Scene approximation.
RWTexture3D<uint> VoxelRepUAV : prebound_s(bindpoint_s(VoxelRepUAV, u0));

#endif