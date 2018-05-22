#include "globals.shi"
#include "globalIllum.shi"

#ifdef FINE_GRID
	StructuredBuffer<VOXEL> gridBuffer: register(CUSTOM0_SB_BP);
#else
	StructuredBuffer<VOXEL> gridBuffer: register(CUSTOM1_SB_BP);
#endif

GLOBAL_POINT_LIGHT_UB(pointLightUB);

cbuffer CUSTOM_UB: register(CUSTOM_UB_BP)
{
	struct
	{
    matrix gridViewProjMatrices[6];
		float4 gridCellSizes; 
	  float4 gridPositions[2];
		float4 snappedGridPositions[2];
	}customUB;
};

struct GS_OUTPUT
{
  float4 position: SV_POSITION;
  uint rtIndex : SV_RenderTargetArrayIndex; 
};

struct FS_OUTPUT
{
  float4 fragColor0: SV_TARGET0;
	float4 fragColor1: SV_TARGET1;
	float4 fragColor2: SV_TARGET2;
};

FS_OUTPUT main(GS_OUTPUT input) 
{
	FS_OUTPUT output;

	// get index of current voxel
	int3 voxelPos = int3(input.position.xy,input.rtIndex);
	int gridIndex = GetGridIndex(voxelPos);

	// get voxel data and early out, if voxel has no geometry info 
	VOXEL voxel = gridBuffer[gridIndex];
  if(voxel.occlusion==0) 
    discard;

	// get world-space position of voxel
	int3 offset = voxelPos-int3(16,16,16);
#ifdef FINE_GRID
	float3 position = (float3(offset.x,offset.y,offset.z)*customUB.gridCellSizes.x)+customUB.snappedGridPositions[0].xyz;
#else
	float3 position = (float3(offset.x,offset.y,offset.z)*customUB.gridCellSizes.z)+customUB.snappedGridPositions[1].xyz;
#endif

	// early out, if voxel is outside of point-light radius
  float3 lightVec = pointLightUB.position-position;
  float lightVecLen = length(lightVec);
  if(lightVecLen > pointLightUB.radius)
    discard;

	float3 lightVecN = lightVec/lightVecLen;

	// decode color of voxel
	float3 albedo = DecodeColor(voxel.colorMask);

	// get normal of voxel that is closest to the light-direction
	float nDotL;
	float3 normal = GetClosestNormal(voxel.normalMasks,lightVecN,nDotL);

	// compute diffuse illumination
	float att = saturate(1.0f-(1.0f/pointLightUB.radius)*lightVecLen); 
	float4 vDiffuse = pointLightUB.color*max(nDotL,0.0f)*float4(albedo,1.0f)*att*pointLightUB.multiplier;

	// turn illuminated voxel into virtual point light, represented by the second order spherical harmonics coeffs
  float4 coeffs = ClampedCosineCoeffs(normal);
	float3 flux = vDiffuse.rgb;
	float4 redSHCoeffs = coeffs*flux.r;
	float4 greenSHCoeffs = coeffs*flux.g;
	float4 blueSHCoeffs = coeffs*flux.b;

	// output red/ green/ blue SH-coeffs 
	output.fragColor0 = redSHCoeffs;
	output.fragColor1 = greenSHCoeffs;
	output.fragColor2 = blueSHCoeffs;

  return output;
}
