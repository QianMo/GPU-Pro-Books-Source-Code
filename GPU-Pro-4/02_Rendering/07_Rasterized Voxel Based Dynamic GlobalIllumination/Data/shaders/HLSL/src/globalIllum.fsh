#include "globals.shi"
#include "globalIllum.shi"

#ifndef NO_TEXTURE
Texture2D colorMap: register(COLOR_TEX_BP); // albedoGloss
SamplerState colorMapSampler: register(COLOR_SAM_BP);
#endif
Texture2D normalMap: register(NORMAL_TEX_BP); // normalDepth
SamplerState normalMapSampler: register(NORMAL_SAM_BP);

// FINE_GRID
Texture2DArray customMap0: register(CUSTOM0_TEX_BP); // redSHCoeffs 
SamplerState customMap0Sampler: register(CUSTOM0_SAM_BP);
Texture2DArray customMap1: register(CUSTOM1_TEX_BP); // greenSHCoeffs 
SamplerState customMap1Sampler: register(CUSTOM1_SAM_BP);
Texture2DArray customMap2: register(CUSTOM2_TEX_BP); // blueSHCoeffs
SamplerState customMap2Sampler: register(CUSTOM2_SAM_BP);

// COARSE_GRID
Texture2DArray customMap3: register(CUSTOM3_TEX_BP); // redSHCoeffs 
SamplerState customMap3Sampler: register(CUSTOM3_SAM_BP);
Texture2DArray customMap4: register(CUSTOM4_TEX_BP); // greenSHCoeffs 
SamplerState customMap4Sampler: register(CUSTOM4_SAM_BP);
Texture2DArray customMap5: register(CUSTOM5_TEX_BP); // blueSHCoeffs
SamplerState customMap5Sampler: register(CUSTOM5_SAM_BP);

GLOBAL_CAMERA_UB(cameraUB);

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
  float2 texCoords: TEXCOORD;
	float3 frustumRay: FRUSTUM_RAY;
};

struct FS_OUTPUT
{
  float4 fragColor: SV_TARGET;
};

// reconstruct world-space position from depth
float4 DecodePosition(in float depth,in float3 frustumRay)
{
  float4 position;
	float3 frustumRayN = normalize(frustumRay);
	position.xyz = cameraUB.position+(frustumRayN*depth);
	position.w = 1.0f;
	return position;
}

// After calculating the texCoords into the 2D texture arrays, the SH-coeffs are trilinearly sampled and
// finally a SH-lighting is done to generate the diffuse global illumination.
float3 GetDiffuseIllum(in float3 offset,in float4 surfaceNormalLobe,in Texture2DArray redSHCoeffsMap,
											 in Texture2DArray greenSHCoeffsMap,in Texture2DArray blueSHCoeffsMap)
{
	// get texCoords into 2D texture arrays
  float3 texCoords = float3(16.5f,16.5f,16.0f)+offset;
	texCoords.xy /= 32.0f;

	// Since hardware already does the filtering in each 2D texture slice, manually only the filtering into
	// the third dimension has to be done.
  int lowZ = floor(texCoords.z);
  int highZ = min(lowZ+1,32-1);
  float highZWeight = texCoords.z-lowZ;
  float lowZWeight = 1.0f-highZWeight;
  float3 texCoordsLow = float3(texCoords.x,texCoords.y,lowZ);
  float3 texCoordsHigh = float3(texCoords.x,texCoords.y,highZ);

	// sample red/ green/ blue SH-coeffs trilinearly from the 2D texture arrays
  float4 redSHCoeffs = lowZWeight*redSHCoeffsMap.Sample(customMap0Sampler,texCoordsLow) + highZWeight*redSHCoeffsMap.Sample(customMap0Sampler,texCoordsHigh);
  float4 greenSHCoeffs = lowZWeight*greenSHCoeffsMap.Sample(customMap0Sampler,texCoordsLow) + highZWeight*greenSHCoeffsMap.Sample(customMap0Sampler,texCoordsHigh);
  float4 blueSHCoeffs = lowZWeight*blueSHCoeffsMap.Sample(customMap0Sampler,texCoordsLow) + highZWeight*blueSHCoeffsMap.Sample(customMap0Sampler,texCoordsHigh);

	// Do diffuse SH-lighting by simply calculating the dot-product between the SH-coeffs from the virtual 
	// point lights and the surface SH-coeffs.
	float3 vDiffuse;
	vDiffuse.r = dot(redSHCoeffs,surfaceNormalLobe);
	vDiffuse.g = dot(greenSHCoeffs,surfaceNormalLobe);
	vDiffuse.b = dot(blueSHCoeffs,surfaceNormalLobe);

  return vDiffuse;
}

FS_OUTPUT main(GS_OUTPUT input) 
{
  FS_OUTPUT output;
	float4 bumpDepth = normalMap.Sample(normalMapSampler,input.texCoords); 
	float4 position = DecodePosition(bumpDepth.w,input.frustumRay);

#ifndef NO_TEXTURE
	float3 albedo = colorMap.Sample(colorMapSampler,input.texCoords).rgb;
#endif
	 
	// get surface SH-coeffs
	float3 normal = normalize(bumpDepth.xyz); 
  float4 surfaceNormalLobe = ClampedCosineCoeffs(normal);

	// get offset into fine resolution grid
	float3 offset = (position.xyz-customUB.snappedGridPositions[0].xyz)*customUB.gridCellSizes.y;

	// The distance for lerping between fine and coarse resolution grid has to be calculated with
	// the unsnapped grid-center, in order to avoid artefacts in the lerp area.
	float3 lerpOffset = (position.xyz-customUB.gridPositions[0].xyz)*customUB.gridCellSizes.y;
	float lerpDist = length(lerpOffset);

	// get diffuse global illumination from fine resolution grid
	float3 fineDiffuseIllum = GetDiffuseIllum(offset,surfaceNormalLobe,customMap0,customMap1,customMap2);
	
	// get offset into coarse resolution grid
	offset = (position.xyz-customUB.snappedGridPositions[1].xyz)*customUB.gridCellSizes.w;	

	// get diffuse global illumination from coarse resolution grid
	float3 coarseDiffuseIllum = GetDiffuseIllum(offset,surfaceNormalLobe,customMap3,customMap4,customMap5);
	
	// lerp between results from both grids
	float factor = saturate((lerpDist-12.0f)*0.25f);
	float3 diffuseIllum = lerp(fineDiffuseIllum,coarseDiffuseIllum,factor);

	diffuseIllum = max(diffuseIllum,float3(0.0f,0.0f,0.0f));
	diffuseIllum /= PI;

#ifndef NO_TEXTURE	
	float3 outputColor = diffuseIllum*albedo;
#else
  float3 outputColor = diffuseIllum;
#endif
	
	output.fragColor = float4(outputColor,1.0f);

	return output;
}