#include "globals.shi"
#include "globalIllum.shi"

Texture2D colorMap: register(COLOR_TEX_BP); // shadowMap
SamplerComparisonState colorMapSampler: register(COLOR_SAM_BP);

#ifdef FINE_GRID
	StructuredBuffer<VOXEL> gridBuffer: register(CUSTOM0_SB_BP);
#else
	StructuredBuffer<VOXEL> gridBuffer: register(CUSTOM1_SB_BP);
#endif

GLOBAL_DIR_LIGHT_UB(dirLightUB);

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

#define PCF_NUM_SAMPLES 16
#define SHADOW_FILTER_RADIUS 2.0f
#define SHADOW_BIAS 0.02f

// poisson disk samples
static float2 filterKernel[PCF_NUM_SAMPLES] = { float2(-0.94201624f,-0.39906216f),
																							  float2(0.94558609f,-0.76890725f),
																							  float2(-0.094184101f,-0.92938870f),
																							  float2(0.34495938f,0.29387760f),
																							  float2(-0.91588581f,0.45771432f),
																							  float2(-0.81544232f,-0.87912464f),
																							  float2(-0.38277543f,0.27676845f),
																							  float2(0.97484398f,0.75648379f),
																							  float2(0.44323325f,-0.97511554f),
																							  float2(0.53742981f,-0.47373420f),
																							  float2(-0.26496911f,-0.41893023f),
																							  float2(0.79197514f,0.19090188f),
																							  float2(-0.24188840f,0.99706507f),
																							  float2(-0.81409955f,0.91437590f),
																							  float2(0.19984126f,0.78641367f),
																							  float2(0.14383161f,-0.14100790f) };

// compute shadow-term using 16x PCF in combination with hardware shadow filtering
float ComputeShadowTerm(in float4 positionWS)
{
  float4 result = mul(dirLightUB.shadowViewProjTexMatrix,positionWS);
	result.xyz /= result.w;
	float filterRadius = SHADOW_FILTER_RADIUS*dirLightUB.invShadowMapSize;
  float shadowTerm = 0.0f;
	[unroll]
  for(int i=0;i<PCF_NUM_SAMPLES;i++)
  {
    float2 offset = filterKernel[i]*filterRadius;
		float2 texCoords = result.xy+offset;
		texCoords.y = 1.0f-texCoords.y;
	  shadowTerm += colorMap.SampleCmp(colorMapSampler,texCoords,result.z-SHADOW_BIAS);
  }
  shadowTerm /= PCF_NUM_SAMPLES;
	return shadowTerm;
}

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
 
  float3 lightVecN = -dirLightUB.direction;

	// decode color of voxel
	float3 albedo = DecodeColor(voxel.colorMask);

	// get normal of voxel that is closest to the light-direction
	float nDotL;
	float3 normal = GetClosestNormal(voxel.normalMasks,lightVecN,nDotL);

	// compute shadowTerm by re-using shadowMap from direct illumination
	float shadowTerm = ComputeShadowTerm(float4(position,1.0f));

	// compute diffuse illumination
	float4 vDiffuse = dirLightUB.color*max(nDotL,0.0f)*float4(albedo,1.0f)*dirLightUB.multiplier*shadowTerm;

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
