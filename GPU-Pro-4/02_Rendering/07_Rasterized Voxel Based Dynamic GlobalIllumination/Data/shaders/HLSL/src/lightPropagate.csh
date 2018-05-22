#include "globals.shi"
#include "globalIllum.shi"

Texture2DArray customMap0: register(CUSTOM0_TEX_BP); // redSHCoeffs
Texture2DArray customMap1: register(CUSTOM1_TEX_BP); // greenSHCoeffs
Texture2DArray customMap2: register(CUSTOM2_TEX_BP); // blueSHCoeffs

#ifdef USE_OCCLUSION 
#ifdef FINE_GRID
	StructuredBuffer<VOXEL> gridBuffer: register(CUSTOM0_SB_BP);
#else
	StructuredBuffer<VOXEL> gridBuffer: register(CUSTOM1_SB_BP);
#endif
#endif

RWTexture2DArray<float4> redOutputTexture: register(u0); // redSHCoeffs
RWTexture2DArray<float4> greenOutputTexture: register(u1); // greenSHCoeffs
RWTexture2DArray<float4> blueOutputTexture: register(u2); // blueSHCoeffs

#define FLUX_AMPLIFIER 1.75f
#define OCCLUSION_AMPLIFIER 1.0f

// solid angles (normalized), subtended by the face onto the neighbor cell center
#define SOLID_ANGLE_A 0.0318842778f // (22.95668f/(4*180.0f)) 
#define SOLID_ANGLE_B 0.0336955972f // (24.26083f/(4*180.0f))

// directions to 6 neighbor cell centers
static float3 directions[6] = { float3(0.0f,0.0f,1.0f),
                                float3(1.0f,0.0f,0.0f),
	                              float3(0.0f,0.0f,-1.0f),
                                float3(-1.0f,0.0f,0.0f),
																float3(0.0f,1.0f,0.0f),
																float3(0.0f,-1.0f,0.0f) };

// SH-coeffs for six faces
static float4 faceCoeffs[6] = { float4(0.8862269521f,0.0f,1.0233267546f,0.0f),  // ClampedCosineCoeffs(directions[0])
																float4(0.8862269521f,0.0f,0.0f,-1.0233267546f), // ClampedCosineCoeffs(directions[1])
																float4(0.8862269521f,0.0f,-1.0233267546f,0.0f), // ClampedCosineCoeffs(directions[2])
																float4(0.8862269521f,0.0f,0.0f,1.0233267546f),  // ClampedCosineCoeffs(directions[3])
																float4(0.8862269521f,-1.0233267546f,0.0f,0.0f), // ClampedCosineCoeffs(directions[4])
																float4(0.8862269521f,1.0233267546,0.0f,0.0f) }; // ClampedCosineCoeffs(directions[5])
  
// offsets to 6 neighbor cell centers  
static int3 offsets[6] = { int3(0,0,1),
                           int3(1,0,0),
                           int3(0,0,-1),
                           int3(-1,0,0),
													 int3(0,1,0),
													 int3(0,-1,0) };

[numthreads(8, 8, 8)]
void main(uint3 GroupID : SV_GroupID, uint3 DispatchThreadID : SV_DispatchThreadID,
		      uint3 GroupThreadID : SV_GroupThreadID, uint GroupIndex : SV_GroupIndex)
{ 
	// get grid-position of current cell  
  int3 elementPos = DispatchThreadID.xyz;
   
	// initialize SH-coeffs with values from current cell 
	float4 sumRedSHCoeffs = customMap0.Load(int4(elementPos,0));
  float4 sumGreenSHCoeffs = customMap1.Load(int4(elementPos,0));
	float4 sumBlueSHCoeffs = customMap2.Load(int4(elementPos,0));

	[unroll]
  for(int i=0;i<6;i++)
  {
		// get grid-position of 6 neighbor cells
		int3 samplePos = elementPos+offsets[i];

		// continue, if cell out of bounds
    if((samplePos.x<0)||(samplePos.x>31)||(samplePos.y<0)||(samplePos.y>31)||(samplePos.z<0)||(samplePos.z>31))
      continue;

		// load SH-coeffs for neighbor cell
    float4 redSHCoeffs = customMap0.Load(int4(samplePos,0));
    float4 greenSHCoeffs = customMap1.Load(int4(samplePos,0));
	  float4 blueSHCoeffs = customMap2.Load(int4(samplePos,0));

#ifdef USE_OCCLUSION
		float4 occlusionCoeffs = float4(0.0f,0.0f,0.0f,0.0f);

		// get index of corresponding voxel
		int gridIndex = GetGridIndex(samplePos);
		VOXEL voxel = gridBuffer[gridIndex];

		// If voxel contains geometry info, find closest normal to current direction. In this way the highest
		// occlusion can be generated. Then get SH-coeffs for retrieved normal.
		if(voxel.occlusion > 0)
		{	
			float dotProduct;
			float3 occlusionNormal = GetClosestNormal(voxel.normalMasks,-directions[i],dotProduct);
			occlusionCoeffs = ClampedCosineCoeffs(occlusionNormal);
		} 
#endif

		[unroll]
    for(int j=0;j<6;j++)
    {
			// get direction for face of current cell to current neighbor cell center
			float3 neighborCellCenter = directions[i];
			float3 facePosition = directions[j]*0.5f;
			float3 dir = facePosition-neighborCellCenter;
			float fLength = length(dir);
			dir /= fLength;

			// get corresponding solid angle
      float solidAngle = 0.0f;
			if(fLength>0.5f)
				solidAngle = (fLength>=1.5f) ? SOLID_ANGLE_A : SOLID_ANGLE_B;

			// get SH-coeffs for direction 
      float4 dirSH = SH(dir);  

			// calculate flux from neigbor cell to face of current cell 
			float3 flux;
      flux.r = dot(redSHCoeffs,dirSH);
      flux.g = dot(greenSHCoeffs,dirSH);
			flux.b = dot(blueSHCoeffs,dirSH);
			flux = max(0.0f,flux)*solidAngle*FLUX_AMPLIFIER;

#ifdef USE_OCCLUSION 
			// apply occlusion
      float occlusion = 1.0f-saturate(OCCLUSION_AMPLIFIER*dot(occlusionCoeffs,dirSH));
			flux *= occlusion;
#endif

			// add contribution to SH-coeffs sums
      float4 coeffs = faceCoeffs[j];
      sumRedSHCoeffs += coeffs*flux.r;
		  sumGreenSHCoeffs += coeffs*flux.g;
      sumBlueSHCoeffs += coeffs*flux.b;
		}
	}

	// write out generated red/ green/ blue SH-coeffs
	redOutputTexture[elementPos] = sumRedSHCoeffs;
	greenOutputTexture[elementPos] = sumGreenSHCoeffs;
	blueOutputTexture[elementPos] = sumBlueSHCoeffs;
}
