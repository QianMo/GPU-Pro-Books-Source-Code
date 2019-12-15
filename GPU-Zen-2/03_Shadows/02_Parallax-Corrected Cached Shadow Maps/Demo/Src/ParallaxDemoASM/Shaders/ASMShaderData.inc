#define gs_ASMMaxRefinement 4
#define gs_ASMTileBorderTexels 8
#define gs_ASMTileSize 256
#define gs_ASMDEMDownsampleLevel 1
#define gs_ASMDEMTileSize ( gs_ASMTileSize >> gs_ASMDEMDownsampleLevel )
#define gs_ASMBorderlessTileSize ( gs_ASMTileSize - 2 * gs_ASMTileBorderTexels )

#define gs_ASMDepthAtlasTextureWidth 4096
#define gs_ASMDepthAtlasTextureHeight 4096
#define gs_ASMDEMAtlasTextureWidth ( gs_ASMDepthAtlasTextureWidth >> gs_ASMDEMDownsampleLevel )
#define gs_ASMDEMAtlasTextureHeight ( gs_ASMDepthAtlasTextureHeight >> gs_ASMDEMDownsampleLevel )

#define gs_ASMTileFarPlane 2000
#define gs_ASMDistanceMax 500
#define gs_ASMLargestTileWorldSize 120

cbuffer ASM_ResolveShaderData : register(b3)
{
    float4x4 LongRangeIndexTexMat;
    float4x4 LongRangePreRenderIndexTexMat;
    float3 LongRangeReceiverWarpVector;
    float LongRangeDefaultShadowFactor;
    float3 LongRangePreRenderReceiverWarpVector;
    float LongRangePreRenderDefaultShadowFactor;
    float3 LongRangeBlockerSearchVector;
    float __Unused0;
    float3 LongRangeLightDir;
    float __Unused1;
    float3 LongRangePreRenderLightDir;
    float __Unused2;
    float3 LongRangePreRenderBlockerSearchVector;
    float __Unused3;
};

