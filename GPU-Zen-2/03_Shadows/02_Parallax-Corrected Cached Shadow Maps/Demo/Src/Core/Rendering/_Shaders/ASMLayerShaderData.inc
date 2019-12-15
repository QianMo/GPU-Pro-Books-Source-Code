cbuffer ASMLayerShaderData : register(b1)
{
    float4x4 g_ASMIndirectionTexMat;
    float4 g_ASMTileData;
    float2 g_ASMOneOverDepthAtlasSize;
};

#ifndef __cplusplus

float3 GetASMTileCoord( float3 indexCoord, float4 indirectionTextureData, float2 oneOverDepthAtlasSize )
{
    indexCoord.xy = floor( abs( indirectionTextureData.w ) ) * oneOverDepthAtlasSize * indexCoord.xy + indirectionTextureData.xy;
    indexCoord.z = indexCoord.z + indirectionTextureData.z;
    return indexCoord;
}

#endif
