#define QUADS_ARRAY_REGS 192

#ifndef __cplusplus

cbuffer AtlasQuads_ConstantBuffer : register(b3)
{
  float4 QuadsData[ QUADS_ARRAY_REGS ];
};

struct AtlasQuads_VS2PS
{
    float4 position : SV_Position;
#if ATLAS_QUADS_TEXCOORD
    float2 uv       : TEXCOORD0;
#endif
#if ATLAS_QUADS_MISC
    float4 miscData : TEXCOORD1;
#endif
};

struct AtlasQuads_QuadConstants
{
    float4 posData;
#if ATLAS_QUADS_MISC
    float4 miscData;
#endif
#if ATLAS_QUADS_TEXCOORD
    float4 texCoordData;
#endif
};

float2 ScaleOffset( float2 a, float4 p )
{
    return a * p.xy + p.zw;
}

AtlasQuads_VS2PS AtlasQuadsVS( uint vertexID : SV_VertexID )
{
    const uint verticesPerQuad = 6;
    uint quadID = vertexID / verticesPerQuad;
    uint quadVertexID = vertexID - quadID * verticesPerQuad;

    float2 pos = float2(-1.0,-1.0 );
    if( quadVertexID == 1 ) pos = float2(-1.0, 1.0 );
    if( quadVertexID == 2 ) pos = float2( 1.0,-1.0 );
    if( quadVertexID == 3 ) pos = float2( 1.0,-1.0 );
    if( quadVertexID == 4 ) pos = float2(-1.0, 1.0 );
    if( quadVertexID == 5 ) pos = float2( 1.0, 1.0 );

    const uint registersPerQuad = 1 + ATLAS_QUADS_TEXCOORD + ATLAS_QUADS_MISC;
    float4 quadData[ registersPerQuad ];
    [unroll] for( uint i = 0; i < registersPerQuad; ++i )
       quadData[ i ] = QuadsData[ quadID * registersPerQuad + i ];
    AtlasQuads_QuadConstants q = (AtlasQuads_QuadConstants)quadData;

    AtlasQuads_VS2PS output = (AtlasQuads_VS2PS)0;
    output.position = float4( ScaleOffset( pos, q.posData ), 0, 1 );
#if ATLAS_QUADS_MISC
    output.miscData = q.miscData;
#endif
#if ATLAS_QUADS_TEXCOORD
    output.uv = ScaleOffset( float2( 0.5,-0.5 ) * pos + 0.5, q.texCoordData );
#endif

    return output;
}

#endif
