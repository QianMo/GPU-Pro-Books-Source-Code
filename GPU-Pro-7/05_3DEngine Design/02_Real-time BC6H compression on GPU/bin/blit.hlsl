Texture2D       SrcTextureA     : register( t0 );
SamplerState    PointSampler    : register( s0 );

struct PSInput
{
    float4 m_pos : SV_POSITION;
};

cbuffer MainCB : register( b0 )
{
    float2  ScreenSizeRcp;
    float2  TextureSizeRcp;
    float2  TexelBias;
    float   TexelScale;
    float   Exposure;
};

PSInput VSMain( uint vertexID : SV_VertexID )
{
    PSInput output;

    float x = vertexID >> 1;
    float y = vertexID & 1;

    output.m_pos = float4( 2.0f * x - 1.0f, 2.0f * y - 1.0f, 0.0f, 1.0f );

    return output;
}

float3 PSMain( PSInput i ) : SV_Target
{
    float2 uv = ( i.m_pos * TexelScale + TexelBias ) * TextureSizeRcp;

    float3 img = SrcTextureA.SampleLevel( PointSampler, uv, 0.0f ) * Exposure;
    return img;
}