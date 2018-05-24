struct VSInput
{
    float2 position : POSITION0;
    float2 uv       : UV0;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float2 uv       : UV0;
};

cbuffer UpsampleConstants : register( b0 )
{
    float g_HalfInputTexelSizeX;
    float g_HalfInputTexelSizeY;
};

Texture2D< float > g_DepthTexture : register( t0 );
SamplerState g_DepthSampler : register( s0 );

PSInput VSMain( VSInput input )
{
    PSInput output = ( PSInput )0;
    output.position = float4( input.position, 0.0, 1.0 );
    output.uv = input.uv;
    return output;
}

float PSMain( PSInput input ) : SV_Target
{
    float min_depth = 1.0;

    float2 offsets[ 4 ] =
    {
        float2( -g_HalfInputTexelSizeX, -g_HalfInputTexelSizeY ),
        float2(  g_HalfInputTexelSizeX, -g_HalfInputTexelSizeY ),
        float2( -g_HalfInputTexelSizeX,  g_HalfInputTexelSizeY ),
        float2(  g_HalfInputTexelSizeX,  g_HalfInputTexelSizeY )
    };
    
    [ unroll ]
    for ( uint i = 0; i < 4; ++i )
    {
        float2 uv = input.uv + offsets[ i ];
        min_depth = min( min_depth, g_DepthTexture.Sample( g_DepthSampler, uv ) );
    }

    return min_depth;
}