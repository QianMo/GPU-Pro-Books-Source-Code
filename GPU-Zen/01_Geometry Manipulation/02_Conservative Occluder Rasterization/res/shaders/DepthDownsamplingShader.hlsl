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

cbuffer DownsampleConstants : register( b0 )
{
    bool g_IsInputSizeEven;
    float g_InputTexelSizeX;
    float g_InputTexelSizeY;
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

    if ( g_IsInputSizeEven )
    {
        float4 texels = g_DepthTexture.Gather( g_DepthSampler, input.uv );

        min_depth = min( min( texels.x, texels.y ), min( texels.z, texels.w ) );
    }
    else
    {
        float2 offsets[ 9 ] =
        {
            float2( -g_InputTexelSizeX, -g_InputTexelSizeY ),
            float2(                0.0, -g_InputTexelSizeY ),
            float2(  g_InputTexelSizeX, -g_InputTexelSizeY ),
            float2( -g_InputTexelSizeX,                0.0 ),
            float2(                0.0,                0.0 ),
            float2(  g_InputTexelSizeX,                0.0 ),
            float2( -g_InputTexelSizeX,  g_InputTexelSizeY ),
            float2(                0.0,  g_InputTexelSizeY ),
            float2(  g_InputTexelSizeX,  g_InputTexelSizeY )
        };
        
        [ unroll ]
        for ( uint i = 0; i < 9; ++i )
        {
            float2 uv = input.uv + offsets[ i ];
            min_depth = min( min_depth, g_DepthTexture.Sample( g_DepthSampler, uv ) );
        }
    }

    return min_depth;
}