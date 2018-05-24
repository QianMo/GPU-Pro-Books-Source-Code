struct VSInput
{
    float2 position : POSITION0;
    float2 uv       : UV0;
};

struct PSInput
{
    float4 position       : SV_POSITION;
    float2 uv             : UV0;
};

Texture2D<float> g_DepthTexture : register( t0 );
SamplerState g_DepthSampler : register( s0 );

PSInput VSMain( VSInput input )
{
    PSInput output = ( PSInput )0;
    output.position = float4( input.position, 0.0f, 1.0f );
    output.uv = input.uv;
    return output;
}

float4 PSMain( PSInput input ) : SV_Target
{
    float depth_sample = g_DepthTexture.Sample( g_DepthSampler, input.uv );

    float2 depth_range = float2( 0.0, 0.05 );
    depth_sample = depth_sample / ( depth_range.y - depth_range.x ) + depth_range.x;

    return float4( depth_sample, depth_sample, depth_sample, 1.0 );
}