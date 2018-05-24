struct VSInput
{
    float2 position : POSITION0;
};

struct PSInput
{
    float4 position : SV_POSITION;
};

Texture2D< float > g_RasterizedDepth : register( t0 );
Texture2D< float > g_ReprojectedDepth : register( t1 );

PSInput VSMain( VSInput input )
{
    PSInput output = ( PSInput )0;
    output.position = float4( input.position, 0.0, 1.0 );
    return output;
}

float PSMain( PSInput input ) : SV_Target
{
    uint2 index = uint2( input.position.xy );

    float rasterized_depth = g_RasterizedDepth[ index ];
        
    float reprojected_depth = g_ReprojectedDepth[ index ];
    reprojected_depth = reprojected_depth == 0xffffffff ? 0.0 : reprojected_depth;

    return max( rasterized_depth, reprojected_depth );
}