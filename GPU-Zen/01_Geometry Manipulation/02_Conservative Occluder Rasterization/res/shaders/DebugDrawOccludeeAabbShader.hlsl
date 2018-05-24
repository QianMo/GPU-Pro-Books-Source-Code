struct VSInput
{
    float3 position_os : POSITION0;
};

struct PSInput
{
    float4 position : SV_POSITION;
    uint instance_id : INSTANCEID0;
};

struct Aabb
{
    float3 center;
    float3 extent;
};

cbuffer OccludeeCollectionConstants : register( b0 )
{
    float4x4 g_ViewProjection;
};

StructuredBuffer< Aabb > g_OccludeeAabbBuffer : register( t0 );
ByteAddressBuffer g_VisibilityBuffer : register( t1 );

PSInput VSMain( VSInput input, uint instance_id : SV_InstanceID )
{
    Aabb aabb = g_OccludeeAabbBuffer[ instance_id ];

    PSInput output = ( PSInput )0;
    output.position = float4( input.position_os, 1.0 );
    output.position.xyz *= aabb.extent;
    output.position.xyz += aabb.center;
    output.position = mul( output.position, g_ViewProjection );
    output.instance_id = instance_id;
    return output;
}

float4 PSMain( PSInput input ) : SV_Target
{
    if ( ( g_VisibilityBuffer.Load( ( input.instance_id / 32 ) * 4 ) >> ( input.instance_id % 32 ) ) & 1 )
    {
        return float4( 0.0, 0.0, 1.0, 1.0 );
    }
    return float4( 1.0, 0.0, 0.0, 1.0 );
}