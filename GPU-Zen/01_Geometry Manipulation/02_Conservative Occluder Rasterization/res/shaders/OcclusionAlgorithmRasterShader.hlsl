struct VSInput
{
    float3 position_os : POSITION0;
};

struct PSInput
{
    float4 position : SV_POSITION;
    nointerpolation uint instance_id : INSTANCEID0;
};

struct Aabb
{
    float3 center;
    float3 extent;
};

cbuffer OcclusionQueryConstantsBuffer : register( b0 )
{
    float4 g_FrustumPlanes[ 6 ];
    float4x4 g_ViewProjection;
    float4x4 g_ViewProjectionFlippedZ;
    float4 g_CameraPosition;
    uint g_Width;
    uint g_Height;
    float g_NearZ;
    float g_FarZ;
};
StructuredBuffer< Aabb > g_OccludeeAabbBuffer : register( t0 );
RWByteAddressBuffer g_VisibilityBuffer : register( u0 );

PSInput VSMain( VSInput input, uint instance_id : SV_InstanceID )
{
    Aabb aabb = g_OccludeeAabbBuffer[ instance_id ];

    float3 aabb_min = aabb.center - aabb.extent - g_NearZ;
    float3 aabb_max = aabb.center + aabb.extent + g_NearZ;
    if ( g_CameraPosition.x > aabb_min.x && g_CameraPosition.x < aabb_max.x &&
         g_CameraPosition.y > aabb_min.y && g_CameraPosition.y < aabb_max.y &&
         g_CameraPosition.z > aabb_min.z && g_CameraPosition.z < aabb_max.z )
    {
        g_VisibilityBuffer.InterlockedOr( ( instance_id / 32 ) * 4, 1 << ( instance_id % 32 ) );
    }

    PSInput output = ( PSInput )0;
    output.position = float4( input.position_os, 1.0 );
    output.position.xyz *= aabb.extent;
    output.position.xyz += aabb.center;
    output.position = mul( output.position, g_ViewProjectionFlippedZ );
    output.instance_id = instance_id;
    return output;
}

[ earlydepthstencil ]
void PSMain( PSInput input )
{
    g_VisibilityBuffer.InterlockedOr( ( input.instance_id / 32 ) * 4, 1 << ( input.instance_id % 32 ) );
}