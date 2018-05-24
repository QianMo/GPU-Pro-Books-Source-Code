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
StructuredBuffer< Aabb > g_AabbBuffer : register( t0 );
Texture2D< float > g_DepthHierarchyEven : register( t1 );
Texture2D< float > g_DepthHierarchyOdd : register( t2 );
RWByteAddressBuffer g_VisibilityBuffer : register( u0 );
SamplerState g_DepthHierarchySampler : register( s0 );

bool IsInsideFrustum( Aabb aabb )
{
    [ unroll ]
    for ( int i = 0; i < 6; ++i )
    {
        float4 plane = g_FrustumPlanes[ i ];

        float d =
            aabb.center.x * plane.x +
            aabb.center.y * plane.y +
            aabb.center.z * plane.z;

        float r =
            aabb.extent.x * abs( plane.x ) +
            aabb.extent.y * abs( plane.y ) +
            aabb.extent.z * abs( plane.z );

        if ( d + r < -plane.w )
        {
            return false;
        }
    }

    return true;
}

[ numthreads( BLOCK_SIZE_X, 1, 1 ) ]
void CSMain( uint3 dispatch_thread_id : SV_DispatchThreadId )
{
    Aabb aabb = g_AabbBuffer[ dispatch_thread_id.x ];
    if ( IsInsideFrustum( aabb ) )
    {
        float3 aabb_min = aabb.center - aabb.extent;
        float3 aabb_max = aabb.center + aabb.extent;

        float4 positions_ws[ 8 ];
        positions_ws[ 0 ] = float4( aabb_min.x, aabb_min.y, aabb_min.z, 1.0 );
        positions_ws[ 1 ] = float4( aabb_max.x, aabb_min.y, aabb_min.z, 1.0 );
        positions_ws[ 2 ] = float4( aabb_max.x, aabb_max.y, aabb_min.z, 1.0 );
        positions_ws[ 3 ] = float4( aabb_min.x, aabb_max.y, aabb_min.z, 1.0 );
        positions_ws[ 4 ] = float4( aabb_min.x, aabb_min.y, aabb_max.z, 1.0 );
        positions_ws[ 5 ] = float4( aabb_max.x, aabb_min.y, aabb_max.z, 1.0 );
        positions_ws[ 6 ] = float4( aabb_max.x, aabb_max.y, aabb_max.z, 1.0 );
        positions_ws[ 7 ] = float4( aabb_min.x, aabb_max.y, aabb_max.z, 1.0 );

        float3 aabb_min_ndc = float3( 1.0, 1.0, 1.0 );
        float3 aabb_max_ndc = float3( -1.0, -1.0, -1.0 );

        for ( uint i = 0; i < 8; ++i )
        {
            float4 position_cs = mul( positions_ws[ i ], g_ViewProjectionFlippedZ );
            float3 position_ndc = position_cs.xyz / position_cs.w;

            aabb_min_ndc = min( aabb_min_ndc, position_ndc );
            aabb_max_ndc = max( aabb_max_ndc, position_ndc );
        }

        aabb_min_ndc.xy = saturate( float2( 0.5, -0.5 ) * aabb_min_ndc.xy + float2( 0.5, 0.5 ) );
        aabb_max_ndc.xy = saturate( float2( 0.5, -0.5 ) * aabb_max_ndc.xy + float2( 0.5, 0.5 ) );

        float w = max( ( aabb_max_ndc.x - aabb_min_ndc.x ) * g_Width, ( aabb_max_ndc.y - aabb_min_ndc.y ) * g_Height );
        float lod = ceil( log2( w ) );

        float4 samples;
        if ( uint( lod ) % 2 == 0 )
        {
            samples.x = g_DepthHierarchyEven.SampleLevel( g_DepthHierarchySampler, float2( aabb_min_ndc.x, aabb_max_ndc.y ), lod );
            samples.y = g_DepthHierarchyEven.SampleLevel( g_DepthHierarchySampler, float2( aabb_max_ndc.x, aabb_max_ndc.y ), lod );
            samples.z = g_DepthHierarchyEven.SampleLevel( g_DepthHierarchySampler, float2( aabb_min_ndc.x, aabb_min_ndc.y ), lod );
            samples.w = g_DepthHierarchyEven.SampleLevel( g_DepthHierarchySampler, float2( aabb_max_ndc.x, aabb_min_ndc.y ), lod );
        }
        else
        {
            samples.x = g_DepthHierarchyOdd.SampleLevel( g_DepthHierarchySampler, float2( aabb_min_ndc.x, aabb_max_ndc.y ), lod );
            samples.y = g_DepthHierarchyOdd.SampleLevel( g_DepthHierarchySampler, float2( aabb_max_ndc.x, aabb_max_ndc.y ), lod );
            samples.z = g_DepthHierarchyOdd.SampleLevel( g_DepthHierarchySampler, float2( aabb_min_ndc.x, aabb_min_ndc.y ), lod );
            samples.w = g_DepthHierarchyOdd.SampleLevel( g_DepthHierarchySampler, float2( aabb_max_ndc.x, aabb_min_ndc.y ), lod );
        }

        float min_depth = min( min( samples.x, samples.y ), min( samples.z, samples.w ) );

        if ( aabb_max_ndc.z >= min_depth || aabb_min_ndc.z < 0.0 )
        {
            g_VisibilityBuffer.InterlockedOr( ( dispatch_thread_id.x / 32 ) * 4, 1 << ( dispatch_thread_id.x % 32 ) );
        }
    }
}
