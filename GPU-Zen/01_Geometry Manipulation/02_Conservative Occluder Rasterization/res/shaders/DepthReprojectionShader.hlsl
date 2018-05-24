cbuffer DepthGeneratorConstants : register( b0 )
{
    float4x4 g_ViewProjectionFlippedZ;
    float4x4 g_PreviousToCurrentViewProjectionFlippedZ;
    float4 g_CameraPosition;
    uint g_DepthWidth;
    uint g_DepthHeight;
    uint g_FullscreenWidth;
    uint g_FullscreenHeight;
    float g_NearZ;
    float g_FarZ;
    uint g_SilhouetteEdgeBufferOffset;
};

Texture2D< float > g_PreviousDepth : register( t0 );
RWTexture2D< uint > g_DepthReprojection : register( u0 );

[ numthreads( BLOCK_SIZE_X, BLOCK_SIZE_Y, 1 ) ]
void CSMain( uint3 dispatch_thread_id : SV_DispatchThreadID )
{
    if ( dispatch_thread_id.x >= 0 && dispatch_thread_id.x < g_DepthWidth &&
         dispatch_thread_id.y >= 0 && dispatch_thread_id.y < g_DepthHeight )
    {
        uint2 previous_index = dispatch_thread_id.xy;
        float2 previous_uv = ( float2( previous_index ) + float2( 0.5, 0.5 ) ) / float2( g_DepthWidth, g_DepthHeight );
        float previous_depth = g_PreviousDepth[ previous_index ];
        float4 previous_position_cs = float4( previous_uv * float2( 2.0, -2.0 ) - float2( 1.0, -1.0 ), previous_depth, 1.0 );
        float4 current_position_cs = mul( previous_position_cs, g_PreviousToCurrentViewProjectionFlippedZ );
        float current_depth = saturate( current_position_cs.z / current_position_cs.w );
        float2 current_uv = ( current_position_cs.xy / current_position_cs.w ) * float2( 0.5, -0.5 ) + 0.5;
        uint2 current_index = uint2( current_uv * float2( g_DepthWidth, g_DepthHeight ) );

        int2 offsets[] =
        {
            int2( -1, -1 ),
            int2(  0, -1 ),
            int2(  1, -1 ),
            int2( -1,  0 ),
            int2(  0,  0 ),
            int2(  1,  0 ),
            int2( -1,  1 ),
            int2(  0,  1 ),
            int2(  1,  1 )
        };
        [ unroll ]
        for ( uint i = 0; i < 9; ++i )
        {
            uint2 reprojected_depth_index = current_index + offsets[ i ];
            if ( reprojected_depth_index.x >= 0 && reprojected_depth_index.x < g_DepthWidth &&
                 reprojected_depth_index.y >= 0 && reprojected_depth_index.y < g_DepthHeight )
            {
                InterlockedMin( g_DepthReprojection[ reprojected_depth_index ], asuint( current_depth ) );
            }
        }
    }
}