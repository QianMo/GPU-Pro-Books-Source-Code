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

cbuffer ModelConstants : register( b1 )
{
    uint g_InstanceOffset;
    uint g_FaceCount;
};

StructuredBuffer< float4x4 > g_WorldMatrixBuffer : register( t0 );
StructuredBuffer< float3 > g_VertexBuffer : register( t1 );
StructuredBuffer< uint > g_IndexBufferAdj : register( t2 );
RWStructuredBuffer< float2 > g_SilhouetteEdgeBuffer : register( u0 );
RWByteAddressBuffer g_SilhouetteEdgeCountBuffer : register( u1 );

bool IsTriangleFrontFacingWS( float3 t_0, float3 t_1, float3 t_2, float3 eye )
{
    float3 center = ( t_0 + t_1 + t_2 ) / 3.0;
    float3 view_direction = normalize( center - eye );
    float3 normal = normalize( cross( t_1 - t_0, t_2 - t_0 ) );
    return dot( view_direction, normal ) >= 0.0;
}

uint IsOutsideFrustumCS( float4 p )
{
    uint is_outside_frustum = 0;
    is_outside_frustum |= ( ( p.x < -p.w ? 1 : 0 ) << 0 );
    is_outside_frustum |= ( ( p.x >  p.w ? 1 : 0 ) << 1 );
    is_outside_frustum |= ( ( p.y < -p.w ? 1 : 0 ) << 2 );
    is_outside_frustum |= ( ( p.y >  p.w ? 1 : 0 ) << 3 );
    is_outside_frustum |= ( ( p.z < -p.w ? 1 : 0 ) << 4 );
    is_outside_frustum |= ( ( p.z >  p.w ? 1 : 0 ) << 5 );
    return is_outside_frustum;
}

float CalculatePlaneIntersectionCS( uint plane_index, float4 p_0, float4 p_1 )
{
    switch ( plane_index )
    {
        case 0: return ( -p_0.w - p_0.x ) / ( p_1.x - p_0.x + p_1.w - p_0.w );
        case 1: return ( p_0.w - p_0.x ) / ( p_1.x - p_0.x - p_1.w + p_0.w );
        case 2: return ( -p_0.w - p_0.y ) / ( p_1.y - p_0.y + p_1.w - p_0.w );
        case 3: return ( p_0.w - p_0.y ) / ( p_1.y - p_0.y - p_1.w + p_0.w );
        case 4: return ( -p_0.w - p_0.z ) / ( p_1.z - p_0.z + p_1.w - p_0.w );
        case 5: return ( p_0.w - p_0.z ) / ( p_1.z - p_0.z - p_1.w + p_0.w );
    }
    return 0.0;
}

bool IsLineSegmentInsideFrustumCS( float4 p_0, float4 p_1 )
{
    uint is_outside_frustum_0 = IsOutsideFrustumCS( p_0 );
    uint is_outside_frustum_1 = IsOutsideFrustumCS( p_1 );

    if ( is_outside_frustum_0 == 0 || is_outside_frustum_1 == 0 )
    {
        return true;
    }
    else if ( ( is_outside_frustum_0 & is_outside_frustum_1 ) != 0 )
    {
        return false;
    }

    float t_0 = 0.0;
    float t_1 = 1.0;

    [ unroll ]
    for ( uint i = 0; i < 6; ++i )
    {
        if ( ( is_outside_frustum_0 >> i ) & 1 )
        {
            t_0 = max( t_0, CalculatePlaneIntersectionCS( i, p_0, p_1 ) );
        }
        else if ( ( is_outside_frustum_1 >> i ) & 1 )
        {
            t_1 = min( t_1, CalculatePlaneIntersectionCS( i, p_0, p_1 ) );
        }
    }

    if ( t_0 >= t_1 )
    {
        return false;
    }

    return true;
}

// Returns the constants m and b in the linear equation y = m * x + b
void CalculateLineConstants( float2 p_0, float2 p_1, out float m, out float b )
{
    float dx = ( p_1.x - p_0.x );
    dx = abs( dx ) > 0.001 ? dx : ( sign( dx ) != 0 ? sign( dx ) * 0.001 : 0.001 );
    m = ( p_1.y - p_0.y ) / dx;
    b = p_0.y - m * p_0.x;
}

void AddSilhouetteEdge( uint instance_index, float m, float b )
{
    uint count, offset = instance_index * g_SilhouetteEdgeBufferOffset;
    g_SilhouetteEdgeCountBuffer.InterlockedAdd( instance_index * 4, 1, count );
    g_SilhouetteEdgeBuffer[ offset + count ] = float2( m, b );
}

[ numthreads( BLOCK_SIZE_X, 1, 1 ) ]
void CSMain( uint3 dispatch_thread_id : SV_DispatchThreadID )
{
    uint vertex_count = g_FaceCount * 3;
    uint instance_index = dispatch_thread_id.x / vertex_count + g_InstanceOffset;
    uint vertex_index = dispatch_thread_id.x % vertex_count;
    uint triangle_index = vertex_index / 3;
    uint edge_index = vertex_index % 3;
    uint index_buffer_offset = triangle_index * 6;

    float4x4 world_matrix = g_WorldMatrixBuffer[ instance_index ];

    float3 triangle_positions_ws[ 3 ];
    triangle_positions_ws[ 0 ] = mul( float4( g_VertexBuffer[ g_IndexBufferAdj[ index_buffer_offset + 0 ] ], 1.0 ), world_matrix ).xyz;
    triangle_positions_ws[ 1 ] = mul( float4( g_VertexBuffer[ g_IndexBufferAdj[ index_buffer_offset + 2 ] ], 1.0 ), world_matrix ).xyz;
    triangle_positions_ws[ 2 ] = mul( float4( g_VertexBuffer[ g_IndexBufferAdj[ index_buffer_offset + 4 ] ], 1.0 ), world_matrix ).xyz;
    
    float3 neighbor_positions_ws[ 3 ];
    neighbor_positions_ws[ 0 ] = triangle_positions_ws[ edge_index ];
    neighbor_positions_ws[ 1 ] = mul( float4( g_VertexBuffer[ g_IndexBufferAdj[ index_buffer_offset + edge_index * 2 + 1 ] ], 1.0 ), world_matrix ).xyz;
    neighbor_positions_ws[ 2 ] = triangle_positions_ws[ ( edge_index + 1 ) % 3 ];

    bool is_triangle_front_facing = IsTriangleFrontFacingWS( triangle_positions_ws[ 0 ], triangle_positions_ws[ 1 ], triangle_positions_ws[ 2 ], g_CameraPosition.xyz );
    bool is_neighbor_front_facing = IsTriangleFrontFacingWS( neighbor_positions_ws[ 0 ], neighbor_positions_ws[ 1 ], neighbor_positions_ws[ 2 ], g_CameraPosition.xyz );
    
    if ( is_triangle_front_facing && !is_neighbor_front_facing )
    {
        float4 edge_position_cs_0 = mul( float4( neighbor_positions_ws[ 0 ], 1.0 ), g_ViewProjectionFlippedZ );
        float4 edge_position_cs_1 = mul( float4( neighbor_positions_ws[ 2 ], 1.0 ), g_ViewProjectionFlippedZ );
        
        if ( IsLineSegmentInsideFrustumCS( edge_position_cs_0, edge_position_cs_1 ) )
        {
            float2 edge_position_ss_0 = ( edge_position_cs_0.xy / edge_position_cs_0.w + float2( 1.0, -1.0 ) ) * float2( 0.5 * g_DepthWidth, -0.5 * g_DepthHeight );
            float2 edge_position_ss_1 = ( edge_position_cs_1.xy / edge_position_cs_1.w + float2( 1.0, -1.0 ) ) * float2( 0.5 * g_DepthWidth, -0.5 * g_DepthHeight );

            float m, b;
            CalculateLineConstants( edge_position_ss_0, edge_position_ss_1, m, b );

            AddSilhouetteEdge( instance_index, m, b );
        }
    }
}