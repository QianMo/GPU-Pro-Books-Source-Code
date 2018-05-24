struct VSInput
{
    float3 position_os : POSITION;
};

struct GSInput
{
    float4 position : SV_POSITION;
#ifdef INNER_CONSERVATIVE
    nointerpolation uint silhouette_edge_index : SILHOUETTEEDGEINDEX0;
#endif
};

struct PSInput
{
    noperspective centroid float4 position : SV_POSITION;
    float4 normal_cs : NORMAL;
#ifdef INNER_CONSERVATIVE
    nointerpolation uint silhouette_edge_index : SILHOUETTEEDGEINDEX0;
#endif
};

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
};

StructuredBuffer< float4x4 > g_WorldMatrixBuffer : register( t0 );
StructuredBuffer< float2 > g_SilhouetteEdgeBuffer : register( t1 );
StructuredBuffer< uint > g_SilhouetteEdgeCountBuffer : register( t2 );

GSInput VSMain( VSInput input, uint instance_id : SV_InstanceID )
{
    uint instance_index = g_InstanceOffset + instance_id;
    float4x4 world_matrix = g_WorldMatrixBuffer[ instance_index ];

    GSInput output = ( GSInput )0;
    output.position = mul( mul( float4( input.position_os, 1.0 ), world_matrix ), g_ViewProjectionFlippedZ );
#ifdef INNER_CONSERVATIVE
    output.silhouette_edge_index = instance_index;
#endif
    return output;
}

[ maxvertexcount( 3 ) ]
void GSMain( triangle GSInput input[ 3 ], inout TriangleStream< PSInput > output_stream )
{
    PSInput output = ( PSInput )0;
    output.normal_cs.xyz = normalize( cross( input[ 1 ].position.xyw - input[ 0 ].position.xyw, input[ 2 ].position.xyw - input[ 0 ].position.xyw ) );
    output.normal_cs.w = -dot( output.normal_cs.xyz, input[ 0 ].position.xyw );
#ifdef INNER_CONSERVATIVE
    output.silhouette_edge_index = input[ 0 ].silhouette_edge_index;
#endif

    [ unroll ]
    for ( uint i = 0; i < 3; ++i )
    {
        output.position = input[ i ].position;
        output_stream.Append( output );
    }
    output_stream.RestartStrip();
}

bool IsLineIntersectingRectangle( float m, float b, float4 rect )
{
    float y_0 = m * rect.x + b;
    float y_1 = m * rect.z + b;

    float min_y = min( y_0, y_1 );
    float max_y = max( y_0, y_1 );

    return rect.y <= max_y && min_y <= rect.w;
}

bool IsPixelFullyCovered( float2 position, uint instance_index )
{
    float4 rect = float4(
        position - float2( 0.5, 0.5 ),
        position + float2( 0.5, 0.5 ) );

    uint count = g_SilhouetteEdgeCountBuffer[ instance_index ];
    uint offset = instance_index * g_SilhouetteEdgeBufferOffset;
    [ loop ]
    for ( uint i = 0; i < count; ++i )
    {
        float2 edge = g_SilhouetteEdgeBuffer[ offset + i ];
        if ( IsLineIntersectingRectangle( edge.x, edge.y, rect ) )
            return false;
    }

    return true;
}

float PSMain( PSInput input ) : SV_DepthLessEqual
{
#ifdef INNER_CONSERVATIVE
    clip( IsPixelFullyCovered( input.position.xy, input.silhouette_edge_index ) ? 1 : -1 );
#endif

#ifdef FULLSCREEN
    float2 half_pixel_size = float2( 1.0 / g_FullscreenWidth, 1.0 / g_FullscreenHeight );
#else
    float2 half_pixel_size = float2( 1.0 / g_DepthWidth, 1.0 / g_DepthHeight );
#endif
    float2 half_pixel_offset = float2( 0.5, -0.5 ) * sign( input.normal_cs.xy );
    float3 ray_direction_cs = normalize( float3( float2( 2.0, -2.0 ) * ( input.position.xy + half_pixel_offset ) * half_pixel_size + float2( -1.0, 1.0 ), 1.0 ) );
    float t = -input.normal_cs.w / dot( input.normal_cs.xyz, ray_direction_cs );
    float z = 0.0;
    if ( t > 0.0 )
    {
        z = max( 0.0, 0.5 - 0.5 * ( -2.0 * g_FarZ * g_NearZ / ( ( g_FarZ - g_NearZ ) * ray_direction_cs.z * t ) + ( g_FarZ + g_NearZ ) / ( g_FarZ - g_NearZ ) ) );
    }
    return z;
}