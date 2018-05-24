cbuffer VisibilityConstants : register( b0 )
{
    uint g_VisibilityBufferOffset;
};
ByteAddressBuffer g_VisibilityBuffer : register( t0 ); // Per all instances
StructuredBuffer< uint > g_InstanceModelMappingsBuffer : register( t1 ); // Per instance
StructuredBuffer< uint > g_ModelInstanceOffsetBuffer : register( t2 ); // Per model
RWStructuredBuffer< uint > g_ModelInstanceCountBuffer : register( u0 ); // Per model, can be resetted
RWStructuredBuffer< uint > g_InstanceIndexMappingsBuffer : register( u1 ); // Per instance

[ numthreads( BLOCK_SIZE_X, 1, 1 ) ]
void CSMain( uint3 dispatch_thread_id : SV_DispatchThreadID )
{
    uint instance_index = dispatch_thread_id.x;
    uint visibility_index = g_VisibilityBufferOffset + instance_index;

    if ( ( g_VisibilityBuffer.Load( ( visibility_index / 32 ) * 4 ) >> ( visibility_index % 32 ) ) & 1 ) // If visible
    {
        uint model_index = g_InstanceModelMappingsBuffer[ instance_index ];
        uint instance_offset = g_ModelInstanceOffsetBuffer[ model_index ];

        uint instance_mappings_index;
        InterlockedAdd( g_ModelInstanceCountBuffer[ model_index ], 1, instance_mappings_index );

        g_InstanceIndexMappingsBuffer[ instance_offset + instance_mappings_index ] = instance_index;
    }
}