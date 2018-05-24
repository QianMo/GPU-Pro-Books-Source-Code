struct IndirectCommand
{
    uint instance_offset;
    uint material_index;
    uint2 vertex_buffer_view_buffer_location;
    uint vertex_buffer_view_size_in_bytes;
    uint vertex_buffer_view_stride_in_bytes;
    uint2 index_buffer_view_buffer_location;
    uint index_buffer_view_size_in_bytes;
    uint index_buffer_view_format;
    uint draw_arguments_index_count_per_instance;
    uint draw_arguments_instance_count;
    uint draw_arguments_start_index_location;
    int draw_arguments_base_vertex_location;
    uint draw_arguments_start_instance_location;

    float padding;
};

StructuredBuffer< uint > g_ModelInstanceCountBuffer : register( t0 );
StructuredBuffer< IndirectCommand > g_InputCommandBuffer : register( t1 );
AppendStructuredBuffer< IndirectCommand > g_OutputCommandBuffer : register( u0 );

[ numthreads( BLOCK_SIZE_X, 1, 1 ) ]
void CSMain( uint3 dispatch_thread_id : SV_DispatchThreadID )
{
    uint model_index = dispatch_thread_id.x;
    uint instance_count = g_ModelInstanceCountBuffer[ model_index ];
    if ( instance_count > 0 )
    {
        IndirectCommand indirect_command = g_InputCommandBuffer[ model_index ];
        indirect_command.draw_arguments_instance_count = instance_count;
        g_OutputCommandBuffer.Append( indirect_command );
    }
}