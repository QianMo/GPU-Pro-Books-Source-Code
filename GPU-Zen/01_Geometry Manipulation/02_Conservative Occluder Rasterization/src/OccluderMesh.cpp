#include "OccluderMesh.h"

#include <assert.h>

COccluderMesh::COccluderMesh() :
    m_VertexBuffer( nullptr ),
    m_VertexSrvBuffer( nullptr ),
    m_VertexBufferUpload( nullptr ),
    m_IndexBuffer( nullptr ),
    m_IndexBufferUpload( nullptr ),
    m_IndexBufferAdj( nullptr ),
    m_IndexBufferAdjUpload( nullptr ),
    m_IndexBufferLine( nullptr ),
    m_IndexBufferLineUpload( nullptr ),

    m_IndexCount( 0 ),
    m_IndexLineCount( 0 ),
    m_FaceCount( 0 )
{

}

void COccluderMesh::Create( ID3D12Device* device,
                            NGraphics::CCommandContext* graphics_context,
                            NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
                            std::vector< DirectX::XMFLOAT3 >* vertices,
                            std::vector< UINT >* indices,
                            std::vector< UINT >* indices_line,
                            std::vector< UINT >* indices_adj )
{
    assert( indices->size() * 2 == indices_adj->size() );

    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();

    UINT vertex_buffer_size = static_cast< UINT >( sizeof( DirectX::XMFLOAT3 ) * vertices->size() );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( vertex_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_VertexBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( vertex_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_VertexSrvBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( vertex_buffer_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_VertexBufferUpload ) ) );
    UINT index_buffer_size = static_cast< UINT >( sizeof( UINT ) * indices->size() );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( index_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_IndexBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( index_buffer_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_IndexBufferUpload ) ) );
    UINT index_buffer_line_size = static_cast< UINT >( sizeof( UINT ) * indices_line->size() );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( index_buffer_line_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_IndexBufferLine ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( index_buffer_line_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_IndexBufferLineUpload ) ) );
    UINT index_buffer_adj_size = static_cast< UINT >( sizeof( UINT ) * indices_adj->size() );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( index_buffer_adj_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_IndexBufferAdj ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( index_buffer_adj_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_IndexBufferAdjUpload ) ) );

    BYTE* mapped_vertex_buffer_upload = nullptr;
    BYTE* mapped_index_buffer_upload = nullptr;
    BYTE* mapped_index_buffer_line_upload = nullptr;
    BYTE* mapped_index_buffer_adj_upload = nullptr;
    HR( m_VertexBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_vertex_buffer_upload ) ) );
    HR( m_IndexBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_index_buffer_upload ) ) );
    HR( m_IndexBufferLineUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_index_buffer_line_upload ) ) );
    HR( m_IndexBufferAdjUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_index_buffer_adj_upload ) ) );
    memcpy( mapped_vertex_buffer_upload, &( *vertices )[ 0 ], vertex_buffer_size );
    memcpy( mapped_index_buffer_upload, &( *indices )[ 0 ], index_buffer_size );
    memcpy( mapped_index_buffer_line_upload, &( *indices_line )[ 0 ], index_buffer_line_size );
    memcpy( mapped_index_buffer_adj_upload, &( *indices_adj )[ 0 ], index_buffer_adj_size );
    m_VertexBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_IndexBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_IndexBufferLineUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_IndexBufferAdjUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

    command_list->CopyBufferRegion( m_VertexBuffer, 0, m_VertexBufferUpload, 0, vertex_buffer_size );
    command_list->CopyBufferRegion( m_VertexSrvBuffer, 0, m_VertexBufferUpload, 0, vertex_buffer_size );
    command_list->CopyBufferRegion( m_IndexBuffer, 0, m_IndexBufferUpload, 0, index_buffer_size );
    command_list->CopyBufferRegion( m_IndexBufferLine, 0, m_IndexBufferLineUpload, 0, index_buffer_line_size );
    command_list->CopyBufferRegion( m_IndexBufferAdj, 0, m_IndexBufferAdjUpload, 0, index_buffer_adj_size );
    const CD3DX12_RESOURCE_BARRIER copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_VertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_VertexSrvBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_IndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_IndexBufferLine, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_IndexBufferAdj, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE )
    };
    command_list->ResourceBarrier( _countof( copy_barriers ), copy_barriers );

    m_VertexBufferView.BufferLocation = m_VertexBuffer->GetGPUVirtualAddress();
    m_VertexBufferView.StrideInBytes = sizeof( DirectX::XMFLOAT3 );
    m_VertexBufferView.SizeInBytes = vertex_buffer_size;

    m_IndexBufferView.BufferLocation = m_IndexBuffer->GetGPUVirtualAddress();
    m_IndexBufferView.Format = DXGI_FORMAT_R32_UINT;
    m_IndexBufferView.SizeInBytes = index_buffer_size;

    m_IndexBufferLineView.BufferLocation = m_IndexBufferLine->GetGPUVirtualAddress();
    m_IndexBufferLineView.Format = DXGI_FORMAT_R32_UINT;
    m_IndexBufferLineView.SizeInBytes = index_buffer_line_size;

    m_IndexBufferAdjView.BufferLocation = m_IndexBufferAdj->GetGPUVirtualAddress();
    m_IndexBufferAdjView.Format = DXGI_FORMAT_R32_UINT;
    m_IndexBufferAdjView.SizeInBytes = index_buffer_adj_size;

    m_VertexSrvBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_IndexBufferAdjSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = static_cast< UINT >( vertices->size() );
    srv_desc.Buffer.StructureByteStride = sizeof( DirectX::XMFLOAT3 );
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_VertexSrvBuffer, &srv_desc, m_VertexSrvBufferSrv.m_Cpu );

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = static_cast< UINT >( indices_adj->size() );
    srv_desc.Buffer.StructureByteStride = sizeof( UINT );
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_IndexBufferAdj, &srv_desc, m_IndexBufferAdjSrv.m_Cpu );

    m_IndexCount = static_cast< UINT >( indices->size() );
    m_IndexLineCount = static_cast< UINT >( indices_line->size() );
    m_IndexAdjCount = static_cast< UINT >( indices_adj->size() );
    m_FaceCount = m_IndexCount / 3;
}

void COccluderMesh::Destroy()
{
    m_FaceCount = 0;
    m_IndexLineCount = 0;
    m_IndexCount = 0;

    SAFE_RELEASE( m_IndexBufferAdjUpload );
    SAFE_RELEASE( m_IndexBufferAdj );
    SAFE_RELEASE( m_IndexBufferLineUpload );
    SAFE_RELEASE( m_IndexBufferLine );
    SAFE_RELEASE( m_IndexBufferUpload );
    SAFE_RELEASE( m_IndexBuffer );
    SAFE_RELEASE( m_VertexBufferUpload );
    SAFE_RELEASE( m_VertexSrvBuffer );
    SAFE_RELEASE( m_VertexBuffer );
}

void COccluderMesh::Draw( ID3D12GraphicsCommandList* command_list, UINT instance_count )
{
    command_list->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
    command_list->IASetVertexBuffers( 0, 1, &m_VertexBufferView );
    command_list->IASetIndexBuffer( &m_IndexBufferView );
    command_list->DrawIndexedInstanced( m_IndexCount, instance_count, 0, 0, 0 );
}

void COccluderMesh::DrawLine( ID3D12GraphicsCommandList* command_list, UINT instance_count )
{
    command_list->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_LINELIST );
    command_list->IASetVertexBuffers( 0, 1, &m_VertexBufferView );
    command_list->IASetIndexBuffer( &m_IndexBufferLineView );
    command_list->DrawIndexedInstanced( m_IndexLineCount, instance_count, 0, 0, 0 );
}

void COccluderMesh::DrawAdj( ID3D12GraphicsCommandList* command_list, UINT instance_count )
{
    command_list->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ );
    command_list->IASetVertexBuffers( 0, 1, &m_VertexBufferView );
    command_list->IASetIndexBuffer( &m_IndexBufferAdjView );
    command_list->DrawIndexedInstanced( m_IndexAdjCount, instance_count, 0, 0, 0 );
}

D3D12_VERTEX_BUFFER_VIEW COccluderMesh::GetVertexBufferView() const
{
    return m_VertexBufferView;
}
D3D12_INDEX_BUFFER_VIEW COccluderMesh::GetIndexBufferView() const
{
    return m_IndexBufferView;
}

NGraphics::SDescriptorHandle COccluderMesh::GetVertexSrvBufferSrv() const
{
    return m_VertexSrvBufferSrv;
}
NGraphics::SDescriptorHandle COccluderMesh::GetIndexBufferAdjSrv() const
{
    return m_IndexBufferAdjSrv;
}

const UINT COccluderMesh::GetIndexCount() const
{
    return m_IndexCount;
}
const UINT COccluderMesh::GetFaceCount() const
{
    return m_FaceCount;
}

void CCubeOccluderMesh::Create( ID3D12Device* device,
                                NGraphics::CCommandContext* graphics_context,
                                NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    std::vector< DirectX::XMFLOAT3 > vertices;
    vertices.insert( vertices.begin(), {
        DirectX::XMFLOAT3( -1.0f, -1.0f, -1.0f ),
        DirectX::XMFLOAT3( -1.0f, -1.0f,  1.0f ),
        DirectX::XMFLOAT3( -1.0f,  1.0f, -1.0f ),
        DirectX::XMFLOAT3( -1.0f,  1.0f,  1.0f ),
        DirectX::XMFLOAT3( 1.0f, -1.0f, -1.0f ),
        DirectX::XMFLOAT3( 1.0f, -1.0f,  1.0f ),
        DirectX::XMFLOAT3( 1.0f,  1.0f, -1.0f ),
        DirectX::XMFLOAT3( 1.0f,  1.0f,  1.0f )
    } );
    std::vector<Index> indices;
    indices.insert( indices.begin(), {
        0, 2, 1,
        1, 2, 3,
        4, 5, 6,
        5, 7, 6,
        0, 1, 5,
        0, 5, 4,
        2, 6, 7,
        2, 7, 3,
        0, 4, 6,
        0, 6, 2,
        1, 3, 7,
        1, 7, 5
    } );
    std::vector<Index> indices_line;
    indices_line.insert( indices_line.begin(), {
        0, 1,
        0, 2,
        0, 4,
        1, 3,
        1, 5,
        2, 3,
        2, 6,
        3, 7,
        4, 5,
        4, 6,
        5, 7,
        6, 7
    } );
    std::vector<Index> indices_adj;
    indices_adj.insert( indices_adj.begin(), {
        0, 6, 2, 3, 1, 4,
        1, 0, 2, 7, 3, 7,
        4, 0, 5, 7, 6, 0,
        5, 1, 7, 2, 6, 4,
        0, 3, 1, 7, 5, 4,
        0, 1, 5, 6, 4, 6,
        2, 0, 6, 5, 7, 3,
        2, 6, 7, 1, 3, 1,
        0, 5, 4, 5, 6, 2,
        0, 4, 6, 7, 2, 1,
        1, 2, 3, 2, 7, 5,
        1, 3, 7, 6, 5, 0
    } );

    COccluderMesh::Create( device, graphics_context, descriptor_heaps, &vertices, &indices, &indices_line, &indices_adj );
}

void CCylinderOccluderMesh::Create( ID3D12Device* device,
                                    NGraphics::CCommandContext* graphics_context,
                                    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    std::vector<DirectX::XMFLOAT3> vertices( 2 * m_CylinderSliceCount + 2 );
    std::vector<UINT> indices( m_CylinderSliceCount * 12 );
    std::vector<UINT> indices_line( m_CylinderSliceCount * 6 + 2 );
    std::vector<UINT> indices_adj( m_CylinderSliceCount * 24 );
    UINT vertices_index = 0;
    UINT indices_index = 0;
    UINT indices_line_index = 0;
    UINT indices_adj_index = 0;

    float theta = 2.0f * DirectX::XM_PI / static_cast< float >( m_CylinderSliceCount );

    for ( UINT i = 0; i < m_CylinderSliceCount; ++i )
    {
        float x = cosf( static_cast< float >( i ) * theta );
        float z = sinf( static_cast< float >( i ) * theta );
        vertices[ vertices_index++ ] = DirectX::XMFLOAT3( x, -1.0f, z );
    }

    for ( UINT i = 0; i < m_CylinderSliceCount; ++i )
    {
        float x = cosf( static_cast< float >( i ) * theta );
        float z = sinf( static_cast< float >( i ) * theta );
        vertices[ vertices_index++ ] = DirectX::XMFLOAT3( x, 1.0f, z );
    }

    vertices[ vertices_index++ ] = DirectX::XMFLOAT3( 0.0f, -1.0f, 0.0f );
    vertices[ vertices_index++ ] = DirectX::XMFLOAT3( 0.0f, 1.0f, 0.0f );

    UINT bottom_center_index = vertices_index - 2;
    UINT top_center_index = vertices_index - 1;

    for ( UINT i = 0; i < m_CylinderSliceCount; ++i )
    {
        indices[ indices_index++ ] = i;
        indices[ indices_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );
        indices[ indices_index++ ] = m_CylinderSliceCount + i;

        indices[ indices_index++ ] = i;
        indices[ indices_index++ ] = ( ( i + 1 ) % m_CylinderSliceCount );
        indices[ indices_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );

        indices_line[ indices_line_index++ ] = i;
        indices_line[ indices_line_index++ ] = m_CylinderSliceCount + i;

        indices_line[ indices_line_index++ ] = m_CylinderSliceCount + i;
        indices_line[ indices_line_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );

        indices_line[ indices_line_index++ ] = i;
        indices_line[ indices_line_index++ ] = ( ( i + 1 ) % m_CylinderSliceCount );

        indices_adj[ indices_adj_index++ ] = i;
        indices_adj[ indices_adj_index++ ] = ( ( i + 1 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = top_center_index;
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + i;
        indices_adj[ indices_adj_index++ ] = ( i == 0 ? m_CylinderSliceCount - 1 : i - 1 );

        indices_adj[ indices_adj_index++ ] = i;
        indices_adj[ indices_adj_index++ ] = bottom_center_index;
        indices_adj[ indices_adj_index++ ] = ( ( i + 1 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + ( ( i + 2 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + i;
    }
    indices_line[ indices_line_index++ ] = m_CylinderSliceCount - 1;
    indices_line[ indices_line_index++ ] = m_CylinderSliceCount * 2 - 1;

    for ( UINT i = 0; i < m_CylinderSliceCount; ++i )
    {
        indices[ indices_index++ ] = bottom_center_index;
        indices[ indices_index++ ] = ( ( i + 1 ) % m_CylinderSliceCount );
        indices[ indices_index++ ] = i;

        indices_adj[ indices_adj_index++ ] = bottom_center_index;
        indices_adj[ indices_adj_index++ ] = ( ( i + 2 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = ( ( i + 1 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + i;
        indices_adj[ indices_adj_index++ ] = i;
        indices_adj[ indices_adj_index++ ] = ( i == 0 ? m_CylinderSliceCount - 1 : i - 1 );
    }

    for ( UINT i = 0; i < m_CylinderSliceCount; ++i )
    {
        indices[ indices_index++ ] = top_center_index;
        indices[ indices_index++ ] = m_CylinderSliceCount + i;
        indices[ indices_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );

        indices_adj[ indices_adj_index++ ] = top_center_index;
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + ( i == 0 ? m_CylinderSliceCount - 1 : i - 1 );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + i;
        indices_adj[ indices_adj_index++ ] = i;
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + ( ( i + 1 ) % m_CylinderSliceCount );
        indices_adj[ indices_adj_index++ ] = m_CylinderSliceCount + ( ( i + 2 ) % m_CylinderSliceCount );
    }

    COccluderMesh::Create( device, graphics_context, descriptor_heaps, &vertices, &indices, &indices_line, &indices_adj );
}