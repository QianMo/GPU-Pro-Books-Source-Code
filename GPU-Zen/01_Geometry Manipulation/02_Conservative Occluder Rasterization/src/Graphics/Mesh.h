#pragma once

#include "GraphicsDefines.h"

#include <d3d12.h>
#include <d3dx12.h>
#include <string>
#include <vector>

namespace NGraphics
{
    template< class CVertex >
    class CMesh
    {
    private:
        ID3D12Resource* m_VertexBuffer;
        ID3D12Resource* m_VertexBufferUpload;
        ID3D12Resource* m_IndexBuffer;
        ID3D12Resource* m_IndexBufferUpload;

        D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
        D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;

        UINT m_IndexCount;

        D3D_PRIMITIVE_TOPOLOGY m_Topology;

    public:
        CMesh();

        void Create( ID3D12Device* device, ID3D12GraphicsCommandList* command_list,
                     std::vector< CVertex >* vertices, std::vector< Index >* indices,
                     D3D_PRIMITIVE_TOPOLOGY topology = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
        void Destroy();

        void Draw( ID3D12GraphicsCommandList* command_list, UINT instance_count = 1 );

        void SetName( const std::wstring& name );

        const UINT GetIndexCount() const;
        const D3D12_VERTEX_BUFFER_VIEW GetVertexBufferView() const;
        const D3D12_INDEX_BUFFER_VIEW GetIndexBufferView() const;

    private:
        const DXGI_FORMAT GetIndexBufferFormat();
    };

    template< class CVertex >
    CMesh< CVertex >::CMesh() :
        m_VertexBuffer( nullptr ),
        m_VertexBufferUpload( nullptr ),
        m_IndexBuffer( nullptr ),
        m_IndexBufferUpload( nullptr ),
        m_IndexCount( 0 ),
        m_Topology( D3D_PRIMITIVE_TOPOLOGY_UNDEFINED )
    {
    }

    template< class CVertex >
    void CMesh< CVertex >::Create( ID3D12Device* device, ID3D12GraphicsCommandList* command_list,
                                   std::vector< CVertex >* vertices, std::vector< Index >* indices,
                                   D3D_PRIMITIVE_TOPOLOGY topology )
    {
        UINT vertex_buffer_size = static_cast< UINT >( sizeof( CVertex ) * vertices->size() );
        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( vertex_buffer_size ),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS( &m_VertexBuffer ) ) );
        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( vertex_buffer_size ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_VertexBufferUpload ) ) );

        D3D12_SUBRESOURCE_DATA vertex_data;
        ZeroMemory( &vertex_data, sizeof( vertex_data ) );
        vertex_data.pData = reinterpret_cast< const BYTE* >( vertices->data() );
        vertex_data.RowPitch = vertex_buffer_size;
        vertex_data.SlicePitch = vertex_data.RowPitch;

        UpdateSubresources( command_list, m_VertexBuffer, m_VertexBufferUpload, 0, 0, 1, &vertex_data );

        command_list->ResourceBarrier( 1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_VertexBuffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER ) );

        m_VertexBufferView.BufferLocation = m_VertexBuffer->GetGPUVirtualAddress();
        m_VertexBufferView.StrideInBytes = sizeof( CVertex );
        m_VertexBufferView.SizeInBytes = vertex_buffer_size;

        m_IndexCount = static_cast< UINT >( indices->size() );
        UINT index_buffer_size = sizeof( Index ) * m_IndexCount;
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

        D3D12_SUBRESOURCE_DATA index_data;
        ZeroMemory( &index_data, sizeof( index_data ) );
        index_data.pData = reinterpret_cast< const BYTE* >( indices->data() );
        index_data.RowPitch = index_buffer_size;
        index_data.SlicePitch = index_data.RowPitch;

        UpdateSubresources( command_list, m_IndexBuffer, m_IndexBufferUpload, 0, 0, 1, &index_data );

        command_list->ResourceBarrier( 1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_IndexBuffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ) );

        m_IndexBufferView.BufferLocation = m_IndexBuffer->GetGPUVirtualAddress();
        m_IndexBufferView.Format = GetIndexBufferFormat();
        m_IndexBufferView.SizeInBytes = index_buffer_size;

        m_Topology = topology;
    }

    template< class CVertex >
    void CMesh< CVertex >::Destroy()
    {
        m_Topology = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED;
        m_IndexCount = 0;

        SAFE_RELEASE( m_VertexBufferUpload );
        SAFE_RELEASE( m_VertexBuffer );
        SAFE_RELEASE( m_IndexBufferUpload );
        SAFE_RELEASE( m_IndexBuffer );
    }

    template< class CVertex >
    void CMesh< CVertex >::Draw( ID3D12GraphicsCommandList* command_list, UINT instance_count )
    {
        command_list->IASetPrimitiveTopology( m_Topology );
        command_list->IASetVertexBuffers( 0, 1, &m_VertexBufferView );
        command_list->IASetIndexBuffer( &m_IndexBufferView );
        command_list->DrawIndexedInstanced( m_IndexCount, instance_count, 0, 0, 0 );
    }

    template< class CVertex >
    void CMesh< CVertex >::SetName( const std::wstring& name )
    {
        m_VertexBuffer->SetName( std::wstring( name + L" Vertex Buffer" ).c_str() );
        m_VertexBufferUpload->SetName( std::wstring( name + L" Vertex Buffer Upload" ).c_str() );
        m_IndexBuffer->SetName( std::wstring( name + L" Index Buffer" ).c_str() );
        m_IndexBufferUpload->SetName( std::wstring( name + L" Index Buffer Upload" ).c_str() );
    }

    template< class CVertex >
    const UINT CMesh< CVertex >::GetIndexCount() const
    {
        return m_IndexCount;
    }

    template< class CVertex >
    const D3D12_VERTEX_BUFFER_VIEW CMesh< CVertex >::GetVertexBufferView() const
    {
        return m_VertexBufferView;
    }

    template< class CVertex >
    const D3D12_INDEX_BUFFER_VIEW CMesh< CVertex >::GetIndexBufferView() const
    {
        return m_IndexBufferView;
    }

    template< class CVertex >
    const DXGI_FORMAT CMesh< CVertex >::GetIndexBufferFormat()
    {
        DXGI_FORMAT index_buffer_format;
        switch ( sizeof( Index ) )
        {
            case 1:
                index_buffer_format = DXGI_FORMAT_R8_UINT;
                break;
            case 2:
                index_buffer_format = DXGI_FORMAT_R16_UINT;
                break;
            default:
                index_buffer_format = DXGI_FORMAT_R32_UINT;
                break;
        }
        return index_buffer_format;
    }
}