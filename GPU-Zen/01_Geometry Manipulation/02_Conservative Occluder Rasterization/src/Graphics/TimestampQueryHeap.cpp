#include "TimestampQueryHeap.h"
#include "CommandContext.h"
#include "GraphicsDefines.h"

#include <assert.h>

namespace NGraphics
{
    CTimestampQueryHeap::CTimestampQueryHeap() :
        m_QueryHeap( nullptr ),
        m_ReadbackResource( nullptr ),
        m_TimestampCount( 0 ),
        m_TimestampFrequency( 0 )
    {
    }

    void CTimestampQueryHeap::Create( ID3D12Device* device, ID3D12CommandQueue* command_queue, UINT timestamp_count )
    {
        m_TimestampCount = timestamp_count;

        D3D12_QUERY_HEAP_DESC heap_desc;
        ZeroMemory( &heap_desc, sizeof( heap_desc ) );
        heap_desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heap_desc.Count = m_TimestampCount;
        HR( device->CreateQueryHeap( &heap_desc, IID_PPV_ARGS( &m_QueryHeap ) ) );

        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_READBACK ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( sizeof( UINT64 ) * heap_desc.Count ),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS( &m_ReadbackResource ) ) );

        command_queue->GetTimestampFrequency( &m_TimestampFrequency );
    }

    void CTimestampQueryHeap::Destroy()
    {
        m_TimestampCount = 0;
        m_TimestampFrequency = 0;

        SAFE_RELEASE( m_ReadbackResource );
        SAFE_RELEASE( m_QueryHeap );
    }

    void CTimestampQueryHeap::SetTimestampQuery( ID3D12GraphicsCommandList* command_list, UINT index )
    {
        assert( index < m_TimestampCount );
        command_list->EndQuery( m_QueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, index );
        command_list->ResolveQueryData( m_QueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, index, 1, m_ReadbackResource, index * sizeof( UINT64 ) );
    }

    UINT64 CTimestampQueryHeap::GetTimestamp( UINT index )
    {
        assert( index < m_TimestampCount );
        BYTE* mapped_data = nullptr;
        HR( m_ReadbackResource->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_data ) ) );
        UINT64 result = reinterpret_cast< UINT64* >( mapped_data )[ index ];
        m_ReadbackResource->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
        return result;
    }

    float CTimestampQueryHeap::GetTimeDifference( UINT start_index, UINT stop_index )
    {
        return ( static_cast< float >( GetTimestamp( stop_index ) - GetTimestamp( start_index ) ) / static_cast< float >( m_TimestampFrequency ) );
    }
}