#include "CommandContext.h"
#include "GraphicsDefines.h"

namespace NGraphics
{
    CCommandContext::CCommandContext() :
        m_CommandQueue( nullptr ),
        m_CommandAllocator( nullptr ),
        m_CommandList( nullptr ),
        m_Fence( nullptr ),
        m_FenceValue( 0 ),
        m_FenceEvent( nullptr )
    {
    }

    void CCommandContext::Create( ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, D3D12_FENCE_FLAGS fence_flags )
    {
        D3D12_COMMAND_QUEUE_DESC command_queue_desc;
        ZeroMemory( &command_queue_desc, sizeof( command_queue_desc ) );
        command_queue_desc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        command_queue_desc.Type = type;
        HR( device->CreateCommandQueue( &command_queue_desc, IID_PPV_ARGS( &m_CommandQueue ) ) );

        HR( device->CreateCommandAllocator( type, IID_PPV_ARGS( &m_CommandAllocator ) ) );

        HR( device->CreateCommandList( 0, type, m_CommandAllocator, nullptr, IID_PPV_ARGS( &m_CommandList ) ) );

        HR( device->CreateFence( 0, fence_flags, IID_PPV_ARGS( &m_Fence ) ) );
        m_FenceValue = 1;
        m_FenceEvent = CreateEventEx( nullptr, FALSE, FALSE, EVENT_ALL_ACCESS );
    }

    void CCommandContext::Destroy()
    {
        SAFE_CLOSE( m_FenceEvent );
        m_FenceValue = 0;
        SAFE_RELEASE( m_Fence );
        SAFE_RELEASE( m_CommandList );
        SAFE_RELEASE( m_CommandAllocator );
        SAFE_RELEASE( m_CommandQueue );
    }

    void CCommandContext::WaitForGpu()
    {
        m_CommandQueue->Signal( m_Fence, m_FenceValue );

        if ( m_Fence->GetCompletedValue() < m_FenceValue )
        {
            m_Fence->SetEventOnCompletion( m_FenceValue, m_FenceEvent );
            WaitForSingleObject( m_FenceEvent, INFINITE );
        }

        ++m_FenceValue;
    }

    void CCommandContext::ResetCommandList( ID3D12PipelineState* initial_pipeline_state )
    {
        HR( m_CommandAllocator->Reset() );
        HR( m_CommandList->Reset( m_CommandAllocator, initial_pipeline_state ) );
    }

    void CCommandContext::CloseCommandList()
    {
        HR( m_CommandList->Close() );
    }

    void CCommandContext::ExecuteCommandList()
    {
        ID3D12CommandList* command_lists[] = { m_CommandList };
        m_CommandQueue->ExecuteCommandLists( _countof( command_lists ), command_lists );
    }

    ID3D12CommandQueue* CCommandContext::GetCommandQueue() const
    {
        return m_CommandQueue;
    }

    ID3D12CommandAllocator* CCommandContext::GetCommandAllocator() const
    {
        return m_CommandAllocator;
    }

    ID3D12GraphicsCommandList* CCommandContext::GetCommandList() const
    {
        return m_CommandList;
    }

    ID3D12Fence* CCommandContext::GetFence() const
    {
        return m_Fence;
    }

    UINT64 CCommandContext::GetFenceValue() const
    {
        return m_FenceValue;
    }

    HANDLE CCommandContext::GetFenceEvent() const
    {
        return m_FenceEvent;
    }
}