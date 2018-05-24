#pragma once

#include <d3d12.h>

namespace NGraphics
{
    class CCommandContext
    {
    private:
        ID3D12CommandQueue* m_CommandQueue;
        ID3D12CommandAllocator* m_CommandAllocator;
        ID3D12GraphicsCommandList* m_CommandList;

        ID3D12Fence* m_Fence;
        UINT64 m_FenceValue;
        HANDLE m_FenceEvent;

    public:
        CCommandContext();

        void Create( ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, D3D12_FENCE_FLAGS fence_flags );
        void Destroy();

        void WaitForGpu();

        void ResetCommandList( ID3D12PipelineState* initial_pipeline_state = nullptr );
        void CloseCommandList();
        void ExecuteCommandList();

        ID3D12CommandQueue* GetCommandQueue() const;
        ID3D12CommandAllocator* GetCommandAllocator() const;
        ID3D12GraphicsCommandList* GetCommandList() const;
        ID3D12Fence* GetFence() const;
        UINT64 GetFenceValue() const;
        HANDLE GetFenceEvent() const;
    };
}