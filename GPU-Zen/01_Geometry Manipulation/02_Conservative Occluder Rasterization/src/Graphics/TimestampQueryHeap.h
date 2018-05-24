#pragma once

#include <d3d12.h>

namespace NGraphics
{
    class CTimestampQueryHeap
    {
    private:
        ID3D12QueryHeap* m_QueryHeap;
        ID3D12Resource* m_ReadbackResource;
        UINT m_TimestampCount;
        UINT64 m_TimestampFrequency;

    public:
        CTimestampQueryHeap();

        void Create( ID3D12Device* device, ID3D12CommandQueue* command_queue, UINT timestamp_count );
        void Destroy();

        void SetTimestampQuery( ID3D12GraphicsCommandList* command_list, UINT index );

        UINT64 GetTimestamp( UINT index );
        float GetTimeDifference( UINT start_index, UINT stop_index );
    };
}