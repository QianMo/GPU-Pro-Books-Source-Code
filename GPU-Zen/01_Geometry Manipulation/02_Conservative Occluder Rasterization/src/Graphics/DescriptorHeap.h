#pragma once

#include <d3d12.h>
#include <d3dx12.h>

namespace NGraphics
{
    struct SDescriptorHandle
    {
        CD3DX12_CPU_DESCRIPTOR_HANDLE m_Cpu;
        CD3DX12_GPU_DESCRIPTOR_HANDLE m_Gpu;
    };

    class CDescriptorHeap
    {
    private:
        ID3D12DescriptorHeap* m_Heap;
        UINT m_Capacity;
        UINT m_Size;
        UINT m_Offset;

    public:
        CDescriptorHeap();

        void Create( ID3D12Device* device, UINT size, D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_DESCRIPTOR_HEAP_FLAGS flags );
        void Destroy();

        SDescriptorHandle GenerateHandle();

        ID3D12DescriptorHeap* GetHeap() const;

        const UINT GetCapacity() const;
        const UINT GetSize() const;
    };
}