#include "DescriptorHeap.h"
#include "GraphicsDefines.h"

namespace NGraphics
{
    CDescriptorHeap::CDescriptorHeap() :
        m_Heap( nullptr ),
        m_Capacity( 0 ),
        m_Size( 0 ),
        m_Offset( 0 )
    {
    }
        
    void CDescriptorHeap::Create( ID3D12Device* device, UINT size, D3D12_DESCRIPTOR_HEAP_TYPE type, D3D12_DESCRIPTOR_HEAP_FLAGS flags )
    {
        D3D12_DESCRIPTOR_HEAP_DESC shader_heap_desc;
        ZeroMemory( &shader_heap_desc, sizeof( shader_heap_desc ) );
        shader_heap_desc.NumDescriptors = size;
        shader_heap_desc.Type = type;
        shader_heap_desc.Flags = flags;
        HR( device->CreateDescriptorHeap( &shader_heap_desc, IID_PPV_ARGS( &m_Heap ) ) );

        m_Capacity = size;
        m_Size = 0;
        m_Offset = device->GetDescriptorHandleIncrementSize( type );
    }

    void CDescriptorHeap::Destroy()
    {
        SAFE_RELEASE( m_Heap );
        m_Capacity = 0;
        m_Size = 0;
        m_Offset = 0;
    }

    SDescriptorHandle CDescriptorHeap::GenerateHandle()
    {
        assert( m_Size < m_Capacity );

        SDescriptorHandle handle;
        handle.m_Cpu = CD3DX12_CPU_DESCRIPTOR_HANDLE( m_Heap->GetCPUDescriptorHandleForHeapStart(), m_Size, m_Offset );
        handle.m_Gpu = CD3DX12_GPU_DESCRIPTOR_HANDLE( m_Heap->GetGPUDescriptorHandleForHeapStart(), m_Size, m_Offset );

        ++m_Size;

        return handle;
    }

    ID3D12DescriptorHeap* CDescriptorHeap::GetHeap() const
    {
        return m_Heap;
    }

    const UINT CDescriptorHeap::GetCapacity() const
    {
        return m_Capacity;
    }
    const UINT CDescriptorHeap::GetSize() const
    {
        return m_Size;
    }
}