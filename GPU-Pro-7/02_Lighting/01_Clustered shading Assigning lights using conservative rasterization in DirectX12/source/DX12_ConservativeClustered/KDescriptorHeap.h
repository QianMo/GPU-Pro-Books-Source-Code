#pragma once
#include <D3D12.h>
#include "Types.h"
#include "d3dx12.h"

class KDescriptorHeap
{
public:
	KDescriptorHeap();
	~KDescriptorHeap();

	HRESULT CreateDescriptorHeap(uint32 num_descriptors, D3D12_DESCRIPTOR_HEAP_TYPE descriptor_type, D3D12_DESCRIPTOR_HEAP_FLAGS descriptor_flags);
	
	CD3DX12_CPU_DESCRIPTOR_HANDLE GetNewCPUHandle();

	CD3DX12_GPU_DESCRIPTOR_HANDLE GetGPUHandleAtHead()	{ return CD3DX12_GPU_DESCRIPTOR_HANDLE(m_DescriptorHeap->GetGPUDescriptorHandleForHeapStart(), m_Size - 1, m_DescIncrSize); }
	CD3DX12_CPU_DESCRIPTOR_HANDLE GetCPUHandleAtHead()	{ return CD3DX12_CPU_DESCRIPTOR_HANDLE(m_DescriptorHeap->GetCPUDescriptorHandleForHeapStart(), m_Size - 1, m_DescIncrSize); }

	CD3DX12_CPU_DESCRIPTOR_HANDLE GetCPUHandleAtStart() { return CD3DX12_CPU_DESCRIPTOR_HANDLE(m_DescriptorHeap->GetCPUDescriptorHandleForHeapStart()); }
	CD3DX12_GPU_DESCRIPTOR_HANDLE GetGPUHandleAtStart() { return CD3DX12_GPU_DESCRIPTOR_HANDLE(m_DescriptorHeap->GetGPUDescriptorHandleForHeapStart()); }

	uint32 GetIncrSize()								{ return m_DescIncrSize; }
	uint32 GetCapacity()								{ return m_Capacity; }
	uint32 GetSize()									{ return m_Size; }

	ID3D12DescriptorHeap* GetHeap()						{ return m_DescriptorHeap; }

	CD3DX12_CPU_DESCRIPTOR_HANDLE GetCPUHandleAt(int32 index);
	CD3DX12_GPU_DESCRIPTOR_HANDLE GetGPUHandleAt(int32 index);



private:

	ID3D12DescriptorHeap* m_DescriptorHeap;

	CD3DX12_CPU_DESCRIPTOR_HANDLE m_CPUHead;

	uint32 m_DescIncrSize;
	uint32 m_Capacity;
	uint32 m_Size;

};