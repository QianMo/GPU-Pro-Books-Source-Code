#pragma once
#include "KGraphicsDevice.h"

class KDepthTarget
{
public:
	KDepthTarget();
	~KDepthTarget();

	void CreateDepthTarget(int32 width, int32 height);

	D3D12_CPU_DESCRIPTOR_HANDLE GetDSVCPUHandle()	{ return m_DSVCPUHandle; }
	D3D12_GPU_DESCRIPTOR_HANDLE GetDSVGPUHandle()	{ return m_DSVGPUHandle; }

	D3D12_CPU_DESCRIPTOR_HANDLE GetSRVCPUHandle()	{ return m_SRVCPUHandle; }
	D3D12_GPU_DESCRIPTOR_HANDLE GetSRVGPUHandle()	{ return m_SRVGPUHandle; }

	ID3D12Resource* GetResource()					{ return m_DTResource; }

private:

	ID3D12Resource* m_DTResource;

	D3D12_CPU_DESCRIPTOR_HANDLE m_DSVCPUHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE m_DSVGPUHandle;

	D3D12_CPU_DESCRIPTOR_HANDLE m_SRVCPUHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE m_SRVGPUHandle;
};