#pragma once
#include "KGraphicsDevice.h"

class KRenderTarget
{
public:
	KRenderTarget();
	~KRenderTarget();

	void CreateRenderTarget(DXGI_FORMAT tex_format, int32 width, int32 height, const float clear_color[4] = 0);
	void CreateRenderTargetArray(uint32 num_slices, DXGI_FORMAT tex_format, int32 width, int32 height, const  float clear_color[4] = 0);

	D3D12_CPU_DESCRIPTOR_HANDLE GetRTVCPUHandle() { return m_RTVCPUHandle; }
	D3D12_GPU_DESCRIPTOR_HANDLE GetRTVGPUHandle() { return m_RTVGPUHandle; }

	D3D12_CPU_DESCRIPTOR_HANDLE GetSRVCPUHandle() { return m_SRVCPUHandle; }
	D3D12_GPU_DESCRIPTOR_HANDLE GetSRVGPUHandle() { return m_SRVGPUHandle; }

	ID3D12Resource* GetResource() { return m_RTResource; }

private:

	ID3D12Resource* m_RTResource;

	D3D12_CPU_DESCRIPTOR_HANDLE m_RTVCPUHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE m_RTVGPUHandle;

	D3D12_CPU_DESCRIPTOR_HANDLE m_SRVCPUHandle;
	D3D12_GPU_DESCRIPTOR_HANDLE m_SRVGPUHandle;
};