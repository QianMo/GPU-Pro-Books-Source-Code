#pragma once
#include "KGraphicsDevice.h"

class KUAVTex2D
{
public:
	KUAVTex2D();
	~KUAVTex2D();

	void CreateUAVTex2DArray(DXGI_FORMAT tex_format, uint32 width, uint32 height, uint32 depth);

	D3D12_GPU_DESCRIPTOR_HANDLE KUAVTex2D::GetUAVGPUHandle() { return m_UAVGPUHandle; }
	D3D12_CPU_DESCRIPTOR_HANDLE KUAVTex2D::GetUAVCPUHandle() { return m_UAVCPUHandle; }

private:
	ID3D12Resource* m_Resource;

	D3D12_GPU_DESCRIPTOR_HANDLE m_UAVGPUHandle;
	D3D12_CPU_DESCRIPTOR_HANDLE m_UAVCPUHandle;

};