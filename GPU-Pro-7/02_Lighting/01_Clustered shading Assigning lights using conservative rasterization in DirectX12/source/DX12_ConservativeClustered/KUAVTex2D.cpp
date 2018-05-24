#include "KUAVTex2D.h"
#include "SharedContext.h"

KUAVTex2D::KUAVTex2D()
{

}

KUAVTex2D::~KUAVTex2D()
{
	m_Resource->Release();
}

void KUAVTex2D::CreateUAVTex2DArray(DXGI_FORMAT tex_format, uint32 width, uint32 height, uint32 depth)
{
	// Create the renderTargetTexture
	CD3DX12_RESOURCE_DESC texDesc(
		D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		0,		// alignment
		width, height, depth,
		1,		// mip levels
		tex_format,
		1, 0,	// sample count/quality
		D3D12_TEXTURE_LAYOUT_UNKNOWN,
		D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&texDesc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_Resource));

	D3D12_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
	ZeroMemory(&UAVDesc, sizeof(UAVDesc));
	UAVDesc.Format = tex_format;
	UAVDesc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
	UAVDesc.Texture2DArray.ArraySize = depth;
	UAVDesc.Texture2DArray.FirstArraySlice = 0;
	UAVDesc.Texture2DArray.MipSlice = 0;

	m_UAVCPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_UAVGPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();

	shared_context.gfx_device->GetDevice()->CreateUnorderedAccessView(m_Resource, nullptr, &UAVDesc, m_UAVCPUHandle);
}


