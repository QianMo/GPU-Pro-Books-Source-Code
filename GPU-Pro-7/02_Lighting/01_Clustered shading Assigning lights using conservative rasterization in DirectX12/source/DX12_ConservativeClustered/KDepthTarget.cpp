#include "KDepthTarget.h"
#include "SharedContext.h"
#include "d3dx12.h"

KDepthTarget::KDepthTarget()
{

}

KDepthTarget::~KDepthTarget()
{
	m_DTResource->Release();
}

void KDepthTarget::CreateDepthTarget(int32 width, int32 height)
{
	// Create the depth texture
	CD3DX12_RESOURCE_DESC depthTexDesc(
		D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		0,		// alignment
		width, height, 1,
		1,		// mip levels
		DXGI_FORMAT_R32_TYPELESS,
		1, 0,	// sample count/quality
		D3D12_TEXTURE_LAYOUT_UNKNOWN,
		D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

	// Performance tip: Tell the runtime at resource creation the desired clear value. 
	D3D12_CLEAR_VALUE clearValue;
	clearValue.Format = DXGI_FORMAT_D32_FLOAT;
	clearValue.DepthStencil.Depth = 1.0f;
	clearValue.DepthStencil.Stencil = 0;

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&depthTexDesc,
		D3D12_RESOURCE_STATE_DEPTH_WRITE,
		&clearValue,
		IID_PPV_ARGS(&m_DTResource));

	D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc;
	ZeroMemory(&depthStencilViewDesc, sizeof(depthStencilViewDesc));
	depthStencilViewDesc.Format = DXGI_FORMAT_D32_FLOAT;
	depthStencilViewDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
	depthStencilViewDesc.Texture2D.MipSlice = 0;

	m_DSVCPUHandle = shared_context.gfx_device->GetDescHeapDSV()->GetNewCPUHandle();
	m_DSVGPUHandle = shared_context.gfx_device->GetDescHeapDSV()->GetGPUHandleAtHead();
	
	shared_context.gfx_device->GetDevice()->CreateDepthStencilView(m_DTResource, &depthStencilViewDesc, m_DSVCPUHandle);

	D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc;
	ZeroMemory(&SRVDesc, sizeof(SRVDesc));
	SRVDesc.Format = DXGI_FORMAT_R32_FLOAT;
	SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	SRVDesc.Texture2D.MipLevels = 1;
	SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	m_SRVCPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_SRVGPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();

	shared_context.gfx_device->GetDevice()->CreateShaderResourceView(m_DTResource, &SRVDesc, m_SRVCPUHandle);
}


