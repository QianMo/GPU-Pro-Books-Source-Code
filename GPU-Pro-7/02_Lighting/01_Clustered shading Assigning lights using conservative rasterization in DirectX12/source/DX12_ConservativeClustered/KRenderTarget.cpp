#include "KRenderTarget.h"
#include "SharedContext.h"
#include "d3dx12.h"

KRenderTarget::KRenderTarget()
{

}

KRenderTarget::~KRenderTarget()
{
	m_RTResource->Release();
}

void KRenderTarget::CreateRenderTarget(DXGI_FORMAT tex_format, int32 width, int32 height, const float clear_color[4])
{
	// Create the renderTargetTexture
	CD3DX12_RESOURCE_DESC texDesc(
		D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		0,		// alignment
		width, height, 1,
		1,		// mip levels
		tex_format,
		1, 0,	// sample count/quality
		D3D12_TEXTURE_LAYOUT_UNKNOWN,
		D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

	// Performance tip: Tell the runtime at resource creation the desired clear value. 
	D3D12_CLEAR_VALUE clearValue;
	clearValue.Format = tex_format;
	if(clear_color == 0)
	{
		float clearColor[] = {0,0,0,1};
		memcpy(&clearValue.Color[0], &clearColor[0], 4 * sizeof(float));
	}
	else
		memcpy(&clearValue.Color[0], &clear_color[0], 4 * sizeof(float));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&texDesc,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		&clearValue,
		IID_PPV_ARGS(&m_RTResource));

	D3D12_RENDER_TARGET_VIEW_DESC RTVDesc;
	ZeroMemory(&RTVDesc, sizeof(RTVDesc));
	RTVDesc.Format = tex_format;
	RTVDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
	RTVDesc.Texture2D.MipSlice = 0;

	m_RTVCPUHandle = shared_context.gfx_device->GetDescHeapRTV()->GetNewCPUHandle();
	m_RTVGPUHandle = shared_context.gfx_device->GetDescHeapRTV()->GetGPUHandleAtHead();
	
	shared_context.gfx_device->GetDevice()->CreateRenderTargetView(m_RTResource, &RTVDesc, m_RTVCPUHandle);

	D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc;
	ZeroMemory(&SRVDesc, sizeof(SRVDesc));
	SRVDesc.Format = tex_format;
	SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	SRVDesc.Texture2D.MipLevels = 1;
	SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	m_SRVCPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_SRVGPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();

	shared_context.gfx_device->GetDevice()->CreateShaderResourceView(m_RTResource, &SRVDesc, m_SRVCPUHandle);
}


void KRenderTarget::CreateRenderTargetArray(uint32 num_slices, DXGI_FORMAT tex_format, int32 width, int32 height, const float clear_color[4] /*= 0*/)
{
	// Create the renderTargetTexture
	CD3DX12_RESOURCE_DESC texDesc(
		D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		0,		// alignment
		width, height, num_slices,
		1,		// mip levels
		tex_format,
		1, 0,	// sample count/quality
		D3D12_TEXTURE_LAYOUT_UNKNOWN,
		D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

	// Performance tip: Tell the runtime at resource creation the desired clear value. 
	D3D12_CLEAR_VALUE clearValue;
	clearValue.Format = tex_format;
	if (clear_color == 0)
	{
		float clearColor[] = { 0, 0, 0, 1 };
		memcpy(&clearValue.Color[0], &clearColor[0], 4 * sizeof(float));
	}
	else
		memcpy(&clearValue.Color[0], &clear_color[0], 4 * sizeof(float));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&texDesc,
		D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		&clearValue,
		IID_PPV_ARGS(&m_RTResource));

	D3D12_RENDER_TARGET_VIEW_DESC RTVDesc;
	ZeroMemory(&RTVDesc, sizeof(RTVDesc));
	RTVDesc.Format = tex_format;
	RTVDesc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
	RTVDesc.Texture2DArray.ArraySize = num_slices;

	m_RTVCPUHandle = shared_context.gfx_device->GetDescHeapRTV()->GetNewCPUHandle();
	m_RTVGPUHandle = shared_context.gfx_device->GetDescHeapRTV()->GetGPUHandleAtHead();

	shared_context.gfx_device->GetDevice()->CreateRenderTargetView(m_RTResource, &RTVDesc, m_RTVCPUHandle);

	D3D12_SHADER_RESOURCE_VIEW_DESC SRVDesc;
	ZeroMemory(&SRVDesc, sizeof(SRVDesc));
	SRVDesc.Format = tex_format;
	SRVDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DARRAY;
	SRVDesc.Texture2DArray.ArraySize = num_slices;
	SRVDesc.Texture2DArray.MipLevels = 1;
	SRVDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	m_SRVCPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_SRVGPUHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();

	shared_context.gfx_device->GetDevice()->CreateShaderResourceView(m_RTResource, &SRVDesc, m_SRVCPUHandle);
}

