#include "GBuffer.h"
#include "KGraphicsDevice.h"
#include "SharedContext.h"
#include "Log.h"
#include "d3dx12.h"

using namespace Log;

GBuffer::GBuffer()
{
	HRESULT hr;

	KShader vertexShader(L"../assets/shaders/GBuffer.vertex", "main", "vs_5_0");
	KShader pixelShader(L"../assets/shaders/GBuffer.pixel", "main", "ps_5_0");

	RootDescriptorRange root_desc_range[] =
	{
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_CBV, SHADER_VISIBILITY::ALL },
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::PIXEL },
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_SAMPLER, SHADER_VISIBILITY::PIXEL },
	};

	m_RootSignature = shared_context.gfx_device->CreateRootSignature(3, root_desc_range);

	D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODesc;
	ZeroMemory(&PSODesc, sizeof(PSODesc));
	PSODesc.InputLayout = vertexShader.GetInputLayout();
	PSODesc.pRootSignature = m_RootSignature;
	PSODesc.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	PSODesc.PS = { reinterpret_cast<BYTE*>(pixelShader.GetBufferPointer()), pixelShader.GetBufferSize() };
	PSODesc.RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
	PSODesc.RasterizerState.DepthClipEnable = TRUE;
	PSODesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
	PSODesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	PSODesc.DepthStencilState.DepthEnable = TRUE;
	PSODesc.DepthStencilState.StencilEnable = FALSE;
	PSODesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
	PSODesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	PSODesc.SampleMask = UINT_MAX;
	PSODesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	PSODesc.NumRenderTargets = 3;
	PSODesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	PSODesc.RTVFormats[1] = DXGI_FORMAT_R16G16B16A16_FLOAT;
	PSODesc.RTVFormats[2] = DXGI_FORMAT_R16G16B16A16_FLOAT;
	PSODesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	PSODesc.SampleDesc.Count = 1;

	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODesc, IID_PPV_ARGS(&m_Pso));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create PSO");

	int tempw = shared_context.gfx_device->GetWindowWidth();
	int temph = shared_context.gfx_device->GetWindowHeight();

	m_DepthTarget[0].CreateDepthTarget(tempw, temph);
	m_DepthTarget[1].CreateDepthTarget(tempw, temph);

	m_ColorRT[0].CreateRenderTarget(DXGI_FORMAT_R8G8B8A8_UNORM, tempw, temph);
	m_NormalRT[0].CreateRenderTarget(DXGI_FORMAT_R16G16B16A16_FLOAT, tempw, temph);
	m_PositionRT[0].CreateRenderTarget(DXGI_FORMAT_R16G16B16A16_FLOAT, tempw, temph);

	m_ColorRT[1].CreateRenderTarget(DXGI_FORMAT_R8G8B8A8_UNORM, tempw, temph);
	m_NormalRT[1].CreateRenderTarget(DXGI_FORMAT_R16G16B16A16_FLOAT, tempw, temph);
	m_PositionRT[1].CreateRenderTarget(DXGI_FORMAT_R16G16B16A16_FLOAT, tempw, temph);
}

GBuffer::~GBuffer()
{
	m_Pso->Release();
	m_RootSignature->Release();
}


