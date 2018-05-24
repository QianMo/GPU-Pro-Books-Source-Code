#include "LightShapeRenderer.h"
#include "Constants.h"
#include "SharedContext.h"
#include "Log.h"
#include "d3dx12.h"

using namespace Log;

LightShapeRenderer::LightShapeRenderer()
{
	HRESULT hr;

	KShader vertexShader(L"../assets/shaders/PointLightShape.vertex", "main", "vs_5_0");
	KShader spotvertexShader(L"../assets/shaders/SpotLightShape.vertex", "main", "vs_5_0");
	KShader pixelShader(L"../assets/shaders/LightShapeRender.pixel", "main", "ps_5_0");

	// create input layout
	D3D12_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R8G8B8A8_SNORM, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
	};

	RootDescriptorRange root_desc_range[] =
	{
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_CBV, SHADER_VISIBILITY::VERTEX },
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::VERTEX },
	};

	m_RootSignature = shared_context.gfx_device->CreateRootSignature(2, root_desc_range);

	D3D12_RASTERIZER_DESC rasterDescFront;
	rasterDescFront.AntialiasedLineEnable = FALSE;
	rasterDescFront.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF;
	rasterDescFront.CullMode = D3D12_CULL_MODE_FRONT;
	rasterDescFront.DepthBias = D3D12_DEFAULT_DEPTH_BIAS;
	rasterDescFront.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP;
	rasterDescFront.DepthClipEnable = TRUE;
	rasterDescFront.FillMode = D3D12_FILL_MODE_SOLID;
	rasterDescFront.ForcedSampleCount = 0;
	rasterDescFront.FrontCounterClockwise = FALSE;
	rasterDescFront.MultisampleEnable = FALSE;
	rasterDescFront.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS;

	D3D12_GRAPHICS_PIPELINE_STATE_DESC PSODescFront;
	ZeroMemory(&PSODescFront, sizeof(PSODescFront));
	PSODescFront.InputLayout = { layout, 1 };
	PSODescFront.pRootSignature = m_RootSignature;
	PSODescFront.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	PSODescFront.PS = { reinterpret_cast<BYTE*>(pixelShader.GetBufferPointer()), pixelShader.GetBufferSize() };
	PSODescFront.RasterizerState = rasterDescFront;
	PSODescFront.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	PSODescFront.BlendState.AlphaToCoverageEnable = FALSE;
	PSODescFront.BlendState.IndependentBlendEnable = FALSE;
	PSODescFront.BlendState.RenderTarget[0].BlendEnable = TRUE;
	PSODescFront.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
	PSODescFront.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
	PSODescFront.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
	PSODescFront.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
	PSODescFront.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_SRC_ALPHA;
	PSODescFront.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
	PSODescFront.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
	PSODescFront.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	PSODescFront.DepthStencilState.DepthEnable = TRUE;
	PSODescFront.DepthStencilState.StencilEnable = FALSE;
	PSODescFront.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
	PSODescFront.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	PSODescFront.SampleMask = UINT_MAX;
	PSODescFront.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	PSODescFront.NumRenderTargets = 1;
	PSODescFront.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
	PSODescFront.DSVFormat = DXGI_FORMAT_D32_FLOAT;
	PSODescFront.SampleDesc.Count = 1;

	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSO));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSO");

	PSODescFront.VS = { reinterpret_cast<BYTE*>(spotvertexShader.GetBufferPointer()), spotvertexShader.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_SpotPSO));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_SpotPSO");
}

LightShapeRenderer::~LightShapeRenderer()
{
	m_PSO->Release();
	m_SpotPSO->Release();
	m_RootSignature->Release();
}

