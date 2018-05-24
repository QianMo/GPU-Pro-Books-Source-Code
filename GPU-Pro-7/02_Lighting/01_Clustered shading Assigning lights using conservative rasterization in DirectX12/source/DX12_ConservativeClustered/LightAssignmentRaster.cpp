#include "LightAssignmentRaster.h"
#include "Constants.h"
#include <vector>
#include "SharedContext.h"
#include "Log.h"
#include "d3dx12.h"

using namespace Log;

LightAssignmentRaster::LightAssignmentRaster(ID3D12GraphicsCommandList* gfx_command_list)
{
	HRESULT hr;

	KShader vertexShaderSpot(L"../assets/shaders/LASpotLight.vertex", "main", "vs_5_0");
	KShader vertexShader(L"../assets/shaders/LAPointLight.vertex", "main", "vs_5_0");
	KShader pixelShaderFront(L"../assets/shaders/LAfront.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES);
	KShader pixelShaderFrontLinear(L"../assets/shaders/LAfront.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES_LINEAR);
	KShader geometryShader(L"../assets/shaders/LA.geometry", "main", "gs_5_0");
	KShader oldpixelShaderFront(L"../assets/shaders/LAold.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES);
	KShader oldpixelShaderFrontLinear(L"../assets/shaders/LAold.pixel", "main", "ps_5_0", Constants::SHADER_DEFINES_LINEAR);
	KShader oldgeometryShader(L"../assets/shaders/LAold.geometry", "main", "gs_5_0");

	// create input layout
	D3D12_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R8G8B8A8_SNORM, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
	};

	RootDescriptorRange root_desc_range[] =
	{
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_CBV, SHADER_VISIBILITY::ALL },
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::VERTEX },
	};

	m_RootSignature = shared_context.gfx_device->CreateRootSignature(2, root_desc_range);
	
	D3D12_RASTERIZER_DESC rasterDescFront;
	rasterDescFront.AntialiasedLineEnable = FALSE;
	rasterDescFront.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
	rasterDescFront.CullMode = D3D12_CULL_MODE_NONE;
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
	PSODescFront.GS = { reinterpret_cast<BYTE*>(geometryShader.GetBufferPointer()), geometryShader.GetBufferSize() };
	PSODescFront.PS = { reinterpret_cast<BYTE*>(pixelShaderFront.GetBufferPointer()), pixelShaderFront.GetBufferSize() };
	PSODescFront.RasterizerState = rasterDescFront;
	PSODescFront.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
	PSODescFront.BlendState.RenderTarget[0].BlendEnable = TRUE;
	PSODescFront.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_MIN;
	PSODescFront.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
	PSODescFront.DepthStencilState.DepthEnable = FALSE;
	PSODescFront.DepthStencilState.StencilEnable = FALSE;
	PSODescFront.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
	PSODescFront.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
	PSODescFront.SampleMask = UINT_MAX;
	PSODescFront.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	PSODescFront.NumRenderTargets = 1;
	PSODescFront.RTVFormats[0] = DXGI_FORMAT_R8G8_UNORM;
	PSODescFront.SampleDesc.Count = 1;

	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOPoint));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOPoint");

	//Reuse pso desc
	PSODescFront.VS = { reinterpret_cast<BYTE*>(vertexShaderSpot.GetBufferPointer()), vertexShaderSpot.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOSpot));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOSpot");

	PSODescFront.PS = { reinterpret_cast<BYTE*>(pixelShaderFrontLinear.GetBufferPointer()), pixelShaderFrontLinear.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOSpotLinear));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOSpotLinear");

		PSODescFront.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOPointLinear));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOPointLinear");


	//Create old pipeline shaders
	PSODescFront.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	PSODescFront.GS = { reinterpret_cast<BYTE*>(oldgeometryShader.GetBufferPointer()), oldgeometryShader.GetBufferSize() };
	PSODescFront.PS = { reinterpret_cast<BYTE*>(oldpixelShaderFront.GetBufferPointer()), oldpixelShaderFront.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOPointOld));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOPointOld");

	PSODescFront.VS = { reinterpret_cast<BYTE*>(vertexShaderSpot.GetBufferPointer()), vertexShaderSpot.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOSpotOld));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOSpotOld");

	PSODescFront.PS = { reinterpret_cast<BYTE*>(oldpixelShaderFrontLinear.GetBufferPointer()), oldpixelShaderFrontLinear.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOSpotOldLinear));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOSpotLinearOld");

	PSODescFront.VS = { reinterpret_cast<BYTE*>(vertexShader.GetBufferPointer()), vertexShader.GetBufferSize() };
	hr = shared_context.gfx_device->GetDevice()->CreateGraphicsPipelineState(&PSODescFront, IID_PPV_ARGS(&m_PSOPointOldLinear));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_PSOPointLinearOld");

	

	//////////////////////////////////////////////////////////////////////////
	//COMPUTE SHADER SET UP

	KShader computeShader(L"../assets/shaders/LA.compute", "main", "cs_5_1", Constants::SHADER_DEFINES);

	RootDescriptorRange compute_root_desc_range[] =
	{
		{ 1, ROOT_DESCRIPTOR_TYPE::RANGE_SRV, SHADER_VISIBILITY::ALL },
		{ 2, ROOT_DESCRIPTOR_TYPE::RANGE_UAV, SHADER_VISIBILITY::ALL },
		{ 1, ROOT_DESCRIPTOR_TYPE::CBV, SHADER_VISIBILITY::ALL },
	};

	m_ComputeRootSig = shared_context.gfx_device->CreateRootSignature(3, compute_root_desc_range);

	D3D12_COMPUTE_PIPELINE_STATE_DESC computePSODesc;
	ZeroMemory(&computePSODesc, sizeof(computePSODesc));
	computePSODesc.pRootSignature = m_ComputeRootSig;
	computePSODesc.CS = { reinterpret_cast<BYTE*>(computeShader.GetBufferPointer()), computeShader.GetBufferSize() };

	hr = shared_context.gfx_device->GetDevice()->CreateComputePipelineState(&computePSODesc, IID_PPV_ARGS(&m_ComputePSO));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create m_ComputePSO");


	//Set up Linked Light List buffers.
	//For building the light list:
	//RWByteAddressBuffer UAV containing a uint for every cluster(clustsX*clustsY*clustsZ*sizeof(uint32))
	//--Clear this to some not-ever-used value each frame
	//RWStructuredBuffer UAV containing the linked list nodes. Init to some safe size depending on max nr of lights.
	//--Add a UAVCounter and set it to 0 each frame

	//For shading using the light list:
	//Standard shader resource views of the resources above
	m_SobUAVHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_SobUAVHandleGPU = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();
	m_LLLUAVHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_LLLUAVHandleGPU = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();

	m_SobSRVHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_SobSRVHandleGPU = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();
	m_LLLSRVHandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetNewCPUHandle();
	m_LLLSRVHandleGPU = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();

	//////////////////////////////////////////////////////////////////////////
	//StartOffsetBuffer
	//////////////////////////////////////////////////////////////////////////
	CD3DX12_RESOURCE_DESC start_buffer_resource_desc(D3D12_RESOURCE_DIMENSION_BUFFER, 0, (Constants::NR_OF_CLUSTS) * sizeof(uint32), 1, 1, 1,
		DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE);
	CD3DX12_RESOURCE_DESC start_buffer_uav_resource_desc = start_buffer_resource_desc;
	start_buffer_uav_resource_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&start_buffer_uav_resource_desc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_StartOffsetBuffer));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&start_buffer_resource_desc,
		D3D12_RESOURCE_STATE_COPY_SOURCE,
		nullptr,
		IID_PPV_ARGS(&m_StartOffsetBufferClear));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&start_buffer_resource_desc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_SobReadBackRes[0]));


	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&start_buffer_resource_desc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_SobReadBackRes[1]));


	//Raw buffer view
	D3D12_UNORDERED_ACCESS_VIEW_DESC uav_view_desc;
	ZeroMemory(&uav_view_desc, sizeof(uav_view_desc));
	uav_view_desc.Format = DXGI_FORMAT_R32_TYPELESS; //Needs to be DXGI_FORMAT_R32_TYPELESS for RAW
	uav_view_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	uav_view_desc.Buffer.FirstElement = 0;
	uav_view_desc.Buffer.NumElements = Constants::NR_OF_CLUSTS;
	uav_view_desc.Buffer.StructureByteStride = 0; //Needs to be zero, otherwise interpreted as structured buffer
	uav_view_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
	uav_view_desc.Buffer.CounterOffsetInBytes = 0; //Needs to be zero if not used

	shared_context.gfx_device->GetDevice()->CreateUnorderedAccessView(m_StartOffsetBuffer, nullptr, &uav_view_desc, m_SobUAVHandle);

	//Also set up SRV for it
	D3D12_SHADER_RESOURCE_VIEW_DESC sob_srv_view_desc;
	ZeroMemory(&sob_srv_view_desc, sizeof(sob_srv_view_desc));
	sob_srv_view_desc.Format = DXGI_FORMAT_R32_UINT;
	sob_srv_view_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	sob_srv_view_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	sob_srv_view_desc.Buffer.FirstElement = 0;
	sob_srv_view_desc.Buffer.NumElements = Constants::NR_OF_CLUSTS;
	sob_srv_view_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	sob_srv_view_desc.Buffer.StructureByteStride = 0;

	shared_context.gfx_device->GetDevice()->CreateShaderResourceView(m_StartOffsetBuffer, &sob_srv_view_desc, m_SobSRVHandle);

	//////////////////////////////////////////////////////////////////////////
	//UAV COUNTER  /////////////////////////////////////////////////////////// 
	//////////////////////////////////////////////////////////////////////////
	CD3DX12_RESOURCE_DESC uav_counter_resource_desc(D3D12_RESOURCE_DIMENSION_BUFFER, 0, sizeof(uint32), 1, 1, 1,
		DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE);
	CD3DX12_RESOURCE_DESC uav_counter_uav_resource_desc = uav_counter_resource_desc;
	uav_counter_uav_resource_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&uav_counter_uav_resource_desc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_StartOffsetBufferCounter));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&uav_counter_resource_desc,
		D3D12_RESOURCE_STATE_COPY_SOURCE,
		nullptr,
		IID_PPV_ARGS(&m_StartOffsetBufferCounterClear));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&uav_counter_resource_desc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_UAVCounterReadBackRes[0]));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&uav_counter_resource_desc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_UAVCounterReadBackRes[1]));


	//////////////////////////////////////////////////////////////////////////
	//Linked Light List //////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	CD3DX12_RESOURCE_DESC lll_resource_desc(D3D12_RESOURCE_DIMENSION_BUFFER, 0, (Constants::LIGHT_INDEX_LIST_COUNT) * sizeof(uint32) * 2, 1, 1, 1,
		DXGI_FORMAT_UNKNOWN, 1, 0, D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12_RESOURCE_FLAG_NONE);
	CD3DX12_RESOURCE_DESC lll_uav_resource_desc = lll_resource_desc;
	lll_uav_resource_desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&lll_uav_resource_desc,
		D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		nullptr,
		IID_PPV_ARGS(&m_LinkedIndexList));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&lll_resource_desc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_LLLReadBackRes[0]));

	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&lll_resource_desc,
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_LLLReadBackRes[1]));

	//Structured buffer uav
	D3D12_UNORDERED_ACCESS_VIEW_DESC lll_uav_view_desc;
	ZeroMemory(&lll_uav_view_desc, sizeof(lll_uav_view_desc));
	lll_uav_view_desc.Format = DXGI_FORMAT_UNKNOWN; //Needs to be UNKNOWN for structured buffer
	lll_uav_view_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
	lll_uav_view_desc.Buffer.FirstElement = 0;
	lll_uav_view_desc.Buffer.NumElements = Constants::LIGHT_INDEX_LIST_COUNT;
	lll_uav_view_desc.Buffer.StructureByteStride = sizeof(uint32) * 2; //2 uint32s in struct
	lll_uav_view_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE; //Not a raw view
	lll_uav_view_desc.Buffer.CounterOffsetInBytes = 0; //First element in UAV counter resource

	shared_context.gfx_device->GetDevice()->CreateUnorderedAccessView(m_LinkedIndexList, m_StartOffsetBufferCounter, &lll_uav_view_desc, m_LLLUAVHandle);

	//Also set up SRV for it
	D3D12_SHADER_RESOURCE_VIEW_DESC lll_srv_view_desc;
	ZeroMemory(&lll_srv_view_desc, sizeof(lll_srv_view_desc));
	lll_srv_view_desc.Format = DXGI_FORMAT_UNKNOWN;
	lll_srv_view_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	lll_srv_view_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	lll_srv_view_desc.Buffer.FirstElement = 0;
	lll_srv_view_desc.Buffer.NumElements = Constants::LIGHT_INDEX_LIST_COUNT;
	lll_srv_view_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
	lll_srv_view_desc.Buffer.StructureByteStride = sizeof(uint32) * 2; //2 unit32s in struct

	shared_context.gfx_device->GetDevice()->CreateShaderResourceView(m_LinkedIndexList, &lll_srv_view_desc, m_LLLSRVHandle);

	//////////////////////////////////////////////////////////////////////////
	//Render targets for conservative raster    //////////////////////////////
	//////////////////////////////////////////////////////////////////////////

	m_pointlightShellTarget[0].CreateRenderTargetArray(Constants::NR_POINTLIGHTS, DXGI_FORMAT_R8G8_UNORM, Constants::NR_X_CLUSTS, Constants::NR_Y_CLUSTS, COLOR::WHITE);
	m_pointlightShellTarget[1].CreateRenderTargetArray(Constants::NR_POINTLIGHTS, DXGI_FORMAT_R8G8_UNORM, Constants::NR_X_CLUSTS, Constants::NR_Y_CLUSTS, COLOR::WHITE);

	m_spotlightShellTarget[0].CreateRenderTargetArray(Constants::NR_SPOTLIGHTS, DXGI_FORMAT_R8G8_UNORM, Constants::NR_X_CLUSTS, Constants::NR_Y_CLUSTS, COLOR::WHITE);
	m_spotlightShellTarget[1].CreateRenderTargetArray(Constants::NR_SPOTLIGHTS, DXGI_FORMAT_R8G8_UNORM, Constants::NR_X_CLUSTS, Constants::NR_Y_CLUSTS, COLOR::WHITE);

	//////////////////////////////////////////////////////////////////////////
	//Fill clear buffers    //////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////
	{
		shared_context.gfx_device->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
			D3D12_HEAP_FLAG_NONE,
			&start_buffer_resource_desc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_StartOffsetBufferClearUpload));

		std::vector<uint32> tempClearData(Constants::NR_OF_CLUSTS, 0x3FFFFFFF);

		D3D12_SUBRESOURCE_DATA clearData = {};
		clearData.pData = &tempClearData[0];
		clearData.RowPitch = Constants::NR_OF_CLUSTS * sizeof(uint32);
		clearData.SlicePitch = clearData.RowPitch;

		shared_context.gfx_device->TransitionResource(gfx_command_list, m_StartOffsetBufferClear, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
		UpdateSubresources<1>(gfx_command_list, m_StartOffsetBufferClear, m_StartOffsetBufferClearUpload, 0, 0, 1, &clearData);
		shared_context.gfx_device->TransitionResource(gfx_command_list, m_StartOffsetBufferClear, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
	}

	{
		shared_context.gfx_device->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
			D3D12_HEAP_FLAG_NONE,
			&uav_counter_resource_desc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&m_StartOffsetBufferCounterClearUpload));

		uint32 tempClearData = 0;

		D3D12_SUBRESOURCE_DATA clearData = {};
		clearData.pData = &tempClearData;
		clearData.RowPitch = sizeof(uint32);
		clearData.SlicePitch = clearData.RowPitch;

		shared_context.gfx_device->TransitionResource(gfx_command_list, m_StartOffsetBufferCounterClear, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
		UpdateSubresources<1>(gfx_command_list, m_StartOffsetBufferCounterClear, m_StartOffsetBufferCounterClearUpload, 0, 0, 1, &clearData);
		shared_context.gfx_device->TransitionResource(gfx_command_list, m_StartOffsetBufferCounterClear, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
	}
}

LightAssignmentRaster::~LightAssignmentRaster()
{
	m_PSOPoint->Release();
	m_PSOSpot->Release();
	m_PSOSpotOld->Release();
	m_PSOPointOld->Release();
	m_PSOPointLinear->Release();
	m_PSOSpotLinear->Release();
	m_PSOPointOldLinear->Release();
	m_PSOSpotOldLinear->Release();
	m_RootSignature->Release();
	m_ComputePSO->Release();
	m_ComputeRootSig->Release();
	m_StartOffsetBuffer->Release();
	m_LinkedIndexList->Release();
	m_StartOffsetBufferCounter->Release();
	m_StartOffsetBufferCounterClear->Release();
	m_StartOffsetBufferCounterClearUpload->Release();
	m_StartOffsetBufferClear->Release();
	m_StartOffsetBufferClearUpload->Release();
	m_SobReadBackRes[0]->Release();
	m_LLLReadBackRes[0]->Release();
	m_UAVCounterReadBackRes[0]->Release();
	m_SobReadBackRes[1]->Release();
	m_LLLReadBackRes[1]->Release();
	m_UAVCounterReadBackRes[1]->Release();
}

ID3D12Resource* LightAssignmentRaster::GetStartOffsetReadResource()		{ return m_SobReadBackRes[!shared_context.gfx_device->GetSwapIndex()]; }
ID3D12Resource* LightAssignmentRaster::GetLinkedLightListReadResource() { return m_LLLReadBackRes[!shared_context.gfx_device->GetSwapIndex()]; }
ID3D12Resource* LightAssignmentRaster::GetUAVCounterReadResource()		{ return m_UAVCounterReadBackRes[!shared_context.gfx_device->GetSwapIndex()]; }

void LightAssignmentRaster::ClearUAVs(ID3D12GraphicsCommandList* gfx_command_list)
{
	ID3D12Resource* resources[] =  {m_StartOffsetBufferCounter, m_StartOffsetBuffer};
	shared_context.gfx_device->TransitionResources(2, gfx_command_list, resources, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
	gfx_command_list->CopyResource(m_StartOffsetBufferCounter, m_StartOffsetBufferCounterClear);
	gfx_command_list->CopyResource(m_StartOffsetBuffer, m_StartOffsetBufferClear);
	shared_context.gfx_device->TransitionResources(2, gfx_command_list, resources, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}

void LightAssignmentRaster::ReadBackDebugData(ID3D12GraphicsCommandList* gfx_command_list, int32 swap_index)
{
	ID3D12Resource* resources[] = { m_StartOffsetBufferCounter, m_StartOffsetBuffer, m_LinkedIndexList };
	shared_context.gfx_device->TransitionResources(3, gfx_command_list, resources, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
	gfx_command_list->CopyResource(m_SobReadBackRes[swap_index], m_StartOffsetBuffer);
	gfx_command_list->CopyResource(m_UAVCounterReadBackRes[swap_index], m_StartOffsetBufferCounter);
	gfx_command_list->CopyResource(m_LLLReadBackRes[swap_index], m_LinkedIndexList);
	shared_context.gfx_device->TransitionResources(3, gfx_command_list, resources, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
}