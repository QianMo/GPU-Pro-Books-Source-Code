#include "KGraphicsDevice.h"
#include "SharedContext.h"
#include "Log.h"
#include "d3dx12.h"

using namespace Log;

KGraphicsDevice::KGraphicsDevice()
	: m_SwapIndex(0)
{

}

KGraphicsDevice::~KGraphicsDevice()
{
	//Swap chain can not be full screen when released
	m_SwapChain->SetFullscreenState(FALSE, nullptr);
	m_CommandQueue->Release();
	m_RenderTarget[0]->Release();
	m_RenderTarget[1]->Release();
	m_CommandAllocator->Release();
	m_SwapChain->Release();
	m_Device->Release();
	m_TimeStampQueryHeap->Release();
	m_TimeStampQueryReadBackRes->Release();
	m_Fence->Release();
	CloseHandle(m_HandleEvent);
}

HRESULT KGraphicsDevice::Init(int32 window_width, int32 window_height)
{
	HRESULT hr;

	m_WindowWidth = window_width;
	m_WindowHeight = window_height;

	//Setup SDL window
	HWND handle;

	if (SDL_Init(SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0)
	{
		PRINT(LogLevel::FATAL_ERROR, "SDL_Init failed: %s", SDL_GetError());
	}

	IMG_Init(IMG_INIT_JPG | IMG_INIT_PNG | IMG_INIT_TIF);

	m_MainWindow = SDL_CreateWindow("Conservative Clustered Shading DirectX12", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, m_WindowWidth, m_WindowHeight, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);

	SDL_SysWMinfo info;
	//Must init info struct with SDL version info, see documentation for explanation
	SDL_VERSION(&info.version);

	if (SDL_GetWindowWMInfo(m_MainWindow, &info))
		handle = info.info.win.window;
	else
		PRINT(LogLevel::FATAL_ERROR, "Failed to get WMInfo: %s", SDL_GetError());

	uint32 flags = 0;

#ifdef _DEBUG
	{
		ID3D12Debug* debugController;
		D3D12GetDebugInterface(IID_PPV_ARGS(&debugController));
		debugController->EnableDebugLayer();

		if (debugController)
			debugController->Release();
	}
#endif

	DXGI_SWAP_CHAIN_DESC descSwapChain;
	ZeroMemory(&descSwapChain, sizeof(descSwapChain));
	descSwapChain.BufferCount = 2;
	descSwapChain.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
	descSwapChain.BufferUsage = DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT;
	descSwapChain.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
	descSwapChain.OutputWindow = handle;
	descSwapChain.SampleDesc.Count = 1;
	descSwapChain.Windowed = true;

	hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_Device));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create D3D12 Device");

	D3D12_FEATURE_DATA_D3D12_OPTIONS opts;
	hr = m_Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to CheckFeatureSupport");
	
	//Print hardware opts
	PrintHWopts(opts);

	uint32 node_count = m_Device->GetNodeCount();
	PRINT(LogLevel::DEBUG_PRINT, "Device node count: %d", node_count);

	IDXGIFactory* dxgifactory;
	hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgifactory));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create IDXGIFactory");

	D3D12_COMMAND_QUEUE_DESC queueDesc;
	ZeroMemory(&queueDesc, sizeof(queueDesc));
	queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
	queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
	hr = m_Device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_CommandQueue));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create Command Queue");

	hr = dxgifactory->CreateSwapChain(m_CommandQueue, &descSwapChain, &m_SwapChain);
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to create SwapChain");

	
	dxgifactory->Release();

	//Set up descriptor heaps
	m_DescHeapCBV_SRV.CreateDescriptorHeap(50, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
	m_DescHeapDSV.CreateDescriptorHeap(2, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
	m_DescHeapRTV.CreateDescriptorHeap(20, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
	m_DescHeapSampler.CreateDescriptorHeap(1, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);

	m_Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_CommandAllocator));

	m_Device->SetStablePowerState(TRUE);

	hr = m_CommandQueue->GetTimestampFrequency(&m_Freq);
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed GetTimestampFrequency");

	m_ViewPort = { 0.0f, 0.0f, (float)m_WindowWidth, (float)m_WindowHeight, 0.0f, 1.0f };
	m_ScissorRect = { 0, 0, m_WindowWidth, m_WindowHeight };

	m_RTDescriptor[0] = m_DescHeapRTV.GetNewCPUHandle();
	m_RTDescriptor[1] = m_DescHeapRTV.GetNewCPUHandle();

	//Get back buffer and create RTVs
	m_SwapChain->GetBuffer(0, IID_PPV_ARGS(&m_RenderTarget[0]));
	m_Device->CreateRenderTargetView(m_RenderTarget[0], nullptr, m_RTDescriptor[0]);
	m_RenderTarget[0]->SetName(L"RENDER TARGET");

	m_SwapChain->GetBuffer(1, IID_PPV_ARGS(&m_RenderTarget[1]));
	m_Device->CreateRenderTargetView(m_RenderTarget[1], nullptr, m_RTDescriptor[1]);

	//Create time stamp query heap
	const uint32 num_time_queries = 24;
	D3D12_QUERY_HEAP_DESC desc;
	desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
	desc.Count = num_time_queries;
	desc.NodeMask = 0;

	hr = m_Device->CreateQueryHeap(&desc, IID_PPV_ARGS(&m_TimeStampQueryHeap));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to CreateQueryHeap");

	m_Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_READBACK, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(num_time_queries * sizeof(uint64)),
		D3D12_RESOURCE_STATE_COPY_DEST,
		nullptr,
		IID_PPV_ARGS(&m_TimeStampQueryReadBackRes));
	if (FAILED(hr))
		PRINT(LogLevel::FATAL_ERROR, "Failed to CreateCommittedResource for query readback buffer");

	//Create fence
	m_Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_Fence));
	m_CurrentFence = 1;

	m_HandleEvent = CreateEventEx(nullptr, FALSE, FALSE, EVENT_ALL_ACCESS);

	return hr;
}

void KGraphicsDevice::TransitionResource(ID3D12GraphicsCommandList* gfx_command_list, ID3D12Resource* resource, D3D12_RESOURCE_STATES state_before, D3D12_RESOURCE_STATES state_after)
{
	D3D12_RESOURCE_BARRIER descBarrier = {};
	descBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	descBarrier.Transition.pResource = resource;
	descBarrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	descBarrier.Transition.StateBefore = state_before;
	descBarrier.Transition.StateAfter = state_after;
	gfx_command_list->ResourceBarrier(1, &descBarrier);
}

void KGraphicsDevice::TransitionResources(int32 num_resources, ID3D12GraphicsCommandList* gfx_command_list, ID3D12Resource** resource, D3D12_RESOURCE_STATES state_before, D3D12_RESOURCE_STATES state_after)
{
	std::vector<D3D12_RESOURCE_BARRIER> descBarrier(num_resources, D3D12_RESOURCE_BARRIER());
	for (int32 i = 0; i < num_resources; ++i)
	{
		descBarrier[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		descBarrier[i].Transition.pResource = resource[i];
		descBarrier[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		descBarrier[i].Transition.StateBefore = state_before;
		descBarrier[i].Transition.StateAfter = state_after;
	}
	gfx_command_list->ResourceBarrier(num_resources, &descBarrier[0]);	
}

void KGraphicsDevice::Present()
{
	m_SwapChain->Present(0, 0);

	m_SwapIndex = (1 + m_SwapIndex) % 2;

	WaitForGPU();
}

void KGraphicsDevice::WaitForGPU()
{
	const uint64 fence = m_CurrentFence;
	m_CommandQueue->Signal(m_Fence, fence);
	++m_CurrentFence;

	if (m_Fence->GetCompletedValue() < fence)
	{
		m_Fence->SetEventOnCompletion(fence, m_HandleEvent);
		WaitForSingleObject(m_HandleEvent, INFINITE);
	}
}

void KGraphicsDevice::SetTimeStampQuery(uint32 timestamp_query, ID3D12GraphicsCommandList* gfx_command_list)
{
	gfx_command_list->EndQuery(m_TimeStampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, timestamp_query);
	gfx_command_list->ResolveQueryData(m_TimeStampQueryHeap, D3D12_QUERY_TYPE_TIMESTAMP, timestamp_query, 1, m_TimeStampQueryReadBackRes, timestamp_query * sizeof(uint64));
}

uint64 KGraphicsDevice::QueryTimeStamp(uint32 timestamp_query)
{
	m_TimeStampQueryReadBackRes->Map(0, &CD3DX12_RANGE(0, 0), reinterpret_cast<void**>(&m_TimeStampQueryMem));
	uint64 data = ((uint64*)m_TimeStampQueryMem)[timestamp_query];
	m_TimeStampQueryReadBackRes->Unmap(0, nullptr);
	return data;
}

uint64 KGraphicsDevice::GetFreq()
{
	return m_Freq;
}

KBuffer KGraphicsDevice::CreateBuffer(uint32 num_elements, uint32 element_size, KBufferType type, D3D12_HEAP_TYPE heap_type /*= D3D12_HEAP_TYPE_DEFAULT*/)
{
	KBuffer buffer;

	UINT64 aligned_width = element_size * num_elements;
	if (type == KBufferType::CONSTANT)
		aligned_width = (aligned_width + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1) & ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1);

	m_Device->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(heap_type, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(aligned_width),
		D3D12_RESOURCE_STATE_GENERIC_READ,
		nullptr,
		IID_PPV_ARGS(&buffer.resource));

	if(heap_type == D3D12_HEAP_TYPE_UPLOAD)
		buffer.resource->Map(0, nullptr, reinterpret_cast<void**>(&buffer.mem));

	if(type == KBufferType::STRUCTURED)
		CreateShaderResourceView(&buffer, num_elements, element_size, type);
	if(type == KBufferType::CONSTANT)
		CreateConstantBufferView(&buffer, num_elements, element_size);

	return buffer;
}

void KGraphicsDevice::CreateShaderResourceView(KBuffer* buffer, uint32 num_elements, uint32 element_size, KBufferType type)
{
	//Set up shader resource view
	D3D12_BUFFER_SRV buffer_srv;
	buffer_srv.FirstElement = 0;
	buffer_srv.NumElements = num_elements;
	buffer_srv.StructureByteStride = element_size;
	buffer_srv.Flags = D3D12_BUFFER_SRV_FLAG_NONE;

	D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
	ZeroMemory(&srv_desc, sizeof(srv_desc));
	srv_desc.Format = DXGI_FORMAT_UNKNOWN;
	srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
	srv_desc.Buffer = buffer_srv;
	srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;

	buffer->srv.cpu_handle = m_DescHeapCBV_SRV.GetNewCPUHandle();
	buffer->srv.gpu_handle = m_DescHeapCBV_SRV.GetGPUHandleAtHead();

	m_Device->CreateShaderResourceView(buffer->resource, &srv_desc, buffer->srv.cpu_handle);
}

void KGraphicsDevice::CreateConstantBufferView(KBuffer* buffer, uint32 num_elements, uint32 element_size)
{
	uint32 size_in_bytes = num_elements * element_size;

	size_in_bytes = (size_in_bytes + D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1) & ~(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT - 1);

	D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc;
	cbv_desc.BufferLocation = buffer->resource->GetGPUVirtualAddress();
	cbv_desc.SizeInBytes = size_in_bytes;

	buffer->cbv.cpu_handle = m_DescHeapCBV_SRV.GetNewCPUHandle();
	buffer->cbv.gpu_handle = m_DescHeapCBV_SRV.GetGPUHandleAtHead();

	m_Device->CreateConstantBufferView(&cbv_desc, buffer->cbv.cpu_handle);
}

void KGraphicsDevice::PrintHWopts(D3D12_FEATURE_DATA_D3D12_OPTIONS& opts)
{
	//Double Precision Float Shader Ops
	if (opts.DoublePrecisionFloatShaderOps)
		PRINT(LogLevel::HELP_PRINT, "DoublePrecisionFloatShaderOps: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "DoublePrecisionFloatShaderOps: FALSE");

	//Output Merger Logic Op
	if (opts.OutputMergerLogicOp)
		PRINT(LogLevel::HELP_PRINT, "OutputMergerLogicOp: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "OutputMergerLogicOp: FALSE");

	//Min Precision Support
	switch (opts.MinPrecisionSupport)
	{
	case D3D12_SHADER_MIN_PRECISION_SUPPORT_NONE:
		PRINT(LogLevel::HELP_PRINT, "MinPrecisionSupport: D3D12_SHADER_MIN_PRECISION_NONE");
		break;
	case D3D12_SHADER_MIN_PRECISION_SUPPORT_10_BIT:
		PRINT(LogLevel::HELP_PRINT, "MinPrecisionSupport: D3D12_SHADER_MIN_PRECISION_10_BIT");
		break;
	case D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT:
		PRINT(LogLevel::HELP_PRINT, "MinPrecisionSupport: D3D12_SHADER_MIN_PRECISION_16_BIT");
		break;
	default:
		break;
	}

	//Tiled Resource Tier
	switch (opts.TiledResourcesTier)
	{
	case D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED:
		PRINT(LogLevel::HELP_PRINT, "TiledResourcesTier: D3D12_TILED_RESOURCES_NOT_SUPPORTED");
		break;
	case D3D12_TILED_RESOURCES_TIER_1:
		PRINT(LogLevel::HELP_PRINT, "TiledResourcesTier: D3D12_TILED_RESOURCES_TIER_1");
		break;
	case D3D12_TILED_RESOURCES_TIER_2:
		PRINT(LogLevel::HELP_PRINT, "TiledResourcesTier: D3D12_TILED_RESOURCES_TIER_2");
		break;
	case D3D12_TILED_RESOURCES_TIER_3:
		PRINT(LogLevel::HELP_PRINT, "TiledResourcesTier: D3D12_TILED_RESOURCES_TIER_3");
		break;
	default:
		break;
	}

	//Resource Binding Tier
	switch (opts.ResourceBindingTier)
	{
	case D3D12_RESOURCE_BINDING_TIER_1:
		PRINT(LogLevel::HELP_PRINT, "ResourceBindingTier: D3D12_RESOURCE_BINDING_TIER_1");
		break;
	case D3D12_RESOURCE_BINDING_TIER_2:
		PRINT(LogLevel::HELP_PRINT, "ResourceBindingTier: D3D12_RESOURCE_BINDING_TIER_2");
		break;
	case D3D12_RESOURCE_BINDING_TIER_3:
		PRINT(LogLevel::HELP_PRINT, "ResourceBindingTier: D3D12_RESOURCE_BINDING_TIER_3");
		break;
	default:
		break;
	}

	//PS Specified Stencil Ref Supported
	if (opts.PSSpecifiedStencilRefSupported)
		PRINT(LogLevel::HELP_PRINT, "PSSpecifiedStencilRefSupported: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "PSSpecifiedStencilRefSupported: FALSE");

	//Typed UAV Load Additional Formats
	if (opts.TypedUAVLoadAdditionalFormats)
		PRINT(LogLevel::HELP_PRINT, "TypedUAVLoadAdditionalFormats: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "TypedUAVLoadAdditionalFormats: FALSE");

	//ROVs Supported
	if (opts.ROVsSupported)
		PRINT(LogLevel::HELP_PRINT, "ROVsSupported: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "ROVsSupported: FALSE");

	//Conservative Rasterization Tier
	switch (opts.ConservativeRasterizationTier)
	{
	case D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED:
		PRINT(LogLevel::HELP_PRINT, "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_NOT_SUPPORTED");
		break;
	case D3D12_CONSERVATIVE_RASTERIZATION_TIER_1:
		PRINT(LogLevel::HELP_PRINT, "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_TIER_1");
		break;
	case D3D12_CONSERVATIVE_RASTERIZATION_TIER_2:
		PRINT(LogLevel::HELP_PRINT, "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_TIER_2");
		break;
	case D3D12_CONSERVATIVE_RASTERIZATION_TIER_3:
		PRINT(LogLevel::HELP_PRINT, "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_TIER_3");
		break;
	default:
		break;
	}

	//Max GPU Virtual Address Bits Per Resource
	PRINT(LogLevel::HELP_PRINT, "MaxGPUVirtualAddressBitsPerResource: %d", opts.MaxGPUVirtualAddressBitsPerResource);

	//Standard Swizzle 64KB Supported
	if (opts.StandardSwizzle64KBSupported)
		PRINT(LogLevel::HELP_PRINT, "StandardSwizzle64KBSupported: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "StandardSwizzle64KBSupported: FALSE");


	//Cross Node Sharing Tier
	switch (opts.CrossNodeSharingTier)
	{
	case D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED:
		PRINT(LogLevel::HELP_PRINT, "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARING_NOT_SUPPORTED");
		break;
	case D3D12_CROSS_NODE_SHARING_TIER_1_EMULATED:
		PRINT(LogLevel::HELP_PRINT, "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARING_TIER_1_EMULATED");
		break;
	case D3D12_CROSS_NODE_SHARING_TIER_1:
		PRINT(LogLevel::HELP_PRINT, "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARING_TIER_1");
		break;
	case D3D12_CROSS_NODE_SHARING_TIER_2:
		PRINT(LogLevel::HELP_PRINT, "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARING_TIER_2");
		break;
	default:
		break;
	}

	//Cross Adapter Row Major Texture Supported
	if (opts.CrossAdapterRowMajorTextureSupported)
		PRINT(LogLevel::HELP_PRINT, "CrossAdapterRowMajorTextureSupported: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "CrossAdapterRowMajorTextureSupported: FALSE");

	//Cross Adapter Row Major Texture Supported
	if (opts.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation)
		PRINT(LogLevel::HELP_PRINT, "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation: TRUE");
	else
		PRINT(LogLevel::HELP_PRINT, "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation: FALSE");

	//Resource heap tier
	switch (opts.ResourceHeapTier)
	{
	case D3D12_RESOURCE_HEAP_TIER_1:
		PRINT(LogLevel::HELP_PRINT, "ResourceHeapTier: D3D12_RESOURCE_HEAP_TIER_1");
		break;
	case D3D12_RESOURCE_HEAP_TIER_2:
		PRINT(LogLevel::HELP_PRINT, "ResourceHeapTier: D3D12_RESOURCE_HEAP_TIER_2");
		break;
	default:
		break;
	}
}

ID3D12RootSignature* KGraphicsDevice::CreateRootSignature(uint32 num_ranges, RootDescriptorRange* descriptor_ranges)
{
	HRESULT hr = S_OK;

	//Root descriptor part
	ID3DBlob* out_blob = nullptr;
	ID3DBlob* error_blob = nullptr;
	//D3D12 ERROR: ID3D12Device::CreateRootSignature: Root Signature declares 6 Descriptor Tables that contain SRVs in them. A Root Signature cannot contain more than 5 Descriptor Tables with SRVs on D3D12_RESOURCE_BINDING_TIER_2 level device. Yes, this is a strange limitation.  To work around this for this level of hardware consider merging descriptor tables - which admittedly can result in wasting descriptor heap space. [ STATE_CREATION ERROR #678: CREATE_ROOT_SIGNATURE_NOT_SUPPORTED_ON_DEVICE]
	
	std::vector<CD3DX12_ROOT_PARAMETER> root_parameters(num_ranges);
	std::vector<CD3DX12_DESCRIPTOR_RANGE> root_desc_ranges(num_ranges);

	uint32 base_shader_register_srv = 0;
	uint32 base_shader_register_cbv = 0;
	uint32 base_shader_register_sam = 0;
	uint32 base_shader_register_uav = 0;

	for(uint32 i = 0; i < num_ranges; ++i)
	{
		switch (descriptor_ranges[i].root_desc_type)
		{
		case ROOT_DESCRIPTOR_TYPE::RANGE_SRV:
			root_desc_ranges[i].Init((D3D12_DESCRIPTOR_RANGE_TYPE)descriptor_ranges[i].root_desc_type, descriptor_ranges[i].num_descriptors, base_shader_register_srv);
			base_shader_register_srv += descriptor_ranges[i].num_descriptors;
			root_parameters[i].InitAsDescriptorTable(1, &root_desc_ranges[i], (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
			break;
		case ROOT_DESCRIPTOR_TYPE::RANGE_CBV:
			root_desc_ranges[i].Init((D3D12_DESCRIPTOR_RANGE_TYPE)descriptor_ranges[i].root_desc_type, descriptor_ranges[i].num_descriptors, base_shader_register_cbv);
			base_shader_register_cbv += descriptor_ranges[i].num_descriptors;
			root_parameters[i].InitAsDescriptorTable(1, &root_desc_ranges[i], (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
			break;
		case ROOT_DESCRIPTOR_TYPE::RANGE_SAMPLER:
			root_desc_ranges[i].Init((D3D12_DESCRIPTOR_RANGE_TYPE)descriptor_ranges[i].root_desc_type, descriptor_ranges[i].num_descriptors, base_shader_register_sam);
			base_shader_register_sam += descriptor_ranges[i].num_descriptors;
			root_parameters[i].InitAsDescriptorTable(1, &root_desc_ranges[i], (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
			break;
		case ROOT_DESCRIPTOR_TYPE::RANGE_UAV:
			if(descriptor_ranges[i].shader_visibility == SHADER_VISIBILITY::PIXEL)
				root_desc_ranges[i].Init((D3D12_DESCRIPTOR_RANGE_TYPE)descriptor_ranges[i].root_desc_type, descriptor_ranges[i].num_descriptors, base_shader_register_uav + 1);
			else
				root_desc_ranges[i].Init((D3D12_DESCRIPTOR_RANGE_TYPE)descriptor_ranges[i].root_desc_type, descriptor_ranges[i].num_descriptors, base_shader_register_uav);

			base_shader_register_uav += descriptor_ranges[i].num_descriptors;
			root_parameters[i].InitAsDescriptorTable(1, &root_desc_ranges[i], (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
			break;
		case ROOT_DESCRIPTOR_TYPE::SRV:
			for (uint32 j = 0; j < descriptor_ranges[i].num_descriptors; ++j)
			{
				root_parameters[i].InitAsShaderResourceView(base_shader_register_srv, 0, (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
				++base_shader_register_srv;
			}
			break;
		case ROOT_DESCRIPTOR_TYPE::CBV:
			for (uint32 j = 0; j < descriptor_ranges[i].num_descriptors; ++j)
			{
				root_parameters[i].InitAsConstantBufferView(base_shader_register_cbv, 0, (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
				++base_shader_register_cbv;
			}
			break;
		case ROOT_DESCRIPTOR_TYPE::UAV:
			for (uint32 j = 0; j < descriptor_ranges[i].num_descriptors; ++j)
			{
				root_parameters[i].InitAsUnorderedAccessView(base_shader_register_uav, 0, (D3D12_SHADER_VISIBILITY)descriptor_ranges[i].shader_visibility);
				++base_shader_register_uav;
			}
			break;
		case ROOT_DESCRIPTOR_TYPE::CONSTANTS:
			//TODO
			break;
		default:
			break;
		}
	}

	CD3DX12_ROOT_SIGNATURE_DESC root_signature_desc;
	root_signature_desc.Init(num_ranges, &root_parameters[0], 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
	hr = D3D12SerializeRootSignature(&root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, &out_blob, &error_blob);
	if (FAILED(hr))
	{
		if (error_blob)
			PRINT(LogLevel::FATAL_ERROR, (char*)error_blob->GetBufferPointer());
	}

	ID3D12RootSignature* root_signature;
	shared_context.gfx_device->GetDevice()->CreateRootSignature(0, out_blob->GetBufferPointer(), out_blob->GetBufferSize(), __uuidof(ID3D12RootSignature), (void**)&root_signature);

	if (error_blob)
		error_blob->Release();

	if (out_blob)
		out_blob->Release();

	return root_signature;
}

KSampler KGraphicsDevice::CreateSampler(D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address_mode, D3D12_COMPARISON_FUNC comp_func /*= D3D12_COMPARISON_ALWAYS*/,
	float mip_lod_bias /*= 0.0f*/, uint32 max_aniso /*= 16*/, float boder_color /*= 0.0f*/, float min_lod /*= 0.0f*/, float max_lod /*= D3D12_FLOAT32_MAX*/)
{
	KSampler sampler;

	D3D12_SAMPLER_DESC sampler_desc;
	sampler_desc.Filter = filter;
	sampler_desc.AddressU = address_mode;
	sampler_desc.AddressV = address_mode;
	sampler_desc.AddressW = address_mode;
	sampler_desc.MipLODBias = mip_lod_bias;
	sampler_desc.MaxAnisotropy = max_aniso;
	sampler_desc.ComparisonFunc = comp_func;
	sampler_desc.BorderColor[0] = sampler_desc.BorderColor[1] = sampler_desc.BorderColor[2] = sampler_desc.BorderColor[3] = boder_color;
	sampler_desc.MinLOD = min_lod;
	sampler_desc.MaxLOD = max_lod;

	sampler.cpu_handle = m_DescHeapSampler.GetNewCPUHandle();
	sampler.gpu_handle = m_DescHeapSampler.GetGPUHandleAtHead();

	m_Device->CreateSampler(&sampler_desc, sampler.cpu_handle);

	return sampler;
}
