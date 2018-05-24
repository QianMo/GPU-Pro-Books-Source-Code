#pragma once
#include <D3D12.h>
#include "Types.h"
#include <SDL.h>
#include <SDL_syswm.h>
#include <SDL_image.h>
#include <vector>
#include "KDescriptorHeap.h"
#include "KRenderTarget.h"
#include "KShader.h"

enum class KBufferType : uint16
{
	NONE = 0,
	STRUCTURED,
	CONSTANT,
};

enum class SHADER_VISIBILITY
{
	ALL = D3D12_SHADER_VISIBILITY_ALL,
	VERTEX = D3D12_SHADER_VISIBILITY_VERTEX,
	HULL = D3D12_SHADER_VISIBILITY_HULL,
	DOM = D3D12_SHADER_VISIBILITY_DOMAIN,
	GEOMETRY = D3D12_SHADER_VISIBILITY_GEOMETRY,
	PIXEL = D3D12_SHADER_VISIBILITY_PIXEL,
};

enum class ROOT_DESCRIPTOR_TYPE 
{
	RANGE_SRV = D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
	RANGE_UAV = D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
	RANGE_CBV = D3D12_DESCRIPTOR_RANGE_TYPE_CBV ,
	RANGE_SAMPLER = D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
	SRV = (RANGE_SAMPLER + 1),
	UAV = (SRV + 1),
	CBV = (UAV + 1),
	CONSTANTS = (CBV + 1),
};

struct KResourceView
{
	D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle;
	D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle;
};

struct KShaderResourceView	: public KResourceView {};
struct KConstantBufferView	: public KResourceView {};
struct KRenderTargetView	: public KResourceView {};
struct KDepthStencilView	: public KResourceView {};
struct KUnorderedAccessView : public KResourceView {};

struct KSampler : public KResourceView {};

struct KResource
{
	ID3D12Resource*		resource;
	uint8*				mem;
};

struct KBuffer : public KResource
{
	KShaderResourceView		srv;
	KConstantBufferView		cbv;
	KUnorderedAccessView	uav;
};

struct RootDescriptorRange
{
	uint32 num_descriptors;
	ROOT_DESCRIPTOR_TYPE root_desc_type;
	SHADER_VISIBILITY shader_visibility;
};

class KGraphicsDevice
{
public:
	KGraphicsDevice();
	~KGraphicsDevice();

	HRESULT Init(int32 window_width, int32 window_height);
	void TransitionResource(ID3D12GraphicsCommandList* gfx_command_list, ID3D12Resource* resource, D3D12_RESOURCE_STATES state_before, D3D12_RESOURCE_STATES state_after);
	void TransitionResources(int32 num_resources, ID3D12GraphicsCommandList* gfx_command_list, ID3D12Resource** resource, D3D12_RESOURCE_STATES state_before, D3D12_RESOURCE_STATES state_after);
	void Present();
	void WaitForGPU();

	//Getters
	KDescriptorHeap* KGraphicsDevice::GetDescHeapRTV()					{ return &m_DescHeapRTV; }
	KDescriptorHeap* KGraphicsDevice::GetDescHeapCBV_SRV()				{ return &m_DescHeapCBV_SRV; }
	KDescriptorHeap* KGraphicsDevice::GetDescHeapDSV()					{ return &m_DescHeapDSV; }
	KDescriptorHeap* KGraphicsDevice::GetDescHeapSampler()				{ return &m_DescHeapSampler; }

	ID3D12CommandQueue* KGraphicsDevice::GetCommandQueue()				{ return m_CommandQueue; }
	ID3D12CommandAllocator* KGraphicsDevice::GetCommandAllocator()		{ return m_CommandAllocator;}

	D3D12_CPU_DESCRIPTOR_HANDLE KGraphicsDevice::GetRTDescHandle()		{ return m_RTDescriptor[m_SwapIndex]; }

	ID3D12Resource* KGraphicsDevice::GetRTResource()					{ return m_RenderTarget[m_SwapIndex]; }

	SDL_Window* KGraphicsDevice::GetMainWindow()						{ return m_MainWindow; }

	D3D12_VIEWPORT KGraphicsDevice::GetViewPort()						{ return m_ViewPort; }
	D3D12_RECT KGraphicsDevice::GetScissorRect()						{ return m_ScissorRect; }

	int32 KGraphicsDevice::GetWindowWidth()								{ return m_WindowWidth; }
	int32 KGraphicsDevice::GetWindowHeight()							{ return m_WindowHeight; }

	ID3D12Device* KGraphicsDevice::GetDevice()							{ return m_Device; }
	int32 KGraphicsDevice::GetSwapIndex()								{ return m_SwapIndex; }

	void SetTimeStampQuery(uint32 timestamp_query, ID3D12GraphicsCommandList* gfx_command_list);
	uint64 QueryTimeStamp(uint32 timestamp_query);

	uint64 GetFreq();
	
	KBuffer CreateBuffer(uint32 num_elements, uint32 element_size, KBufferType type, D3D12_HEAP_TYPE heap_type = D3D12_HEAP_TYPE_DEFAULT);

	KSampler CreateSampler(D3D12_FILTER filter, D3D12_TEXTURE_ADDRESS_MODE address_mode, D3D12_COMPARISON_FUNC comp_func = D3D12_COMPARISON_FUNC_ALWAYS,
		float mip_lod_bias = 0.0f, uint32 max_aniso = 16, float boder_color = 0.0f, float min_lod = 0.0f, float max_lod = D3D12_FLOAT32_MAX);

	ID3D12RootSignature* CreateRootSignature(uint32 num_ranges, RootDescriptorRange* descriptor_ranges);
	
private:

	ID3D12Device*				m_Device;
	IDXGISwapChain*				m_SwapChain;
	ID3D12Resource*				m_RenderTarget[2];
	ID3D12CommandQueue*			m_CommandQueue;
	ID3D12CommandAllocator*		m_CommandAllocator;

	D3D12_CPU_DESCRIPTOR_HANDLE m_RTDescriptor[2];

	KDescriptorHeap m_DescHeapDSV;
	KDescriptorHeap m_DescHeapCBV_SRV;
	KDescriptorHeap m_DescHeapRTV;
	KDescriptorHeap m_DescHeapSampler;

	int32 m_SwapIndex;

	ID3D12Fence*	m_Fence;
	uint64			m_CurrentFence;
	HANDLE			m_HandleEvent;

	uint64			m_Freq;

	D3D12_VIEWPORT	m_ViewPort;
	D3D12_RECT		m_ScissorRect;

	int32			m_WindowWidth;
	int32			m_WindowHeight;

	SDL_Window*		m_MainWindow;

	ID3D12QueryHeap* m_TimeStampQueryHeap;
	ID3D12Resource* m_TimeStampQueryReadBackRes;
	uint8*			m_TimeStampQueryMem;

	void CreateShaderResourceView(KBuffer* buffer, uint32 num_elements, uint32 element_size, KBufferType type);
	void CreateConstantBufferView(KBuffer* buffer, uint32 num_elements, uint32 element_size);

	void PrintHWopts(D3D12_FEATURE_DATA_D3D12_OPTIONS& opts);
};