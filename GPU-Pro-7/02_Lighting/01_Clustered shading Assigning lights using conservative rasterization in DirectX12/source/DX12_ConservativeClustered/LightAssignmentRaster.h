#pragma once
#include "KGraphicsDevice.h"
#include "KUAVTex2D.h"
#include "KRenderTarget.h"

class LightAssignmentRaster
{
public:
	LightAssignmentRaster(ID3D12GraphicsCommandList* gfx_command_list);
	~LightAssignmentRaster();

	KRenderTarget* GetSpotRT(int32 swap_index) {	return &m_spotlightShellTarget[swap_index];}

	ID3D12PipelineState* GetPSOSpot()					{ return m_PSOSpot; }
	ID3D12PipelineState* GetPSOSpotOld()				{ return m_PSOSpotOld; }
	ID3D12PipelineState* GetPSOPointOld()				{ return m_PSOPointOld; }
	ID3D12PipelineState* GetPSOSpotLinear()				{ return m_PSOSpotLinear; }
	ID3D12PipelineState* GetPSOPointLinear()			{ return m_PSOPointLinear; }
	ID3D12PipelineState* GetPSOPointLinearOld()			{ return m_PSOPointOldLinear; }
	ID3D12PipelineState* GetPSOSpotLinearOld()			{ return m_PSOSpotOldLinear; }
	ID3D12PipelineState* GetComputePSO()				{ return m_ComputePSO; }
	ID3D12PipelineState* GetPSOPoint()					{ return m_PSOPoint; }

	ID3D12RootSignature* GetRootSig()					{ return m_RootSignature; }
	ID3D12RootSignature* GetComputeRootSig()			{ return m_ComputeRootSig; }

	KRenderTarget* GetColorRT(int32 swap_index)			{ return &m_pointlightShellTarget[swap_index]; }

	D3D12_CPU_DESCRIPTOR_HANDLE GetSobUAVCPUHandle()	{ return m_SobUAVHandle; }														
	D3D12_CPU_DESCRIPTOR_HANDLE GetLLLUAVCPUHandle()	{ return m_LLLUAVHandle; }														
	D3D12_CPU_DESCRIPTOR_HANDLE GetSobSRVCPUHandle()	{ return m_SobSRVHandle; }														
	D3D12_CPU_DESCRIPTOR_HANDLE GetLLLSRVCPUHandle()	{ return m_LLLSRVHandle; }	

	D3D12_GPU_DESCRIPTOR_HANDLE GetSobUAVGPUHandle()	{ return m_SobUAVHandleGPU; }														
	D3D12_GPU_DESCRIPTOR_HANDLE GetLLLUAVGPUHandle()	{ return m_LLLUAVHandleGPU; }														
	D3D12_GPU_DESCRIPTOR_HANDLE GetSobSRVGPUHandle()	{ return m_SobSRVHandleGPU; }												
	D3D12_GPU_DESCRIPTOR_HANDLE GetLLLSRVGPUHandle()	{ return m_LLLSRVHandleGPU; }

	ID3D12Resource* GetStartOffsetResource()			{ return m_StartOffsetBuffer; }
	ID3D12Resource* GetLinkedLightListResource()		{ return m_LinkedIndexList; }
	ID3D12Resource* GetUAVCounterResource()				{ return m_StartOffsetBufferCounter; }

	ID3D12Resource* GetStartOffsetReadResource();
	ID3D12Resource* GetLinkedLightListReadResource();
	ID3D12Resource* GetUAVCounterReadResource();

	void ClearUAVs(ID3D12GraphicsCommandList* gfx_command_list);

	void ReadBackDebugData(ID3D12GraphicsCommandList* gfx_command_list, int32 swap_index);

private:

	ID3D12PipelineState* m_PSOPoint;
	ID3D12PipelineState* m_PSOPointLinear;
	ID3D12PipelineState* m_PSOPointOld;
	ID3D12PipelineState* m_PSOPointOldLinear;
	ID3D12PipelineState* m_PSOSpot;
	ID3D12PipelineState* m_PSOSpotLinear;
	ID3D12PipelineState* m_PSOSpotOld;
	ID3D12PipelineState* m_PSOSpotOldLinear;
	ID3D12RootSignature* m_RootSignature;
	
	//Compute stuff
	ID3D12PipelineState* m_ComputePSO;
	ID3D12RootSignature* m_ComputeRootSig;

	KRenderTarget m_pointlightShellTarget[2];
	KRenderTarget m_spotlightShellTarget[2];

	//Light Linked List stuff
	ID3D12Resource* m_StartOffsetBuffer;
	ID3D12Resource* m_LinkedIndexList;
	ID3D12Resource* m_StartOffsetBufferCounter;

	ID3D12Resource* m_StartOffsetBufferCounterClear;
	ID3D12Resource* m_StartOffsetBufferCounterClearUpload;
	ID3D12Resource* m_StartOffsetBufferClear;
	ID3D12Resource* m_StartOffsetBufferClearUpload;

	D3D12_CPU_DESCRIPTOR_HANDLE m_SobUAVHandle;
	D3D12_CPU_DESCRIPTOR_HANDLE m_LLLUAVHandle;

	D3D12_CPU_DESCRIPTOR_HANDLE m_SobSRVHandle;
	D3D12_CPU_DESCRIPTOR_HANDLE m_LLLSRVHandle;

	D3D12_GPU_DESCRIPTOR_HANDLE m_SobUAVHandleGPU;
	D3D12_GPU_DESCRIPTOR_HANDLE m_LLLUAVHandleGPU;

	D3D12_GPU_DESCRIPTOR_HANDLE m_SobSRVHandleGPU;
	D3D12_GPU_DESCRIPTOR_HANDLE m_LLLSRVHandleGPU;

	ID3D12Resource* m_SobReadBackRes[2];
	ID3D12Resource* m_LLLReadBackRes[2];
	ID3D12Resource* m_UAVCounterReadBackRes[2];

};