/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/RaySet.h"
#include "RayTracing/VoxelRep.h"

#include "RayTracing/Ray.h"

#include "IncrementalGPUTimer.h"

#include <beGraphics/beMaterial.h>

#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beDeviceContext.h>

#include <cuda_D3D11_interop.h>
#include "Morton.h"

#include <beMath/beVector.h>

#include <beGraphics/DX/beError.h>
#include <lean/logging/errors.h>

// #define DATA_PARALLEL

namespace app
{

namespace tracing
{

const uint4 Zeroes[4] = { 0 };
const uint4 MinusOnes[4] = { -1, -1, -1, -1 };

// Constructor.
RaySet::RaySet(uint4 maxRayCount, besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
	: m_maxRayCount(maxRayCount)
{
	beg::api::Device *device = ToImpl(*renderer->Device());

	m_constBuffer = beg::Any::CreateConstantBuffer(device, sizeof(RaySetLayout));

	// Rays
	{
		m_rayDescBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(float) * 6, m_maxRayCount
			);
		m_rayDescSRV = beg::Any::CreateSRV(m_rayDescBuffer);
		m_rayDescUAV = beg::Any::CreateCountingUAV(m_rayDescBuffer);

		m_rayGeometryBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(uint4) * 5, m_maxRayCount
			);
		m_rayGeometrySRV = beg::Any::CreateSRV(m_rayGeometryBuffer);
		m_rayGeometryUAV = beg::Any::CreateCountingUAV(m_rayGeometryBuffer);

		m_rayLightBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(uint4) * 2, m_maxRayCount
			);
		m_rayLightSRV = beg::Any::CreateSRV(m_rayLightBuffer);
		m_rayLightUAV = beg::Any::CreateCountingUAV(m_rayLightBuffer);

		m_rayDebugBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(uint4), m_maxRayCount
			);
		m_rayDebugSRV = beg::Any::CreateSRV(m_rayDebugBuffer);
		m_rayDebugUAV = beg::Any::CreateCountingUAV(m_rayDebugBuffer);
	}
}

// Destructor.
RaySet::~RaySet()
{
}

// Binds the given material.
void RayGridGen::BindMaterial(beg::Material *material)
{
	m_material = material;
	m_effect = ToImpl(m_material->GetEffects()[0])->Get();

	m_march = beg::Any::ValidateEffectVariable(m_effect->GetTechniqueByName("March"), LSS);
	m_inject = beg::Any::ValidateEffectVariable(m_effect->GetTechniqueByName("Inject"), LSS);
	m_inline = beg::Any::ValidateEffectVariable(m_effect->GetTechniqueByName("CompactRays"), LSS);
	m_pFilter = m_effect->GetTechniqueByName("FilterRays");
	if (!m_pFilter->IsValid()) m_pFilter = nullptr;

	m_boundedTraversalLimitsVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("BoundedTraversalLimits")->AsVector(), LSS);

	m_voxelRepConstVar = beg::Any::ValidateEffectVariable(m_effect->GetConstantBufferByName("VoxelRepConstants"), LSS);
	m_raySetConstVar = beg::Any::ValidateEffectVariable(m_effect->GetConstantBufferByName("RaySetConstants"), LSS);

	m_rayDescUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayQueueUAV")->AsUnorderedAccessView(), LSS);
	m_rayDescVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayQueue")->AsShaderResource(), LSS);
	m_rayDebugUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayDebugUAV")->AsUnorderedAccessView(), LSS);
	m_rayGeometryVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGeometry")->AsShaderResource(), LSS);
	m_rayGeometryUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGeometryUAV")->AsUnorderedAccessView(), LSS);
	m_voxelRepVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("Voxels")->AsShaderResource(), LSS);

	m_activeRayListVar = m_effect->GetVariableBySemantic("ActiveRayList")->AsShaderResource();
	m_activeRayListUAVVar = m_effect->GetVariableBySemantic("ActiveRayListUAV")->AsUnorderedAccessView();

	m_gridIdxListVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridIdxList")->AsShaderResource(), LSS);
	m_gridIdxListUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridIdxListUAV")->AsUnorderedAccessView(), LSS);
	m_rayLinkListVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayLinkList")->AsShaderResource(), LSS);
	m_rayLinkListUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayLinkListUAV")->AsUnorderedAccessView(), LSS);

	m_rayInlineListVar = m_effect->GetVariableBySemantic("RayList")->AsShaderResource();
	m_rayInlineListUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayListUAV")->AsUnorderedAccessView(), LSS);

	m_rayGridIdxBaseAddressVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridIdxBaseAddress")->AsScalar(), LSS);
	m_rayLinkBaseAddressVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayLinkBaseAddress")->AsScalar(), LSS);

	m_gridBeginVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridBegin")->AsShaderResource(), LSS);
	m_gridEndVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridEnd")->AsShaderResource(), LSS);
	m_gridBeginUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridBeginUAV")->AsUnorderedAccessView(), LSS);
	m_gridEndUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("RayGridEndUAV")->AsUnorderedAccessView(), LSS);

	m_groupDispatchUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("GroupDispatchUAV")->AsUnorderedAccessView(), LSS);
	m_counterUAVVar = beg::Any::ValidateEffectVariable(m_effect->GetVariableBySemantic("CounterUAV")->AsUnorderedAccessView(), LSS);
}

// Constructor.
RayGridGen::RayGridGen(const lean::utf8_ntri &file, uint4 maxRayCount, bem::vector<uint4, 3> resolution, uint4 avgRayLength,
	besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
	: m_resolution(resolution),
	m_maxRayCount(maxRayCount),
	m_maxRayLinkCount(maxRayCount * avgRayLength),
	
	m_sorter( cuda::CreateB40Sorter(0) ) // We provide auxiliary storage buffers ourselves
{
	beg::api::Device *device = ToImpl(*renderer->Device());
	
	// Effect
	BindMaterial( resourceManager->MaterialCache()->NewByFile(file, "RayGrid") );
	
	// Ray Links & Ray Inline storage
	{
		m_main.m_rayListBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(uint4),
				2 * m_maxRayLinkCount,
				D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS
			);
		m_main.m_rayListCGR = CreateCGR(m_main.m_rayListBuffer);
		BE_THROW_CUDA_ERROR_MSG(
				cudaGraphicsResourceSetMapFlags(m_main.m_rayListCGR, cudaGraphicsMapFlagsNone),
				"cudaGraphicsResourceSetMapFlags()"
			);

		m_aux.m_rayListBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(uint4),
				3 * m_maxRayLinkCount,
				D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS
			);
		m_aux.m_rayListCGR = CreateCGR(m_aux.m_rayListBuffer);
		BE_THROW_CUDA_ERROR_MSG(
				cudaGraphicsResourceSetMapFlags(m_aux.m_rayListCGR, cudaGraphicsMapFlagsNone),
				"cudaGraphicsResourceSetMapFlags()"
			);

		const uint4 segmentElements = m_maxRayLinkCount;
		const uint4 segmentSize = sizeof(uint4) * segmentElements;
		
		m_main.m_rayLinkSRV = beg::Any::CreateRawSRV(m_main.m_rayListBuffer, 0, segmentSize * 0, segmentSize);
		m_main.m_rayLinkUAV = beg::Any::CreateRawUAV(m_main.m_rayListBuffer, 0, segmentSize * 0, segmentSize);
		m_main.m_rayLinkOffset = segmentElements * 0;
		m_aux.m_rayLinkSRV = beg::Any::CreateRawSRV(m_aux.m_rayListBuffer, 0, segmentSize * 0, segmentSize);
		m_aux.m_rayLinkUAV = beg::Any::CreateRawUAV(m_aux.m_rayListBuffer, 0, segmentSize * 0, segmentSize);
		m_aux.m_rayLinkOffset = segmentElements * 0;
		
		m_main.m_rayGridIdxSRV = beg::Any::CreateRawSRV(m_main.m_rayListBuffer, 0, segmentSize * 1, segmentSize);
		m_main.m_rayGridIdxUAV = beg::Any::CreateRawUAV(m_main.m_rayListBuffer, 0, segmentSize * 1, segmentSize);
		m_main.m_rayGridIdxOffset = segmentElements * 1;
		m_aux.m_rayGridIdxSRV = beg::Any::CreateRawSRV(m_aux.m_rayListBuffer, 0, segmentSize * 1, segmentSize);
		m_aux.m_rayGridIdxUAV = beg::Any::CreateRawUAV(m_aux.m_rayListBuffer, 0, segmentSize * 1, segmentSize);
		m_aux.m_rayGridIdxOffset = segmentElements * 1;

		m_main.m_rayGridIdxLinkUAV = beg::Any::CreateRawUAV(m_main.m_rayListBuffer, 0, 0, 2 * segmentSize);
		m_aux.m_rayGridIdxLinkUAV = beg::Any::CreateRawUAV(m_aux.m_rayListBuffer, 0, 0, 2 * segmentSize);

		m_rayListSRV = beg::Any::CreateRawSRV(m_aux.m_rayListBuffer);
		m_rayListUAV = beg::Any::CreateRawUAV(m_aux.m_rayListBuffer);
	}

	// Grid
	{
		m_rayGridBegin = beg::Any::CreateTexture3D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32_UINT,
				resolution[0], resolution[1], resolution[2]
			);
		m_rayGridBeginSRV = beg::Any::CreateSRV(m_rayGridBegin);
		m_rayGridBeginUAV = beg::Any::CreateUAV(m_rayGridBegin);

		m_rayGridEnd = beg::Any::CreateTexture3D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32_UINT,
				resolution[0], resolution[1], resolution[2]
			);
		m_rayGridEndSRV = beg::Any::CreateSRV(m_rayGridEnd);
		m_rayGridEndUAV = beg::Any::CreateUAV(m_rayGridEnd);
	}

	{
		m_dispatchBuffer = beg::Any::CreateStructuredBuffer(device, D3D11_BIND_UNORDERED_ACCESS,
			sizeof(uint4), 4, D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS);
		m_dispatchUAV = beg::Any::CreateRawUAV(m_dispatchBuffer);
	}

	{
		m_counterBuffer = beg::Any::CreateStructuredBuffer(device, D3D11_BIND_UNORDERED_ACCESS, sizeof(uint4), 4);
		m_counterUAV = beg::Any::CreateCountingUAV(m_counterBuffer);
	}

	{
		m_stagingBuffer = beg::Any::CreateStagingBuffer(device, sizeof(uint4), sizeof(RaySetLayout) / sizeof(uint4),
			D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE);
	}
}

// Destructor.
RayGridGen::~RayGridGen()
{
}

// Processes / commits changes.
void RayGridGen::Commit()
{
	// Hot swap
	if (beg::Material *successor = m_material->GetSuccessor())
		BindMaterial(successor);
}

// Gets the number of tracing steps.
uint4 RayGridGen::GetTraceStepCount() const
{
	return 3;
}

// Computes the number of groups to dispatch.
void RayGridGen::ComputeDispatchGroupCount(beg::api::EffectTechnique *technique, uint4 passID,
	beg::api::UnorderedAccessView *dispatchUAV, beg::api::DeviceContext *context) const
{
	m_groupDispatchUAVVar->SetUnorderedAccessView(dispatchUAV);

	technique->GetPassByIndex(passID)->Apply(0, context);

	context->Dispatch(1, 1, 1);

	technique->GetPassByIndex(passID)->Unbind(context);
}

// Scatters the rays into the ray grid.
void RayGridGen::March(RaySet &raySet, VoxelRep &voxelRep, uint4 traceStep, besc::RenderContext &context)
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());

	{
		// Max number of layers (3rd component) traversed and ray links (4th component) inserted per ray in this pass
		int4 boundedTraversalLimits[4] = { 0, 0, 1, 4 };

		if (traceStep >= 1)
		{
			memcpy(boundedTraversalLimits, boundedTraversalLimits + 2, sizeof(uint4) * 2);
			boundedTraversalLimits[2] = 4 - boundedTraversalLimits[0];
			boundedTraversalLimits[3] = 12 - boundedTraversalLimits[1];
		}

		if (traceStep >= 2)
		{
			memcpy(boundedTraversalLimits, boundedTraversalLimits + 2, sizeof(uint4) * 2);
			boundedTraversalLimits[2] = -1;
			boundedTraversalLimits[3] = -1;
		}

		m_boundedTraversalLimitsVar->SetIntVector(boundedTraversalLimits);
	}

	m_voxelRepConstVar->SetConstantBuffer(voxelRep.Constants());
	m_raySetConstVar->SetConstantBuffer(raySet.Constants());

	m_voxelRepVar->SetResource(voxelRep.Voxels());
	m_rayGeometryVar->SetResource(raySet.RayGeometry());
	m_rayGeometryUAVVar->SetUnorderedAccessView(raySet.RayGeometryUAV());
	m_rayDescUAVVar->SetUnorderedAccessView(raySet.RayDescUAV(), 0);
	m_rayDescVar->SetResource(raySet.RayDescs());
	m_rayDebugUAVVar->SetUnorderedAccessView(raySet.RayDebugUAV(), 0);

	DoubleBuffered &filterBuffer = m_aux;

	// Collect active rays
	if (m_pFilter && traceStep > 0)
	{
		m_counterUAVVar->SetUnorderedAccessView(m_counterUAV, 0);
		m_activeRayListVar->SetResource(filterBuffer.m_rayLinkSRV);
		m_activeRayListUAVVar->SetUnorderedAccessView(filterBuffer.m_rayLinkUAV);

		ComputeDispatchGroupCount(m_pFilter, 1, m_dispatchUAV, deviceContext);
		m_pFilter->GetPassByIndex(0)->Apply(0, deviceContext);

		deviceContext->DispatchIndirect(m_dispatchBuffer, 0);
		
		beg::Any::UnbindAllComputeTargets(deviceContext);

		deviceContext->CopyStructureCount(
				raySet.Constants(),
				offsetof(RaySetLayout, ActiveRayCount),
				m_counterUAV
			);
	}

	// Ugh: Can't write to part of constant buffer
	beg::Any::CopyBuffer(deviceContext, m_stagingBuffer, 0, raySet.Constants(), 0, sizeof(RaySetLayout));
	beg::Any::WriteBufferByMap(deviceContext, m_stagingBuffer, offsetof(RaySetLayout, MaxRayLinkCount), &m_maxRayLinkCount, 0, sizeof(m_maxRayLinkCount));
	beg::Any::CopyBuffer(deviceContext, raySet.Constants(), 0, m_stagingBuffer, 0, sizeof(RaySetLayout));

	DoubleBuffered &marchBuffer = m_main;

	m_gridIdxListUAVVar->SetUnorderedAccessView(marchBuffer.m_rayGridIdxLinkUAV);
	m_rayLinkListUAVVar->SetUnorderedAccessView(marchBuffer.m_rayGridIdxLinkUAV);
	m_counterUAVVar->SetUnorderedAccessView(m_counterUAV, 0);
	m_rayGridIdxBaseAddressVar->SetInt(marchBuffer.m_rayGridIdxOffset * sizeof(uint4));
	m_rayLinkBaseAddressVar->SetInt(marchBuffer.m_rayLinkOffset * sizeof(uint4));

#ifndef DATA_PARALLEL
	m_march->GetPassByIndex(traceStep >= 1)->Apply(0, deviceContext);

	// Spawn fixed number of persistent threads
	static const uint4 GroupCount = 8 * 16;
	deviceContext->Dispatch(GroupCount, 1, 1);
#else
	ComputeDispatchGroupCount(m_march, 2 + (traceStep >= 1), m_dispatchUAV, deviceContext);
	m_march->GetPassByIndex(traceStep >= 1)->Apply(0, deviceContext);

	deviceContext->DispatchIndirect(m_dispatchBuffer, 0);
#endif
	
	beg::Any::UnbindAllComputeTargets(deviceContext);

	// Fetch list node count
	deviceContext->CopyStructureCount(m_stagingBuffer, offsetof(RaySetLayout, RayLinkCount), m_counterUAV);

	// This read back is unnecessary, but the CUDA radix sort implementation by B40C requires it
	// TODO: Replace with a competitive DirectCompute radix sort implementation & move this code to GPU
	beg::Any::ReadBufferData(deviceContext, m_stagingBuffer, &m_rayLinkCount, sizeof(m_rayLinkCount), offsetof(RaySetLayout, RayLinkCount));
	m_rayLinkCount = min(m_rayLinkCount, m_maxRayLinkCount);
	beg::Any::WriteBufferByMap(deviceContext, m_stagingBuffer, offsetof(RaySetLayout, RayLinkCount), &m_rayLinkCount, 0, sizeof(m_rayLinkCount));
	beg::Any::CopyBuffer(deviceContext, raySet.Constants(), 0, m_stagingBuffer, 0, sizeof(RaySetLayout));
}

// Scatters the rays into the ray grid.
void RayGridGen::Sort(RaySet &raySet, besc::RenderContext &context)
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());

	cudaGraphicsResource *graphicsResources[] = {
			m_main.m_rayListCGR.get(),
			m_aux.m_rayListCGR.get()
		};
	BE_LOG_CUDA_ERROR_MSG(
		cudaGraphicsMapResources(lean::arraylen(graphicsResources), graphicsResources),
		"cudaGraphicsMapResources");

	void *buffer, *auxBuffer;
	size_t unused;
	BE_LOG_CUDA_ERROR_MSG(
		cudaGraphicsResourceGetMappedPointer(&buffer, &unused, m_main.m_rayListCGR.get()),
		"cudaGraphicsResourceGetMappedPointer");
	BE_LOG_CUDA_ERROR_MSG(
		cudaGraphicsResourceGetMappedPointer(&auxBuffer, &unused, m_aux.m_rayListCGR.get()),
		"cudaGraphicsResourceGetMappedPointer");

	// Sort ray links by cell keys
	bool bSwapped = m_sorter->SortSwap(static_cast<uint4*>(buffer) + m_main.m_rayGridIdxOffset, static_cast<uint4*>(buffer) + m_main.m_rayLinkOffset, 
		static_cast<uint4*>(auxBuffer) + m_aux.m_rayGridIdxOffset, static_cast<uint4*>(auxBuffer) + m_aux.m_rayLinkOffset, 
		m_rayLinkCount);

	BE_LOG_CUDA_ERROR_MSG(
		cudaGraphicsUnmapResources(lean::arraylen(graphicsResources), graphicsResources),
		"cudaGraphicsUnmapResources");

	// Copy sorted links back to main store if number of sort passes uneven
	if (bSwapped)
	{
		beg::Any::CopyBuffer(deviceContext,
				m_main.m_rayListBuffer, m_main.m_rayGridIdxOffset * sizeof(uint4),
				m_aux.m_rayListBuffer, m_aux.m_rayGridIdxOffset * sizeof(uint4), m_rayLinkCount * sizeof(uint4)
			);
		beg::Any::CopyBuffer(deviceContext,
				m_main.m_rayListBuffer, m_main.m_rayLinkOffset * sizeof(uint4),
				m_aux.m_rayListBuffer, m_aux.m_rayLinkOffset * sizeof(uint4), m_rayLinkCount * sizeof(uint4)
			);
	}

	beg::Any::UnbindAllComputeTargets(deviceContext);
}

// Scatters the rays into the ray grid.
void RayGridGen::Inject(RaySet &raySet, besc::RenderContext &context)
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());

	m_raySetConstVar->SetConstantBuffer(raySet.Constants());

	m_gridIdxListVar->SetResource(m_main.m_rayGridIdxSRV);
	m_rayLinkListVar->SetResource(m_main.m_rayLinkSRV);
	m_gridBeginUAVVar->SetUnorderedAccessView(m_rayGridBeginUAV);
	m_gridEndUAVVar->SetUnorderedAccessView(m_rayGridEndUAV);
	
	m_inject->GetPassByIndex(0)->Apply(0, deviceContext);
	ComputeDispatchGroupCount(m_inject, 1, m_dispatchUAV, deviceContext);
	m_inject->GetPassByIndex(0)->Apply(0, deviceContext);

	deviceContext->DispatchIndirect(m_dispatchBuffer, 0);
	
	beg::Any::UnbindAllComputeTargets(deviceContext);
}

// Inlines the rays with the list of ray links.
void RayGridGen::Inline(RaySet &raySet, besc::RenderContext &context)
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());

	m_raySetConstVar->SetConstantBuffer(raySet.Constants());
	
	m_rayDescVar->SetResource(raySet.RayDescs());
	m_gridIdxListVar->SetResource(m_main.m_rayGridIdxSRV);
	m_rayLinkListVar->SetResource(m_main.m_rayLinkSRV);
	m_rayInlineListUAVVar->SetUnorderedAccessView(m_rayListUAV);
	
	m_inline->GetPassByIndex(0)->Apply(0, deviceContext);
	ComputeDispatchGroupCount(m_inline, 1, m_dispatchUAV, deviceContext);
	m_inline->GetPassByIndex(0)->Apply(0, deviceContext);

	deviceContext->DispatchIndirect(m_dispatchBuffer, 0);
	
	beg::Any::UnbindAllComputeTargets(deviceContext);
}

// Scatters the rays into the ray grid.
void RayGridGen::ConstructRayGrid(RaySet &raySet, VoxelRep &voxelRep, uint4 traceStep, besc::RenderContext &context,
	IncrementalGPUTimer *pMarch, IncrementalGPUTimer *pSort, IncrementalGPUTimer *pInject, IncrementalGPUTimer *pInline)
{
	beg::api::DeviceContext *deviceContext = ToImpl(context.Context());

	// End == 0 indicates empty cell
	deviceContext->ClearUnorderedAccessViewUint(m_rayGridEndUAV, Zeroes);

	if (pMarch) pMarch->Begin(deviceContext);
	March(raySet, voxelRep, traceStep, context);
	if (pMarch) pMarch->End(deviceContext);

	if (pSort) pSort->Begin(deviceContext);
	Sort(raySet, context);
	if (pSort) pSort->End(deviceContext);


	if (pInject) pInject->Begin(deviceContext);
	Inject(raySet, context);
	if (pInject) pInject->End(deviceContext);

	if (pInline) pInline->Begin(deviceContext);
	Inline(raySet, context);
	if (pInline) pInline->End(deviceContext);
}

} // namespace

} // namespace
