#include "stdafx.h"

#ifdef RAY_CASTING

#include "CastingPipeline.h"

#include <beScene/beRenderingPipeline.h>
#include <beScene/beRenderContext.h>

#include <beScene/bePipe.h>
#include <beScene/bePipelineProcessor.h>

#include <beScene/beQuadProcessor.h>

#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beSetup.h>
#include <beGraphics/Any/beTextureTargetPool.h>
#include <beGraphics/Any/beStateManager.h>

#include <beGraphics/Any/beQuery.h>

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>

#include <random>
#include <AntTweakBar.h>

#include <beGraphics/DX/beError.h>
#include <lean/logging/errors.h>

#include <lean/meta/math.h>

// #define RAY_LISTS
// #define INLINED_RAYS

const uint4 MaxBundleCount = 6;
const uint4 MaxSliceSets = 1;
const uint4 BundleResolution = 512;
const uint4 BundleMipCount = 1 + lean::log2<uint4, BundleResolution>::value;

const uint4 Zeroes[4] = { 0 };
const uint4 MinusOnes[4] = { -1, -1, -1, -1 };

struct SceneApproximation
{
	bem::fvec2 Resolution;
	bem::fvec2 PixelWidth;

	float LayerCount;
	float LayerWidth;
	float OneOverLayerWidth;
	float _pad2;

	float SliceCount;
	float SliceWidth;
	float OneOverSliceWidth;
	float SlicesPerLayer;

	uint4 BundleStride;
	float MinDepth;
	float MaxDepth;
	float Depth;
};

struct RuntimeInfo
{
	uint4 totalRayCount;
	uint4 totalNodeCount;
};

struct Mat3Pad
{
	bem::fvec3 right;
	float _pad1;
	bem::fvec3 up;
	float _pad2;
	bem::fvec3 dir;
	float _pad3;
};

LEAN_INLINE Mat3Pad ToMat3Pad(const bem::fmat3 &m)
{
	Mat3Pad p = { m[0], 0.0f, m[1], 0.0f, m[2], 0.0f };
	return p;
}

struct BundleDesc
{
	Mat3Pad orientation;
	float fov;
	float farPlane;
	float _pad4[2];
};

struct BundleDescLayout
{
	BundleDesc Descs[MaxBundleCount];
	Mat3Pad BundleSpace;
	uint4 RayOffsets[MaxBundleCount][4];
};

const uint4 testasdfasd = sizeof(BundleDescLayout);

struct IntermediateBundleBounds
{
	bem::ivec3 Min;
	bem::ivec3 Max;
};

struct BundleData
{
	IntermediateBundleBounds IntermediateBounds[MaxBundleCount];
	bem::fmat4 ViewProjection[MaxBundleCount];
	bem::fmat4 Projection[MaxBundleCount];
	bem::fmat4 View[MaxBundleCount];
};

struct BundleDebugInfo
{
	bem::fvec3 min;
	bem::fvec3 max;

	bem::fvec3 rayOrig;
	bem::fvec3 rayDir;
	bem::fvec4 rayStart;
	bem::fvec4 rayEnd;

	bem::fmat4 proj;
};

struct StagingData
{
	BundleData bundleData;
	BundleDebugInfo bundleDebugInfo[MaxBundleCount];
	RuntimeInfo runtimeInfo;
	uint4 totalHitCount;
};

struct tw_delete_policy
{
	static void release(TwBar *bar)
	{
		TwDeleteBar(bar);
	}
};

struct HistopyramidBinder
{
	static const uint4 GroupSize = 8;
	static const uint4 GroupDepth = 4;

	lean::com_ptr<beg::API::Effect> effect;

	beg::API::EffectTechnique *technique;
	beg::API::EffectShaderResource *counts;
	beg::API::EffectUnorderedAccessView *countsUAV;
	beg::API::EffectUnorderedAccessView *offsetsUAV;
	beg::API::EffectUnorderedAccessView *listUAV;

	void Construct(beg::API::Effect *effect, beg::API::EffectTechnique *technique)
	{
		this->effect = effect;
		this->technique = beg::Any::Validate(technique);

		this->counts = beg::Any::ValidateEffectVariable( effect->GetVariableBySemantic("Histopyramid")->AsShaderResource(), LSS, "Histopyramid" );
		this->countsUAV = beg::Any::ValidateEffectVariable( effect->GetVariableBySemantic("HistopyramidUAV")->AsUnorderedAccessView(), LSS, "HistopyramidUAV" );
		this->offsetsUAV = beg::Any::ValidateEffectVariable( effect->GetVariableBySemantic("OffsetFieldUAV")->AsUnorderedAccessView(), LSS, "OffsetFieldUAV" );
		this->listUAV = beg::Any::ValidateEffectVariable( effect->GetVariableBySemantic("RayListUAV")->AsUnorderedAccessView(), LSS, "RayListUAV" );
	}

	void Dispatch(beg::api::DeviceContext *context,
		beg::api::ShaderResourceView *const *pyramidSRVs,
		beg::api::UnorderedAccessView *const *pyramidUAVs,
		beg::api::ShaderResourceView *pyramidSRV,
		beg::api::UnorderedAccessView *offsetUAV,
		beg::api::UnorderedAccessView *listUAV,
		uint4 resolution, uint4 elements)
	{
		// Append end of list
		{
			this->countsUAV->SetUnorderedAccessView(pyramidUAVs[0]);

			this->technique->GetPassByIndex(2)->Apply(0, context);

			uint4 groups = resolution / GroupSize;
			uint4 elementGroups = elements / GroupDepth;
			context->Dispatch(groups, groups, elementGroups);
		}

		beg::Any::UnbindAllComputeTargets(context);

		// Compute histopyramid
		for (uint4 mipRes = resolution / 2, mipElements = elements / 2, level = 1;
			mipRes > 0;
			mipRes /= 2, mipElements /= 2, ++level)
		{
//			context->ClearUnorderedAccessViewUint(pyramidUAVs[level], Zeroes);

			this->counts->SetResource(pyramidSRVs[level - 1]);
			this->countsUAV->SetUnorderedAccessView(pyramidUAVs[level]);

			this->technique->GetPassByIndex(0)->Apply(0, context);

			uint4 groups = max(mipRes / GroupSize, 1U);
			uint4 elementGroups = max(mipElements / GroupDepth, 1U);
			context->Dispatch(groups, groups, elementGroups);
		}
		
		beg::Any::UnbindAllComputeTargets(context);

		// Compute offsets
		{
			this->counts->SetResource(pyramidSRV);
			this->offsetsUAV->SetUnorderedAccessView(offsetUAV);
			this->listUAV->SetUnorderedAccessView(listUAV);

			this->technique->GetPassByIndex(1)->Apply(0, context);

			uint4 groups = resolution / GroupSize;
			uint4 elementGroups = elements / GroupDepth;
			context->Dispatch(groups, groups, elementGroups);
		}

		beg::Any::UnbindAllComputeTargets(context);
	}
};

struct IncrementalGPUTimer
{
	static const uint4 BufferCount = 3;

	lean::com_ptr<ID3D11Query> startQuery[BufferCount];
	lean::com_ptr<ID3D11Query> endQuery[BufferCount];
	bool waiting[BufferCount];
	uint4 bufferIdx;

	uint8 ticks;
	float ms;

	void Construct(ID3D11Device *device)
	{
		for (uint4 b = 0; b < BufferCount; ++b)
		{
			startQuery[b] = beg::Any::CreateTimestampQuery(device);
			endQuery[b] = beg::Any::CreateTimestampQuery(device);
		}
		Reset();
	}

	void Begin(ID3D11DeviceContext *context)
	{
		if (waiting[bufferIdx])
			ReadData(context, bufferIdx);

		context->End(startQuery[bufferIdx]);
	}
	void End(ID3D11DeviceContext *context)
	{
		context->End(endQuery[bufferIdx]);
		waiting[bufferIdx] = true;

		bufferIdx = (bufferIdx + 1) % BufferCount;
	}

	uint8 ReadData(ID3D11DeviceContext *context, uint4 bufferIdx)
	{
		if (waiting[bufferIdx])
		{
			uint8 beginStamp = beg::Any::GetTimestamp(context, startQuery[bufferIdx]);
			uint8 endStamp = beg::Any::GetTimestamp(context, endQuery[bufferIdx]);

			ticks += endStamp - beginStamp;

			waiting[bufferIdx] = false;
		}

		return ticks;
	}

	uint8 ReadData(ID3D11DeviceContext *context)
	{
		for (uint4 b = 0; b < BufferCount; ++b)
			ReadData(context, b);

		return ticks;
	}

	float ToMS(uint8 frequency)
	{
		ms = ticks * 1000000 / frequency / 1000.0f;
		return ms;
	}

	void Reset()
	{
		for (uint4 b = 0; b < BufferCount; ++b)
			waiting[b] = false;
		bufferIdx = 0;
		ticks = 0;
		ms = 0.0f;
	}
};

struct CastingPipeline::M
{
	lean::resource_ptr<besc::EffectDrivenRenderer> renderer;
	lean::resource_ptr<besc::ResourceManager> resourceManager;

	uint4 geometryStageID;
	uint4 approxStageID;
	uint4 tracingStageID;

	uint4 rayBundleResolution;
	uint4 rayBundleMipCount;
	uint4 rayBundleCount;
	uint4 rayBundleLayerCount;

	uint4 rayBundleStride;
	uint4 tracingWorkerCount;
	uint4 tracingQueueSize;

	lean::com_ptr<ID3D11Buffer> sceneApproxBuffer;
	lean::com_ptr<ID3D11Buffer> rayBundleDescBuffer;

	lean::com_ptr<ID3D11Buffer> rayBundleDataBuffer;
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleDataSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleDataUAV;

	lean::com_ptr<ID3D11Buffer> rayBundleDebugBuffer;
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleDebugUAV;

	lean::com_ptr<ID3D11Buffer> rayDescBuffer;
	lean::com_ptr<ID3D11ShaderResourceView> rayDescSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayDescUAV;

	lean::com_ptr<ID3D11Buffer> rayListNodeBuffer;
	lean::com_ptr<ID3D11ShaderResourceView> rayListNodeSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayListNodeUAV;

	lean::com_ptr<ID3D11Texture3D> rayBundleCountTexture;
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleCountSRV;
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleCountMipSRVs[BundleMipCount];
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleCountUAV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleCountMipUAVs[BundleMipCount];

	lean::com_ptr<ID3D11Texture3D> rayBundleHeadTexture;
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleHeadSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleHeadUAV;

	lean::com_ptr<ID3D11Texture2D> rayBundleApproxIntTexture;
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleApproxIntSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleApproxIntUAV;
	lean::com_ptr<ID3D11Texture3D> rayBundleApproxTexture;
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleApproxSRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayBundleApproxUAV;

	lean::com_ptr<ID3D11Texture2D> rayBundleDepthBuffer;
	lean::com_ptr<ID3D11DepthStencilView> rayBundleDepthDSV;

	lean::com_ptr<ID3D11Buffer> queueBuffer;
	lean::com_ptr<ID3D11UnorderedAccessView> queueUAV;

	lean::com_ptr<ID3D11Buffer> rayGeometryBuffer;
	lean::com_ptr<ID3D11ShaderResourceView> rayGeometrySRV;
	lean::com_ptr<ID3D11UnorderedAccessView> rayGeometryUAV;

	static const uint4 DebugTextureCount = 3;
	lean::com_ptr<ID3D11Texture2D> rayBundleDebugTexture[DebugTextureCount];
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleDebugSRV[DebugTextureCount];
	static const uint4 Debug4TextureCount = 1;
	lean::com_ptr<ID3D11Texture2D> rayBundleDebug4Texture[Debug4TextureCount];
	lean::com_ptr<ID3D11ShaderResourceView> rayBundleDebug4SRV[Debug4TextureCount];

	lean::com_ptr<ID3D11Buffer> runtimeInfoBuffer;

	lean::com_ptr<ID3D11Buffer> stagingBuffer;

	struct Stage
	{
		enum T
		{
			Setup = 0,
			Bound,
			Prepare,
			Readback,
			Scatter,
			Histopyramid,
			Approx,
			TraceMask,
			Trace,
			Gather,
			Lighting,
			Frame,

			Count
		};
	};

	struct ComputeStage
	{
		lean::resource_ptr<const beg::Any::Technique> technique;
		lean::resource_ptr<beg::Setup> setup;

		ID3DX11Effect *effect;
		ID3DX11EffectUnorderedAccessViewVariable *bundleDebug;

		static void FromMaterial(ComputeStage &cs, besc::Material &material, uint4 layerID)
		{
			cs.technique = ToImpl(material.GetTechnique(layerID));
			cs.setup = material.GetTechniqueSetup(layerID);
			cs.effect = *cs.technique->GetEffect();
			cs.bundleDebug = cs.effect->GetVariableByName("BundleDebugUAV")->AsUnorderedAccessView();
		}
	};

	struct SetupStage : public ComputeStage
	{
		ID3DX11EffectConstantBuffer *bundleDescs;
		ID3DX11EffectUnorderedAccessViewVariable *bundleData;
	} setupStage;

	struct BoundStage
	{
		lean::resource_ptr<besc::QuadProcessor> processor;
		uint4 layerID;
		lean::resource_ptr<beg::Setup> setup;
		
		ID3DX11Effect *effect;
		ID3DX11EffectConstantBuffer *bundleDescs;
		ID3DX11EffectUnorderedAccessViewVariable *bundleData;
		ID3DX11EffectUnorderedAccessViewVariable *rayQueue;
		ID3DX11EffectScalarVariable *activeBundleIdx;
	} boundStage;

	struct PrepareStage : public ComputeStage
	{
		ID3DX11EffectConstantBuffer *bundleDescs;
		ID3DX11EffectUnorderedAccessViewVariable *bundleData;
	} prepareStage;

	struct ScheduleStage : public ComputeStage
	{
		ID3DX11EffectShaderResourceVariable *bundleData;
		ID3DX11EffectUnorderedAccessViewVariable *rayQueue;
		ID3DX11EffectUnorderedAccessViewVariable *bundleHeads;
		ID3DX11EffectUnorderedAccessViewVariable *rayList;
		ID3DX11EffectConstantBuffer *sceneApprox;
		ID3DX11EffectShaderResourceVariable *bundleApprox;
		ID3DX11EffectScalarVariable *activeBundleIdx;
		ID3DX11EffectVectorVariable *bundleResolution;
	} scheduleStage;

	struct GatherStage
	{
		lean::resource_ptr<besc::QuadProcessor> processor;
		lean::resource_ptr<beg::Setup> setup;
		uint4 layerID;
		
		ID3DX11Effect *effect;
	} gatherStage;

	struct DebugStage
	{
		lean::resource_ptr<besc::QuadProcessor> processor;
		lean::resource_ptr<beg::Setup> setup;
		uint4 layerID;
		
		ID3DX11Effect *effect;
		ID3DX11EffectShaderResourceVariable *bundleDebug;
		ID3DX11EffectShaderResourceVariable *bundleDebug4;
		ID3DX11EffectVectorVariable *bundleResolution;
	} debugStage;

	struct TraceMaskStage
	{
		lean::resource_ptr<besc::QuadProcessor> processor;
		lean::resource_ptr<beg::Setup> setup;
		uint4 layerID;
		
		ID3DX11Effect *effect;
	} traceMaskStage;

	struct ArrayToUint4
	{
		lean::com_ptr<ID3DX11Effect> effect;
		ID3DX11EffectTechnique *technique;
		ID3DX11EffectShaderResourceVariable *source;
		ID3DX11EffectUnorderedAccessViewVariable *dest;
		uint4 blockSize;
	} arrayToUint4;

	HistopyramidBinder histopyramidBinder;

	BundleDescLayout rayBundleDescs;

	bool bTracingEnabled;
	bool bDebugOutput;
	uint4 rayCount;
	uint4 listNodeCount;
	float avgRayLength;
	uint4 hitTests;
	float avgHitTests;

	lean::com_ptr<ID3D11Query> timingQuery;
	IncrementalGPUTimer timers[Stage::Count];

	lean::scoped_ptr<TwBar, tw_delete_policy> tweakBar;

	M(besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
		: renderer( LEAN_ASSERT_NOT_NULL(renderer) ),
		resourceManager( LEAN_ASSERT_NOT_NULL(resourceManager) ),
		
		rayBundleResolution(BundleResolution),
		rayBundleMipCount(BundleMipCount),
		rayBundleCount(6),
		rayBundleLayerCount(32),
		rayBundleStride(32),

		tracingWorkerCount(BundleResolution),
		tracingQueueSize(1), // BundleResolution * 256

		bTracingEnabled(true),
		bDebugOutput(false)
	{
		besc::RenderingPipeline &pipeline = *renderer->Pipeline();
		
		// Pipeline stages
		geometryStageID = pipeline.GetStageID("GeometryPipelineStage");

		if (geometryStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "GeometryPipelineStage");

		tracingStageID = pipeline.GetStageID("TracingPipelineStage");

		if (tracingStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "TracingPipelineStage");

		approxStageID = pipeline.GetStageID("ApproximationPipelineStage");

		if (approxStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "ApproximationPipelineStage");

		beg::Any::Device &device = ToImpl(*renderer->Device());
		beg::SwapChainDesc swapChainDesc = device.GetHeadSwapChain(0)->GetDesc();

		sceneApproxBuffer = beg::Any::CreateConstantBuffer(device, sizeof(SceneApproximation));

		// Bundle buffers
		rayBundleDescBuffer = beg::Any::CreateConstantBuffer(device, sizeof(BundleDescLayout));

		rayBundleDataBuffer = beg::Any::CreateStructuredBuffer(
				device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(BundleData), 1
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayBundleDataBuffer, nullptr, rayBundleDataSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayBundleDataBuffer, nullptr, rayBundleDataUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		rayBundleDebugBuffer = beg::Any::CreateStructuredBuffer(
				device,
				D3D11_BIND_UNORDERED_ACCESS,
				sizeof(BundleDebugInfo), MaxBundleCount
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayBundleDebugBuffer, nullptr, rayBundleDebugUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		// Ray buffers
		D3D11_UNORDERED_ACCESS_VIEW_DESC rayDescUAVDesc;
		rayDescUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		rayDescUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		rayDescUAVDesc.Buffer.FirstElement = 0;
		rayDescUAVDesc.Buffer.NumElements = swapChainDesc.Display.Width * swapChainDesc.Display.Height;
		rayDescUAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;

		rayDescBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(float) * 6, rayDescUAVDesc.Buffer.NumElements
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayDescBuffer, nullptr, rayDescSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayDescBuffer, &rayDescUAVDesc, rayDescUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		D3D11_UNORDERED_ACCESS_VIEW_DESC rayListNodeUAVDesc;
		rayListNodeUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		rayListNodeUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		rayListNodeUAVDesc.Buffer.FirstElement = 0;
		rayListNodeUAVDesc.Buffer.NumElements = 40 * swapChainDesc.Display.Width * swapChainDesc.Display.Height;
		rayListNodeUAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;

		rayListNodeBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
#ifdef RAY_LISTS
				2 * sizeof(uint4),
#else
#ifdef INLINED_RAYS
				sizeof(uint4) + 6 * sizeof(float),
#else
				sizeof(uint4),
#endif
#endif
				rayListNodeUAVDesc.Buffer.NumElements
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayListNodeBuffer, nullptr, rayListNodeSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayListNodeBuffer, &rayListNodeUAVDesc, rayListNodeUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		rayBundleCountTexture = beg::Any::CreateTexture3D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32_UINT,
				rayBundleResolution, rayBundleResolution, rayBundleStride, rayBundleMipCount
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayBundleCountTexture, nullptr, rayBundleCountSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayBundleCountTexture, nullptr, rayBundleCountUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );
		for (uint4 j = 0; j < rayBundleMipCount; ++j)
		{
			D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
			SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE3D;
			SRVDesc.Format = DXGI_FORMAT_R32_UINT;
			SRVDesc.Texture3D.MostDetailedMip = j;
			SRVDesc.Texture3D.MipLevels = 1;

			BE_THROW_DX_ERROR_MSG(
				device->CreateShaderResourceView(rayBundleCountTexture, &SRVDesc, rayBundleCountMipSRVs[j].rebind()),
				"ID3D11Device::CreateShaderResourceView()" );

			D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
			UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_TEXTURE3D;
			UAVDesc.Format = DXGI_FORMAT_R32_UINT;
			UAVDesc.Texture3D.MipSlice = j;
			UAVDesc.Texture3D.FirstWSlice = 0;
			UAVDesc.Texture3D.WSize = -1;

			BE_THROW_DX_ERROR_MSG(
				device->CreateUnorderedAccessView(rayBundleCountTexture, &UAVDesc, rayBundleCountMipUAVs[j].rebind()),
				"ID3D11Device::CreateUnorderedAccessView()" );
		}

		rayBundleHeadTexture = beg::Any::CreateTexture3D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32_UINT,
				rayBundleResolution, rayBundleResolution, rayBundleStride
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayBundleHeadTexture, nullptr, rayBundleHeadSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayBundleHeadTexture, nullptr, rayBundleHeadUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		rayBundleApproxIntTexture = beg::Any::CreateTexture2D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32_UINT,
				rayBundleResolution, rayBundleResolution, 4 * MaxSliceSets
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayBundleApproxIntTexture, nullptr, rayBundleApproxIntSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayBundleApproxIntTexture, nullptr, rayBundleApproxIntUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		rayBundleApproxTexture = beg::Any::CreateTexture3D(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				DXGI_FORMAT_R32G32B32A32_UINT,
				rayBundleResolution, rayBundleResolution, MaxSliceSets
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayBundleApproxTexture, nullptr, rayBundleApproxSRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayBundleApproxTexture, nullptr, rayBundleApproxUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		rayBundleDepthBuffer = beg::Any::CreateTexture2D(device,
				D3D11_BIND_DEPTH_STENCIL,
				DXGI_FORMAT_D16_UNORM,
				rayBundleResolution, rayBundleResolution
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateDepthStencilView(rayBundleDepthBuffer, nullptr, rayBundleDepthDSV.rebind()),
			"ID3D11Device::CreateDepthStencilView()" );

		rayGeometryBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
				sizeof(uint4) * 5, rayDescUAVDesc.Buffer.NumElements
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateShaderResourceView(rayGeometryBuffer, nullptr, rayGeometrySRV.rebind()),
			"ID3D11Device::CreateShaderResourceView()" );
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(rayGeometryBuffer, nullptr, rayGeometryUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		for (uint4 i = 0; i < DebugTextureCount; ++i)
		{
			rayBundleDebugTexture[i] = beg::Any::CreateTexture2D(device,
					D3D11_BIND_SHADER_RESOURCE,
					DXGI_FORMAT_R32_UINT,
					rayBundleResolution, rayBundleResolution, rayBundleCount
				);
			BE_THROW_DX_ERROR_MSG(
				device->CreateShaderResourceView(rayBundleDebugTexture[i], nullptr, rayBundleDebugSRV[i].rebind()),
				"ID3D11Device::CreateShaderResourceView()" );
		}
		for (uint4 i = 0; i < Debug4TextureCount; ++i)
		{
			rayBundleDebug4Texture[i] = beg::Any::CreateTexture2D(device,
					D3D11_BIND_SHADER_RESOURCE,
					DXGI_FORMAT_R32G32B32A32_UINT,
					rayBundleResolution, rayBundleResolution, rayBundleCount
				);
			BE_THROW_DX_ERROR_MSG(
				device->CreateShaderResourceView(rayBundleDebug4Texture[i], nullptr, rayBundleDebug4SRV[i].rebind()),
				"ID3D11Device::CreateShaderResourceView()" );
		}

		D3D11_UNORDERED_ACCESS_VIEW_DESC queueUAVDesc;
		queueUAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		queueUAVDesc.Format = DXGI_FORMAT_UNKNOWN;
		queueUAVDesc.Buffer.FirstElement = 0;
		queueUAVDesc.Buffer.NumElements = tracingQueueSize;
		queueUAVDesc.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_COUNTER;

		queueBuffer = beg::Any::CreateStructuredBuffer(device,
				D3D11_BIND_UNORDERED_ACCESS,
				sizeof(float) * 8 * 3 + sizeof(uint4), queueUAVDesc.Buffer.NumElements
			);
		BE_THROW_DX_ERROR_MSG(
			device->CreateUnorderedAccessView(queueBuffer, &queueUAVDesc, queueUAV.rebind()),
			"ID3D11Device::CreateUnorderedAccessView()" );

		stagingBuffer = beg::Any::CreateStagingBuffer(device, sizeof(StagingData));

		// Ray bundle shader code
		lean::resource_ptr<besc::Material> rayBundleMaterial = lean::new_resource<besc::Material>(
				resourceManager->EffectCache()->GetEffect("Hardwired/RayBundles.fx", nullptr, 0),
				*resourceManager->EffectCache(),
				*resourceManager->TextureCache()
			);

		// TODO: Kinda dirty
		// Extract runtime info buffer
		ToImpl(*rayBundleMaterial->GetEffect())->GetConstantBufferByName("RuntimeInfo")->GetConstantBuffer(runtimeInfoBuffer.rebind());

		// Bound stage
		boundStage.processor = lean::new_resource<besc::QuadProcessor>(renderer->Device(), renderer->ProcessingDrivers());
		boundStage.processor->SetMaterial(rayBundleMaterial);
		
		boundStage.layerID = 0;
		boundStage.setup = rayBundleMaterial->GetTechniqueSetup(1);
		boundStage.effect = ToImpl(boundStage.setup->GetEffect())->Get();
		
		boundStage.bundleDescs = boundStage.effect->GetConstantBufferByName("BundleDescs");
		boundStage.bundleData = boundStage.effect->GetVariableByName("BundleDataUAV")->AsUnorderedAccessView();
		boundStage.rayQueue = boundStage.effect->GetVariableByName("RayQueueUAV")->AsUnorderedAccessView();
		boundStage.activeBundleIdx = boundStage.effect->GetVariableByName("ActiveBundleIndex")->AsScalar();

		// Setup stage
		ComputeStage::FromMaterial(setupStage, *rayBundleMaterial, 0);
		setupStage.bundleDescs = setupStage.effect->GetConstantBufferByName("BundleDescs");
		setupStage.bundleData = setupStage.effect->GetVariableByName("BundleDataUAV")->AsUnorderedAccessView();

		// Prepare stage
		ComputeStage::FromMaterial(prepareStage, *rayBundleMaterial, 2);
		prepareStage.bundleDescs = prepareStage.effect->GetConstantBufferByName("BundleDescs");
		prepareStage.bundleData = prepareStage.effect->GetVariableByName("BundleDataUAV")->AsUnorderedAccessView();

		// Scatter stage
		ComputeStage::FromMaterial(scheduleStage, *rayBundleMaterial, 3);
		scheduleStage.bundleData = scheduleStage.effect->GetVariableByName("BundleDataBuffer")->AsShaderResource();
		scheduleStage.rayQueue = scheduleStage.effect->GetVariableByName("RayQueueUAV")->AsUnorderedAccessView();
		scheduleStage.bundleHeads = scheduleStage.effect->GetVariableByName("BundleHeadsUAV")->AsUnorderedAccessView();
		scheduleStage.rayList = scheduleStage.effect->GetVariableByName("RayListUAV")->AsUnorderedAccessView();
		scheduleStage.sceneApprox = scheduleStage.effect->GetConstantBufferByName("SceneApproximationConstants");
		scheduleStage.bundleApprox = scheduleStage.effect->GetVariableByName("SceneApproximationTexture")->AsShaderResource();
		scheduleStage.activeBundleIdx = boundStage.effect->GetVariableByName("ActiveBundleIndex")->AsScalar();
		scheduleStage.bundleResolution = scheduleStage.effect->GetVariableByName("BundleHeadResolution")->AsVector();

		// Debug stage
		debugStage.processor = boundStage.processor;
		debugStage.layerID = 1;
		debugStage.setup = rayBundleMaterial->GetTechniqueSetup(4);

		debugStage.effect = boundStage.effect;
		debugStage.bundleDebug = scheduleStage.effect->GetVariableByName("BundleDebugTexture")->AsShaderResource();
		debugStage.bundleDebug4 = scheduleStage.effect->GetVariableByName("BundleDebug4Texture")->AsShaderResource();
		debugStage.bundleResolution = scheduleStage.effect->GetVariableByName("BundleHeadResolution")->AsVector();

		// Gather stage
		gatherStage.processor = boundStage.processor;
		gatherStage.layerID = 2;
		gatherStage.setup = rayBundleMaterial->GetTechniqueSetup(5);

		gatherStage.effect = boundStage.effect;

		// Trace mask stage
		traceMaskStage.processor = boundStage.processor;
		traceMaskStage.layerID = 3;
		traceMaskStage.setup = rayBundleMaterial->GetTechniqueSetup(6);

		traceMaskStage.effect = boundStage.effect;

		// Helper shader code
		arrayToUint4.effect = ToImpl(resourceManager->EffectCache()->GetEffect("Hardwired/MemCopy.fx", nullptr, 0))->Get();
		arrayToUint4.technique = arrayToUint4.effect->GetTechniqueByName("ArrayToUint4");
		arrayToUint4.source = arrayToUint4.effect->GetVariableByName("SrcUintTA")->AsShaderResource();
		arrayToUint4.dest = arrayToUint4.effect->GetVariableByName("DestUintT")->AsUnorderedAccessView();
		arrayToUint4.blockSize = 8;

		histopyramidBinder.Construct(boundStage.effect, boundStage.effect->GetTechniqueByName("PrefixSum"));

		// Timing
		timingQuery = beg::Any::CreateTimingQuery(device);
		for (uint4 i = 0; i < Stage::Count; ++i)
			timers[i].Construct(device);

		tweakBar = TwNewBar("Tracing Pipeline");
		TwAddVarRO(tweakBar, "setup", TW_TYPE_FLOAT, &timers[Stage::Setup].ms, "group=timing");
		TwAddVarRO(tweakBar, "bound", TW_TYPE_FLOAT, &timers[Stage::Bound].ms, "group=timing");
		TwAddVarRO(tweakBar, "prepare", TW_TYPE_FLOAT, &timers[Stage::Prepare].ms, "group=timing");
		TwAddVarRO(tweakBar, "readback", TW_TYPE_FLOAT, &timers[Stage::Readback].ms, "group=timing");
		TwAddVarRO(tweakBar, "approx", TW_TYPE_FLOAT, &timers[Stage::Approx].ms, "group=timing");
		TwAddVarRO(tweakBar, "scatter", TW_TYPE_FLOAT, &timers[Stage::Scatter].ms, "group=timing");
		TwAddVarRO(tweakBar, "histopyramid", TW_TYPE_FLOAT, &timers[Stage::Histopyramid].ms, "group=timing");
		TwAddVarRO(tweakBar, "trace mask", TW_TYPE_FLOAT, &timers[Stage::TraceMask].ms, "group=timing");
		TwAddVarRO(tweakBar, "trace", TW_TYPE_FLOAT, &timers[Stage::Trace].ms, "group=timing");
		TwAddVarRO(tweakBar, "gather", TW_TYPE_FLOAT, &timers[Stage::Gather].ms, "group=timing");
		TwAddVarRO(tweakBar, "lighting", TW_TYPE_FLOAT, &timers[Stage::Lighting].ms, "group=timing");
		TwAddVarRO(tweakBar, "frame", TW_TYPE_FLOAT, &timers[Stage::Frame].ms, "group=timing");

		TwAddVarRO(tweakBar, "ray count", TW_TYPE_UINT32, &rayCount, "group=info");
		TwAddVarRO(tweakBar, "node count", TW_TYPE_UINT32, &listNodeCount, "group=info");
		TwAddVarRO(tweakBar, "avg length", TW_TYPE_FLOAT, &avgRayLength, "group=info");
		TwAddVarRO(tweakBar, "hit tests", TW_TYPE_UINT32, &hitTests, "group=info");
		TwAddVarRO(tweakBar, "avg hit tests", TW_TYPE_FLOAT, &avgHitTests, "group=info");

		TwAddVarRW(tweakBar, "enabled", TW_TYPE_BOOL8, &bTracingEnabled, "group=control");
		TwAddVarRW(tweakBar, "debug", TW_TYPE_BOOL8, &bDebugOutput, "group=control");
		TwAddVarRW(tweakBar, "layers", TW_TYPE_UINT32, &rayBundleLayerCount, "group=control");
	}
};

// Constructor.
CastingPipeline::CastingPipeline(besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
	: m( new M(renderer, resourceManager) )
{
}

// Destructor.
CastingPipeline::~CastingPipeline()
{
}

namespace ScatterPass
{
	enum T
	{
		Count = 0,
		Write = 1
	};
}

void Scatter(CastingPipeline::M &m, const beg::Any::DeviceContext &deviceContext, ScatterPass::T pass, uint4 bundleIdx)
{
	const float BundleResolution[4] = {
		(float) m.rayBundleResolution, (float) m.rayBundleResolution,
		1.0f / m.rayBundleResolution, 1.0f / m.rayBundleResolution
	};

	// Scatter rays to ray lists
	m.scheduleStage.activeBundleIdx->SetInt(bundleIdx);
	m.scheduleStage.bundleResolution->SetFloatVector(BundleResolution);

	if (pass == ScatterPass::Count)
		deviceContext->ClearUnorderedAccessViewUint(m.rayBundleCountUAV, Zeroes);
#ifdef RAY_LISTS
	if (pass == ScatterPass::Write)
		deviceContext->ClearUnorderedAccessViewUint(m.rayBundleHeadUAV, MinusOnes);
#endif

	m.scheduleStage.bundleData->SetResource(m.rayBundleDataSRV);
	
	m.scheduleStage.rayQueue->SetUnorderedAccessView(m.rayDescUAV, 0);
	if (pass == ScatterPass::Count)
		m.scheduleStage.bundleHeads->SetUnorderedAccessView(m.rayBundleCountUAV);
	if (pass == ScatterPass::Write)
		m.scheduleStage.bundleHeads->SetUnorderedAccessView(m.rayBundleHeadUAV);

	m.scheduleStage.rayList->SetUnorderedAccessView(m.rayListNodeUAV, bundleIdx == 0 ? 0 : -1);
	
	m.scheduleStage.sceneApprox->SetConstantBuffer(m.sceneApproxBuffer);
	m.scheduleStage.bundleApprox->SetResource(m.rayBundleApproxSRV);

	m.scheduleStage.bundleDebug->SetUnorderedAccessView(m.rayBundleDebugUAV);

	m.scheduleStage.setup->Apply(deviceContext);
	m.scheduleStage.technique->Get()->GetPassByIndex(pass)->Apply(0, deviceContext);

	deviceContext->Dispatch(64, 1, 1);

	beg::Any::UnbindAllComputeTargets(deviceContext);
}

/// Inverse of bitsep1.
uint4 bitcomp1(uint4 x)
{
	// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

/// 2 dimensions from morton code.
bem::vector<uint4, 2> bitunzip2(uint4 c)
{
	return bem::vec( bitcomp1(c), bitcomp1(c >> 1) );
}

// Renders the scene.
void CastingPipeline::Render(besc::SceneController *scene, const besc::PerspectiveDesc &perspectiveDesc, besc::Pipe *pipe, besc::PipelineProcessor *pProcessor)
{
	LEAN_PIMPL_M

	const float BundleResolution[4] = {
			(float) m.rayBundleResolution, (float) m.rayBundleResolution,
			1.0f / m.rayBundleResolution, 1.0f / m.rayBundleResolution
		};

	besc::RenderingPipeline &pipeline = *m.renderer->Pipeline();
	besc::RenderContext &renderContext = *scene->GetRenderContext();
	const beg::Any::DeviceContext &deviceContext = ToImpl(renderContext.Context());

	// Reset timers
	for (uint4 i = 0; i < M::Stage::Count; ++i)
		m.timers[i].Reset();

	deviceContext->Begin(m.timingQuery);
	m.timers[M::Stage::Frame].Begin(deviceContext);

	// Enable cross-pipeline results
	pipe->KeepResults();

	bool bTrace = true;
	bool bSchedule = bTrace || true;
	bool bCount = !bTrace && false;
	bool bApprox = bSchedule || true;
	bool bGather = bTrace || true;
	bool bDebugOutput = m.bTracingEnabled && m.bDebugOutput;

	if (m.bTracingEnabled)
	{
		lean::com_ptr<const beg::ColorTextureTarget> gBufferTarget;
		ID3D11ShaderResourceView *gBuffer;

		// Render main G-Buffer
		{
			besc::PipelinePerspective &perspective = *pipeline.AddPerspective(perspectiveDesc, pipe, nullptr, 1U << m.geometryStageID);
			scene->Render();
			m.renderer->InvalidateCaches();

			gBufferTarget = pipe->GetColorTarget("SceneGeometryTarget");

			if (!gBufferTarget)
			{
				LEAN_LOG_ERROR_CTX("G-Buffer target missing", "SceneGeometryTarget");
				return;
			}

			gBuffer = gBufferTarget->GetTexture();

			if (!gBuffer)
			{
				LEAN_LOG_ERROR_CTX("G-Buffer not readable", "SceneGeometryTarget");
				return;
			}
		}

		StagingData stagingData;

		// TODO: Allow for variable number?
		LEAN_ASSERT(m.rayBundleCount == 6);

		// IDEA: Timing feedback to optimize dir?
		bem::fmat3 bundleSpace = bem::mat_rot_yxz<3>(bem::Constants::pi<float>::quarter, bem::Constants::pi<float>::quarter, 0.0f);
		m.rayBundleDescs.BundleSpace = ToMat3Pad(bundleSpace);

		// Generate bundle ray directions
		for (uint4 i = 0; i < m.rayBundleCount; ++i)
		{
			BundleDesc &bundleDesc = m.rayBundleDescs.Descs[i];

			bundleDesc.orientation.dir = bem::nvec<3, float>(i / 2, (i % 2) ? -1.0f : 1.0f);
			
			bundleDesc.orientation.up = (abs( dot(bundleDesc.orientation.dir, perspectiveDesc.CamUp) ) < 0.9f)
				? perspectiveDesc.CamUp
				: perspectiveDesc.CamRight;

			bundleDesc.orientation.right = normalize( cross(bundleDesc.orientation.up, bundleDesc.orientation.dir) );
			bundleDesc.orientation.up = cross(bundleDesc.orientation.dir, bundleDesc.orientation.right);

			bundleDesc.orientation.right = mul(bundleDesc.orientation.right, bundleSpace);
			bundleDesc.orientation.up = mul(bundleDesc.orientation.up, bundleSpace);
			bundleDesc.orientation.dir = mul(bundleDesc.orientation.dir, bundleSpace);

			bundleDesc.fov = bem::pi<float>::half;
			bundleDesc.farPlane = 256.0f;
		}

		// NOTE: Partly uninitialized, filled during bound stage
		deviceContext->UpdateSubresource(m.rayBundleDescBuffer, 0, nullptr, &m.rayBundleDescs, 0, 0);

		{
			// Setup ray bundles
			m.timers[M::Stage::Setup].Begin(deviceContext);

			m.setupStage.bundleDescs->SetConstantBuffer(m.rayBundleDescBuffer);
			m.setupStage.bundleData->SetUnorderedAccessView(m.rayBundleDataUAV);

			m.setupStage.setup->Apply(deviceContext);
			m.setupStage.technique->Get()->GetPassByIndex(0)->Apply(0, deviceContext);

			deviceContext->Dispatch(m.rayBundleCount, 1, 1);

			beg::Any::UnbindAllComputeTargets(deviceContext);

			m.timers[M::Stage::Setup].End(deviceContext);


			// Distribute G-Buffer pixels to ray perspectives (Back pointer & count & view port)
			m.timers[M::Stage::Bound].Begin(deviceContext);

			besc::PipelinePerspective &perspective = *pipeline.AddPerspective(perspectiveDesc, pipe);

			m.boundStage.bundleDescs->SetConstantBuffer(m.rayBundleDescBuffer);
			m.boundStage.bundleData->SetUnorderedAccessView(m.rayBundleDataUAV);
			
			for (uint4 bundleIdx = 0; bundleIdx < m.rayBundleCount; ++bundleIdx)
			{
				m.boundStage.rayQueue->SetUnorderedAccessView(m.rayDescUAV, (bundleIdx == 0) ? 0 : -1);
				m.boundStage.activeBundleIdx->SetInt(bundleIdx);

				m.boundStage.processor->Render(m.boundStage.layerID, &perspective, renderContext);

				beg::Any::UnbindAllRenderTargets(deviceContext);


				// Fetch local ray offset
				deviceContext->CopyStructureCount(
						m.rayBundleDescBuffer,
						offsetof(BundleDescLayout, RayOffsets) + bundleIdx * 4 * sizeof(uint4),
						m.rayDescUAV
					);
			}

			pipeline.ClearPerspectives();
			m.renderer->InvalidateCaches();

			m.timers[M::Stage::Bound].End(deviceContext);


			// Fetch total ray count
			deviceContext->CopyStructureCount(
					m.runtimeInfoBuffer,
					offsetof(RuntimeInfo, totalRayCount),
					m.rayDescUAV
				);


			// Prepare ray bundle processing
			m.timers[M::Stage::Prepare].Begin(deviceContext);

			m.prepareStage.bundleDescs->SetConstantBuffer(m.rayBundleDescBuffer);
			m.prepareStage.bundleData->SetUnorderedAccessView(m.rayBundleDataUAV);

			m.prepareStage.bundleDebug->SetUnorderedAccessView(m.rayBundleDebugUAV);

			m.prepareStage.setup->Apply(deviceContext);
			m.prepareStage.technique->Get()->GetPassByIndex(0)->Apply(0, deviceContext);

			deviceContext->Dispatch(m.rayBundleCount, 1, 1);

			m.timers[M::Stage::Prepare].End(deviceContext);

			beg::Any::UnbindAllComputeTargets(deviceContext);


			// DEBUG: Fetch debug info
			beg::Any::CopyBuffer(deviceContext,
					m.stagingBuffer, offsetof(StagingData, bundleDebugInfo), 
					m.rayBundleDebugBuffer, 0, sizeof(stagingData.bundleDebugInfo)
				);

			m.timers[M::Stage::Readback].Begin(deviceContext);
			
			// Fetch bundle data
			beg::Any::CopyBuffer(deviceContext,
					m.stagingBuffer, offsetof(StagingData, bundleData), 
					m.rayBundleDataBuffer, 0, sizeof(stagingData.bundleData)
				);

			deviceContext->Flush();

			// Read back
			beg::Any::ReadBufferData(deviceContext, m.stagingBuffer, &stagingData, sizeof(stagingData));

			m.timers[M::Stage::Readback].End(deviceContext);
		}

//		uint4 rayCount = 0;

		for (uint4 bundleIdx = 0; bundleIdx < m.rayBundleCount; ++bundleIdx)
		{
			const bem::fmat4 &bundleView = stagingData.bundleData.View[bundleIdx];
			const bem::fmat4 &bundleProj = stagingData.bundleData.Projection[bundleIdx];

			bem::fmat4 bundleCam = bem::mat_transform_inverse(
					bem::fvec3(bundleView[3]),
					bem::fvec3(bundleView[2]), bem::fvec3(bundleView[1]), bem::fvec3(bundleView[0])
				);

			besc::PerspectiveDesc bundlePerspectiveDesc(
					bem::fvec3(bundleCam[3]),
					bem::fvec3(bundleCam[0]),
					bem::fvec3(bundleCam[1]),
					bem::fvec3(bundleCam[2]),
					bundleView,
					bundleProj,
					0.0f,
					m.rayBundleDescs.Descs[bundleIdx].farPlane,
					perspectiveDesc.Flipped,
					perspectiveDesc.Time,
					perspectiveDesc.TimeStep,
					bundleIdx
				);

			SceneApproximation approx;
			approx.Depth = m.rayBundleDescs.Descs[bundleIdx].farPlane;
			approx.MinDepth = 0.0f;
			approx.MaxDepth = approx.Depth;

			approx.LayerCount = (float) m.rayBundleLayerCount;
			approx.LayerWidth = 0.6f * approx.Depth / approx.LayerCount;
			approx.OneOverLayerWidth = 1.0f / approx.LayerWidth;

			approx.SliceCount = MaxSliceSets * 128.0f;
			approx.SliceWidth = 0.4f * approx.Depth / approx.SliceCount;
			approx.OneOverSliceWidth = 1.0f / approx.SliceWidth;
			approx.SlicesPerLayer = approx.SliceCount / approx.LayerCount;
			
			approx.BundleStride = m.rayBundleStride;
			approx.Resolution = bem::vec<float>(BundleResolution[0], BundleResolution[1]);
			approx.PixelWidth = bem::vec<float>(BundleResolution[2], BundleResolution[3]);

			// Compute voxel approximation
			if (bApprox)
			{
				m.timers[M::Stage::Approx].Begin(deviceContext);


				besc::PipelinePerspective &bundlePerspective = *pipeline.AddPerspective(bundlePerspectiveDesc, nullptr, nullptr, 1U << m.approxStageID);

				deviceContext->UpdateSubresource(m.sceneApproxBuffer, 0, nullptr, &approx, 0, 0);
			
				ToImpl(renderContext.StateManager()).Revert();
				ToImpl(renderContext.StateManager()).Reset();

				deviceContext->VSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
				deviceContext->DSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
				deviceContext->GSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
				deviceContext->PSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
			
				deviceContext->ClearUnorderedAccessViewUint(m.rayBundleApproxIntUAV, Zeroes);
			
				deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr, 0, 1, &m.rayBundleApproxIntUAV.get(), nullptr);

				D3D11_VIEWPORT approxViewport = { 0.0f, 0.0f, approx.Resolution[0], approx.Resolution[1], 0.0f, 1.0f };
				deviceContext->RSSetViewports(1, &approxViewport);

				ToImpl(renderContext.StateManager()).Override(beg::Any::StateMasks::RenderTargets);
				ToImpl(renderContext.StateManager()).RecordOverridden();

				scene->Render();
				m.renderer->InvalidateCaches();

				beg::Any::UnbindAllRenderTargets(deviceContext);


				m.arrayToUint4.source->SetResource(m.rayBundleApproxIntSRV);
				m.arrayToUint4.dest->SetUnorderedAccessView(m.rayBundleApproxUAV);
				m.arrayToUint4.technique->GetPassByIndex(0)->Apply(0, deviceContext);
				deviceContext->Dispatch(m.rayBundleResolution / m.arrayToUint4.blockSize, m.rayBundleResolution / m.arrayToUint4.blockSize, 1);

				beg::Any::UnbindAllComputeTargets(deviceContext);

				// DEBUG
				if (bDebugOutput)
				{
					D3D11_BOX debugBox = { 0, 0, 0, m.rayBundleResolution, m.rayBundleResolution, 1 };
					deviceContext->CopySubresourceRegion(m.rayBundleDebug4Texture[0], bundleIdx, 0, 0, 0,
						m.rayBundleApproxTexture, 0, &debugBox);
				}

				m.timers[M::Stage::Approx].End(deviceContext);
			}

			// Scatter rays to ray lists
			if (bSchedule)
			{
				D3D11_BOX debugBox = { 0, 0, 0, m.rayBundleResolution, m.rayBundleResolution, 1 };

#ifndef RAY_LISTS
				m.timers[M::Stage::Scatter].Begin(deviceContext);
				Scatter(m, deviceContext, ScatterPass::Count, bundleIdx);
				m.timers[M::Stage::Scatter].End(deviceContext);

				// DEBUG
				if (bDebugOutput)
					deviceContext->CopySubresourceRegion(m.rayBundleDebugTexture[0], bundleIdx, 0, 0, 0,
						m.rayBundleCountTexture, 0, &debugBox);

				m.timers[M::Stage::Histopyramid].Begin(deviceContext);

				m.histopyramidBinder.Dispatch( deviceContext,
						&m.rayBundleCountMipSRVs->get(),
						&m.rayBundleCountMipUAVs->get(),
						m.rayBundleCountSRV,
						m.rayBundleHeadUAV,
						m.rayListNodeUAV,
						m.rayBundleResolution, m.rayBundleStride
					);

				m.timers[M::Stage::Histopyramid].End(deviceContext);

/*				uint4 localRayCount = -1;
				beg::Any::DebugFetchTextureData(deviceContext, m.rayBundleCountTexture, &localRayCount, sizeof(localRayCount), 9);
//				rayCount += localRayCount;

				static uint4 headCounts[32][512][512];
				beg::Any::DebugFetchTextureData(deviceContext, m.rayBundleCountTexture, headCounts, sizeof(headCounts));

				static uint4 headOffsets[32][512][512];
				beg::Any::DebugFetchTextureData(deviceContext, m.rayBundleHeadTexture, headOffsets, sizeof(headOffsets));

				uint4 lastOffset = 0;
				uint4 checkRayCount = 0;

				uint4 listCount = 0, wrongListCount = 0;

				for (uint4 i = 0; i < 512 * 512; ++i)
				{
					bem::vector<uint4, 2> v = bitunzip2(i);
					uint4 nextOffset = headOffsets[0][v[1]][v[0]];
					uint4 nextCount = headCounts[0][v[1]][v[0]];

					if (nextOffset != -1)
					{
						LEAN_ASSERT(nextOffset > lastOffset);

						uint4 listLength = nextOffset - lastOffset;

						wrongListCount += (nextCount != listLength);
						++listCount;

						checkRayCount += listLength;
						lastOffset = nextOffset;
					}
					else
						LEAN_ASSERT(nextCount == 0);
				}

				LEAN_ASSERT(checkRayCount == localRayCount);

				for (int i = 0; i < 32; ++i)
					for (int j = 0; j < 512; ++j)
						for (int k = 0; k < 512; ++k)
							if (headOffsets[i][j][k] == localRayCount - 1)
								goto LABEL;
LABEL:
*/
				// DEBUG
				if (bDebugOutput)
					deviceContext->CopySubresourceRegion(m.rayBundleDebugTexture[1], bundleIdx, 0, 0, 0,
						m.rayBundleHeadTexture, 0, &debugBox);
#endif
				m.timers[M::Stage::Scatter].Begin(deviceContext);
				Scatter(m, deviceContext, ScatterPass::Write, bundleIdx);
				m.timers[M::Stage::Scatter].End(deviceContext);

				// DEBUG
				if (bDebugOutput)
					deviceContext->CopySubresourceRegion(m.rayBundleDebugTexture[2], bundleIdx, 0, 0, 0,
						m.rayBundleHeadTexture, 0, &debugBox);


				// DEBUG: Fetch total node count
				deviceContext->CopyStructureCount(
						m.runtimeInfoBuffer,
						offsetof(RuntimeInfo, totalNodeCount),
						m.rayListNodeUAV
					);
//				deviceContext->CopySubresourceRegion(
//					m.runtimeInfoBuffer, 0, offsetof(RuntimeInfo, totalNodeCount), 0, 0,
//					m.rayBundleCount, D3D11CalcSubresource(m.rayBundleMipCount - 1, 0, m.rayBundleMipCount), );
				beg::Any::CopyBuffer(deviceContext,
						m.stagingBuffer, offsetof(StagingData, runtimeInfo), 
						m.runtimeInfoBuffer, 0, sizeof(stagingData.runtimeInfo)
					);
				// Fetch runtime info
				beg::Any::CopyBuffer(deviceContext,
						m.stagingBuffer, offsetof(StagingData, runtimeInfo), 
						m.runtimeInfoBuffer, 0, sizeof(stagingData.runtimeInfo)
					);

				// DEBUG: Fetch debug info
				beg::Any::CopyBuffer(deviceContext,
						m.stagingBuffer, offsetof(StagingData, bundleDebugInfo), 
						m.rayBundleDebugBuffer, 0, sizeof(stagingData.bundleDebugInfo)
					);

				// DEBUG: Fetch total hit count
				deviceContext->CopyStructureCount(
						m.stagingBuffer,
						offsetof(StagingData, totalHitCount),
						m.queueUAV
					);


				// Read back
//				beg::Any::ReadBufferData(deviceContext, m.stagingBuffer, &stagingData, sizeof(stagingData));
				int eos = -1;

				// INFO
				m.rayCount = stagingData.runtimeInfo.totalRayCount;
				m.listNodeCount = stagingData.runtimeInfo.totalNodeCount;
				m.avgRayLength = (float) m.listNodeCount / m.rayCount;
				m.hitTests = stagingData.totalHitCount;
				m.avgHitTests = (float) m.hitTests / m.rayCount;
			}

			// Perform local ray tracing
			if (bTrace)
			{
				besc::PipelinePerspective &bundlePerspective = *pipeline.AddPerspective(bundlePerspectiveDesc, nullptr, nullptr, 1U << m.tracingStageID);

				m.timers[M::Stage::TraceMask].Begin(deviceContext);

				deviceContext->UpdateSubresource(m.sceneApproxBuffer, 0, nullptr, &approx, 0, 0);
				
				ToImpl(renderContext.StateManager()).Revert();
				ToImpl(renderContext.StateManager()).Reset();

				deviceContext->VSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
				deviceContext->DSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
				deviceContext->GSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());
				deviceContext->PSSetConstantBuffers(2, 1, &m.sceneApproxBuffer.get());

				deviceContext->PSSetShaderResources(11, 1, &m.rayDescSRV.get());
				deviceContext->PSSetShaderResources(12, 1, &m.rayListNodeSRV.get());
				deviceContext->PSSetShaderResources(13, 1, &m.rayBundleHeadSRV.get());
				
				D3D11_VIEWPORT approxViewport = { 0.0f, 0.0f, approx.Resolution[0], approx.Resolution[1], 0.0f, 1.0f };
				
				if (bundleIdx == 0)
					deviceContext->ClearUnorderedAccessViewUint(m.rayGeometryUAV, MinusOnes);
//				deviceContext->ClearUnorderedAccessViewUint(m.traceQueueUAV, MinusOnes);
				deviceContext->ClearDepthStencilView(m.rayBundleDepthDSV, D3D11_CLEAR_DEPTH, 0.0f, 0);

				// Mask
				deviceContext->OMSetRenderTargets(0, nullptr, m.rayBundleDepthDSV);
				deviceContext->RSSetViewports(1, &approxViewport);

				ToImpl(renderContext.StateManager()).Override(beg::Any::StateMasks::RenderTargets);
				ToImpl(renderContext.StateManager()).RecordOverridden();

				m.traceMaskStage.processor->Render(m.traceMaskStage.layerID, &bundlePerspective, renderContext);

				m.timers[M::Stage::TraceMask].End(deviceContext);


				m.timers[M::Stage::Trace].Begin(deviceContext);

				// Tracing
				ToImpl(renderContext.StateManager()).Revert();
				ToImpl(renderContext.StateManager()).Reset();

				ID3D11UnorderedAccessView *tracingUAVs[2] = { m.rayGeometryUAV.get(), m.queueUAV.get() };
				uint4 tracingUAVCounts[2] = { -1, (bundleIdx == 0) ? 0 : -1 };
				deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, m.rayBundleDepthDSV, 0, lean::arraylen(tracingUAVs), tracingUAVs, tracingUAVCounts);
				deviceContext->RSSetViewports(1, &approxViewport);

				ToImpl(renderContext.StateManager()).Override(beg::Any::StateMasks::RenderTargets);
				ToImpl(renderContext.StateManager()).RecordOverridden();

				scene->Render();
				m.renderer->InvalidateCaches();

				beg::Any::UnbindAllRenderTargets(deviceContext);

				m.timers[M::Stage::Trace].End(deviceContext);
			}
		}

		// Loop over ray targets
		
			// Compute lighting on ray lists

		// Compute direct lighting
		if (true)
		{
			m.timers[M::Stage::Lighting].Begin(deviceContext);

			besc::PipelinePerspective &perspective = *pipeline.AddPerspective(perspectiveDesc, pipe, nullptr, ~(1U << m.geometryStageID), true);
			scene->Render();
			m.renderer->InvalidateCaches();

			m.timers[M::Stage::Lighting].End(deviceContext);
		}

		// Gather ray fragments (using back pointer) & compute direct lighting
		if (bGather)
		{
			m.timers[M::Stage::Gather].Begin(deviceContext);

			deviceContext->PSSetShaderResources(14, 1, &m.rayGeometrySRV.get());

			besc::PipelinePerspective &perspective = *pipeline.AddPerspective(perspectiveDesc, pipe);
			m.gatherStage.processor->Render(m.gatherStage.layerID, &perspective, renderContext);
			pipeline.ClearPerspectives();
			m.renderer->InvalidateCaches();

			m.timers[M::Stage::Gather].End(deviceContext);
		}
	}

	// Classic rendering
	if (!m.bTracingEnabled)
	{
		besc::PipelinePerspective &perspective = *pipeline.AddPerspective(perspectiveDesc, pipe);
		scene->Render();
		m.renderer->InvalidateCaches();
	}

	// Apply final processing
	if (pProcessor || bDebugOutput)
	{
		besc::PipelinePerspective &perspective = *pipeline.AddPerspective(perspectiveDesc, pipe);

		if (pProcessor)
			pProcessor->Render(&perspective, renderContext);

		if (bDebugOutput)
		{
			m.debugStage.bundleDebug->SetResourceArray(const_cast<ID3D11ShaderResourceView**>(&m.rayBundleDebugSRV->get()), 0, M::DebugTextureCount);
			m.debugStage.bundleDebug4->SetResourceArray(const_cast<ID3D11ShaderResourceView**>(&m.rayBundleDebug4SRV->get()), 0, M::Debug4TextureCount);
			m.debugStage.bundleResolution->SetFloatVector(BundleResolution);

			m.debugStage.processor->Render(m.debugStage.layerID, &perspective, renderContext);
		}

		pipeline.ClearPerspectives();
		m.renderer->InvalidateCaches();
	}

	pipe->KeepResults(false);
	pipe->Release();

	m.timers[M::Stage::Frame].End(deviceContext);
	deviceContext->End(m.timingQuery);

	{
		uint8 frequency = beg::Any::GetTimingFrequency(deviceContext, m.timingQuery);

		for (uint4 i = 0; i < M::Stage::Count; ++i)
		{
			m.timers[i].ReadData(deviceContext);
			m.timers[i].ToMS(frequency);
		}
	}
}

#endif