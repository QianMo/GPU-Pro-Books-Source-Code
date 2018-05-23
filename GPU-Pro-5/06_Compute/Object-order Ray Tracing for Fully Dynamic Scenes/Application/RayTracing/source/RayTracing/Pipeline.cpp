/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/Pipeline.h"

#include "RayTracing/VoxelRep.h"
#include "RayTracing/RayGen.h"
#include "RayTracing/RaySet.h"

#include <beScene/beShaderDrivenPipeline.h>
#include <beGraphics/beEffectCache.h>

#include "RayTracing/TracingEffectBinderPool.h"
#include "RayTracing/RenderableEffectDriverCache.h"
#include "RayTracing/LightEffectDriverCache.h"

#include <beScene/beRenderingPipeline.h>
#include <beScene/beRenderContext.h>

#include <beScene/bePipe.h>
#include <beScene/bePerspectivePool.h>
#include <beScene/bePipelineProcessor.h>

#include <beScene/beQuadProcessor.h>

#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beDevice.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beTexture.h>
#include <beGraphics/Any/beTextureTargetPool.h>
#include <beGraphics/Any/beStateManager.h>

#include <beGraphics/Any/beQuery.h>
#include "IncrementalGPUTimer.h"

#include "HLSLPacking.h"

#include <beMath/beVector.h>
#include <beMath/beMatrix.h>

#include <AntTweakBar.h>

#include <beGraphics/DX/beError.h>
#include <lean/logging/errors.h>

#include <lean/containers/array.h>

#include <lean/meta/math.h>

// #define SMALL_BAR
// #define SHOW_INTERSECTION_STATISTICS // requires RECORD_INTERSECTION_STATISTICS in RayTracing/Materials/Textured.fx

 #define TIMING(x) x
// #define TIMING(x) nullptr

namespace app
{

namespace tracing
{

const bem::vector<uint4, 3> GridResolution = bem::vec<uint4>(128, 128, 128);
const uint4 GridMipCount = 1 + lean::log2<uint4, 128>::value;

const uint4 Zeroes[4] = { 0 };
const uint4 MinusOnes[4] = { -1, -1, -1, -1 };
const uint4 MaxInts[4] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };

struct DebugInfo
{
	uint4 ErrorRays;
	uint4 LastErrorRay;
	uint4 TotalRayTriangle;
	uint4 _1;

	bem::fvec3 ErrorRayDir;
	uint4 _2;
};

struct StagingData
{
	VoxelRepLayout voxelRep;
	RaySetLayout raySet;
	
	uint4 totalTestCount;
	uint4 totalHitCount;
	uint4 totalAtmoCount;
	uint4 _pad1;

	DebugInfo debugInfo;
};

struct tw_delete_policy
{
	static void release(TwBar *bar)
	{
		TwDeleteBar(bar);
	}
};

namespace
{

/// Creates a tracing renderer from the given renderer.
lean::resource_ptr<besc::EffectDrivenRenderer, true> CreateTracingRenderer(besc::EffectDrivenRenderer &renderer, bec::ComponentMonitor *pMonitor,
																		   TracingEffectBinderPool *tracingPool)
{
	lean::resource_ptr<RenderableEffectDriverCache> renderableCache = new_resource RenderableEffectDriverCache(
		renderer.RenderableDrivers(), renderer.Pipeline(), renderer.PerspectiveEffectBinderPool(), tracingPool);
	lean::resource_ptr<LightEffectDriverCache> lightCache = new_resource LightEffectDriverCache(
		renderer.LightDrivers(), renderer.Pipeline(), renderer.PerspectiveEffectBinderPool(), tracingPool);

	return besc::CreateEffectDrivenRenderer(renderer, pMonitor,
			renderer.PerspectiveEffectBinderPool(),
			renderer.ProcessingDrivers(),
			renderableCache, nullptr, nullptr,
			lightCache, nullptr
		);
}

/// Gets the number of pixels in the givne swap chain.
uint4 GetPixelCount(const beg::SwapChain &swapChain)
{
	beg::SwapChainDesc desc = swapChain.GetDesc();
	return desc.Display.Width * desc.Display.Height;
}

} // namespace

struct Pipeline::M
{
	uint4 maxRayCount;
	uint4 maxTriBatchSize;

	VoxelRep voxelRep;
	RaySet raySet;

	lean::resource_ptr<besc::ResourceManager> resourceManager;
	lean::resource_ptr<TracingEffectBinderPool> tracingPool;
	lean::resource_ptr<besc::EffectDrivenRenderer> renderer;

	uint4 geometryStageID;
	uint4 shadowStageID;
	uint4 voxelRepStageID;
	uint4 tracingStageID;
	uint4 partialTracingStageID;
	uint4 traceLightingStageID;
	uint4 processingStageID;

	VoxelRepGen voxelRepStage;
	RayGen rayGenStage;
	RayGridGen rayGridStage;
	
	lean::com_ptr<beg::api::Buffer> stagingBuffer;
	lean::com_ptr<beg::api::Buffer> debugInfoBuffer;
	lean::com_ptr<beg::api::UnorderedAccessView> debugInfoUAV;

	uint4 frameIndex;
	
	// Tweakbar stuff
	struct Stage
	{
		enum T
		{
			GBuffer,
			Shadow,
			
			Clear,

			VoxelRep,
			
			Readback,

			RayGen,
			
			Tracing,

			RayGrid,
			RayMarch,
			RaySort,
			RayInject,
			RayInline,

			Intersect,
			IntersectGBuffer,
			IntersectLighting,
			
			RayGather,
			Lighting,
			Processing,
			
			Frame,

			Count
		};
	};

	bool bTracingEnabled;
	bool bGenerateCoherent;
	bool bPartialTracing;
	bool bReadDebug;
	
	uint4 rayCount;
	uint4 rayLinkCount;
	uint4 maxRayLinkCount;
	float avgRayLength;
	float maxAvgRayLength;
	uint4 hitTests;
	uint4 hits;
	float avgHitTests;
	float avgHits;
	float hitRatio;

	uint4 maxTraceStep;

	lean::com_ptr<beg::api::Query> timingQuery;
	lean::array<IncrementalGPUTimer, Stage::Count> timers;

	lean::scoped_ptr<TwBar, lean::stable_ref, tw_delete_policy> tweakBar;

	M(besc::EffectDrivenRenderer *originalRenderer, besc::ResourceManager *resourceManager)
		: maxRayCount( GetPixelCount(*originalRenderer->Device()->GetHeadSwapChain(0)) ),
		maxTriBatchSize( 1024 * 128 ),
		
		voxelRep(GridResolution, originalRenderer, resourceManager),
		raySet(maxRayCount, originalRenderer, resourceManager),

		resourceManager( LEAN_ASSERT_NOT_NULL(resourceManager) ),
		tracingPool( new_resource TracingEffectBinderPool(ToImpl(*originalRenderer->Device()), &raySet) ),
		renderer( CreateTracingRenderer(*LEAN_ASSERT_NOT_NULL(originalRenderer), resourceManager->Monitor(), tracingPool) ),
		
		voxelRepStage("Hardwired/VoxelRep.fx", GridResolution, renderer, resourceManager),
		rayGenStage("Hardwired/RayGen.fx", renderer, resourceManager),
		rayGridStage("Hardwired/RayGrid.fx", maxRayCount, GridResolution, 4, renderer, resourceManager),

		stagingBuffer( beg::Any::CreateStagingBuffer(ToImpl(*renderer->Device()), sizeof(StagingData)) ),

		debugInfoBuffer( beg::Any::CreateStructuredBuffer(ToImpl(*renderer->Device()), D3D11_BIND_UNORDERED_ACCESS, sizeof(DebugInfo)) ),
		debugInfoUAV( beg::Any::CreateCountingUAV(debugInfoBuffer) ),

		timingQuery( beg::Any::CreateTimingQuery(ToImpl(*renderer->Device())) ),
		timers( ToImpl(*renderer->Device()) ),

		frameIndex(0),

		bTracingEnabled(true),
		bGenerateCoherent(true),
		bPartialTracing(false),
		bReadDebug(true),
		
		maxTraceStep(-1)
	{
		tracingPool->SetDebugUAV(raySet.RayDebugUAV());

		besc::RenderingPipeline &pipeline = *renderer->Pipeline();
		
		// Pipeline stages
		besc::LoadRenderingPipeline(pipeline,
				resourceManager->EffectCache()->GetByFile("Pipelines/Tracing/Pipeline.fx", nullptr, 0),
				*renderer->RenderableDrivers()
			);

		geometryStageID = pipeline.GetStageID("GeometryPipelineStage");
		if (geometryStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "GeometryPipelineStage");

		shadowStageID = pipeline.GetStageID("ShadowPipelineStage");
		if (shadowStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "ShadowPipelineStage");

		processingStageID = pipeline.GetStageID("ProcessingPipelineStage");
		if (processingStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "ProcessingPipelineStage");

		voxelRepStageID = pipeline.GetStageID("VoxelRepPipelineStage");
		if (voxelRepStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "VoxelRepPipelineStage");

		tracingStageID = pipeline.GetStageID("TracingPipelineStage");
		if (tracingStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "TracingPipelineStage");

		partialTracingStageID = pipeline.GetStageID("PartialTracingPipelineStage");
		if (partialTracingStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "PartialTracingPipelineStage");

		traceLightingStageID = pipeline.GetStageID("TraceLightingPipelineStage");
		if (traceLightingStageID == besc::InvalidPipelineStage)
			LEAN_THROW_ERROR_CTX("Unknown pipeline stage", "TraceLightingPipelineStage");

		// Tweak bar stuff
		TwBar* metricsBar = LEAN_ASSERT_NOT_NULL( TwGetBarByName("Metrics") );
		int margins[2], area[2];
		TwGetParam(metricsBar, nullptr, "position", TW_PARAM_INT32, lean::arraylen(margins), margins);
		TwGetParam(metricsBar, nullptr, "size", TW_PARAM_INT32, lean::arraylen(area), area);

		tweakBar = TwNewBar( ("Pipeline" + identityString(this)).c_str() );
		TwSetParam(tweakBar.get(), nullptr, "label", TW_PARAM_CSTRING, 1, "Tracing Pipeline");

		TwAddVarRO(tweakBar.get(), "clear", TW_TYPE_FLOAT, timers[Stage::Clear].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "gbuffer", TW_TYPE_FLOAT, timers[Stage::GBuffer].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "shadow", TW_TYPE_FLOAT, timers[Stage::Shadow].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "voxrep", TW_TYPE_FLOAT, timers[Stage::VoxelRep].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "raygen", TW_TYPE_FLOAT, timers[Stage::RayGen].GetDataMS(), "group=timing");

		TwAddVarRO(tweakBar.get(), "trace", TW_TYPE_FLOAT, timers[Stage::Tracing].GetDataMS(), "group=timing");

		TwAddVarRO(tweakBar.get(), "+ raygrid", TW_TYPE_FLOAT, timers[Stage::RayGrid].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "  + march", TW_TYPE_FLOAT, timers[Stage::RayMarch].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "  + sort", TW_TYPE_FLOAT, timers[Stage::RaySort].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "  + inject", TW_TYPE_FLOAT, timers[Stage::RayInject].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "  + inline", TW_TYPE_FLOAT, timers[Stage::RayInline].GetDataMS(), "group=timing");

		TwAddVarRO(tweakBar.get(), "+ intersect", TW_TYPE_FLOAT, timers[Stage::Intersect].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "  + hit", TW_TYPE_FLOAT, timers[Stage::IntersectGBuffer].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "  + shade", TW_TYPE_FLOAT, timers[Stage::IntersectLighting].GetDataMS(), "group=timing");

		TwAddVarRO(tweakBar.get(), "raygather", TW_TYPE_FLOAT, timers[Stage::RayGather].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "lighting", TW_TYPE_FLOAT, timers[Stage::Lighting].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "processing", TW_TYPE_FLOAT, timers[Stage::Processing].GetDataMS(), "group=timing");
		TwAddVarRO(tweakBar.get(), "frame", TW_TYPE_FLOAT, timers[Stage::Frame].GetDataMS(), "group=timing");
#ifndef SMALL_BAR
		TwAddVarRO(tweakBar.get(), "readback", TW_TYPE_FLOAT, timers[Stage::Readback].GetDataMS(), "group=timing");
#endif

		TwAddVarRO(tweakBar.get(), "ray count", TW_TYPE_UINT32, &rayCount, "group=info");
		TwAddVarRO(tweakBar.get(), "node count", TW_TYPE_UINT32, &rayLinkCount, "group=info");
		TwAddVarRO(tweakBar.get(), "avg length", TW_TYPE_FLOAT, &avgRayLength, "group=info");
		TwAddVarRO(tweakBar.get(), "max avg length", TW_TYPE_FLOAT, &maxAvgRayLength, "group=info");
#ifdef SHOW_INTERSECTION_STATISTICS
		TwAddVarRO(tweakBar.get(), "hit tests", TW_TYPE_UINT32, &hitTests, "group=info");
		TwAddVarRO(tweakBar.get(), "hits", TW_TYPE_UINT32, &hits, "group=info");
		TwAddVarRO(tweakBar.get(), "avg hit tests", TW_TYPE_FLOAT, &avgHitTests, "group=info");
		TwAddVarRO(tweakBar.get(), "avg hits", TW_TYPE_FLOAT, &avgHits, "group=info");
		TwAddVarRO(tweakBar.get(), "hit %", TW_TYPE_FLOAT, &hitRatio, "group=info");
#endif
		TwAddVarRW(tweakBar.get(), "enabled", TW_TYPE_BOOLCPP, &bTracingEnabled, "group=control");
#ifndef SMALL_BAR
		TwAddVarRW(tweakBar.get(), "read dbg", TW_TYPE_BOOLCPP, &bReadDebug, "group=control");
		TwAddVarRW(tweakBar.get(), "step", TW_TYPE_INT32, &maxTraceStep, "group=control");
		TwAddVarRW(tweakBar.get(), "coh gen", TW_TYPE_BOOLCPP, &bGenerateCoherent, "group=control");
		TwAddVarRW(tweakBar.get(), "partial", TW_TYPE_BOOLCPP, &bPartialTracing, "group=control");
#endif

		int size[2] = { 240, 450 };
#ifdef SHOW_INTERSECTION_STATISTICS
		size[1] += 80;
#endif
#ifndef SMALL_BAR
		size[1] += 80;
#endif
		size[1] = min(size[1], area[1] - 2 * margins[1]);
		int pos[2] = { margins[0], margins[1] };
		TwSetParam(tweakBar.get(), nullptr, "position", TW_PARAM_INT32, 2, pos);
		TwSetParam(tweakBar.get(), nullptr, "size", TW_PARAM_INT32, 2, size);
		TwSetParam(tweakBar.get(), nullptr, "color", TW_PARAM_CSTRING, 1, "45 45 45");

#ifdef SMALL_BAR
		TwSetParam(tweakBar.get(), nullptr, "iconified", TW_PARAM_CSTRING, 1, "true");
#endif
	}
};

// Constructor.
Pipeline::Pipeline(besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager)
	: m( new M(renderer, resourceManager) )
{
}

// Destructor.
Pipeline::~Pipeline()
{
}

// Processes / commits changes.
void Pipeline::Commit()
{
	LEAN_PIMPL();

	m.voxelRepStage.Commit();
	m.rayGenStage.Commit();
	m.rayGridStage.Commit();
}

// Gets the number of warm-up passes.
uint4 Pipeline::GetWarmupPassCount() const
{
	return 312;
}

// Renders the scene.
void Pipeline::Render(besc::RenderingController *scene, besc::PipelinePerspective &perspective, benchmark_vector *pBenchmark)
{
	LEAN_PIMPL();

	bool bTracingEnabled = m.bTracingEnabled && 32 < m.frameIndex && !(48 < m.frameIndex && m.frameIndex < GetWarmupPassCount() - 4);

	besc::RenderingPipeline &pipeline = *m.renderer->Pipeline();
	besc::Pipe *pipe = perspective.GetPipe();
	besc::RenderContext &renderContext = *scene->GetRenderContext();
	const beg::Any::DeviceContext &deviceContext = ToImpl(renderContext.Context());

	m.renderer->InvalidateCaches();

	// Reset timers
	for (uint4 i = 0; i < M::Stage::Count; ++i)
		m.timers[i].Reset();

	// Enable cross-pipeline results
	pipe->KeepResults();

	deviceContext->Begin(m.timingQuery);
	m.timers[M::Stage::Frame].Begin(deviceContext);

	if (bTracingEnabled)
	{
		const besc::RenderingController::Renderables renderables = scene->GetRenderables();

		uint4 primaryStages = pipeline.GetNormalStages();
		pipeline.Prepare(perspective, renderables.Begin, Size4(renderables), primaryStages);
		pipeline.Optimize(perspective, renderables.Begin, Size4(renderables), primaryStages);

		m.timers[M::Stage::Shadow].Begin(deviceContext);
		pipeline.PreRender(perspective, renderables.Begin, Size4(renderables), renderContext);
		m.timers[M::Stage::Shadow].End(deviceContext);

		lean::com_ptr<const beg::ColorTextureTarget> gBufferTarget;
		ID3D11ShaderResourceView *gBuffer;

		// Render normal G-Buffer (rasterization)
		{
			m.timers[M::Stage::GBuffer].Begin(deviceContext);

			pipeline.RenderStages(perspective, renderables.Begin, Size4(renderables), renderContext, 1U << m.geometryStageID);

			m.timers[M::Stage::GBuffer].End(deviceContext);

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

		// Prepare
		m.voxelRepStage.Bind(m.voxelRep);
		
		// Clear
		{
			m.timers[M::Stage::Clear].Begin(deviceContext);

			deviceContext->ClearUnorderedAccessViewUint(m.voxelRep.VoxelUAV(), Zeroes);
			deviceContext->ClearUnorderedAccessViewUint(m.raySet.RayGeometryUAV(), MaxInts);
			deviceContext->ClearUnorderedAccessViewUint(m.raySet.RayLightUAV(), Zeroes);

			m.timers[M::Stage::Clear].End(deviceContext);
		}

		// Generate rays
		{
			m.timers[M::Stage::RayGen].Begin(deviceContext);

			if (m.bGenerateCoherent)
				m.rayGenStage.GenerateRaysCoherent(perspective, m.raySet, false, renderContext);
			else
				m.rayGenStage.GenerateRays(perspective, m.raySet, false, renderContext);

			// Fetch ray count
			deviceContext->CopyStructureCount(
					m.raySet.Constants(),
					offsetof(RaySetLayout, RayCount),
					m.raySet.RayDescUAV()
				);

			m.timers[M::Stage::RayGen].End(deviceContext);
		}
		
		// Construct omnidirectional tracing perspective
		lean::com_ptr<besc::PipelinePerspective> tracePerspective = m.renderer->PerspectivePool()->GetPerspective(nullptr, nullptr, 0);
		{
			besc::PerspectiveDesc tracePerspectiveDesc = perspective.GetDesc();
			tracePerspectiveDesc.Flags |= besc::PerspectiveFlags::Omnidirectional;
			
			// Disable culling
			for (int i = 0; i < 6; ++i)
				tracePerspectiveDesc.Frustum[i] = bem::fplane3(bem::nvec<3>(i % 3, 1.0f), 2.0e16f);

			tracePerspective->SetDesc(tracePerspectiveDesc);
		}

		// Prepare omnidirectional tracing perspective for rendering
		uint4 tracingStages = (1U << m.voxelRepStageID) | (1U << m.tracingStageID) | (1U << m.traceLightingStageID);
		if (m.bPartialTracing)
			tracingStages |= (1U << m.partialTracingStageID);
		pipeline.Prepare(*tracePerspective, renderables.Begin, Size4(renderables), tracingStages);
		pipeline.Optimize(*tracePerspective, renderables.Begin, Size4(renderables), tracingStages);

		// Render shadow maps for tracing perspective
		m.timers[M::Stage::Shadow].Begin(deviceContext);
		pipeline.PreRender(*tracePerspective, renderables.Begin, Size4(renderables), renderContext);
		m.timers[M::Stage::Shadow].End(deviceContext);
		
		// Construct conservative voxel representation of the scene
		if (true)
		{
			m.timers[M::Stage::VoxelRep].Begin(deviceContext);

			ToImpl(renderContext.StateManager()).Revert();
			ToImpl(renderContext.StateManager()).Reset();

			m.voxelRep.UpdateConstants(deviceContext);
			SetConstantBuffers(deviceContext, 3, 1, &m.voxelRep.Constants());
			
			ID3D11UnorderedAccessView *voxelUAVs[2] = { m.voxelRep.VoxelUAV(), m.raySet.RayDebugUAV() };
			uint4 voxelUAVCounts[2] = { -1, 0 };
			deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr, 0, lean::arraylen(voxelUAVs), voxelUAVs, voxelUAVCounts);
			
			D3D11_VIEWPORT bundleViewport = { 0.0f, 0.0f, (float) m.voxelRep.RasterResolution(), (float) m.voxelRep.RasterResolution(), 0.0f, 1.0f };
			deviceContext->RSSetViewports(1, &bundleViewport);

			ToImpl(renderContext.StateManager()).Override(beg::Any::StateMasks::RenderTargets);
			ToImpl(renderContext.StateManager()).RecordOverridden();

			// Voxelize in the standard graphics pipeline: Render normal scene pass using voxelization shaders
			pipeline.RenderStages(*tracePerspective, renderables.Begin, Size4(renderables), renderContext, 1U << m.voxelRepStageID);

			beg::Any::UnbindAllTargets(deviceContext);
			
			m.timers[M::Stage::VoxelRep].End(deviceContext);

			// Not using mip levels anywhere right now
//			m.voxelRepStage.MipVoxels(m.voxelRep, renderContext);
		}
		
		// Debug read back
		if (m.bReadDebug)
		{
			m.timers[M::Stage::Readback].Begin(deviceContext);
			
			// DEBUG: Voxel rep info
			beg::Any::CopyBuffer(deviceContext,
					m.stagingBuffer, offsetof(StagingData, voxelRep), 
					m.voxelRep.Constants(), 0, sizeof(stagingData.voxelRep)
				);
			// DEBUG: Ray set info
			beg::Any::CopyBuffer(deviceContext,
					m.stagingBuffer, offsetof(StagingData, raySet), 
					m.raySet.Constants(), 0, sizeof(stagingData.raySet)
				);
			// DEBUG: Debug info
			beg::Any::CopyBuffer(deviceContext,
					m.stagingBuffer, offsetof(StagingData, debugInfo), 
					m.debugInfoBuffer, 0, sizeof(stagingData.debugInfo)
				);

			beg::Any::ReadBufferData(deviceContext, m.stagingBuffer, &stagingData, sizeof(stagingData));

			// Clear debug memory
			deviceContext->ClearUnorderedAccessViewUint(m.debugInfoUAV, Zeroes);
			deviceContext->ClearUnorderedAccessViewUint(m.raySet.RayDebugUAV(), Zeroes);

			m.timers[M::Stage::Readback].End(deviceContext);
		}

		uint4 traceStepCount = m.rayGridStage.GetTraceStepCount();
		bool bFirstTraceStep = true;

		// Allow for the inspection of a subset of tracing passes
		if (m.maxTraceStep < traceStepCount - 1)
			traceStepCount = m.maxTraceStep + 1;

		m.rayLinkCount = 0;
		m.maxRayLinkCount = 0;

		m.timers[M::Stage::Tracing].Begin(deviceContext);

		// Perform multi-pass bounded intersection testing
		for (uint4 traceStep = 0; traceStep < traceStepCount; ++traceStep)
		{
			// Construct ray grid
			if (true)
			{
				TIMING( m.timers[M::Stage::RayGrid].Begin(deviceContext) );

				m.rayGridStage.Bind(m.raySet);
				m.rayGridStage.ConstructRayGrid(
					m.raySet, m.voxelRep, traceStep, renderContext,
					TIMING( &m.timers[M::Stage::RayMarch] ), TIMING( &m.timers[M::Stage::RaySort] ),
					TIMING( &m.timers[M::Stage::RayInject] ), TIMING( &m.timers[M::Stage::RayInline] ) );

				m.rayLinkCount += m.rayGridStage.GetRayLinkCount();
				m.maxRayLinkCount = max(m.rayGridStage.GetRayLinkCount(), m.maxRayLinkCount);

				TIMING( m.timers[M::Stage::RayGrid].End(deviceContext) );
			}

			// Do intersection testing
			if (true)
			{
				TIMING( m.timers[M::Stage::Intersect].Begin(deviceContext) );

				// Find hit points
				{
					TIMING( m.timers[M::Stage::IntersectGBuffer].Begin(deviceContext) );

					ToImpl(renderContext.StateManager()).Revert();
					ToImpl(renderContext.StateManager()).Reset();

					SetConstantBuffers(deviceContext, 3, 1, &m.voxelRep.Constants());
					SetConstantBuffers(deviceContext, 4, 1, &m.raySet.Constants());

					SetShaderResources(deviceContext, 7, 1, &m.voxelRep.Voxels());
					SetShaderResources(deviceContext, 8, 1, &m.raySet.RayList());
					SetShaderResources(deviceContext, 10, 1, &m.raySet.RayDescs());
					SetShaderResources(deviceContext, 11, 1, &m.raySet.RayLinks());
					SetShaderResources(deviceContext, 12, 1, &m.raySet.RayGridEnd());
					SetShaderResources(deviceContext, 13, 1, &m.raySet.RayGridBegin());

					D3D11_VIEWPORT tracingViewports[2] = {
							{ 0.0f, 0.0f, (float) m.voxelRep.RasterResolution(), (float) m.voxelRep.RasterResolution(), 0.0f, 1.0f },
							{ 0.0f, 0.0f, 4096.0f, 4096.0f, 0.0f, 1.0f }
						};

					ID3D11UnorderedAccessView *tracingUAVs[2] = { m.raySet.RayGeometryUAV(), m.raySet.RayDebugUAV() };
					uint4 tracingUAVCounts[2] = { (bFirstTraceStep) ? 0 : -1, (bFirstTraceStep) ? 0 : -1 };

					deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr, 0, lean::arraylen(tracingUAVs), tracingUAVs, tracingUAVCounts);
					deviceContext->RSSetViewports(lean::arraylen(tracingViewports), tracingViewports);

					ToImpl(renderContext.StateManager()).Override(beg::Any::StateMasks::RenderTargets);
					ToImpl(renderContext.StateManager()).RecordOverridden();

					uint4 profileTracingStageID = m.bPartialTracing ? m.partialTracingStageID : m.tracingStageID;
					pipeline.RenderStages(*tracePerspective, renderables.Begin, Size4(renderables), renderContext, 1U << profileTracingStageID);
					
					beg::Any::UnbindAllRenderTargets(deviceContext);

					TIMING( m.timers[M::Stage::IntersectGBuffer].End(deviceContext) );

					// Redo complete tracing to fix subsequent passes when parts of tracing are disabled for profiling purposes
					if (m.bPartialTracing)
					{
						pipeline.RenderStages(*tracePerspective, renderables.Begin, Size4(renderables), renderContext, 1U << m.tracingStageID);

						beg::Any::UnbindAllRenderTargets(deviceContext);
					}
				}

				// Light/shade hit points in the last pass
				if (traceStep + 1 == traceStepCount)
				{
					TIMING( m.timers[M::Stage::IntersectLighting].Begin(deviceContext) );

					ToImpl(renderContext.StateManager()).Revert();
					ToImpl(renderContext.StateManager()).Reset();
					
					SetConstantBuffers(deviceContext, 3, 1, &m.voxelRep.Constants());
					SetConstantBuffers(deviceContext, 4, 1, &m.raySet.Constants());

					SetShaderResources(deviceContext, 8, 1, &m.raySet.RayList());
					SetShaderResources(deviceContext, 9, 1, &m.raySet.RayGeometry());
					SetShaderResources(deviceContext, 10, 1, &m.raySet.RayDescs());
					SetShaderResources(deviceContext, 11, 1, &m.raySet.RayLinks());
					SetShaderResources(deviceContext, 12, 1, &m.raySet.RayGridEnd());
					SetShaderResources(deviceContext, 13, 1, &m.raySet.RayGridBegin());

					D3D11_VIEWPORT tracingViewports[2] = {
							{ 0.0f, 0.0f, (float) m.voxelRep.RasterResolution(), (float) m.voxelRep.RasterResolution(), 0.0f, 1.0f },
							{ 0.0f, 0.0f, 4096.0f, 4096.0f, 0.0f, 1.0f }
						};

					ID3D11UnorderedAccessView *tracingUAVs[2] = { m.raySet.RayLightUAV(), m.raySet.RayDebugUAV() };
					uint4 tracingUAVCounts[2] = { 0, -1 }; // (bFirstTraceStep) ? 0 : -1

					deviceContext->OMSetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr, 0, lean::arraylen(tracingUAVs), tracingUAVs, tracingUAVCounts);
					deviceContext->RSSetViewports(lean::arraylen(tracingViewports), tracingViewports);

					ToImpl(renderContext.StateManager()).Override(beg::Any::StateMasks::RenderTargets);
					ToImpl(renderContext.StateManager()).RecordOverridden();

					pipeline.RenderStages(*tracePerspective, renderables.Begin, Size4(renderables), renderContext, 1U << m.traceLightingStageID);

					beg::Any::UnbindAllRenderTargets(deviceContext);

					TIMING( m.timers[M::Stage::IntersectLighting].End(deviceContext) );
				}


				TIMING( m.timers[M::Stage::Intersect].End(deviceContext) );

				if (m.bReadDebug)
				{
					// DEBUG: Fetch hit count
					deviceContext->CopyStructureCount(
							m.stagingBuffer,
							offsetof(StagingData, totalTestCount),
							m.raySet.RayDebugUAV()
						);
					deviceContext->CopyStructureCount(
							m.stagingBuffer,
							offsetof(StagingData, totalHitCount),
							m.raySet.RayGeometryUAV()
						);
					deviceContext->CopyStructureCount(
							m.stagingBuffer,
							offsetof(StagingData, totalAtmoCount),
							m.raySet.RayLightUAV()
						);
				}

				// DEBUG / INFO
				m.rayCount = stagingData.raySet.RayCount;
				// Already read back due to CUDA sorting
//				m.rayLinkCount = stagingData.raySet.RayLinkCount;
//				m.maxRayLinkCount = max(m.maxRayLinkCount, stagingData.raySet.RayLinkCount);
				m.avgRayLength = (float) m.rayLinkCount / m.rayCount;
				m.maxAvgRayLength = (float) m.maxRayLinkCount / m.rayCount;
				m.hitTests = stagingData.totalTestCount;
				m.hits = stagingData.totalHitCount + stagingData.totalAtmoCount;
				m.avgHitTests = (float) m.hitTests / m.rayCount;
				m.avgHits = (float) m.hits / m.rayCount;
				m.hitRatio = (float) m.hits / m.hitTests * 100;
			}

			bFirstTraceStep = false;
		}
		m.timers[M::Stage::Tracing].End(deviceContext);

		// Finalize omnidirectional tracing perspective rendering
		pipeline.PostRender(*tracePerspective, renderables.Begin, Size4(renderables), renderContext);
		pipeline.ReleaseIntermediate(*tracePerspective, renderables.Begin, Size4(renderables));

		// Compute direct lighting
		if (true)
		{
			m.timers[M::Stage::Lighting].Begin(deviceContext);

			pipeline.RenderStages(perspective, renderables.Begin, Size4(renderables), renderContext, ~((1U << m.geometryStageID) | (1U << m.processingStageID)));

			m.timers[M::Stage::Lighting].End(deviceContext);
		}

		// Gather ray fragments & combine with direct lighting
		if (true)
		{
			m.timers[M::Stage::RayGather].Begin(deviceContext);

			m.rayGenStage.GatherRays(
					perspective,
					m.raySet, m.voxelRep,
					m.voxelRep.Voxels(),
					renderContext
				);

			m.timers[M::Stage::RayGather].End(deviceContext);
		}

		// Processing
		if (true)
		{
			m.timers[M::Stage::Processing].Begin(deviceContext);

			pipeline.RenderStages(perspective, renderables.Begin, Size4(renderables), renderContext, 1U << m.processingStageID);
			pipeline.PostRender(perspective, renderables.Begin, Size4(renderables), renderContext);

			m.timers[M::Stage::Processing].End(deviceContext);
		}

		pipeline.ReleaseIntermediate(perspective, renderables.Begin, Size4(renderables));
	}

	// Classic rendering
	if (!bTracingEnabled)
		scene->Render(perspective, renderContext);

	m.timers[M::Stage::Frame].End(deviceContext);
	deviceContext->End(m.timingQuery);

	pipe->KeepResults(false);
	pipe->Release();

	// Timing
	{
		uint8 frequency = beg::Any::GetTimingFrequency(deviceContext, m.timingQuery);

		for (uint4 i = 0; i < M::Stage::Count; ++i)
		{
			m.timers[i].ReadData(deviceContext);
			m.timers[i].ToMS(frequency);
		}

		if (pBenchmark)
		{
#ifdef BENCHMARK_TIMING
#ifdef BENCHMARK_TIMING_TRACE
			pBenchmark->push_back(*m.timers[M::Stage::TraceGBuffer].GetDataMS());
			pBenchmark->push_back(*m.timers[M::Stage::TraceLighting].GetDataMS());
			pBenchmark->push_back(*m.timers[M::Stage::Trace].GetDataMS());
#else
			pBenchmark->push_back(*m.timers[M::Stage::Frame].GetDataMS());
			pBenchmark->push_back(*m.timers[M::Stage::VoxelRep].GetDataMS());
			pBenchmark->push_back(*m.timers[M::Stage::RayMarch].GetDataMS());
			pBenchmark->push_back(*m.timers[M::Stage::RaySort].GetDataMS());
			pBenchmark->push_back(*m.timers[M::Stage::Intersect].GetDataMS());
#endif
#endif
		}
	}

	++m.frameIndex;
}

// Sets the bounds of the scene.
void Pipeline::SetBounds(const bem::fvec3 &min, const bem::fvec3 &max)
{
	LEAN_PIMPL();
	m.voxelRep.SetBounds(min, max);
}

// Sets the number of triangles.
void Pipeline::SetBatchSize(uint4 triCount)
{
	LEAN_PIMPL();
	m.maxTriBatchSize = triCount;
	m.tracingPool->SetupBuffers(triCount, sizeof(float) * (3 + 2) * 3, triCount, 24);
}

// Gets a modified tracing renderer.
besc::EffectDrivenRenderer *Pipeline::GetTracingRenderer()
{
	return m->renderer;
}

} // namespace

} // namespace
