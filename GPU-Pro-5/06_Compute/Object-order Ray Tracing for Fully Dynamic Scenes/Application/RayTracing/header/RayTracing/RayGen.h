#pragma once

#include "Tracing.h"

#include <beScene/beResourceManager.h>
#include <beScene/beEffectDrivenRenderer.h>

#include <beScene/beQuadProcessor.h>
#include <beGraphics/beMaterial.h>

#include <beScene/bePerspective.h>
#include <beScene/bePipe.h>
#include <beScene/beRenderContext.h>

#include <beGraphics/Any/beEffectsAPI.h>

#include <lean/smart/resource_ptr.h>

#include <beMath/beMatrixDef.h>

namespace app
{

namespace tracing
{

class VoxelRep;
class RaySet;

/// Ray generation.
class RayGen
{
private:
	besc::EffectDrivenRenderer *m_renderer;

	lean::resource_ptr<beg::Material> m_material;

	lean::resource_ptr<besc::QuadProcessor> m_processor;
	uint4 m_rayGatherLayer;
	uint4 m_rayGenLayer;
	uint4 m_rayGenCohAllocLayer;
	uint4 m_rayGenCohWriteLayer;
	uint4 m_rayCountLayer;
	uint4 m_rayOffsetLayer;

	beg::api::Effect *m_effect;

	beg::api::EffectConstantBuffer *m_voxelRepConstVar;
	beg::api::EffectConstantBuffer *m_raySetConstVar;

	beg::api::EffectShaderResource *m_rayCountVar;
	beg::api::EffectShaderResource *m_rayOffsetVar;

	beg::api::EffectUnorderedAccessView *m_rayQueueUAVVar;
	beg::api::EffectShaderResource *m_rayDescVar;
	beg::api::EffectShaderResource *m_tracedGeometryVar;
	beg::api::EffectShaderResource *m_tracedLightVar;

	beg::api::EffectShaderResource *m_rayDebugVar;
	beg::api::EffectShaderResource *m_gridDebugVar;

	/// Performs ray compaction.
	void RayCompact(besc::Perspective &perspective, besc::RenderContext &context) const;

	/// Binds the given material.
	void BindMaterial(beg::Material *material);

public:
	/// Constructor.
	RayGen(const lean::utf8_ntri &file, besc::EffectDrivenRenderer *renderer, besc::ResourceManager *resourceManager);
	/// Destructor.
	~RayGen();

	/// Processes / commits changes.
	void Commit();

	/// Generates rays for the given bundle.
	void GenerateRays(besc::Perspective &perspective, RaySet &raySet,
		bool bKeepRays, besc::RenderContext &context) const;

	/// Generates rays for the given bundle.
	void GenerateRaysCoherent(besc::Perspective &perspective, RaySet &raySet,
		bool bKeepRays, besc::RenderContext &context) const;

	/// Gathers rays.
	void GatherRays(besc::Perspective &perspective, RaySet &raySet, VoxelRep &voxelRep,
		beg::api::ShaderResourceView *debugGrid, besc::RenderContext &context) const;
};

} // namespace

} // namespace