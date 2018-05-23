#pragma once

#include "Tracing.h"

#include <beScene/beEffectBinder.h>
#include <beScene/beStateEffectBinder.h>

#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>

#include <lean/smart/resource_ptr.h>
#include <vector>

#include <beScene/beAbstractRenderableEffectDriver.h>

// #define APPEND_CONSUME_TRAVERSAL

namespace app
{

namespace tracing
{

// Prototypes
class TracingEffectBinderPool;

/// Light effect binder.
class TracingEffectBinder : public besc::EffectBinder
{
public:
	struct Pass;
	typedef std::vector<Pass> pass_vector;

private:
	const beGraphics::Any::Technique m_technique;

	lean::resource_ptr<TracingEffectBinderPool> m_pool;

	beGraphics::api::EffectUnorderedAccessView *m_tracedGeometryUAV;
	beGraphics::api::EffectUnorderedAccessView *m_tracedLightUAV;
	beGraphics::api::EffectUnorderedAccessView *m_debugUAV;

	beGraphics::api::EffectConstantBuffer *m_tracingConstantsVar;

	beGraphics::api::EffectShaderResource *m_triangleSRVVar;
	beGraphics::api::EffectShaderResource *m_geometrySRVVar;

	beGraphics::api::EffectShaderResource *m_voxelInVar;
	beGraphics::api::EffectUnorderedAccessView *m_voxelOutVar;

	beGraphics::api::EffectUnorderedAccessView *m_groupDispatchUAV;

	lean::com_ptr<beGraphics::api::InputLayout> m_noInputLayout;

	pass_vector m_passes;

	/// Computes the number of groups to dispatch.
	void ComputeDispatchGroupCount(beg::api::EffectPass *pass, beg::api::UnorderedAccessView *dispatchUAV, beg::api::DeviceContext *context) const;

public:
	/// Constructor.
	TracingEffectBinder(const beGraphics::Any::Technique &technique, TracingEffectBinderPool *pool,
		uint4 passID = static_cast<uint4>(-1));
	/// Destructor.
	~TracingEffectBinder();

	/// Applies stuff.
	bool Apply(beGraphics::Any::API::DeviceContext *pContext) const;
	
	// Draws the given number of primitives.
	bool TracingEffectBinder::Render(uint4 primitiveCount, lean::vcallable<besc::AbstractRenderableEffectDriver::DrawJobSignature> &drawJob,
		uint4 passID, beGraphics::Any::StateManager &stateManager, beg::api::DeviceContext *context) const;

	/// Gets the technique.
	const beGraphics::Technique& GetTechnique() const { return m_technique; }
	/// Gets the effect.
	const beGraphics::Effect& GetEffect() const { return *m_technique.GetEffect(); }
	/// Gets the pool.
	TracingEffectBinderPool* GetPool() const { return m_pool; }
};

} // namespace

} // namespace