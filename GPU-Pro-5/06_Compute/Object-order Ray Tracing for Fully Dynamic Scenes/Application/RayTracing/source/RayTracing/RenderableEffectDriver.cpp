/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/RenderableEffectDriver.h"
#include "RayTracing/RenderableEffectDriverCache.h"

#include "RayTracing/TracingEffectBinderPool.h"

#include <beGraphics/beEffectCache.h>
#include <beGraphics/Any/beEffect.h>

#include <beGraphics/Any/beDeviceContext.h>

namespace app
{

namespace tracing
{

// Constructor.
RenderableEffectDriver::RenderableEffectDriver(const beGraphics::Technique &technique, besc::RenderingPipeline *pipeline,
		besc::PerspectiveEffectBinderPool *perspectivePool, TracingEffectBinderPool *tracingPool, uint4 flags)
	: besc::RenderableEffectDriver(technique, pipeline, perspectivePool, flags),
	m_tracingBinder(ToImpl(technique), tracingPool)
{
}

// Destructor.
RenderableEffectDriver::~RenderableEffectDriver()
{
}

// Draws the given pass.
void RenderableEffectDriver::Render(const besc::QueuedPass *pPass, const besc::RenderableEffectData *pRenderableData, const besc::Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const
{
	struct DrawJob : lean::vcallable_base<DrawJobSignature, DrawJob>
	{
		uint4 primitiveCount;
		const TracingEffectBinder *binder;
		lean::vcallable<DrawJobSignature> *drawJob;

		void operator ()(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context)
		{
			binder->Render(primitiveCount, *drawJob, passIdx, ToImpl(stateManager), ToImpl(context));
		}
	};

	DrawJob drawJobEx;
	// TODO: Something sensible
	// TODO: Only triangles?
	drawJobEx.primitiveCount = reinterpret_cast<const besc::RenderableEffectDataEx&>(*pRenderableData).ElementCount / 3;
	drawJobEx.binder = &m_tracingBinder;
	drawJobEx.drawJob = &drawJob;

	besc::RenderableEffectDriver::Render(pPass, pRenderableData, perspective, drawJobEx, stateManager, context);
}

// Creates an effect binder from the given effect.
lean::resource_ptr<besc::EffectDriver, lean::critical_ref> RenderableEffectDriverCache::CreateEffectBinder(const beg::Technique &technique, uint4 flags) const
{
	lean::resource_ptr<besc::AbstractRenderableEffectDriver> driver;

	BOOL requiresTracing = false;
	ToImpl(technique)->GetAnnotationByName("EnableTracing")->AsScalar()->GetBool(&requiresTracing);

	if (requiresTracing)
		driver = new_resource RenderableEffectDriver(technique, m_pipeline, m_perspectivePool, m_tracingPool, flags);
	else
		driver = m_baseCache->GetEffectBinder(technique, flags);

	return driver.transfer();
}

} // namespace

} // namespace
