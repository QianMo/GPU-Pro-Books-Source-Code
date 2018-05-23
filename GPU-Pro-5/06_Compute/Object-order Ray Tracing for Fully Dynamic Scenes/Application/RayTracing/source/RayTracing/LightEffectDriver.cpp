/******************************************************/
/* Object-order Ray Tracing Demo (c) Tobias Zirr 2013 */
/******************************************************/

#include "stdafx.h"

#include "RayTracing/LightEffectDriver.h"
#include "RayTracing/LightEffectDriverCache.h"

#include <beGraphics/Any/beDeviceContext.h>

namespace app
{

namespace tracing
{

// Constructor.
LightEffectDriver::LightEffectDriver(const beGraphics::Technique &technique, besc::RenderingPipeline *pipeline,
		besc::PerspectiveEffectBinderPool *perspectivePool, TracingEffectBinderPool *tracingPool, uint4 flags)
	: besc::LightEffectDriver(technique, pipeline, perspectivePool, flags),
	m_tracingBinder(ToImpl(technique), tracingPool)
{
}

// Destructor.
LightEffectDriver::~LightEffectDriver()
{
}

// Draws the given pass.
void LightEffectDriver::Render(const besc::QueuedPass *pPass, const besc::LightEffectData *pLightData, const besc::Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const
{
	struct DrawJob : lean::vcallable_base<DrawJobSignature, DrawJob>
	{
		const TracingEffectBinder *binder;
		lean::vcallable<DrawJobSignature> *drawJob;

		void operator ()(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context)
		{
			binder->Render(0, *drawJob, passIdx, ToImpl(stateManager), ToImpl(context));
		}
	};

	DrawJob drawJobEx;
	drawJobEx.binder = &m_tracingBinder;
	drawJobEx.drawJob = &drawJob;

	besc::LightEffectDriver::Render(pPass, pLightData, perspective, drawJobEx, stateManager, context);
}

// Creates an effect binder from the given effect.
lean::resource_ptr<besc::EffectDriver, lean::critical_ref> LightEffectDriverCache::CreateEffectBinder(const beg::Technique &technique, uint4 flags) const
{
	lean::resource_ptr<besc::AbstractLightEffectDriver> driver;

	BOOL requiresTracing = false;
	ToImpl(technique)->GetAnnotationByName("EnableTracing")->AsScalar()->GetBool(&requiresTracing);

	if (requiresTracing)
		driver = new_resource LightEffectDriver(technique, m_pipeline, m_perspectivePool, m_tracingPool, flags);
	else
		driver = m_baseCache->GetEffectBinder(technique, flags);

	return driver.transfer();
}

} // namespace

} // namespace
