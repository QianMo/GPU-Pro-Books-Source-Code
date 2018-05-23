#pragma once

#include "Tracing.h"

#include <beScene/beLightEffectDriver.h>

#include "TracingEffectBinder.h"

namespace app
{

namespace tracing
{

class TracingEffectBinderPool;

/// Light effect driver.
class LightEffectDriver : public besc::LightEffectDriver
{
protected:
	TracingEffectBinder m_tracingBinder;		///< Tracing effect binder.

public:
	/// Constructor.
	LightEffectDriver(const beGraphics::Technique &technique, besc::RenderingPipeline *pipeline,
		besc::PerspectiveEffectBinderPool *perspectivePool, TracingEffectBinderPool *tracingPool, 
		uint4 flags = 0);
	/// Destructor.
	~LightEffectDriver();

	/// Draws the given pass.
	void Render(const besc::QueuedPass *pPass, const besc::LightEffectData *pLightData, const besc::Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const LEAN_OVERRIDE;

	/// Gets the tracing effect binder.
	LEAN_INLINE const TracingEffectBinder& GetTracingBinder() const { return m_tracingBinder; }
};

} // namespace

} // namespace