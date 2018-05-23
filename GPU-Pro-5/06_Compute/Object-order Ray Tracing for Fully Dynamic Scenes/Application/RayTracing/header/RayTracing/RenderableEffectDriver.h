#pragma once

#include "Tracing.h"

#include <beScene/beRenderableEffectDriver.h>

#include "TracingEffectBinder.h"

#include <lean/smart/scoped_ptr.h>

namespace app
{

namespace tracing
{

class TracingEffectBinderPool;

/// Renderable effect driver.
class RenderableEffectDriver : public besc::RenderableEffectDriver
{
protected:
	TracingEffectBinder m_tracingBinder;		///< Tracing effect binder.

public:
	/// Constructor.
	RenderableEffectDriver(const beGraphics::Technique &technique, besc::RenderingPipeline *pipeline,
		besc::PerspectiveEffectBinderPool *perspectivePool, TracingEffectBinderPool *tracingPool, 
		uint4 flags = 0);
	/// Destructor.
	~RenderableEffectDriver();

	/// Draws the given pass.
	void Render(const besc::QueuedPass *pPass, const besc::RenderableEffectData *pRenderableData, const besc::Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const LEAN_OVERRIDE;

	/// Gets the tracing effect binder.
	LEAN_INLINE const TracingEffectBinder& GetTracingBinder() const { return m_tracingBinder; }
};

} // namespace

} // namespace