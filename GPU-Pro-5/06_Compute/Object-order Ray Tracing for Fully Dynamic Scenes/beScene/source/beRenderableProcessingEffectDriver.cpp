/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableProcessingEffectDriver.h"
#include "beScene/beRenderableEffectDriverCache.h"
#include "beScene/bePerspective.h"
#include "beScene/DX11/bePipe.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beStateManager.h>

namespace beScene
{

namespace
{

} // namespace

// Constructor.
RenderableProcessingEffectDriver::RenderableProcessingEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pPipeline, PerspectiveEffectBinderPool *pPool,
																   uint4 flags)
	: RenderableEffectDriver( technique, pPipeline, pPool, flags ),
	m_pipeBinder( ToImpl(technique) )
{
}

// Destructor.
RenderableProcessingEffectDriver::~RenderableProcessingEffectDriver()
{
}

// Draws the given pass.
void RenderableProcessingEffectDriver::Render(const QueuedPass *pass_, const RenderableEffectData *pRenderableData, const Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager_, const beGraphics::DeviceContext &context) const
{
	const PipelineEffectBinderPass* pass = static_cast<const PipelineEffectBinderPass*>(pass_);
	beGraphics::Any::StateManager &stateManager = ToImpl(stateManager_);
	ID3D11DeviceContext *contextDX = ToImpl(context);

	// Prepare
	m_renderableBinder.Apply(pRenderableData, perspective, stateManager, contextDX);

	LightingBinderState lightState;

	// Render passes
	for (uint4 step = 0, nextStep; const StateEffectBinderPass *statePass = statePass = pass->GetPass(step); step = nextStep)
	{
		nextStep = step + 1;

		uint4 passID = statePass->GetPassID();
		uint4 nextPassID = passID;

		// Skip invalid light passes
		if (m_lightBinder.Apply(nextPassID, nullptr, nullptr, lightState, contextDX))
		{
			// Repeat this step, if suggested by the light effect binder
			bool bRepeat = (nextPassID == passID);

			// NOTE: Reset for pipe effect binder
			nextPassID = passID;

			if (m_pipeBinder.Apply(nextPassID, ToImpl(perspective.GetPipe()), perspective.GetDesc().OutputIndex,
				pRenderableData, stateManager, contextDX))
				// Repeat this step, if suggested by the pipe effect binder
				bRepeat |= (nextPassID == passID);
			
			if (bRepeat)
				nextStep = step;

			if (statePass->Apply(stateManager, contextDX))
				drawJob(passID, stateManager, context);
		}
	}
}

// Creates an effect binder from the given effect.
lean::resource_ptr<EffectDriver, true> RenderableEffectDriverCache::CreateEffectBinder(const beGraphics::Technique &technique, uint4 flags) const
{
	AbstractRenderableEffectDriver *driver;

	BOOL requiresProcessing = false;
	ToImpl(technique)->GetAnnotationByName("EnableProcessing")->AsScalar()->GetBool(&requiresProcessing);

	if (requiresProcessing)
		driver = new RenderableProcessingEffectDriver(technique, m_pipeline, m_pool, flags);
	else
		driver = new RenderableEffectDriver(technique, m_pipeline, m_pool, flags);

	return lean::bind_resource(driver);
}

} // namespace