/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableEffectDriver.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beDeviceContext.h>
#include <beGraphics/Any/beStateManager.h>

namespace beScene
{

namespace
{

} // namespace

// Constructor.
RenderableEffectDriver::RenderableEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pPipeline, PerspectiveEffectBinderPool *pPool,
											   uint4 flags)
	: m_pipelineBinder( ToImpl(technique), pPipeline, (flags & RenderableEffectDriverFlags::Setup) ? PipelineEffectBinderFlags::AllowUnclassified : 0 ),
	m_renderableBinder( ToImpl(technique), pPool ),
	m_lightBinder( ToImpl(technique) )
{
}

// Destructor.
RenderableEffectDriver::~RenderableEffectDriver()
{
}

// Draws the given pass.
void RenderableEffectDriver::Render(const QueuedPass *pass_, const RenderableEffectData *pRenderableData, const Perspective &perspective,
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
			if (nextPassID == passID)
				nextStep = step;

			if (statePass->Apply(stateManager, contextDX))
				drawJob(passID, stateManager, context);
		}
	}
}

// Gets the pass identified by the given ID.
RenderableEffectDriver::PassRange RenderableEffectDriver::GetPasses() const
{
	return lean::static_range_cast<RenderableEffectDriver::PassRange>( ToSTD(m_pipelineBinder.GetPasses()) );
}

} // namespace