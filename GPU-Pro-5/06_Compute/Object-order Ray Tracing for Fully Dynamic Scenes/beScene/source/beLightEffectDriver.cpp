/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beLightEffectDriver.h"
#include "beScene/beLightEffectDriverCache.h"

#include "beScene/bePipelinePerspective.h"
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
LightEffectDriver::LightEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pipeline, PerspectiveEffectBinderPool *pool,
									 uint4 flags)
	: m_pipelineBinder( ToImpl(technique), pipeline, 0 ),
	m_perspectiveBinder( ToImpl(technique), pool ),
	m_lightBinder( ToImpl(technique) ),
	m_pipeBinder( ToImpl(technique) )
{
}

// Destructor.
LightEffectDriver::~LightEffectDriver()
{
}

// Draws the given pass.
void LightEffectDriver::Render(const QueuedPass *pass_, const LightEffectData *lightData, const Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager_, const beGraphics::DeviceContext &context) const
{
	const PipelineEffectBinderPass* pass = static_cast<const PipelineEffectBinderPass*>(pass_);
	beGraphics::Any::StateManager &stateManager = ToImpl(stateManager_);
	ID3D11DeviceContext *contextDX = ToImpl(context);

	// Prepare
	m_perspectiveBinder.Apply(perspective, stateManager, contextDX);

	// Render passes
	for (uint4 step = 0, nextStep; const StateEffectBinderPass *statePass = statePass = pass->GetPass(step); step = nextStep)
	{
		nextStep = step + 1;

		uint4 passID = statePass->GetPassID();
		uint4 nextPassID = passID;

		// Skip invalid light passes
		if (m_lightBinder.Apply(nextPassID, *lightData, contextDX))
		{
			// Repeat this step, if suggested by the light effect binder
			bool bRepeat = (nextPassID == passID);

			// NOTE: Reset for pipe effect binder
			nextPassID = passID;

			if (m_pipeBinder.Apply(nextPassID, ToImpl(perspective.GetPipe()), perspective.GetDesc().OutputIndex,
				lightData, stateManager, contextDX))
				// Repeat this step, if suggested by the pipe effect binder
				bRepeat |= (nextPassID == passID);
			
			if (bRepeat)
				nextStep = step;

			if (statePass->Apply(stateManager, contextDX))
				drawJob(passID, stateManager, context);
		}
	}
}

// Gets the pass identified by the given ID.
LightEffectDriver::PassRange LightEffectDriver::GetPasses() const
{
	return lean::static_range_cast<LightEffectDriver::PassRange>( ToSTD(m_pipelineBinder.GetPasses()) );
}

// Creates an effect binder from the given effect.
lean::resource_ptr<EffectDriver, true> LightEffectDriverCache::CreateEffectBinder(const beGraphics::Technique &technique, uint4 flags) const
{
	return new_resource LightEffectDriver(technique, m_pipeline, m_pool);
}

} // namespace