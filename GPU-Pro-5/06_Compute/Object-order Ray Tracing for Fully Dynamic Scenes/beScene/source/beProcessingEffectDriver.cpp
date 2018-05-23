/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beProcessingEffectDriver.h"
#include "beScene/beProcessingEffectDriverCache.h"
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
ProcessingEffectDriver::ProcessingEffectDriver(const beGraphics::Technique &technique, RenderingPipeline *pPipeline, PerspectiveEffectBinderPool *pPool)
	: m_pipelineBinder( ToImpl(technique), pPipeline, PipelineEffectBinderFlags::AllowUnclassified ),
	m_perspectiveBinder( ToImpl(technique), pPool ),
	m_pipeBinder( ToImpl(technique), PipeEffectBinderFlags::NoDefaultMS )
{
}

// Destructor.
ProcessingEffectDriver::~ProcessingEffectDriver()
{
}

/// Draws the given pass.
void ProcessingEffectDriver::Render(const QueuedPass *pass_, const void *pProcessor, const Perspective *pPerspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager_, const beGraphics::DeviceContext &context) const
{
	const PipelineEffectBinderPass* pass = static_cast<const PipelineEffectBinderPass*>(pass_);
	beGraphics::Any::StateManager &stateManager = ToImpl(stateManager_);
	ID3D11DeviceContext *contextDX = ToImpl(context);

	// Prepare
	if (pPerspective)
		m_perspectiveBinder.Apply(*pPerspective, stateManager, contextDX);

	// Render passes
	for (uint4 step = 0, nextStep; const StateEffectBinderPass *statePass = statePass = pass->GetPass(step); step = nextStep)
	{
		nextStep = step + 1;

		uint4 passID = statePass->GetPassID();
		uint4 nextPassID = passID;

		DX11::Pipe *pPipe = nullptr;
		uint4 outIndex = 0;

		if (pPerspective)
		{
			pPipe = ToImpl(pPerspective->GetPipe());
			outIndex = pPerspective->GetDesc().OutputIndex;
		}

		if (m_pipeBinder.Apply(nextPassID, pPipe, outIndex, pProcessor, stateManager, contextDX))
		{
			if (nextPassID == passID)
				// Repeat this step, if suggested by the pipe effect binder
				nextStep = step;

			if (statePass->Apply(stateManager, contextDX))
				drawJob(passID, stateManager, context);
		}
	}
}

// Gets the pass identified by the given ID.
ProcessingEffectDriver::PassRange ProcessingEffectDriver::GetPasses() const
{
	return lean::static_range_cast<ProcessingEffectDriver::PassRange>( ToSTD(m_pipelineBinder.GetPasses()) );
}

// Creates an effect binder from the given effect.
lean::resource_ptr<EffectDriver, true> ProcessingEffectDriverCache::CreateEffectBinder(const beGraphics::Technique &technique, uint4 flags) const
{
	return new_resource ProcessingEffectDriver(technique, m_pipeline, m_pool);
}

} // namespace