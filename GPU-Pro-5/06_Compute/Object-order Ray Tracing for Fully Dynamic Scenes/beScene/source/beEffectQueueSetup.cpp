/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beEffectQueueSetup.h"
#include "beScene/beAbstractRenderableEffectDriver.h"
#include "beScene/beRenderContext.h"
#include "beScene/beRenderingLimits.h"
#include <beGraphics/Any/beStateManager.h>

namespace beScene
{

// Constructor.
EffectQueueSetup::EffectQueueSetup(AbstractRenderableEffectDriver *pEffectDriver)
	: m_pEffectDriver(pEffectDriver)
{
}

// Destructor.
EffectQueueSetup::~EffectQueueSetup()
{
}

// Called before drawing of a specific render queue begins.
void EffectQueueSetup::SetupRendering(uint4 stageID, uint4 queueID, const Perspective &perspective, const RenderContext &context) const
{
	beGraphics::Any::StateManager &stateManager = ToImpl( context.StateManager() );
	stateManager.Revert();

	struct NoDraw : lean::vcallable_base<AbstractRenderableEffectDriver::DrawJobSignature, NoDraw>
	{
		void operator ()(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) { }
	} noDraw;
	AbstractRenderableEffectDriver::PassRange passes = m_pEffectDriver->GetPasses();

	for (uint4 passID = 0; passID < Size(passes); ++passID)
	{
		const QueuedPass *pass = &passes.Begin[passID];
		uint4 passStageID = pass->GetStageID();
		uint4 passQueueID = pass->GetQueueID();

		bool bStageMatch = passStageID == stageID || passStageID == InvalidPipelineStage;
		bool bQueueMatch = passQueueID == queueID || passQueueID == InvalidRenderQueue;

		if (bStageMatch && bQueueMatch)
			m_pEffectDriver->Render(pass, nullptr, perspective, noDraw, stateManager, context.Context());
	}

	stateManager.RecordOverridden();
}

} // namespace
