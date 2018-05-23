/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beStateQueueSetup.h"
#include "beScene/beRenderContext.h"

namespace beScene
{

// Constructor.
StateQueueSetup::StateQueueSetup(const beGraphics::Any::StateSetup &setup)
	: m_stateSetup(setup)
{
}

// Destructor.
StateQueueSetup::~StateQueueSetup()
{
}

// Called before drawing of a specific render queue begins.
void StateQueueSetup::SetupRendering(uint4 stageID, uint4 queueID, const Perspective &perspective, const RenderContext &context) const
{
	context.StateManager().Set(m_stateSetup);
}

} // namespace
