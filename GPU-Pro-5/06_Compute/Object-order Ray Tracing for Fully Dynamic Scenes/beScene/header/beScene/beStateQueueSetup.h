/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_STATE_QUEUE_SETUP
#define BE_SCENE_STATE_QUEUE_SETUP

#include "beScene.h"
#include "beQueueSetup.h"
#include <beGraphics/Any/beStateManager.h>

namespace beScene
{

// Prototypes.
class RenderContext;
class Perspective;

// Queue setup interface.
class StateQueueSetup : public QueueSetup
{
private:
	beGraphics::Any::StateSetup m_stateSetup;

public:
	/// Constructor.
	BE_SCENE_API StateQueueSetup(const beGraphics::Any::StateSetup &setup);
	/// Destructor.
	BE_SCENE_API ~StateQueueSetup();

	/// Called before drawing of a specific render queue begins.
	BE_SCENE_API void SetupRendering(uint4 stageID, uint4 queueID, const Perspective &perspective, const RenderContext &context) const;

	/// Gets the state setup.
	LEAN_INLINE beGraphics::Any::StateSetup& GetSetup() { return m_stateSetup; }
	/// Gets the state setup.
	LEAN_INLINE const beGraphics::Any::StateSetup& GetSetup() const { return m_stateSetup; }
};

} // namespace

#endif