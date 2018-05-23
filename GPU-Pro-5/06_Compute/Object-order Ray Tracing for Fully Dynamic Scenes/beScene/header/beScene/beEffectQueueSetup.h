/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_STATE_QUEUE_SETUP
#define BE_SCENE_STATE_QUEUE_SETUP

#include "beScene.h"
#include "beQueueSetup.h"
#include <lean/smart/resource_ptr.h>

namespace beScene
{

// Prototypes.
class RenderContext;
class Perspective;
class AbstractRenderableEffectDriver;

// Queue setup interface.
class EffectQueueSetup : public QueueSetup
{
private:
	lean::resource_ptr<AbstractRenderableEffectDriver> m_pEffectDriver;

public:
	/// Constructor.
	BE_SCENE_API EffectQueueSetup(AbstractRenderableEffectDriver *pEffectDriver);
	/// Destructor.
	BE_SCENE_API ~EffectQueueSetup();

	/// Called before drawing of a specific render queue begins.
	BE_SCENE_API void SetupRendering(uint4 stageID, uint4 queueID, const Perspective &perspective, const RenderContext &context) const;
};

} // namespace

#endif