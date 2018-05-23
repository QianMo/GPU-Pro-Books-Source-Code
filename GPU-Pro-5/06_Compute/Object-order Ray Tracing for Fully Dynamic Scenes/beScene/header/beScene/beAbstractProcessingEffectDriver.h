/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_ABSTRACT_PROCESSING_EFFECT_DRIVER
#define BE_SCENE_ABSTRACT_PROCESSING_EFFECT_DRIVER

#include "beScene.h"
#include "beEffectDriver.h"
#include "bePassSequence.h"
#include "beQueuedPass.h"
#include "beEffectBinderCache.h"
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>
#include <lean/containers/strided_ptr.h>

#include <lean/functional/callable.h>

namespace beScene
{

// Prototypes.
class Perspective;

/// Processing effect driver base.
class LEAN_INTERFACE AbstractProcessingEffectDriver : public EffectDriver, public PassSequence< QueuedPass, lean::strided_ptr<const QueuedPass> >
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(AbstractProcessingEffectDriver)

public:
	/// Signature of draw job call backed for every pass to be rendered.
	typedef void (DrawJobSignature)(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context);

	/// Draws the given pass.
	virtual void Render(const QueuedPass *pass, const void *pProcessor, const Perspective *pPerspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const = 0;
};

typedef EffectDriverCache<AbstractProcessingEffectDriver> AbstractProcessingEffectDriverCache;

} // namespace

#endif