/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_ABSTRACT_RENDERABLE_EFFECT_DRIVER
#define BE_SCENE_ABSTRACT_RENDERABLE_EFFECT_DRIVER

#include "beScene.h"
#include "beEffectDriver.h"
#include "bePassSequence.h"
#include "beQueuedPass.h"
#include "beRenderableEffectData.h"
#include "beEffectBinderCache.h"
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/beStateManager.h>
#include <lean/containers/strided_ptr.h>

#include <lean/functional/callable.h>

namespace beScene
{

// Prototypes.
class Perspective;
class Renderable;
struct LightJob;

/// Renderable effect driver state.
struct AbstractRenderableDriverState
{
	char Data[4 * sizeof(uint4)];
};

/// Renderable effect driver flags enumeration.
struct RenderableEffectDriverFlags
{
	/// Enumeration.
	enum T
	{
		Setup = 0x1		///< Treats effect as setup effect.
	};
	LEAN_MAKE_ENUM_STRUCT(RenderableEffectDriverFlags)
};

/// Renderable effect driver base.
class LEAN_INTERFACE AbstractRenderableEffectDriver : public EffectDriver, public PassSequence< QueuedPass, lean::strided_ptr<const QueuedPass> >
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(AbstractRenderableEffectDriver)

public:
	/// Signature of draw job call backed for every pass to be rendered.
	typedef void (DrawJobSignature)(uint4 passIdx, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context);

	/// Draws the given pass.
	virtual void Render(const QueuedPass *pass, const RenderableEffectData *pRenderableData, const Perspective &perspective,
		lean::vcallable<DrawJobSignature> &drawJob, beGraphics::StateManager &stateManager, const beGraphics::DeviceContext &context) const = 0;
};

typedef EffectDriverCache<AbstractRenderableEffectDriver> AbstractRenderableEffectDriverCache;

} // namespace

#endif