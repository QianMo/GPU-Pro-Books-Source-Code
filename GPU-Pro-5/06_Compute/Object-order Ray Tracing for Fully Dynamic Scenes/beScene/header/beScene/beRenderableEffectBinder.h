/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_EFFECT_BINDER
#define BE_SCENE_RENDERABLE_EFFECT_BINDER

#include "beScene.h"
#include "beEffectBinder.h"
#include "beRenderableEffectData.h"
#include <beGraphics/beDeviceContext.h>
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beStateManager.h>
#include "bePerspectiveEffectBinder.h"

namespace beScene
{

// Prototypes.
class Perspective;
class Renderable;

/// Renderable effect binder.
class RenderableEffectBinder : public EffectBinder
{
private:
	PerspectiveEffectBinder m_perspectiveBinder;

	beGraphics::Any::API::EffectMatrix *m_pWorld;
	beGraphics::Any::API::EffectMatrix *m_pWorldInverse;

	beGraphics::Any::API::EffectMatrix *m_pWorldViewProj;
	beGraphics::Any::API::EffectMatrix *m_pWorldView;

	beGraphics::Any::API::EffectVector *m_pObjectCamPos;
	beGraphics::Any::API::EffectVector *m_pObjectCamDir;

	beGraphics::Any::API::EffectScalar *m_pID;

public:
	/// Constructor.
	BE_SCENE_API RenderableEffectBinder(const beGraphics::Any::Technique &technique, PerspectiveEffectBinderPool *pPool);
	/// Destructor.
	BE_SCENE_API ~RenderableEffectBinder();

	/// Applies the given renderable & perspective data to the effect bound by this effect binder.
	BE_SCENE_API void Apply(const RenderableEffectData *pRenderableData, const Perspective &perspective,
		beGraphics::Any::StateManager& stateManager, beGraphics::Any::API::DeviceContext *pContext) const;

	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return m_perspectiveBinder.GetEffect(); }
};

} // namespace

#endif