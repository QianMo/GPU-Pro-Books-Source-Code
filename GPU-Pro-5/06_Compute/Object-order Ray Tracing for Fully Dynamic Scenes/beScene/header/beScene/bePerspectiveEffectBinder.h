/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_PERSPECTIVE_EFFECT_BINDER
#define BE_SCENE_PERSPECTIVE_EFFECT_BINDER

#include "beScene.h"
#include "beEffectBinder.h"
#include <beGraphics/Any/beEffect.h>
#include <beGraphics/Any/beEffectsAPI.h>
#include <beGraphics/Any/beStateManager.h>
#include <lean/smart/com_ptr.h>
#include <vector>

namespace beScene
{

// Prototypes.
class Perspective;
class PerspectiveEffectBinderPool;

/// Perspective effect binder.
class PerspectiveEffectBinder : public EffectBinder
{
public:
	struct Rasterizer
	{
		beGraphics::Any::API::EffectRasterizerState *pState;
		uint4 stateIndex;
		lean::com_ptr<beGraphics::Any::API::RasterizerState> pFlippedState;

		Rasterizer(beGraphics::Any::API::EffectRasterizerState *pState,
			uint4 stateIndex,
			beGraphics::Any::API::RasterizerState *pFlippedState)
				: pState(pState),
				stateIndex(stateIndex),
				pFlippedState(pFlippedState) { }
	};
	typedef std::vector<Rasterizer> rasterizer_state_vector;

private:
	lean::resource_ptr<const beGraphics::Any::Effect> m_effect;

	PerspectiveEffectBinderPool *m_pPool;

	beGraphics::Any::API::EffectConstantBuffer *m_pPerspectiveConstants;
	
	rasterizer_state_vector m_rasterizerStates;

public:
	/// Constructor.
	BE_SCENE_API PerspectiveEffectBinder(const beGraphics::Any::Technique &technique, PerspectiveEffectBinderPool *pPool);
	/// Destructor.
	BE_SCENE_API ~PerspectiveEffectBinder();

	/// Applies the given perspective data to the effect bound by this effect binder.
	BE_SCENE_API void Apply(const Perspective &perspective, beGraphics::Any::StateManager& stateManager, beGraphics::Any::API::DeviceContext *pContext) const;

	/// Gets the effect.
	LEAN_INLINE const beGraphics::Effect& GetEffect() const { return *m_effect; }
};

} // namespace

#endif