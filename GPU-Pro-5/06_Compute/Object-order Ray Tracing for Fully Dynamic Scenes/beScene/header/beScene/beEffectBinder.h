/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_EFFECT_BINDER
#define BE_SCENE_EFFECT_BINDER

#include "beScene.h"
#include <beCore/beShared.h>
#include <beGraphics/beEffect.h>

namespace beScene
{

/// Effect binder base.
class LEAN_INTERFACE EffectBinder : public beCore::Shared
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(EffectBinder)

public:
	/// Gets the effect bound.
	virtual const beGraphics::Effect& GetEffect() const = 0;
};

} // namespace

#endif