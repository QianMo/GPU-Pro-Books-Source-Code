/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_EFFECT_DRIVER
#define BE_SCENE_EFFECT_DRIVER

#include "beScene.h"
#include "beEffectBinder.h"
#include <beCore/beShared.h>

namespace beScene
{

/// Effect binder base.
class LEAN_INTERFACE EffectDriver : public beCore::Resource, public EffectBinder
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(EffectDriver)
};

} // namespace

#endif