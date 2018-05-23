/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_EFFECT
#define BE_GRAPHICS_EFFECT

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beManagedResource.h>
#include <beCore/beComponent.h>
#include <lean/tags/noncopyable.h>

namespace beGraphics
{

class EffectCache;
class TextureCache;

/// Effect interface.
class LEAN_INTERFACE Effect : public lean::nonassignable, public beCore::OptionalResource,
	public beCore::ManagedResource<EffectCache>, public beCore::HotResource<Effect>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(Effect)

public:
	/// Gets the component type.
	BE_GRAPHICS_API static const beCore::ComponentType* GetComponentType();
};

/// Technique interface.
class LEAN_INTERFACE Technique : public beCore::OptionalResource, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(Technique)

public:
	/// Gets the effect.
	virtual const Effect* GetEffect() const = 0;
};

} // namespace

#endif