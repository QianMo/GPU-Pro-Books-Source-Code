/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_EFFECTBINDERCACHE
#define BE_SCENE_EFFECTBINDERCACHE

#include "beScene.h"
#include <beCore/beShared.h>
#include "beEffectBinder.h"
#include "beEffectDriver.h"
#include <beGraphics/beEffect.h>

namespace beScene
{

/// Effect binder cache interface.
class LEAN_INTERFACE GenericEffectBinderCache : public lean::noncopyable, public beCore::Resource
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(GenericEffectBinderCache)

public:
	/// Gets an effect binder from the given effect.
	virtual EffectBinder* GetEffectBinder(const beGraphics::Technique &technique, uint4 flags = 0) = 0;
};

/// Effect driver cache interface.
class LEAN_INTERFACE GenericEffectDriverCache : public GenericEffectBinderCache
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(GenericEffectDriverCache)

public:
	/// Gets an effect binder from the given effect.
	virtual EffectDriver* GetEffectBinder(const beGraphics::Technique &technique, uint4 flags = 0) = 0;
};

/// Effect binder cache interface.
template <class EffectBinder>
class LEAN_INTERFACE EffectBinderCache : public GenericEffectBinderCache
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(EffectBinderCache)

public:
	/// Cached effect binder type.
	typedef EffectBinder Binder;

	/// Gets an effect binder from the given effect.
	virtual Binder* GetEffectBinder(const beGraphics::Technique &technique, uint4 flags = 0) = 0;
};

/// Effect binder cache interface.
template <class EffectDriver>
class LEAN_INTERFACE EffectDriverCache : public GenericEffectDriverCache
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(EffectDriverCache)

public:
	/// Cached effect driver type.
	typedef EffectDriver Binder;
	/// Cached effect driver type.
	typedef EffectDriver Driver;

	/// Gets an effect binder from the given effect.
	virtual Driver* GetEffectBinder(const beGraphics::Technique &technique, uint4 flags = 0) = 0;
};

} // namespace

#endif