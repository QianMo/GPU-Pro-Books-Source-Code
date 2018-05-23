/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_GENERIC_EFFECT_DRIVER_CACHE
#define BE_SCENE_GENERIC_EFFECT_DRIVER_CACHE

#include "beScene.h"
#include "beEffectBinderCache.h"
#include "beEffectDriver.h"
#include <lean/smart/resource_ptr.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beScene
{

/// Effect binder cache implementation.
class LEAN_INTERFACE GenericDefaultEffectDriverCache
{
public:
	class M;

private:
	lean::pimpl_ptr<M> m;

protected:
	/// Creates an effect binder from the given effect.
	virtual lean::resource_ptr<EffectDriver, lean::critical_ref> CreateEffectBinder(const beGraphics::Technique &technique, uint4 flags) const = 0;

	/// Constructor.
	BE_SCENE_API GenericDefaultEffectDriverCache();
	/// Destructor.
	BE_SCENE_API ~GenericDefaultEffectDriverCache();

public:
	/// Gets an effect binder from the given effect.
	BE_SCENE_API EffectDriver* GetEffectBinder(const beGraphics::Technique &technique, uint4 flags = 0);
};

/// Effect binder cache implementation.
template < class EffectDriver, class Interface = EffectDriverCache<EffectDriver> >
class LEAN_INTERFACE DefaultEffectDriverCache : public GenericDefaultEffectDriverCache, public Interface
{
public:
	/// Gets an effect binder from the given effect.
	EffectDriver* GetEffectBinder(const beGraphics::Technique &technique, uint4 flags = 0)
	{
		return static_cast<EffectDriver*>( GenericDefaultEffectDriverCache::GetEffectBinder(technique, flags) );
	}
};

} // namespace

#endif