/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_BOUND_MATERIAL_CACHE
#define BE_SCENE_BOUND_MATERIAL_CACHE

#include "beScene.h"
#include <beCore/beShared.h>
#include <lean/tags/noncopyable.h>
#include "beBoundMaterial.h"
#include "beEffectBinderCache.h"
#include <beCore/beComponentMonitor.h>
#include <lean/pimpl/pimpl_ptr.h>

namespace beScene
{

/// Bound material cache.
class LEAN_INTERFACE GenericBoundMaterialCache : public lean::noncopyable, public beCore::Resource
{
public:
	struct M;

private:
	lean::pimpl_ptr<M> m;

protected:
	/// Creates a bound material for the given material.
	virtual lean::resource_ptr<GenericBoundMaterial, lean::critical_ref> CreateBoundMaterial(beGraphics::Material *material) = 0;
	
public:
	/// Constructor.
	BE_SCENE_API GenericBoundMaterialCache();
	/// Destructor.
	BE_SCENE_API virtual ~GenericBoundMaterialCache();

	/// Gets a renderable material wrapping the given material.
	BE_SCENE_API GenericBoundMaterial* GetMaterial(beGraphics::Material *pMaterial);

	/// Sets the component monitor.
	BE_SCENE_API void SetComponentMonitor(beCore::ComponentMonitor *componentMonitor);
	/// Gets the component monitor.
	BE_SCENE_API beCore::ComponentMonitor* GetComponentMonitor() const;

	/// Commits / reacts to changes.
	BE_SCENE_API virtual void Commit();

	/// Gets the bound material component type.
	virtual const beCore::ComponentType* GetComponentType() = 0;
};

/// Bound material cache.
template <class BoundMaterial>
class LEAN_INTERFACE BoundMaterialCache : public GenericBoundMaterialCache
{
public:
	/// Gets a renderable material wrapping the given material.
	LEAN_INLINE BoundMaterial* GetMaterial(beGraphics::Material *pMaterial)
	{
		return static_cast<BoundMaterial*>(GenericBoundMaterialCache::GetMaterial(pMaterial));
	}

	/// Gets the bound material component type.
	BE_SCENE_API const beCore::ComponentType* GetComponentType()
	{
		return BoundMaterial::GetComponentType();
	}
};

} // namespace

#endif