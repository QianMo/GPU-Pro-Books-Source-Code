/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT_MATERIAL_CACHE
#define BE_SCENE_LIGHT_MATERIAL_CACHE

#include "beScene.h"
#include "beBoundMaterialCache.h"
#include "beLightMaterial.h"

namespace beScene
{

/// Light material cache.
class LightMaterialCache : public BoundMaterialCache<LightMaterial>
{
protected:
	lean::resource_ptr<AbstractLightEffectDriverCache> m_driverCache;

	/// Creates a bound material for the given material.
	BE_SCENE_API lean::resource_ptr<GenericBoundMaterial, lean::critical_ref> CreateBoundMaterial(beGraphics::Material *material);

public:
	/// Constructor.
	BE_SCENE_API LightMaterialCache(AbstractLightEffectDriverCache *driverCache)
		: m_driverCache(driverCache) { }
	/// Destructor.
	BE_SCENE_API ~LightMaterialCache() { }
};

/// Creates a new material cache.
BE_SCENE_API lean::resource_ptr<LightMaterialCache, lean::critical_ref> CreateLightMaterialCache(
	AbstractLightEffectDriverCache *driverCache);

} // namespace

#endif