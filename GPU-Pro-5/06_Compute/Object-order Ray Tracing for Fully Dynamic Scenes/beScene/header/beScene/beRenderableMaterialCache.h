/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_MATERIAL_CACHE
#define BE_SCENE_RENDERABLE_MATERIAL_CACHE

#include "beScene.h"
#include "beBoundMaterialCache.h"
#include "beRenderableMaterial.h"

namespace beScene
{

/// Renderable material cache.
class RenderableMaterialCache : public BoundMaterialCache<RenderableMaterial>
{
protected:
	lean::resource_ptr<AbstractRenderableEffectDriverCache> m_driverCache;

	/// Creates a bound material for the given material.
	BE_SCENE_API lean::resource_ptr<GenericBoundMaterial, lean::critical_ref> CreateBoundMaterial(beGraphics::Material *material);

public:
	/// Constructor.
	BE_SCENE_API RenderableMaterialCache(AbstractRenderableEffectDriverCache *driverCache)
		: m_driverCache(driverCache) { }
	/// Destructor.
	BE_SCENE_API ~RenderableMaterialCache() { }
};

/// Creates a new material cache.
BE_SCENE_API lean::resource_ptr<RenderableMaterialCache, lean::critical_ref> CreateRenderableMaterialCache(
	AbstractRenderableEffectDriverCache *driverCache);

} // namespace

#endif