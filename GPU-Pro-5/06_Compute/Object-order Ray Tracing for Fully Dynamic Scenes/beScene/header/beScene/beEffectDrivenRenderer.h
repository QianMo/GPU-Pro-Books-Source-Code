/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_EFFECT_DRIVEN_RENDERER
#define BE_SCENE_EFFECT_DRIVEN_RENDERER

#include "beScene.h"
#include "beRenderer.h"
#include "beEffectBinderCache.h"
#include <beCore/beComponentMonitor.h>
#include <lean/smart/resource_ptr.h>

namespace beScene
{
	
// Prototypes
class PerspectiveEffectBinderPool;
class AbstractRenderableEffectDriver;
class AbstractLightEffectDriver;
class AbstractProcessingEffectDriver;
class RenderableMaterialCache;
class RenderableMeshCache;
class LightMaterialCache;

/// Renderer.
class EffectDrivenRenderer : public Renderer
{
public:
	/// Constructor.
	BE_SCENE_API EffectDrivenRenderer(Renderer &base,
		PerspectiveEffectBinderPool *perspectiveEffectPool,
		EffectDriverCache<AbstractProcessingEffectDriver> *processingDrivers,
		EffectDriverCache<AbstractRenderableEffectDriver> *renderableDrivers,
		RenderableMaterialCache *renderableMaterials,
		RenderableMeshCache *renderableMeshes,
		EffectDriverCache<AbstractLightEffectDriver> *lightDrivers,
		LightMaterialCache *lightMaterials);
	/// Copy constructor.
	BE_SCENE_API EffectDrivenRenderer(const EffectDrivenRenderer &right);
	/// Destructor.
	BE_SCENE_API ~EffectDrivenRenderer();

	/// Invalidates all caches.
	BE_SCENE_API virtual void InvalidateCaches();

	/// Perspective effect binder pool.
	lean::resource_ptr< PerspectiveEffectBinderPool > PerspectiveEffectBinderPool;

	/// Processing effect driver cache. 
	lean::resource_ptr< EffectDriverCache<AbstractProcessingEffectDriver> > ProcessingDrivers;

	/// Renderable effect driver cache.
	lean::resource_ptr< EffectDriverCache<AbstractRenderableEffectDriver> > RenderableDrivers;
	/// Effect-driven renderable material cache.
	lean::resource_ptr< RenderableMaterialCache > RenderableMaterials;
	/// Effect-driven renderable mesh cache.
	lean::resource_ptr< RenderableMeshCache > RenderableMeshes;

	/// Light effect driver cache.
	lean::resource_ptr< EffectDriverCache<AbstractLightEffectDriver> > LightDrivers;
	/// Effect-driven light material cache.
	lean::resource_ptr< LightMaterialCache > LightMaterials;

	/// Commits / reacts to changes.
	BE_SCENE_API virtual void Commit();
};

/// Creates a renderer from the given parameters.
BE_SCENE_API lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(beGraphics::Device *device, beCore::ComponentMonitor *pMonitor);
/// Creates a renderer from the given parameters.
BE_SCENE_API lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(Renderer &renderer, beCore::ComponentMonitor *pMonitor);
/// Creates a renderer from the given parameters.
BE_SCENE_API lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(Renderer &renderer, beCore::ComponentMonitor *pMonitor,
	PerspectiveEffectBinderPool *pPerspectivePool, 
	EffectDriverCache<AbstractProcessingEffectDriver> *pProcessingDrivers,
	EffectDriverCache<AbstractRenderableEffectDriver> *pRenderableDrivers,
	EffectDriverCache<AbstractLightEffectDriver> *pLightDrivers);
/// Creates a renderer from the given parameters.
BE_SCENE_API lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(Renderer &renderer, beCore::ComponentMonitor *pMonitor,
	PerspectiveEffectBinderPool *pPerspectivePool, 
	EffectDriverCache<AbstractProcessingEffectDriver> *pProcessingDrivers,
	EffectDriverCache<AbstractRenderableEffectDriver> *pRenderableDrivers,
	RenderableMaterialCache *pRenderableMaterials,
	RenderableMeshCache *pRenderableMesh,
	EffectDriverCache<AbstractLightEffectDriver> *pLightDrivers,
	LightMaterialCache *pLightMaterials);

} // namespace

#endif