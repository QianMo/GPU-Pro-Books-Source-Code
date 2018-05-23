/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beEffectDrivenRenderer.h"
#include "beScene/bePerspectiveEffectBinderPool.h"
#include "beScene/beProcessingEffectDriverCache.h"
#include "beScene/beRenderableEffectDriverCache.h"
#include "beScene/beRenderableMaterialCache.h"
#include "beScene/beRenderableMeshCache.h"
#include "beScene/beLightEffectDriverCache.h"
#include "beScene/beLightMaterialCache.h"
#include <beGraphics/Any/beDevice.h>

namespace beScene
{

namespace
{

// Creates a perspective effect binder pool.
lean::resource_ptr<PerspectiveEffectBinderPool, true> CreatePerspectiveEffectPool(const beGraphics::Device &device)
{
	return new_resource PerspectiveEffectBinderPool(ToImpl(device));
}

} // namespace

// Constructor.
EffectDrivenRenderer::EffectDrivenRenderer(Renderer &base,
										   class PerspectiveEffectBinderPool *perspectiveEffectPool,
										   EffectDriverCache<AbstractProcessingEffectDriver> *processingDrivers,
										   EffectDriverCache<AbstractRenderableEffectDriver> *renderableDrivers,
										   RenderableMaterialCache *renderableMaterials,
										   RenderableMeshCache *renderableMeshes,
										   EffectDriverCache<AbstractLightEffectDriver> *lightDrivers,
										   LightMaterialCache *lightMaterials)
	: Renderer(base),
	PerspectiveEffectBinderPool( LEAN_ASSERT_NOT_NULL(perspectiveEffectPool) ),
	ProcessingDrivers( LEAN_ASSERT_NOT_NULL(processingDrivers) ),
	RenderableDrivers( LEAN_ASSERT_NOT_NULL(renderableDrivers) ),
	RenderableMaterials( LEAN_ASSERT_NOT_NULL(renderableMaterials) ),
	RenderableMeshes( LEAN_ASSERT_NOT_NULL(renderableMeshes) ),
	LightDrivers( LEAN_ASSERT_NOT_NULL(lightDrivers) ),
	LightMaterials( LEAN_ASSERT_NOT_NULL(lightMaterials) )
{
}

// Copy constructor.
EffectDrivenRenderer::EffectDrivenRenderer(const EffectDrivenRenderer &right)
	: Renderer(right),
	// MONITOR: REDUNDANT
	PerspectiveEffectBinderPool( LEAN_ASSERT_NOT_NULL(right.PerspectiveEffectBinderPool) ),
	ProcessingDrivers( LEAN_ASSERT_NOT_NULL(right.ProcessingDrivers) ),
	RenderableDrivers( LEAN_ASSERT_NOT_NULL(right.RenderableDrivers) ),
	RenderableMaterials( LEAN_ASSERT_NOT_NULL(right.RenderableMaterials) ),
	RenderableMeshes( LEAN_ASSERT_NOT_NULL(right.RenderableMeshes) ),
	LightDrivers( LEAN_ASSERT_NOT_NULL(right.LightDrivers) ),
	LightMaterials( LEAN_ASSERT_NOT_NULL(right.LightMaterials) )
{
}

// Destructor.
EffectDrivenRenderer::~EffectDrivenRenderer()
{
}

// Commits / reacts to changes.
void EffectDrivenRenderer::Commit()
{
	Renderer::Commit();

	RenderableMaterials->Commit();
	RenderableMeshes->Commit();
	LightMaterials->Commit();
}

// Invalidates all caches.
void EffectDrivenRenderer::InvalidateCaches()
{
	Renderer::InvalidateCaches();
	PerspectiveEffectBinderPool->Invalidate();
}

// Creates a renderer from the given parameters.
lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(beGraphics::Device *device, beCore::ComponentMonitor *pMonitor)
{
	return CreateEffectDrivenRenderer( *CreateRenderer(device), pMonitor );
}

// Creates a renderer from the given parameters.
lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(Renderer &renderer, beCore::ComponentMonitor *pMonitor)
{
	return CreateEffectDrivenRenderer(renderer, pMonitor, nullptr, nullptr, nullptr, nullptr);
}

// Creates a renderer from the given parameters.
lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(Renderer &renderer, beCore::ComponentMonitor *pMonitor,
	PerspectiveEffectBinderPool *pPerspectivePool, 
	EffectDriverCache<AbstractProcessingEffectDriver> *pProcessingDrivers,
	EffectDriverCache<AbstractRenderableEffectDriver> *pRenderableDrivers,
	EffectDriverCache<AbstractLightEffectDriver> *pLightDrivers)
{
	return CreateEffectDrivenRenderer( renderer, pMonitor, pPerspectivePool,
		pProcessingDrivers,
		pRenderableDrivers, nullptr, nullptr, pLightDrivers, nullptr);
}

// Creates a renderer from the given parameters.
lean::resource_ptr<EffectDrivenRenderer, true> CreateEffectDrivenRenderer(Renderer &renderer, beCore::ComponentMonitor *pMonitor,
	PerspectiveEffectBinderPool *pPerspectiveEffectPool, 
	EffectDriverCache<AbstractProcessingEffectDriver> *pProcessingDrivers,
	EffectDriverCache<AbstractRenderableEffectDriver> *pRenderableDrivers,
	RenderableMaterialCache *pRenderableMaterials,
	RenderableMeshCache *pRenderableMesh,
	EffectDriverCache<AbstractLightEffectDriver> *pLightDrivers,
	LightMaterialCache *pLightMaterials)
{
	lean::resource_ptr<PerspectiveEffectBinderPool> perspectiveEffectPool = (pPerspectiveEffectPool)
		? lean::secure_resource(pPerspectiveEffectPool)
		: CreatePerspectiveEffectPool(*renderer.Device());

	lean::resource_ptr< EffectDriverCache<AbstractProcessingEffectDriver> > processingDrivers = (pProcessingDrivers)
		? lean::secure_resource(pProcessingDrivers)
		: new_resource ProcessingEffectDriverCache(renderer.Pipeline(), perspectiveEffectPool);

	lean::resource_ptr< EffectDriverCache<AbstractRenderableEffectDriver> > renderableDrivers = (pRenderableDrivers)
		? lean::secure_resource(pRenderableDrivers)
		: new_resource RenderableEffectDriverCache(renderer.Pipeline(), perspectiveEffectPool);
	lean::resource_ptr<RenderableMaterialCache> renderableMaterials = (pRenderableMaterials)
		? lean::secure_resource(pRenderableMaterials)
		: new_resource RenderableMaterialCache(renderableDrivers);
	lean::resource_ptr<RenderableMeshCache> renderableMeshes = (pRenderableMesh)
		? lean::secure_resource(pRenderableMesh)
		: new_resource RenderableMeshCache(renderer.Device()); // TODO: renderableDrivers

	lean::resource_ptr< EffectDriverCache<AbstractLightEffectDriver> > lightDrivers = (pLightDrivers)
		? lean::secure_resource(pLightDrivers)
		: new_resource LightEffectDriverCache(renderer.Pipeline(), perspectiveEffectPool);
	lean::resource_ptr<LightMaterialCache> lightMaterials = (pLightMaterials)
		? lean::secure_resource(pLightMaterials)
		: new_resource LightMaterialCache(lightDrivers);

	if (pMonitor)
	{
		renderableMaterials->SetComponentMonitor(pMonitor);
		renderableMeshes->SetComponentMonitor(pMonitor);
		lightMaterials->SetComponentMonitor(pMonitor);
	}

	return new_resource EffectDrivenRenderer(
			renderer,
			perspectiveEffectPool,
			processingDrivers,
			renderableDrivers, renderableMaterials, renderableMeshes,
			lightDrivers, lightMaterials
		);
}

} // namespace
