/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beRenderableMaterial.h"
#include "beScene/beRenderableMaterialCache.h"

#include "beScene/beLightMaterial.h"
#include "beScene/beLightMaterialCache.h"

namespace beScene
{

extern const beCore::ComponentType RenderableMaterialType = { "RenderableMaterial" };

// Gets the component type.
const beCore::ComponentType* RenderableMaterial::GetComponentType()
{
	return &RenderableMaterialType;
}

// Creates a bound material for the given material.
lean::resource_ptr<GenericBoundMaterial, lean::critical_ref> RenderableMaterialCache::CreateBoundMaterial(beGraphics::Material *material)
{
	return new_resource RenderableMaterial(material, *m_driverCache);
}

// Creates a new material cache.
lean::resource_ptr<RenderableMaterialCache, lean::critical_ref> CreateRenderableMaterialCache(AbstractRenderableEffectDriverCache *driverCache)
{
	return new_resource RenderableMaterialCache(driverCache);
}


extern const beCore::ComponentType LightMaterialType = { "LightMaterial" };

// Gets the component type.
const beCore::ComponentType* LightMaterial::GetComponentType()
{
	return &LightMaterialType;
}

// Creates a bound material for the given material.
lean::resource_ptr<GenericBoundMaterial, lean::critical_ref> LightMaterialCache::CreateBoundMaterial(beGraphics::Material *material)
{
	return new_resource LightMaterial(material, *m_driverCache);
}

// Creates a new material cache.
lean::resource_ptr<LightMaterialCache, lean::critical_ref> CreateLightMaterialCache(AbstractLightEffectDriverCache *driverCache)
{
	return new_resource LightMaterialCache(driverCache);
}

} // namespace
