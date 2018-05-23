/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_RENDERABLE_MATERIAL
#define BE_SCENE_RENDERABLE_MATERIAL

#include "beScene.h"
#include "beBoundMaterial.h"
#include "beAbstractRenderableEffectDriver.h"

#include <beCore/beComponent.h>

namespace beScene
{

/// Renderable material layer.
typedef BoundMaterialTechnique<AbstractRenderableEffectDriver> RenderableMaterialTechnique;

/// Material class.
class RenderableMaterial : public BoundMaterial<AbstractRenderableEffectDriver, RenderableMaterial>
{
public:
	/// Constructor.
	LEAN_INLINE RenderableMaterial(beGraphics::Material *material, EffectDriverCache<AbstractRenderableEffectDriver> &driverCache)
		: BoundMaterial(material, driverCache) { }

	/// Gets the component type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
};

} // namespace

#endif