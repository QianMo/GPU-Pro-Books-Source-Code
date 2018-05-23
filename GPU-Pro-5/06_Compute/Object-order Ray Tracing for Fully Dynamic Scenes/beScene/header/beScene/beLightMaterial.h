/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_LIGHT_MATERIAL
#define BE_SCENE_LIGHT_MATERIAL

#include "beScene.h"
#include "beBoundMaterial.h"
#include "beAbstractLightEffectDriver.h"

#include <beCore/beComponent.h>

namespace beScene
{

/// Light material layer.
typedef BoundMaterialTechnique<AbstractLightEffectDriver> LightMaterialTechnique;

/// Material class.
class LightMaterial : public BoundMaterial<AbstractLightEffectDriver, LightMaterial>
{
public:
	/// Constructor.
	LEAN_INLINE LightMaterial(beGraphics::Material *material, EffectDriverCache<AbstractLightEffectDriver> &driverCache)
		: BoundMaterial(material, driverCache) { }

	/// Gets the component type.
	BE_SCENE_API static const beCore::ComponentType* GetComponentType();
};

} // namespace

#endif