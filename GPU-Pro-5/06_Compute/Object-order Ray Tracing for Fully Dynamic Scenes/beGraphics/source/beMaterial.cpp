/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/beMaterialConfig.h"
#include "beGraphics/beMaterial.h"
#include "beGraphics/beEffect.h"

#include <beCore/beComponentTypes.h>

namespace beGraphics
{

const beCore::ComponentType MaterialConfigType = { "MaterialConfig" };
// TODO: Move to component reflector
// const beCore::ComponentTypePlugin MaterialConfigTypePlugin(&MaterialConfigType);

/// Gets the component type.
const beCore::ComponentType* MaterialConfig::GetComponentType()
{
	return &MaterialConfigType;
}

/// Gets the component type.
const beCore::ComponentType* MaterialConfig::GetType() const
{
	return &MaterialConfigType;
}


const beCore::ComponentType MaterialType = { "Material" };
// TODO: Move to component reflector
const beCore::ComponentTypePlugin MaterialTypePlugin(&MaterialType);

/// Gets the component type.
const beCore::ComponentType* Material::GetComponentType()
{
	return &MaterialType;
}

/// Gets the component type.
const beCore::ComponentType* Material::GetType() const
{
	return &MaterialType;
}


const beCore::ComponentType TextureType = { "Texture" };
// TODO: Move to component reflector
// const beCore::ComponentTypePlugin TextureTypePlugin(&TextureType);

/// Gets the component type.
const beCore::ComponentType* TextureView::GetComponentType()
{
	return &TextureType;
}


const beCore::ComponentType EffectType = { "Effect" };
// TODO: Move to component reflector
// const beCore::ComponentTypePlugin EffectTypePlugin(&EffectType);

/// Gets the component type.
const beCore::ComponentType* Effect::GetComponentType()
{
	return &EffectType;
}

} // namespace
