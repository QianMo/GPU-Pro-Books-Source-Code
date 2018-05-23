/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL
#define BE_GRAPHICS_MATERIAL

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beManagedResource.h>
#include <beCore/beReflectedComponent.h>
#include "beTextureProvider.h"
#include <lean/tags/noncopyable.h>

#include <beCore/beMany.h>
#include <lean/containers/strided_ptr.h>

namespace beGraphics
{

using beCore::PropertyDesc;
class Effect;
class Technique;
class Material;
class MaterialConfig;
class MaterialCache;
class EffectCache;
class DeviceContext;

/// Material technique.
struct MaterialTechnique
{
	Material *Material;
	const Technique *Technique;

	MaterialTechnique(class Material *material,
			const class Technique *technique)
		: Material(material),
		Technique(technique) { }

	/// Applies this technique.
	LEAN_INLINE void Apply(const DeviceContext &context) const;
};

/// Public material binding interface.
class MaterialReflectionBinding : public beCore::ReflectedComponent, public TextureProvider
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(MaterialReflectionBinding)
};

/// Material interface.
class Material : public beCore::Resource, public beCore::OptionalPropertyProvider<beCore::ReflectedComponent>,
	public beCore::ManagedResource<MaterialCache>, public beCore::HotResource<Material>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(Material)

public:
	/// Rebinds the data sources.
	virtual void Rebind() = 0;

	/// Applys the setup.
	virtual void Apply(const MaterialTechnique *technique, const DeviceContext &context) = 0;

	/// Effects.
	typedef beCore::Range<const Effect *const*> Effects;
	/// Gets the effects.
	virtual Effects GetEffects() const = 0;

	/// Gets the number of techniques.
	virtual uint4 GetTechniqueCount() const = 0;
	/// Gets a technique by name.
	virtual uint4 GetTechniqueIdx(const utf8_ntri &name) = 0;
	/// Gets the number of techniques.
	virtual const MaterialTechnique* GetTechnique(uint4 idx) = 0;
	
	/// Gets a technique by name.
	LEAN_INLINE const MaterialTechnique* GetTechniqueByName(const utf8_ntri &name)
	{
		uint4 techniqueIdx = GetTechniqueIdx(name);
		return (techniqueIdx != -1) ? GetTechnique(techniqueIdx) : nullptr;
	}

	/// Gets the number of configurations.
	virtual uint4 GetConfigurationCount() const = 0;
	/// Sets the given configuration.
	virtual void SetConfiguration(uint4 idx, MaterialConfig *config, uint4 layerMask = -1) = 0;
	/// Gets the configurations.
	virtual MaterialConfig* GetConfiguration(uint4 idx, uint4 *pLayerMask = nullptr) const = 0;

	/// Material configuration layers.
	typedef beCore::Range< lean::strided_ptr<MaterialConfig *const> > Configurations;
	/// Sets all layered material configurations (important first).
	virtual void SetConfigurations(MaterialConfig *const *config, uint4 configCount) = 0;
	/// Gets all layered material configurations (important first).
	virtual Configurations GetConfigurations() const = 0;

	/// Gets a merged reflection binding.
	virtual lean::com_ptr<MaterialReflectionBinding, lean::critical_ref> GetFixedBinding() = 0;
	/// Gets a reflection binding for the given configuration.
	virtual lean::com_ptr<MaterialReflectionBinding, lean::critical_ref> GetConfigBinding(uint4 configIdx) = 0;

	/// Gets the component type.
	BE_GRAPHICS_API static const beCore::ComponentType* GetComponentType();
	/// Gets the component type.
	BE_GRAPHICS_API const beCore::ComponentType* GetType() const;
};

// Applies this technique.
LEAN_INLINE void MaterialTechnique::Apply(const DeviceContext &context) const
{
	Material->Apply(this, context);
}

/// Creates a new material.
BE_GRAPHICS_API lean::resource_ptr<Material, lean::critical_ref>
	CreateMaterial(const Effect *const* effects, uint4 effectCount, EffectCache &effectCache);
/// Creates a new material.
BE_GRAPHICS_API lean::resource_ptr<Material, lean::critical_ref> CreateMaterial(const Material &right);

} // namespace

#endif