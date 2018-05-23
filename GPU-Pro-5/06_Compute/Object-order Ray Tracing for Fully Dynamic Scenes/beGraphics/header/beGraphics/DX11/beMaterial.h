/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_DX11
#define BE_GRAPHICS_MATERIAL_DX11

#include "beGraphics.h"
#include "../beMaterial.h"
#include "beEffect.h"
#include "beTexture.h"
#include "beMaterialConfig.h"
#include <D3DX11Effect.h>
#include <lean/smart/resource_ptr.h>
#include <vector>
#include <lean/smart/scoped_ptr.h>

namespace beGraphics
{

class TextureCache;
class MaterialCache;

struct MaterialConfigRevision;

namespace DX11
{

using beCore::PropertyDesc;
class MaterialConfig;

/// Setup implementation.
class Material : public beCore::ResourceToRefCounted< Material, beCore::PropertyFeedbackProvider<beGraphics::Material> >
{
public:
	struct ReflectionBinding;

	typedef lean::simple_vector<lean::resource_ptr<const beg::Effect>, lean::vector_policies::semipod> effects_t;

	struct DataSource;
	typedef lean::simple_vector<DataSource, lean::vector_policies::semipod> datasources_t;

	struct Constants;
	struct ConstantDataLink;
	typedef lean::simple_vector<Constants, lean::containers::vector_policies::semipod> constants_t;
	typedef lean::simple_vector<ConstantDataLink, lean::containers::vector_policies::semipod> constant_data_links_t;

	struct PropertyData;
	typedef lean::simple_vector<char, lean::containers::vector_policies::inipod> backingstore_t;
	typedef lean::simple_vector<PropertyData, lean::containers::vector_policies::inipod> properties_t;

	struct TextureData;
	struct TextureDataLink;
	typedef lean::simple_vector<TextureData, lean::containers::vector_policies::semipod> textures_t;
	typedef lean::simple_vector<TextureDataLink, lean::containers::vector_policies::inipod> texture_data_links_t;
	
	typedef lean::simple_vector<uint4, lean::containers::vector_policies::inipod> indices_t;

	struct Technique;
	typedef lean::simple_vector<Technique, lean::vector_policies::semipod> techniques_t;

	struct Data
	{
		backingstore_t backingStore;
		constants_t constants;
		texture_data_links_t textureDataLinks;

		properties_t properties;
		constant_data_links_t constantDataLinks;

		textures_t textures;
	};

	struct M
	{
		effects_t effects;
		uint4 hiddenEffectCount;

		Data data;
		bool bBindingChanged;

		techniques_t techniques;

		datasources_t dataSources;
		uint4 immutableBaseConfigCount;
	};

private:
	M m;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API Material(const beGraphics::Effect *const* effects, uint4 effectCount, beGraphics::EffectCache &effectCache);
	/// Copies the given material.
	BE_GRAPHICS_DX11_API Material(const Material &right);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~Material();

	/// Rebinds the data sources.
	BE_GRAPHICS_DX11_API void Rebind();

	/// Applys the setup.
	BE_GRAPHICS_DX11_API void Apply(const MaterialTechnique *technique, const beGraphics::DeviceContext &context) LEAN_OVERRIDE;
	
	/// Gets the effects.
	BE_GRAPHICS_DX11_API Effects GetEffects() const;
	/// Gets linked-in effects.
	BE_GRAPHICS_DX11_API Effects GetLinkedEffects() const;

	/// Gets the number of techniques.
	BE_GRAPHICS_DX11_API uint4 GetTechniqueCount() const LEAN_OVERRIDE;
	/// Gets a technique by name.
	BE_GRAPHICS_DX11_API uint4 GetTechniqueIdx(const utf8_ntri &name) LEAN_OVERRIDE;
	/// Gets the number of techniques.
	BE_GRAPHICS_DX11_API const MaterialTechnique* GetTechnique(uint4 idx) LEAN_OVERRIDE;

	/// Gets the number of configurations.
	BE_GRAPHICS_DX11_API uint4 GetConfigurationCount() const LEAN_OVERRIDE;
	/// Sets the given configuration.
	BE_GRAPHICS_DX11_API void SetConfiguration(uint4 idx, beGraphics::MaterialConfig *config, uint4 layerMask = -1) LEAN_OVERRIDE;
	/// Gets the configurations.
	BE_GRAPHICS_DX11_API MaterialConfig* GetConfiguration(uint4 idx, uint4 *pLayerMask = nullptr) const LEAN_OVERRIDE;

	/// Sets all layered material configurations (important first).
	BE_GRAPHICS_DX11_API void SetConfigurations(beGraphics::MaterialConfig *const *config, uint4 configCount) LEAN_OVERRIDE;
	/// Gets all layered material configurations (important first).
	BE_GRAPHICS_DX11_API Configurations GetConfigurations() const LEAN_OVERRIDE;

	/// Gets a merged reflection binding.
	BE_GRAPHICS_DX11_API lean::com_ptr<MaterialReflectionBinding, lean::critical_ref> GetFixedBinding() LEAN_OVERRIDE;
	/// Gets a reflection binding for the given configuration.
	BE_GRAPHICS_DX11_API lean::com_ptr<MaterialReflectionBinding, lean::critical_ref> GetConfigBinding(uint4 configIdx) LEAN_OVERRIDE;

	/// Gets the number of child components.
	BE_GRAPHICS_DX11_API uint4 GetComponentCount() const;
	/// Gets the name of the n-th child component.
	BE_GRAPHICS_DX11_API beCore::Exchange::utf8_string GetComponentName(uint4 idx) const;
	/// Returns true, if the n-th component is issential.
	BE_GRAPHICS_DX11_API bool Material::IsComponentEssential(uint4 idx) const;
	/// Gets the n-th reflected child component, nullptr if not reflected.
	BE_GRAPHICS_DX11_API lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const;

	using beg::Material::GetComponentType;
	/// Gets the type of the n-th child component.
	BE_GRAPHICS_DX11_API const beCore::ComponentType* GetComponentType(uint4 idx) const;
	/// Gets the n-th component.
	BE_GRAPHICS_DX11_API lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const;

	/// Returns true, if the n-th component can be replaced.
	BE_GRAPHICS_DX11_API bool IsComponentReplaceable(uint4 idx) const;
	/// Sets the n-th component.
	BE_GRAPHICS_DX11_API void SetComponent(uint4 idx, const lean::any &pComponent);
	
	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::Material> { typedef Material Type; };

} // namespace

} // namespace

#endif