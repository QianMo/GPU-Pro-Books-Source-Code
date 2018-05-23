/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_CONFIG_DX11
#define BE_GRAPHICS_MATERIAL_CONFIG_DX11

#include "beGraphics.h"
#include "../beMaterialConfig.h"
#include "beEffect.h"
#include "beTexture.h"
#include <D3DX11Effect.h>
#include <lean/smart/resource_ptr.h>
#include <vector>
#include <lean/smart/scoped_ptr.h>
#include <lean/containers/simple_vector.h>
#include <lean/containers/parallel_vector.h>

namespace beGraphics
{

class TextureCache;

namespace DX11
{

using beCore::PropertyDesc;

/// Setup implementation.
class MaterialConfig : public lean::nonassignable,
	public beCore::ResourceToRefCounted< MaterialConfig, beCore::PropertyFeedbackProvider<beGraphics::MaterialConfig> >
{
public:
	/// Property.
	struct Property
	{
		utf8_string name;
		PropertyDesc desc;
		
		/// Constructor.
		Property(const utf8_ntri& name,
			const PropertyDesc &desc)
				: name(name.to<utf8_string>()),
				desc(desc){ }
	};

	/// Property data.
	struct PropertyData
	{
		uint4 offset;
		uint2 count;
		uint1 elementSize;
		bool bSet;
		const std::type_info *type;

		/// Constructor.
		PropertyData(uint4 offset, uint2 count, uint1 elementSize, const std::type_info &type)
			: offset(offset),
			count(count),
			elementSize(elementSize),
			bSet(false),
			type(&type) { }
	};
	
	enum property_desc_t { propertyDescs };
	enum property_data_t { propertyData };
	typedef lean::parallel_vector_t< lean::simple_vector_binder<lean::vector_policies::semipod> >::make<
			Property, property_desc_t,
			PropertyData, property_data_t
		>::type properties_t;
	typedef lean::simple_vector<char, lean::containers::vector_policies::inipod> backingstore_t;

	/// Texture.
	struct Texture
	{
		utf8_string name;

		/// Constructor.
		Texture(const utf8_ntri& name)
			: name(name.to<utf8_string>()) { }
	};

	/// Texture.
	struct TextureData
	{
		lean::resource_ptr<const TextureView> pTexture;
		bool bSet;
		bool bIsColor;
		
		/// Constructor.
		explicit TextureData(bool bIsColor)
			: bSet(false),
			bIsColor(bIsColor) { }
	};

	enum texture_desc_t { textureDescs };
	enum texture_data_t { textureData };
	typedef lean::parallel_vector_t< lean::simple_vector_binder<lean::vector_policies::semipod> >::make<
			Texture, texture_desc_t,
			TextureData, texture_data_t
		>::type textures_t;

	struct M
	{
		backingstore_t backingStore;
		properties_t properties;
		textures_t textures;

		M() { }
	};

protected:
	M m;

	MaterialConfigRevision *const m_revision;

public:
	/// Constructor.
	BE_GRAPHICS_DX11_API MaterialConfig();
	/// Constructor.
	BE_GRAPHICS_DX11_API MaterialConfig(const MaterialConfig &right);
	/// Destructor.
	BE_GRAPHICS_DX11_API ~MaterialConfig();

	/// Adds the given property.
	BE_GRAPHICS_DX11_API uint4 AddProperty(const utf8_ntri& name, const PropertyDesc &desc);
	/// Unsets the given property.
	BE_GRAPHICS_DX11_API void UnsetProperty(uint4 propertyID);

	/// Adds the given texture.
	BE_GRAPHICS_DX11_API uint4 AddTexture(const utf8_ntri& name, bool bIsColor);
	/// Unsets the given texture.
	BE_GRAPHICS_DX11_API void UnsetTexture(uint4 textureID);

	/// Gets the number of properties.
	BE_GRAPHICS_DX11_API uint4 GetPropertyCount() const;
	/// Gets the ID of the given property.
	BE_GRAPHICS_DX11_API uint4 GetPropertyID(const utf8_ntri &name) const;
	/// Gets the ID of the given property.
	BE_GRAPHICS_DX11_API uint4 GetMaterialPropertyID(const utf8_ntri &name, const PropertyDesc &desc) const;
	/// Gets the name of the given property.
	BE_GRAPHICS_DX11_API utf8_ntr GetPropertyName(uint4 id) const;
	/// Gets the type of the given property.
	BE_GRAPHICS_DX11_API PropertyDesc GetPropertyDesc(uint4 id) const;

	/// Sets the given (raw) values.
	BE_GRAPHICS_DX11_API bool SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count);
	/// Gets the given number of (raw) values.
	BE_GRAPHICS_DX11_API bool GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const;

	/// Sets the given (raw) values.
	BE_GRAPHICS_DX11_API bool SetMaterialPropertyRaw(uint4 id, const void *values, size_t bytes);
	/// Gets the given number of (raw) values.
	BE_GRAPHICS_DX11_API bool GetMaterialPropertyRaw(uint4 id, void *values, size_t bytes) const;

	/// Visits a property for modification.
	BE_GRAPHICS_DX11_API bool WriteProperty(uint4 id, beCore::PropertyVisitor &visitor, uint4 flags = beCore::PropertyVisitFlags::None);
	/// Visits a property for reading.
	BE_GRAPHICS_DX11_API bool ReadProperty(uint4 id, beCore::PropertyVisitor &visitor, uint4 flags = beCore::PropertyVisitFlags::None) const;

	/// Gets the number of textures.
	BE_GRAPHICS_DX11_API uint4 GetTextureCount() const;
	/// Gets the ID of the given texture.
	BE_GRAPHICS_DX11_API uint4 GetTextureID(const utf8_ntri &name) const;
	/// Gets the name of the given texture.
	BE_GRAPHICS_DX11_API utf8_ntr GetTextureName(uint4 id) const;
	/// Gets whether the texture is a color texture.
	BE_GRAPHICS_DX11_API bool IsColorTexture(uint4 id) const;

	/// Sets the given texture.
	BE_GRAPHICS_DX11_API void SetTexture(uint4 id, const beGraphics::TextureView *pView);
	/// Gets the given texture.
	BE_GRAPHICS_DX11_API const TextureView* GetTexture(uint4 id) const;

	/// Gets the number of child components.
	BE_GRAPHICS_DX11_API uint4 GetComponentCount() const;
	/// Gets the name of the n-th child component.
	BE_GRAPHICS_DX11_API beCore::Exchange::utf8_string GetComponentName(uint4 idx) const;
	/// Gets the n-th reflected child component, nullptr if not reflected.
	BE_GRAPHICS_DX11_API lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const;

	using beg::MaterialConfig::GetComponentType;
	/// Gets the type of the n-th child component.
	BE_GRAPHICS_DX11_API const beCore::ComponentType* GetComponentType(uint4 idx) const;
	/// Gets the n-th component.
	BE_GRAPHICS_DX11_API lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const;

	/// Returns true, if the n-th component can be replaced.
	BE_GRAPHICS_DX11_API bool IsComponentReplaceable(uint4 idx) const;
	/// Sets the n-th component.
	BE_GRAPHICS_DX11_API void SetComponent(uint4 idx, const lean::any &pComponent);

	/// Gets the revision.
	LEAN_INLINE const MaterialConfigRevision* GetRevision() const { return m_revision; }

	/// Gets the implementation identifier.
	LEAN_INLINE ImplementationID GetImplementationID() const { return DX11Implementation; }
};

template <> struct ToImplementationDX11<beGraphics::MaterialConfig> { typedef MaterialConfig Type; };

} // namespace

} // namespace

#endif