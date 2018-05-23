/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MATERIAL_CONFIG
#define BE_GRAPHICS_MATERIAL_CONFIG

#include "beGraphics.h"
#include <beCore/beShared.h>
#include <beCore/beManagedResource.h>
#include <beCore/beReflectedComponent.h>
#include "beTextureProvider.h"
#include <lean/smart/resource_ptr.h>
#include <lean/tags/noncopyable.h>

namespace beGraphics
{

/// Material configuration versioning information.
struct MaterialConfigRevision
{
	uint4 Structure;	///< Configuration structure revision.
	uint4 Data;			///< Configuration data revision.

	/// Constructor.
	MaterialConfigRevision(uint4 revision = 0)
		: Structure(revision),
		Data(revision) { }
};

using beCore::PropertyDesc;
class MaterialConfigCache;

/// Setup interface.
class MaterialConfig : public beCore::Resource, public beCore::ReflectedComponent, public TextureProvider,
	public beCore::ManagedResource<MaterialConfigCache>, public beCore::HotResource<MaterialConfig>, public Implementation
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(MaterialConfig)

public:
	/// Adds the given property.
	virtual uint4 AddProperty(const utf8_ntri& name, const PropertyDesc &desc) = 0;
	/// Unsets the given property.
	virtual void UnsetProperty(uint4 propertyID) = 0;

	/// Adds the given texture.
	virtual uint4 AddTexture(const utf8_ntri& name, bool bIsColor) = 0;
	/// Unsets the given texture.
	virtual void UnsetTexture(uint4 textureID) = 0;

	/// Gets the revision.
	virtual const MaterialConfigRevision* GetRevision() const = 0;

	/// Gets the component type.
	BE_GRAPHICS_API static const beCore::ComponentType* GetComponentType();	
	/// Gets the component type.
	BE_GRAPHICS_API const beCore::ComponentType* GetType() const;

};

/// Creates a material configuration.
BE_GRAPHICS_API lean::resource_ptr<MaterialConfig, lean::critical_ref> CreateMaterialConfig();
/// Creates a material configuration.
BE_GRAPHICS_API lean::resource_ptr<MaterialConfig, lean::critical_ref> CreateMaterialConfig(const MaterialConfig &right);

/// Transfers all data from the given source config to the given destination config.
BE_GRAPHICS_API void Transfer(MaterialConfig &dest, const MaterialConfig &source);

} // namespace

#endif