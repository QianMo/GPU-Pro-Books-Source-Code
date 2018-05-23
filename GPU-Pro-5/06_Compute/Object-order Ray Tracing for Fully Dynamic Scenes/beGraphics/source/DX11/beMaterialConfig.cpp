/*****************************************************/
/* breeze Engine Graphics Module  (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beGraphicsInternal/stdafx.h"
#include "beGraphics/DX11/beMaterialConfig.h"

#include "beGraphics/DX11/beDeviceContext.h"
#include "beGraphics/DX11/beDevice.h"
#include "beGraphics/DX11/beD3DXEffects11.h"
#include "beGraphics/DX11/beTextureCache.h"
#include "beGraphics/DX11/beTexture.h"

#include <beCore/beBuiltinTypes.h>

#include "beGraphics/DX/beError.h"

#include <beCore/bePropertyVisitor.h>

#include <lean/functional/algorithm.h>
#include <lean/memory/chunk_pool.h>
#include <lean/logging/log.h>

namespace beGraphics
{

namespace DX11
{

// Keep revisions together
namespace { lean::chunk_pool<MaterialConfigRevision, 1024> revisionPool; }

// Constructor.
MaterialConfig::MaterialConfig()
	: m_revision( new(revisionPool.allocate()) MaterialConfigRevision(1) )
{
	LEAN_ASSERT(m_revision);
}

// Constructor.
MaterialConfig::MaterialConfig(const MaterialConfig &right)
	: m(right.m),
	 m_revision( new(revisionPool.allocate()) MaterialConfigRevision(1) )
{
}

// Destructor.
MaterialConfig::~MaterialConfig()
{
	revisionPool.free(m_revision);
}

// Adds the given property.
uint4 MaterialConfig::AddProperty(const utf8_ntri& name, const PropertyDesc &desc)
{
	LEAN_ASSERT(desc.TypeDesc);

	uint2 count = static_cast<uint2>(desc.Count);
	uint1 elementSize = static_cast<uint1>(desc.TypeDesc->Info.size);
	
	uint4 offset = static_cast<uint4>( m.backingStore.size() );
	m.backingStore.resize(offset + count * elementSize);
	
	uint4 propertyID = static_cast<uint4>( m.properties.size() );
	m.properties.push_back( Property(name, desc), PropertyData(offset, count, elementSize, desc.TypeDesc->Info.type) );

	// NOTE: Structure has not changed YET, property still unset
	return propertyID;
}

// Unsets the given property.
void MaterialConfig::UnsetProperty(uint4 propertyID)
{
	LEAN_ASSERT(propertyID < m.properties.size());

	PropertyData &data = m.properties(propertyData)[propertyID];
	m_revision->Structure += data.bSet;
	data.bSet = false;
}

// Adds the given texture.
uint4 MaterialConfig::AddTexture(const utf8_ntri& name, bool bIsColor)
{
	uint4 textureID = static_cast<uint4>( m.textures.size() );
	m.textures.push_back( Texture(name), TextureData(bIsColor) );

	// NOTE: Structure has not changed YET, property still unset
	return textureID;
}

// Unsets the given texture.
void MaterialConfig::UnsetTexture(uint4 textureID)
{
	LEAN_ASSERT(textureID < m.textures.size());

	TextureData &data = m.textures(textureData)[textureID];
	m_revision->Structure += data.bSet;
	data.bSet = false;
}

// Gets the number of properties.
uint4 MaterialConfig::GetPropertyCount() const
{
	return static_cast<uint4>(m.properties.size() + 1);
}

// Gets the ID of the given property.
uint4 MaterialConfig::GetPropertyID(const utf8_ntri &name) const
{
	for (properties_t::const_iterator it = m.properties.begin();
		it != m.properties.end(); ++it)
		if (it->name == name)
			return static_cast<uint4>(it - m.properties.begin());
	return -1;
}

// Gets the ID of the given property.
uint4 MaterialConfig::GetMaterialPropertyID(const utf8_ntri &name, const PropertyDesc &desc) const
{
	for (properties_t::const_iterator it = m.properties.begin();
		it != m.properties.end(); ++it)
		if (it->name == name && it->desc.TypeDesc == desc.TypeDesc && it->desc.Count == desc.Count)
			return static_cast<uint4>(it - m.properties.begin());

	return static_cast<uint4>(-1);
}

// Gets the name of the given property.
utf8_ntr MaterialConfig::GetPropertyName(uint4 id) const
{
	uint4 propertyCount = (uint4) m.properties.size();
	return (id < propertyCount)
		? utf8_ntr(m.properties[id].name)
		: utf8_ntr();
}

// Gets the type of the given property.
PropertyDesc MaterialConfig::GetPropertyDesc(uint4 id) const
{
	uint4 propertyCount = (uint4) m.properties.size();
	return (id < propertyCount)
		? m.properties[id].desc
		: PropertyDesc();
}

// Sets the given (raw) values.
bool MaterialConfig::SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count)
{
	uint4 propertyCount = (uint4) m.properties.size();
	if (id < propertyCount)
	{
		PropertyData &data = m.properties(propertyData)[id];

		// TODO: Realtime Debugging?
		if (*data.type == type)
		{
			memcpy(m.backingStore.data() + data.offset, values, min<uint4>(count, data.count) * data.elementSize);
			++m_revision->Data;
			m_revision->Structure += !data.bSet;
			data.bSet = true;
			EmitPropertyChanged();
			return true;
		}
	}
	return false;
}

// Gets the given number of (raw) values.
bool MaterialConfig::GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const
{
	uint4 propertyCount = (uint4) m.properties.size();
	if (id < propertyCount)
	{
		const PropertyData &data = m.properties(propertyData)[id];

		if (data.bSet && *data.type == type)
		{
			memcpy(values, m.backingStore.data() + data.offset, min<uint4>(count, data.count) * data.elementSize);
			return true;
		}
	}
	return false;
}

// Sets the given (raw) values.
bool MaterialConfig::SetMaterialPropertyRaw(uint4 id, const void *values, size_t bytes)
{
	LEAN_ASSERT(id < m.properties.size());
	PropertyData &data = m.properties(propertyData)[id];

	memcpy(m.backingStore.data() + data.offset, values, min<uint4>(bytes, data.count * data.elementSize));
	++m_revision->Data;
	m_revision->Structure += !data.bSet;
	data.bSet = true;
	EmitPropertyChanged();
	
	return true;
}

// Gets the given number of (raw) values.
bool MaterialConfig::GetMaterialPropertyRaw(uint4 id, void *values, size_t bytes) const
{
	LEAN_ASSERT(id < m.properties.size());
	const PropertyData &data = m.properties(propertyData)[id];

	if (data.bSet)
	{
		memcpy(values, m.backingStore.data() + data.offset, min<uint4>(bytes, data.count * data.elementSize));
		return true;
	}
	else
		return false;
}

// Visits a property for modification.
bool MaterialConfig::WriteProperty(uint4 id, beCore::PropertyVisitor &visitor, uint4 flags)
{
	uint4 propertyCount = (uint4) m.properties.size();
	if (id < propertyCount)
	{
		Property &property = m.properties[id];
		PropertyData &data = m.properties(propertyData)[id];

		bool bModified = visitor.Visit(
				*this,
				id,
				property.desc,
				m.backingStore.data() + data.offset
			);

		if (bModified)
		{
			++m_revision->Data;
			m_revision->Structure += !data.bSet;
			data.bSet = true;
			EmitPropertyChanged();
		}

		return bModified;
	}
	return false;
}

// Visits a property for reading.
bool MaterialConfig::ReadProperty(uint4 id, beCore::PropertyVisitor &visitor, uint4 flags) const
{
	uint4 propertyCount = (uint4) m.properties.size();
	if (id < propertyCount)
	{
		const Property &property = m.properties[id];
		const PropertyData &data = m.properties(propertyData)[id];

		// WARNING: Call read-only overload!
		visitor.Visit(
				*this,
				id,
				property.desc,
				const_cast<const char*>(m.backingStore.data() + data.offset)
			);

		return true;
	}
	return false;
}

// Gets the number of textures.
uint4 MaterialConfig::GetTextureCount() const
{
	return static_cast<uint4>( m.textures.size() );
}

// Gets the ID of the given texture.
uint4 MaterialConfig::GetTextureID(const utf8_ntri &name) const
{
	for (textures_t::const_iterator it = m.textures.begin();
		it != m.textures.end(); ++it)
		if (it->name == name)
			return static_cast<uint4>( it - m.textures.begin() );

	return static_cast<uint4>(-1);
}

// Gets the name of the given texture.
utf8_ntr MaterialConfig::GetTextureName(uint4 id) const
{
	return (id < m.textures.size())
		? utf8_ntr(m.textures[id].name)
		: utf8_ntr("");
}

// Gets whether the texture is a color texture.
bool MaterialConfig::IsColorTexture(uint4 id) const
{
	return (id < m.textures.size())
		? m.textures(textureData)[id].bIsColor
		: false; 
}

// Sets the given texture.
void MaterialConfig::SetTexture(uint4 id, const beGraphics::TextureView *pView)
{
	LEAN_ASSERT(id < m.textures.size());

	TextureData &data = m.textures(textureData)[id];

	data.pTexture = ToImpl(pView);
	++m_revision->Data;
	m_revision->Structure += !data.bSet;
	data.bSet = true;
	EmitPropertyChanged();
}

// Gets the given texture.
const TextureView* MaterialConfig::GetTexture(uint4 id) const
{
	LEAN_ASSERT(id < m.textures.size());

	const TextureData &data = m.textures(textureData)[id];
	return (data.bSet) ? data.pTexture : nullptr;
}

// Gets the number of child components.
uint4 MaterialConfig::GetComponentCount() const
{
	return GetTextureCount();
}

// Gets the name of the n-th child component.
beCore::Exchange::utf8_string MaterialConfig::GetComponentName(uint4 idx) const
{
	beCore::Exchange::utf8_string name;

	utf8_ntr textureName = GetTextureName(idx);
	name.reserve(textureName.size() + lean::ntarraylen(" (Texture)"));

	name.append(textureName.c_str(), textureName.size());
	name.append(" (Texture)", lean::ntarraylen(" (Texture)"));

	return name;
}

// Gets the n-th reflected child component, nullptr if not reflected.
lean::com_ptr<const beCore::ReflectedComponent, lean::critical_ref> MaterialConfig::GetReflectedComponent(uint4 idx) const
{
	return nullptr;
}

// Gets the type of the n-th child component.
const beCore::ComponentType* MaterialConfig::GetComponentType(uint4 idx) const
{
	return beg::TextureView::GetComponentType();
}

// Gets the n-th component.
lean::cloneable_obj<lean::any, true> MaterialConfig::GetComponent(uint4 idx) const
{
	return bec::any_resource_t<beGraphics::TextureView>::t( const_cast<beGraphics::DX11::TextureView*>( GetTexture(idx) ) );
}

// Returns true, if the n-th component can be replaced.
bool MaterialConfig::IsComponentReplaceable(uint4 idx) const
{
	return true;
}

// Sets the n-th component.
void MaterialConfig::SetComponent(uint4 idx, const lean::any &pComponent)
{
	SetTexture(idx, lean::any_cast<beGraphics::TextureView*>(pComponent));
}

} // namespace

// Creates a material configuration.
lean::resource_ptr<MaterialConfig, lean::critical_ref> CreateMaterialConfig()
{
	return new_resource DX11::MaterialConfig();
}

// Creates a material configuration.
lean::resource_ptr<MaterialConfig, lean::critical_ref> CreateMaterialConfig(const MaterialConfig &right)
{
	return new_resource DX11::MaterialConfig( ToImpl(right) );
}

} // namespace
