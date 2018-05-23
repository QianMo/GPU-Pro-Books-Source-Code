/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beComponentTypes.h"
#include "beCore/beComponentReflector.h"

#include <lean/logging/errors.h>

namespace beCore
{

// Constructor.
ComponentTypes::ComponentTypes()
{
}

// Destructor.
ComponentTypes::~ComponentTypes()
{
}

// Adds the given component type, returning a unique address for this type.
const ComponentTypeDesc& ComponentTypes::AddType(const ComponentType *type)
{
	LEAN_ASSERT(type);
	LEAN_ASSERT(type->Name && strlen(type->Name) > 0);

	std::pair<typename_map::iterator, bool> it = m_typeNames.insert(std::make_pair(utf8_nt(type->Name), ComponentTypeDesc(type)));

	if (!it.second)
	{
		// NOTE: Allow for re-initialization on code swap
		if (!it.first->second.Type)
			it.first->second = ComponentTypeDesc(type);
		else
			LEAN_THROW_ERROR_CTX("Component type name collision", type->Name);
	}

	try { m_types[type] = &it.first->second; }  LEAN_ASSERT_NOEXCEPT

	return it.first->second;
}

// Removes the given component type.
void ComponentTypes::RemoveType(const ComponentType *type)
{
	LEAN_ASSERT(type);

	typename_map::iterator it = m_typeNames.find(utf8_nt(type->Name));

	if (it != m_typeNames.end() && it->second.Type == type)
	{
		// NOTE: Invalidate but keep, allow for code swap!
		it->second.Type = nullptr;

		try { m_types.erase(type); }  LEAN_ASSERT_NOEXCEPT
	}
}

// Sets the given reflector for the given type.
void ComponentTypes::SetReflector(const ComponentReflector *reflector)
{
	LEAN_ASSERT(reflector);
	const beCore::ComponentType *type = reflector->GetType();
	LEAN_ASSERT(type);
	typename_map::iterator it = m_typeNames.find(utf8_nt(type->Name));

	if (it != m_typeNames.end() && it->second.Type == type)
		it->second.Reflector = reflector;
	else
		LEAN_THROW_ERROR_CTX("Reflector type not registered", type->Name);
}

// Unsets the given reflector for the given type.
void ComponentTypes::UnsetReflector(const ComponentReflector *pReflector)
{
	if (!pReflector)
		return;

	const beCore::ComponentType *type = pReflector->GetType();
	LEAN_ASSERT(type);
	typename_map::iterator it = m_typeNames.find(utf8_nt(type->Name));

	if (it != m_typeNames.end() && it->second.Type == type && it->second.Reflector == pReflector)
		it->second.Reflector = nullptr;
}

// Gets a component type address for the given component type name.
const ComponentTypeDesc* ComponentTypes::GetDesc(const utf8_ntri &name) const
{
	typename_map::const_iterator it = m_typeNames.find(utf8_nt(name));

	// NOTE: Check for invalidated type descs
	return (it != m_typeNames.end() && it->second.Type)
		? &it->second
		: nullptr;
}

// Gets a component type address for the given component type name.
const ComponentTypeDesc* ComponentTypes::GetDesc(const void *type) const
{
	type_map::const_iterator it = m_types.find( static_cast<const ComponentType*>(type) );

	return (it != m_types.end())
		? it->second
		: nullptr;
}

// Gets all reflectors.
ComponentTypes::TypeDescs ComponentTypes::GetDescs() const
{
	TypeDescs descs;
	descs.reserve(m_types.size());

	for (type_map::const_iterator it = m_types.begin(); it != m_types.end(); ++it)
		descs.push_back(it->second);

	return descs;
}

// Gets the component type register.
ComponentTypes& GetComponentTypes()
{
	static ComponentTypes componentTypes;
	return componentTypes;
}

// Adds a component type.
void AddComponentType(const ComponentType *type, ComponentTypes &types)
{
	types.AddType(type);
}

// Adds a component type.
void RemoveComponentType(const ComponentType *type, ComponentTypes &types)
{
	types.RemoveType(type);
}

} // namespace
