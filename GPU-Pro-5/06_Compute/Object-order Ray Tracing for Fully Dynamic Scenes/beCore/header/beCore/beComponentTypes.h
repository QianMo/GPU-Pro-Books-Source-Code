/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_COMPONENT_TYPES
#define BE_CORE_COMPONENT_TYPES

#include "beCore.h"
#include "beComponent.h"
#include "beExchangeContainers.h"
#include <lean/tags/noncopyable.h>
#include <unordered_map>
#include <lean/strings/hashing.h>

namespace beCore
{

class ComponentReflector;

/// Component type descriptor.
struct ComponentTypeDesc
{
	const ComponentType *Type;				///< Type identifier.
	utf8_ntr Name;							///< Type name.
	const ComponentReflector *Reflector;	///< Type reflector.

	/// Initializing constructor.
	explicit ComponentTypeDesc(const ComponentType *type,
			const ComponentReflector *pReflector = nullptr)
		: Type(type),
		Name(type->Name),
		Reflector(pReflector) { }
};

/// Manages component types & reflectors.
class ComponentTypes : public lean::noncopyable
{
private:
	typedef std::unordered_map< utf8_nt, ComponentTypeDesc, struct lean::hash<utf8_nt> > typename_map;
	typename_map m_typeNames;
	typedef std::unordered_map<const ComponentType*, ComponentTypeDesc*> type_map;
	type_map m_types;

public:
	/// Constructor.
	BE_CORE_API ComponentTypes();
	/// Destructor.
	BE_CORE_API ~ComponentTypes();

	/// Adds the given component type, returning a unique address for this type.
	BE_CORE_API const ComponentTypeDesc& AddType(const ComponentType *type);
	/// Removes the given component type.
	BE_CORE_API void RemoveType(const ComponentType *type);

	/// Sets the given reflector for the given type.
	BE_CORE_API void SetReflector(const ComponentReflector *reflector);
	/// Unsets the given reflector for the given type.
	BE_CORE_API void UnsetReflector(const ComponentReflector *pReflector);

	/// Gets a component type address for the given component type name.
	BE_CORE_API const ComponentTypeDesc* GetDesc(const utf8_ntri &name) const;
	/// Gets a component reflector for the given component type.
	LEAN_INLINE const ComponentReflector* GetReflector(const utf8_ntri &name) const
	{
		const ComponentTypeDesc *pDesc = GetDesc(name);
		return (pDesc) ? pDesc->Reflector : nullptr;
	}

	/// Gets a component type address for the given component type name.
	BE_CORE_API const ComponentTypeDesc* GetDesc(const void *type) const;
	/// Gets a component reflector for the given component type.
	LEAN_INLINE const ComponentReflector* GetReflector(const void *type) const
	{
		const ComponentTypeDesc *pDesc = GetDesc(type);
		return (pDesc) ? pDesc->Reflector : nullptr;
	}

	typedef Exchange::vector_t<const ComponentTypeDesc*>::t TypeDescs;
	/// Gets all component type descriptors.
	BE_CORE_API TypeDescs GetDescs() const;
};

/// Gets the component type register.
BE_CORE_API ComponentTypes& GetComponentTypes();

/// Instantiate this to add a reflector of the given type.
template <class ComponentReflector>
struct ComponentReflectorPlugin : ComponentTypePlugin
{
	/// Reflector.
	ComponentReflector Reflector;

	/// Adds the reflector.
	ComponentReflectorPlugin(const ComponentType *type)
		: ComponentTypePlugin(type)
	{
		GetComponentTypes().SetReflector(&Reflector);
	}
	/// Removes the reflector.
	~ComponentReflectorPlugin()
	{
		GetComponentTypes().UnsetReflector(&Reflector);
	}
};

} // namespace

#endif