/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_COMPONENT
#define BE_CORE_COMPONENT

#include "beCore.h"
#include "beShared.h"

namespace beCore
{

/// Component type identifier. USE POD INITIALIZATION.
struct ComponentType
{
	const char *Name;	///< Type name.
};

/// Generic component interface.
class LEAN_INTERFACE Component : public RefCounted
{
	LEAN_INTERFACE_BEHAVIOR(Component)

public:
	/// Gets the component type.
	virtual const ComponentType* GetType() const = 0;
#ifdef DOXYGEN_READ_THIS
	/// Gets the component type.
	static virtual const ComponentType* GetComponentType() = 0;
#endif
};

class ComponentTypes;

/// Gets the component type register.
BE_CORE_API ComponentTypes& GetComponentTypes();

/// Adds a component type.
BE_CORE_API void AddComponentType(const ComponentType *type, ComponentTypes &types = GetComponentTypes());
/// Adds a component type.
BE_CORE_API void RemoveComponentType(const ComponentType *type, ComponentTypes &types = GetComponentTypes());

/// Instantiate this to add a reflector of the given type.
struct ComponentTypePlugin
{
	/// Component type.
	const ComponentType *const Type;

	/// Adds the reflector.
	ComponentTypePlugin(const ComponentType *type)
		: Type(type) 
	{
		AddComponentType(type);
	}
	/// Removes the reflector.
	~ComponentTypePlugin()
	{
		RemoveComponentType(Type);
	}
};

} // namespace

/// Associates the given component type with the given type.
#define BE_CORE_ASSOCIATE_COMPONENT_TYPE_X(type, dynamic_method, static_method, component_type) \
	const beCore::ComponentType* type::static_method() { return &component_type; } \
	const beCore::ComponentType* type::dynamic_method() const { return &component_type; }
/// Associates the given component type with the given type.
#define BE_CORE_ASSOCIATE_COMPONENT_TYPE(type, component_type) BE_CORE_ASSOCIATE_COMPONENT_TYPE_X(type, GetType, GetComponentType, component_type)
/// Registers a component type for the given type.
#define BE_CORE_REGISTER_COMPONENT_TYPE_ALIAS(type, name) \
	const beCore::ComponentType LEAN_JOIN_VALUES(type, Type) = { name }; \
	const beCore::ComponentTypePlugin LEAN_JOIN_VALUES(type, TypePlugin)(&LEAN_JOIN_VALUES(type, Type));
/// Registers a component type for the given type.
#define BE_CORE_REGISTER_COMPONENT_TYPE(type) \
	BE_CORE_REGISTER_COMPONENT_TYPE_ALIAS(type, LEAN_QUOTE_VALUE(type))
/// Publishes the given type as component type under the given name.
#define BE_CORE_PUBLISH_COMPONENT_ALIAS(type, name) \
	BE_CORE_REGISTER_COMPONENT_TYPE_ALIAS(type, name) \
	BE_CORE_ASSOCIATE_COMPONENT_TYPE(type, LEAN_JOIN_VALUES(type, Type))
/// Publishes the given type as component type.
#define BE_CORE_PUBLISH_COMPONENT(type) \
	BE_CORE_REGISTER_COMPONENT_TYPE(type) \
	BE_CORE_ASSOCIATE_COMPONENT_TYPE(type, LEAN_JOIN_VALUES(type, Type))

#endif