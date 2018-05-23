/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_VALUE_TYPES
#define BE_CORE_VALUE_TYPES

#include "beCore.h"
#include "beValueType.h"
#include "beExchangeContainers.h"
#include <lean/tags/noncopyable.h>
#include <unordered_map>
#include <lean/strings/hashing.h>

namespace beCore
{

/// Manages component types & reflectors.
class ValueTypes : public lean::noncopyable
{
private:
	typedef std::unordered_map< utf8_nt, ValueTypeDesc, struct lean::hash<utf8_nt> > typename_map;
	typename_map m_typeNames;

public:
	/// Constructor.
	BE_CORE_API ValueTypes();
	/// Destructor.
	BE_CORE_API ~ValueTypes();

	/// Adds the given value type, returning a unique address for this type.
	BE_CORE_API const ValueTypeDesc& AddType(const lean::property_type_info &info);
	/// Removes the given value type.
	BE_CORE_API void RemoveType(const lean::property_type_info &info);

	/// Sets the given serializer for the given type.
	BE_CORE_API void SetSerializer(const TextSerializer *text);
	/// Unsets the given serializer for the given type.
	BE_CORE_API void UnsetSerializer(const TextSerializer *text);

	/// Gets a component type address for the given value type name.
	BE_CORE_API const ValueTypeDesc* GetDesc(const utf8_ntri &name) const;
/*	/// Gets a component type address for the given value type desc.
	BE_CORE_API const ValueTypeDesc* GetDesc(const void *type) const;
*/
	typedef Exchange::vector_t<const ValueTypeDesc*>::t TypeDescs;
	/// Gets all component type descriptors.
	BE_CORE_API TypeDescs GetDescs() const;
};

/// Gets the component type register.
BE_CORE_API ValueTypes& GetValueTypes();

} // namespace

#endif