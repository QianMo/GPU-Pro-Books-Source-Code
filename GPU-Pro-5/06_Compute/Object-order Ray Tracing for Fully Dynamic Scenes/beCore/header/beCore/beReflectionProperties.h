/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_REFLECTION_PROPERTIES
#define BE_CORE_REFLECTION_PROPERTIES

#include "beCore.h"
#include "bePropertyProvider.h"
#include <lean/properties/property.h>
#include <lean/properties/property_accessors.h>
#include "beValueType.h"
#include <lean/properties/property_collection.h>
#include "beBuiltinTypes.h"

namespace beCore
{

struct PropertyPersistence
{
	enum T
	{
		None = 0x0,
		Read = 0x1,
		Write = 0x2,
		ReadWrite = Read | Write
	};
};

struct ValueTypeDesc;

/// Reflection property.
struct ReflectionProperty : public lean::ui_property_desc<PropertyProvider, int2, ReflectionProperty, ValueTypeDesc>
{
	uint2 persistence;				///< Persistent property.

	/// Constructs an empty property description.
	ReflectionProperty() { }
	/// Constructs a property description from the given parameters.
	ReflectionProperty(const utf8_ntri &name, const ValueTypeDesc &typeDesc, size_t count, int2 widget, uint2 persistence)
		: ui_property_desc(name, typeDesc, count, widget),
		persistence(persistence) { }
};

/// Constructs a property description from the given parameters.
template <class Type>
LEAN_INLINE ReflectionProperty MakeReflectionPropertyN(const utf8_ntri &name, int2 widget, size_t count, uint2 persistence = PropertyPersistence::ReadWrite)
{
	return ReflectionProperty(name, GetBuiltinType<Type>(), count, widget, persistence);
}

/// Constructs a property description from the given parameters.
template <class Type, size_t Count>
LEAN_INLINE ReflectionProperty MakeReflectionProperty(const utf8_ntri &name, int2 widget, uint2 persistence = PropertyPersistence::ReadWrite)
{
	return ReflectionProperty(name, GetBuiltinType<Type>(), Count, widget, persistence);
}

namespace Impl
{
	template <class Type>
	struct DeducePropertyElements
	{
		typedef Type type;
		static const size_t count = 1;
	};
	template <class Type, size_t Count>
	struct DeducePropertyElements<Type[Count]>
	{
		typedef Type type;
		static const size_t count = Count;
	};
}

/// Constructs a property description from the given parameters.
template <class Type>
LEAN_INLINE ReflectionProperty MakeReflectionProperty(const utf8_ntri &name, int2 widget, uint2 persistence = PropertyPersistence::ReadWrite)
{
	return ReflectionProperty(name,
			GetBuiltinType< typename Impl::DeducePropertyElements<Type>::type >(),
			typename Impl::DeducePropertyElements<Type>::count, widget,
			persistence
		);
}

/// Gets the reflection property range.
template <size_t Count>
LEAN_INLINE lean::range<const ReflectionProperty*> ToPropertyRange(const ReflectionProperty (&properties)[Count])
{
	return lean::make_range(&properties[0], &properties[Count]);
}

/// Property collection.
typedef lean::property_collection<beCore::PropertyProvider, ReflectionProperty> ReflectionProperties;
/// Property range.
typedef lean::range<const ReflectionProperty*> PropertyRange;
/// Gets the reflection property range.
LEAN_INLINE PropertyRange ToPropertyRange(const ReflectionProperties &properties)
{
	return lean::make_range(properties.data(), properties.data_end());
}

/// Gets the ID of the given property.
BE_CORE_API uint4 GetPropertyID(uint4 baseOffset, PropertyRange range, const utf8_ntri &name);
/// Gets the name of the given property.
BE_CORE_API utf8_ntr GetPropertyName(uint4 baseOffset, PropertyRange range, uint4 id);
/// Gets the type of the given property.
BE_CORE_API PropertyDesc GetPropertyDesc(uint4 baseOffset, PropertyRange range, uint4 id);

/// Sets the given (raw) values.
BE_CORE_API bool SetProperty(uint4 baseOffset, PropertyRange range, PropertyProvider &provider, uint4 id, const std::type_info &type, const void *values, size_t count);
/// Gets the given number of (raw) values.
BE_CORE_API bool GetProperty(uint4 baseOffset, PropertyRange range, const PropertyProvider &provider, uint4 id, const std::type_info &type, void *values, size_t count);

/// Visits a property for modification.
BE_CORE_API bool WriteProperty(uint4 baseOffset, PropertyRange range, PropertyProvider &provider, uint4 id, PropertyVisitor &visitor, uint4 flagse);
/// Visits a property for reading.
BE_CORE_API bool ReadProperty(uint4 baseOffset, PropertyRange range, const PropertyProvider &provider, uint4 id, PropertyVisitor &visitor, uint4 flags);

} // namespace

/// @addtogroup GlobalMacros
/// @{

/// Constructs a property getter that provides access to the given number of the given values.
#define BE_CORE_PROPERTY_CONSTANT(constants, count) ::lean::properties::make_property_constant<::beCore::PropertyProvider>(constants, count)

/// Constructs a property setter that provides access to object values using the given setter method.
#define BE_CORE_PROPERTY_SETTER(setter) ::lean::properties::deduce_accessor_binder(setter) \
	.set_base<::beCore::PropertyProvider>().bind_setter<setter>()
/// Constructs a property setter that provides access to object values using the given getter method.
#define BE_CORE_PROPERTY_GETTER(getter) ::lean::properties::deduce_accessor_binder(getter) \
	.set_base<::beCore::PropertyProvider>().bind_getter<getter>()

/// Constructs a property setter that provides access to object values using the given setter method, splitting or merging values of the given type to values of the setter parameter type.
#define BE_CORE_PROPERTY_SETTER_UNION(setter, value_type) ::lean::properties::deduce_accessor_binder(setter) \
	.set_base<::beCore::PropertyProvider>().set_value<value_type>().bind_setter<setter>()
/// Constructs a property getter that provides access to object values using the given getter method, splitting or merging values of the given type to values of the getter parameter (return) type.
#define BE_CORE_PROPERTY_GETTER_UNION(getter, value_type) ::lean::properties::deduce_accessor_binder(getter) \
	.set_base<::beCore::PropertyProvider>().set_value<value_type>().bind_getter<getter>()

/// Associates the given properties with the given type.
#define BE_CORE_ASSOCIATE_PROPERTIES_X(type, dynamic_method, static_method, properties) \
	lean::range<const beCore::ReflectionProperty*> type::static_method() { return ToPropertyRange(properties); } \
	lean::range<const beCore::ReflectionProperty*> type::dynamic_method() const { return ToPropertyRange(properties); }
/// Associates the given properties with the given type.
#define BE_CORE_ASSOCIATE_PROPERTIES(type, properties) \
	BE_CORE_ASSOCIATE_PROPERTIES_X(type, GetReflectionProperties, GetOwnProperties, properties)
	

/// @}

#endif