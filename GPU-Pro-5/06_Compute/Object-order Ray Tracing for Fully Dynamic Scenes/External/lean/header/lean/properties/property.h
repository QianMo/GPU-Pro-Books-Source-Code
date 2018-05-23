/*****************************************************/
/* lean Properties              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PROPERTIES_PROPERTY
#define LEAN_PROPERTIES_PROPERTY

#include "../lean.h"
#include "property_type.h"
#include "../smart/cloneable.h"
#include "../smart/cloneable_obj.h"
#include "../tags/noncopyable.h"
#include <string>
#include "../strings/types.h"
#include "../type_info.h"

namespace lean
{
namespace properties
{

/// Passes data to a specific destination.
template <class Class>
class LEAN_INTERFACE property_setter : public cloneable
{
	LEAN_INTERFACE_BEHAVIOR(property_setter)

public:
	/// Type of the object whose properties are exposed.
	typedef Class object_type;

	/// Passes the given values of the given type to the given object.
	virtual bool operator ()(object_type &object, const std::type_info &type, const void *values, size_t count) = 0;

	/// Passes the given values to the given object.
	template <class Value>
	LEAN_INLINE bool operator ()(object_type &object, const Value *values, size_t count)
	{
		return (*this)(object, typeid(Value), values, count);
	}
};

/// Fetches data from a specific source.
template <class Class>
class LEAN_INTERFACE property_getter : public cloneable
{
	LEAN_INTERFACE_BEHAVIOR(property_getter)

public:
	/// Type of the object whose properties are exposed.
	typedef Class object_type;

	/// Fetches the given number of values of the given type from the given object.
	virtual bool operator ()(const object_type &object, const std::type_info &type, void *values, size_t count) const = 0;

	/// Fetches the given number of values from the given object.
	template <class Value>
	LEAN_INLINE bool operator ()(const object_type &object, Value *values, size_t count) const
	{
		return (*this)(object, typeid(Value), values, count);
	}
};

/// Destribes a property.
template <class Class, class Derived = void, class TypeInfo = property_type_info>
struct property_desc
{
	/// Type of the most derived structure.
	typedef typename first_non_void<Derived, property_desc>::type actual_type;

	/// Property type
	const TypeInfo *type_info;	///< Property type.
	size_t count;				///< Number of elements.

	/// Setter type.
	typedef property_setter<Class> setter_type;
	typedef cloneable_obj<setter_type, true> setter_storage_type;
	/// Getter type.
	typedef property_getter<Class> getter_type;
	typedef cloneable_obj<getter_type, true> getter_storage_type;

	setter_storage_type setter;	///< Value setter.
	getter_storage_type getter;	///< Value getter.

	/// Constructs an empty property description.
	property_desc()
		: type_info(nullptr),
		count(0),
		setter(setter_storage_type::null()),
		getter(getter_storage_type::null()) { }
	/// Constructs a property description from the given parameters.
	property_desc(const TypeInfo &type, size_t count)
		: type_info(&type),
		count(count),
		setter(setter_storage_type::null()),
		getter(getter_storage_type::null()) { }

	/// Sets the setter.
	actual_type& set_setter(const setter_type &setter) { this->setter = setter; return static_cast<actual_type&>(*this); }
	/// Sets the getter.
	actual_type& set_getter(const getter_type &getter) { this->getter = getter; return static_cast<actual_type&>(*this); }
};

/// Describes a named property.
template <class Class, class Derived = void, class TypeInfo = property_type_info>
struct named_property_desc
	: public property_desc<
		Class,
		typename first_non_void< Derived, named_property_desc<Class, Derived, TypeInfo> >::type,
		TypeInfo
	>
{
	/// Type of the most derived structure.
	typedef typename named_property_desc::actual_type actual_type;

	utf8_string name;	///< Property name.

	/// Constructs an empty property description.
	named_property_desc() { }
	/// Constructs a property description from the given parameters.
	named_property_desc(const utf8_ntri &name, const TypeInfo &type, size_t count)
		: typename named_property_desc::property_desc(type, count),
		name(name.to<utf8_string>()) { }
};

/// Describes a UI property.
template <class Class, class Widget, class Derived = void, class TypeInfo = property_type_info>
struct ui_property_desc
	: public named_property_desc<
		Class,
		typename first_non_void< Derived, ui_property_desc<Class, Widget, Derived, TypeInfo> >::type,
		TypeInfo
	>
{
	/// Type of the most derived structure.
	typedef typename ui_property_desc::actual_type actual_type;

	Widget widget;	///< UI widget used to display/edit this property.

	/// Value storage type.
	typedef property_getter<Class> value_type;
	typedef cloneable_obj<value_type, true> value_storage_type;

	value_storage_type default_value;	///< Default value getter.

	value_storage_type min_value;		///< Min value getter.
	value_storage_type value_step;		///< Value step getter.
	value_storage_type max_value;		///< Max value getter.

	/// Constructs an empty property description.
	ui_property_desc()
		: default_value(value_storage_type::null()),
		min_value(value_storage_type::null()),
		value_step(value_storage_type::null()),
		max_value(value_storage_type::null()) { }
	/// Constructs a property description from the given parameters.
	ui_property_desc(const utf8_ntri &name, const TypeInfo &type, size_t count, const Widget &widget)
		: typename ui_property_desc::named_property_desc(name, type, count),
		widget(widget),
		default_value(value_storage_type::null()),
		min_value(value_storage_type::null()),
		value_step(value_storage_type::null()),
		max_value(value_storage_type::null()) { }

	/// Sets the default value getter.
	actual_type& set_default_value(const value_type &getter) { this->default_value = getter; return static_cast<actual_type&>(*this); }
	/// Sets the min value getter.
	actual_type& set_min_value(const value_type &getter) { this->min_value = getter; return static_cast<actual_type&>(*this); }
	/// Sets the value step getter.
	actual_type& set_value_step(const value_type &getter) { this->value_step = getter; return static_cast<actual_type&>(*this); }
	/// Sets the max value getter.
	actual_type& set_max_value(const value_type &getter) { this->max_value = getter; return static_cast<actual_type&>(*this); }
};

/// Passes the given values to the given object using the given setter.
template <class Class, class Value>
LEAN_INLINE bool set_property(Class &object, property_setter<Class> *setter, const Value *values, size_t count)
{
	return (setter) ? (*setter)(object, values, count) : false;
}
/// Passes the given values to the given object using the given setter.
template <class Class, class Value>
LEAN_INLINE bool set_property(Class &object, cloneable_obj<property_setter<Class>, false> &setter, const Value *values, size_t count)
{
	return (*setter)(object, values, count);
}
/// Passes the given values to the given object using the given setter.
template <class Class, class Value>
LEAN_INLINE bool set_property(Class &object, const cloneable_obj<property_setter<Class>, true> &setter, const Value *values, size_t count)
{
	return set_property(object, setter.getptr(), values, count);
}

/// Fetches the given number of values from the given object.
template <class Class, class Value>
LEAN_INLINE bool get_property(const Class &object, const property_getter<Class> *getter, Value *values, size_t count)
{
	return (getter) ? (*getter)(object, values, count) : false;
}
/// Fetches the given number of values from the given object.
template <class Class, class Value, bool PointerSem>
LEAN_INLINE bool get_property(const Class &object, const cloneable_obj<property_getter<Class>, PointerSem> &getter, Value *values, size_t count)
{
	return (PointerSem)
		? get_property(object, getter.getptr(), values, count)
		: (*getter)(object, values, count);
}

/// Invalid property ID.
static const size_t invalid_property_id = static_cast<size_t>(-1);

/// Finds a property by name, returning its ID on success, invalid_property_id on failure.
template <class ID, class Collection, class String>
inline ID find_property(const Collection &collection, const String &name, ID invalidID = static_cast<ID>(-1), ID baseOffset = 0)
{
	for (typename Collection::const_iterator itProperty = collection.begin();
		itProperty != collection.end(); ++itProperty)
		if (itProperty->name == name)
			return (ID) (itProperty - collection.begin()) + baseOffset;

	return invalidID;
}

} // namespace

using properties::property_setter;
using properties::property_getter;
using properties::property_type;

using properties::property_desc;
using properties::named_property_desc;
using properties::ui_property_desc;

using properties::get_property;
using properties::set_property;

using properties::invalid_property_id;
using properties::find_property;

using properties::scoped_property_data;
using properties::delete_property_data_policy;
using properties::destruct_property_data_policy;
using properties::deallocate_property_data_policy;

} // namespace

#endif