/*****************************************************/
/* lean Properties              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PROPERTIES_PROPERTY_TYPES
#define LEAN_PROPERTIES_PROPERTY_TYPES

#include "../lean.h"
#include "property_type.h"
#include "../type_info.h"
#include "../memory/default_heap.h"

namespace lean
{
namespace properties
{

/// Generic property type implementation.
template <class Type, class Heap = default_heap, size_t Alignment = alignof(Type)>
struct generic_property_type : public property_type
{
	/// Value type.
	typedef Type value_type;
	/// Heap type.
	typedef default_heap heap_type;
	/// Alignment
	static const size_t alignment = Alignment;

	/// Gets the size required by the given number of elements.
	size_t size(size_t count) const
	{
		return sizeof(value_type) * count;
	}
	/// Gets the STD lib element typeid.
	const struct type_info& type_info() const
	{
		return get_type_info<value_type>();
	}

	/// Allocates the given number of elements.
	void* allocate(size_t count) const
	{
		return Heap::allocate<Alignment>(size(count));
	}
	/// Constructs the given number of elements.
	void construct(void *elements, size_t count) const
	{
		new (elements) value_type[count];
	}
	/// Destructs the given number of elements.
	void destruct(void *elements, size_t count) const
	{
		for (size_t i = 0; i < count; ++i)
			static_cast<value_type*>(elements)[i].~value_type();
	}
	/// Deallocates the given number of elements.
	void deallocate(void *elements, size_t count) const
	{
		Heap::free<Alignment>(elements);
	}
};

/// Gets the property type info for the given type.
template <class Type>
LEAN_INLINE const property_type& get_property_type()
{
	static generic_property_type<Type> type;
	return type;
}

/// Gets a type info object for the given type.
template <class Type>
inline const property_type_info& get_property_type_info()
{
	static property_type_info info(get_type_info<Type>(), get_property_type<Type>());
	return info;
}

} // namespace

using properties::generic_property_type;
using properties::get_property_type;
using properties::get_property_type_info;

} // namespace

#endif