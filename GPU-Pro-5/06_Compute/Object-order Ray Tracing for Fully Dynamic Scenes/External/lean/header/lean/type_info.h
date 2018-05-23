/*****************************************************/
/* lean built-in types          (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TYPE_INFO
#define LEAN_TYPE_INFO

#include "lean.h"
#include <typeinfo>
#include "meta/strip.h"

namespace lean
{

namespace types
{

/// Enhanced type info.
struct type_info
{
	/// Type info.
	const std::type_info &type;
	/// Size of the type.
	const size_t size;

	/// Constructor.
	type_info(const std::type_info &type, size_t size)
		: type(type),
		size(size) { }
};

namespace impl
{
	template <class Type>
	struct robust_sizeof { static const size_t value = sizeof(Type); };
	template <>
	struct robust_sizeof<void> { static const size_t value = 0; };
}

/// Gets a type info object for the given type.
template <class Type>
inline const type_info& get_type_info()
{
	static type_info info(
			typeid(Type),
			impl::robust_sizeof<typename lean::strip_modref<Type>::type>::value
		);
	return info;
}


/// Checks if the given type info matches the given type.
template <class Type>
LEAN_INLINE bool is_type(const std::type_info &type)
{
	return typeid(Type) == type;
}


/// Gets a pointer to a value of the given type, if the given types match.
template <class Value>
LEAN_INLINE Value* to_type(const std::type_info &type, void *value)
{
	return is_type<Value>(type)
		? static_cast<Value*>(value)
		: nullptr;
}

/// Gets a pointer to a value of the given type, if the given types match.
template <class Value>
LEAN_INLINE const Value* to_type(const std::type_info &type, const void *value)
{
	return to_type<Value>(type, const_cast<void*>(value));
}

/// Gets a pointer to a value of the given type, if the given types match.
template <class Value>
LEAN_INLINE volatile Value* to_type(const std::type_info &type, volatile void *value)
{
	return to_type<Value>(type, const_cast<void*>(value));
}

/// Gets a pointer to a value of the given type, if the given types match.
template <class Value>
LEAN_INLINE const volatile Value* to_type(const std::type_info &type, const volatile void *value)
{
	return to_type<Value>(type, const_cast<void*>(value));
}


/// Gets a pointer to the offset-th element in the given array.
LEAN_INLINE void* get_element(size_t stride, void *value, size_t offset)
{
	return reinterpret_cast<char*>(value) + offset * stride;
}

/// Gets a pointer to the offset-th element in the given array.
LEAN_INLINE const void* get_element(size_t stride, const void *value, size_t offset)
{
	return get_element(stride, const_cast<void*>(value), offset);
}

/// Gets a pointer to the offset-th element in the given array.
LEAN_INLINE volatile void* get_element(size_t stride, volatile void *value, size_t offset)
{
	return get_element(stride, const_cast<void*>(value), offset);
}

/// Gets a pointer to the offset-th element in the given array.
LEAN_INLINE const volatile void* get_element(size_t stride, const volatile void *value, size_t offset)
{
	return get_element(stride, const_cast<void*>(value), offset);
}


} // namespace

using types::type_info;
using types::get_type_info;

using types::is_type;
using types::to_type;
using types::get_element;

} // namespace

#endif