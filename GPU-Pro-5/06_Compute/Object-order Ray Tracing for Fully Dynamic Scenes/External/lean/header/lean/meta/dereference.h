/*****************************************************/
/* lean Meta                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_META_DEREFERENCE
#define LEAN_META_DEREFERENCE

#include "../lean.h"
#include "strip.h"

namespace lean
{
namespace meta
{

namespace impl
{

template <class StrippedValue, class Value>
struct maybe_dereference_once
{
private:
	typedef typename strip_const<typename strip_reference<Value>::type>::type stripped_value_type;

public:
	static const bool dereferenced = false;
	
	typedef Value value_type;

	typedef stripped_value_type& parameter_type;
	typedef value_type& return_type;
	static LEAN_INLINE return_type dereference(parameter_type value) { return value; }

	typedef const stripped_value_type& const_parameter_type;
	typedef const value_type& const_return_type;
	static LEAN_INLINE const_return_type dereference(const_parameter_type value) { return value; }
};

template <class StrippedValue, class Value>
struct maybe_dereference_pointer
{
private:
	struct disable_type { };

public:
	static const bool dereferenced = true;

	typedef Value value_type;
	
	typedef disable_type& parameter_type;
	typedef disable_type& return_type;
	static LEAN_INLINE return_type dereference(parameter_type parameter) { return parameter; }

	typedef Value* const_parameter_type;
	typedef value_type& const_return_type;
	static LEAN_INLINE const_return_type dereference(const_parameter_type pointer) { return *pointer; }
};

template <class ModifiedVoid>
struct maybe_dereference_pointer<void, ModifiedVoid>
{
	static const bool dereferenced = false;

	typedef ModifiedVoid* value_type;
	
	typedef value_type& parameter_type;
	typedef value_type& return_type;
	static LEAN_INLINE return_type dereference(parameter_type value) { return value; }

	typedef const value_type& const_parameter_type;
	typedef const value_type& const_return_type;
	static LEAN_INLINE const_return_type dereference(const_parameter_type value) { return value; }
};

template <class Value, class Pointer>
struct maybe_dereference_once<Value*, Pointer>
	: public maybe_dereference_pointer<typename strip_modifiers<Value>::type, Value> { };

} // namespace

/// Dereferences a given value type once, if the value type is a pointer type.
template <class Type>
struct maybe_dereference_once
{
private:
	typedef  impl::maybe_dereference_once<
		typename strip_modifiers<typename strip_reference<Type>::type>::type,
		Type > internal_dereferencer;

public:
	/// True, if any dereferencing performed.
	static const bool dereferenced = internal_dereferencer::dereferenced;

	/// Value type after dereferencing.
	typedef typename internal_dereferencer::value_type value_type;

	/// Dereferences the given value parameter once, if the value is of a pointer type.
	static LEAN_INLINE typename internal_dereferencer::return_type dereference(typename internal_dereferencer::parameter_type value)
	{
		return internal_dereferencer::dereference(value);
	}
	/// Dereferences the given value parameter once, if the value is of a pointer type.
	static LEAN_INLINE typename internal_dereferencer::const_return_type dereference(typename internal_dereferencer::const_parameter_type value)
	{
		return internal_dereferencer::dereference(value);
	}
};

} // namespace

using meta::maybe_dereference_once;

} // namespace

#endif