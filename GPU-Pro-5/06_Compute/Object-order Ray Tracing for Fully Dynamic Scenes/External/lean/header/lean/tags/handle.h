/*****************************************************/
/* lean Tags                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TAGS_HANDLE
#define LEAN_TAGS_HANDLE

#include "../lean.h"

namespace lean
{
namespace tags
{

/// Opaque handle class that allows to hide internal values while passing them around.
template <class Type, class Owner>
class handle
{
	friend Owner;

public:
	/// Type of the value stored.
	typedef Type value_type;
	/// Type of the owner.
	typedef Owner owner_type;

protected:
	/// Value stored.
	value_type value;

	/// Constructs a handle from the given value.
	LEAN_INLINE explicit handle(const value_type &value)
		: value(value) { }
	
	/// Gets the value stored by this handle.
	LEAN_INLINE const value_type& get() const { return this->value; }
	/// Gets the value stored by this handle.
	LEAN_INLINE operator value_type() const { return this->value; }
};

} // namespace

using tags::handle;

} // namespace

#endif