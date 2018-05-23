/*****************************************************/
/* lean Tags                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TAGS_MOVE_PTR
#define LEAN_TAGS_MOVE_PTR

#include "../lean.h"
#include "../smart/common.h"

namespace lean
{

namespace smart
{
	template <class T, reference_state_t RS, class R>
	class scoped_ptr;
}

namespace tags
{

/// Pointer wrapper class that signals transferral ownership.
template <class Type, class ActualType = Type>
class move_ptr
{
	template <class OtherType, class OtherActualType>
	friend class move_ptr;

	ActualType* *const ptr;

public:
#ifndef LEAN0X_NO_NULLPTR
	/// Constructs a nullptr.
	LEAN_INLINE move_ptr(nullptr_t)
		: ptr(nullptr) { }
#else
	/// Constructs a nullptr.
	LEAN_INLINE move_ptr(int)
		: ptr(nullptr) { }
#endif
	/// Wraps the given pointer whose ownership is to be transferred.
	LEAN_INLINE explicit move_ptr(ActualType *&ptr)
		: ptr(&ptr) { }
	/// Copies the given move_ptr.
	template <class OtherType>
	LEAN_INLINE move_ptr(const move_ptr<OtherType, ActualType> &right)
		: ptr(right.ptr) { }

	/// Detaches the stored pointer.
	LEAN_INLINE Type* transfer()
	{
		if (ptr)
		{
			Type *p = *ptr;
			*ptr = nullptr;
			return p;
		}
		else
			return nullptr;
	}

	/// Peeks at the stored pointer.
	LEAN_INLINE Type* peek() const
	{
		return (ptr) ? *ptr : nullptr;
	}

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	// NOTE: This should actually be implemented as implicit r-value conversion
	// operator, however, r-value methods are not yet supported by the MSVC familiy
	
	/// Wraps the given pointer whose ownership is to be transferred.
	template <reference_state_t RS, class R>
	LEAN_INLINE move_ptr(smart::scoped_ptr<ActualType, RS, R> &&ptr)
		: ptr(ptr.move_ptr().ptr) { }
#endif
};

/// Constructs a move pointer for the given pointer.
template <class Type>
LEAN_INLINE move_ptr<Type> do_move_ptr(Type *&ptr)
{
	return move_ptr<Type>(ptr);
}

/// Constructs a move pointer for the given pointer.
template <class Type, class ActualType>
LEAN_INLINE move_ptr<Type, ActualType> do_move_ptr(ActualType *&ptr)
{
	return move_ptr<Type, ActualType>(ptr);
}

} // namespace

using tags::move_ptr;

using tags::do_move_ptr;

} // namespace

#endif