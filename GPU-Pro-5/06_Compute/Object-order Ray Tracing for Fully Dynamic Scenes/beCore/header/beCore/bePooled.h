/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_POOLED
#define BE_CORE_POOLED

#include "beCore.h"
#include "beShared.h"
#include <lean/smart/scoped_ptr.h>
#include <lean/containers/simple_vector.h>

namespace beCore
{

/// Pooled object that keeps track of the number of users.
template <class Derived>
class Pooled
{
private:
	mutable uint4 m_users;

protected:
	/// Initializes this pooled object with no users.
	LEAN_INLINE Pooled() : m_users(0) { }
	/// Initializes this pooled object with no users.
	LEAN_INLINE Pooled(const Pooled&) : m_users(0) { }
	/// Assignment operator, does not change the number of users.
	LEAN_INLINE Pooled& operator =(const Pooled&) { return *this; }
	/// Hidden destructor.
	LEAN_INLINE ~Pooled() { }

	/// Called when all users have released their references.
	LEAN_INLINE void UsersReleased() { }

public:
	/// Marks this pooled object used.
	LEAN_INLINE void AddUser() const { ++m_users; }
	/// Marks this pooled object unused, when all users have released their references.
	LEAN_INLINE void RemoveUser() const { if (--m_users == 0) static_cast<Derived*>(const_cast<Pooled*>(this))->UsersReleased(); }
	/// Checks whether this pooled object is currently in use.
	LEAN_INLINE bool IsUsed() const { return (m_users > 0); }
};

/// Acquires a reference to the given COM object.
template <class COMType>
LEAN_INLINE void acquire_com(const Pooled<COMType> &object)
{
	object.AddUser();
}
/// Releases a reference to the given COM object.
template <class COMType>
LEAN_INLINE void release_com(const Pooled<COMType> *object)
{
	if (object)
		object->RemoveUser();
}

/// Pooled object that keeps track of the number of users.
template <class Derived>
class LEAN_INTERFACE PooledToRefCounted : public Pooled<Derived>, public RefCounted
{
	// NOTE: Neither pooled nor ref-counted need special member delegation
	LEAN_STATIC_INTERFACE_BEHAVIOR(PooledToRefCounted)

public:
	/// Same as AddUser.
	void AddRef() const LEAN_OVERRIDE { AddUser(); }
	/// Same as RemoveUser.
	void Release() const LEAN_OVERRIDE { RemoveUser(); }
};

} // namespace

#endif