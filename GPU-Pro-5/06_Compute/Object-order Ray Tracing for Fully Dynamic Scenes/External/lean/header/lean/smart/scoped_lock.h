/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_SCOPED_LOCK
#define LEAN_SMART_SCOPED_LOCK

#include "../lean.h"
#include "../tags/noncopyable.h"

namespace lean
{
namespace smart
{

/// Default locking policy.
template <class Lockable>
struct default_lock_policy
{
	/// Calls try_lock on the given lock object.
	static LEAN_INLINE bool try_lock(Lockable &lock)
	{
		return lock.try_lock();
	}
	/// Calls lock on the given lock object.
	static LEAN_INLINE void lock(Lockable &lock)
	{
		lock.lock();
	}
	/// Calls unlock on the given lock object.
	static LEAN_INLINE void unlock(Lockable &lock)
	{
		lock.unlock();
	}
};

/// Allows locked lockable objects to be adopted by a scoped lock object on construction.
enum adopt_lock_t
{
	adopt_lock	///< Adopts a locked lockable object on scoped lock construction
};

/// Automatic lock management class that locks a given object on construction to be unlocked on destruction.
template < class Lockable, class Policy = default_lock_policy<Lockable> >
class scoped_lock : public noncopyable
{
private:
	Lockable &m_lock;

public:
	/// Type of the lock managed by this class.
	typedef Lockable lock_type;
	/// Type of the locking policy used by this class.
	typedef Policy policy_type;

	/// Locks the given object, to be unlocked on destruction.
	LEAN_INLINE explicit scoped_lock(lock_type &lock)
		: m_lock(lock)
	{
		policy_type::lock(m_lock);
	}
	/// Assumes that the given object is already locked, obtaining lock ownership and unlocking the lockable object on destruction.
	LEAN_INLINE scoped_lock(lock_type &lock, adopt_lock_t)
		: m_lock(lock) { }
	/// Unlocks the lock object managed by this class.
	LEAN_INLINE ~scoped_lock()
	{
		policy_type::unlock(m_lock);
	}
	
	/// Gets the lock object managed by this class.
	LEAN_INLINE lock_type& get(void) { return m_lock; };
	/// Gets the lock object managed by this class.
	LEAN_INLINE const lock_type& get(void) const { return m_lock; };
};

} // namespace

using smart::default_lock_policy;
using smart::scoped_lock;

} // namespace

#endif