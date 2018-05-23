/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_SHAREABLE_LOCK_POLICIES
#define LEAN_CONCURRENT_SHAREABLE_LOCK_POLICIES

#include "../lean.h"

namespace lean
{
namespace concurrent
{

/// Shared locking policy.
template <class Lockable>
struct shared_lock_policy
{
	/// Calls try_lock_shared on the given lock object.
	static LEAN_INLINE bool try_lock(Lockable &lock)
	{
		return lock.try_lock_shared();
	}
	/// Calls lock_shared on the given lock object.
	static LEAN_INLINE void lock(Lockable &lock)
	{
		lock.lock_shared();
	}
	/// Calls unlock_shared on the given lock object.
	static LEAN_INLINE void unlock(Lockable &lock)
	{
		lock.unlock_shared();
	}
};

/// Shared lock upgrade locking policy.
template <class Lockable>
struct upgrade_lock_policy
{
	/// Calls try_upgrade_lock on the given lock object.
	static LEAN_INLINE bool try_lock(Lockable &lock)
	{
		return lock.try_upgrade_lock();
	}
	/// Calls upgrade_lock on the given lock object.
	static LEAN_INLINE void lock(Lockable &lock)
	{
		lock.upgrade_lock();
	}
	/// Calls downgrade_lock on the given lock object.
	static LEAN_INLINE void unlock(Lockable &lock)
	{
		lock.downgrade_lock();
	}
};

} // namespace

using concurrent::shared_lock_policy;
using concurrent::upgrade_lock_policy;

} // namespace

#endif