/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_SHAREABLE_SPINLOCK
#define LEAN_CONCURRENT_SHAREABLE_SPINLOCK

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "atomic.h"

// Include automatically to encourage use of scoped_lock
#include "shareable_lock_policies.h"
#include "../smart/scoped_lock.h"

namespace lean
{
namespace concurrent
{

/// Implements a shareable spin lock that is NOT reentrant.
template <class Counter = long>
class shareable_spin_lock : public noncopyable
{
private:
	Counter m_counter;
	Counter m_exclCounter;

public:
	/// Constructs a shareable spin lock.
	shareable_spin_lock()
		: m_counter(0),
		m_exclCounter(0) {  }

	/// Tries to exclusively lock this spin lock, returning false if currently locked / shared by another user.
	LEAN_INLINE bool try_lock()
	{
		return atomic_test_and_set(m_counter, static_cast<Counter>(0), static_cast<Counter>(-1));
	}

	/// Tries to atomically upgrade shared ownership of this spin lock to exclusive ownership,
	/// returning false if currently shared with another user.
	LEAN_INLINE bool try_upgrade_lock()
	{
		return atomic_test_and_set(m_counter, static_cast<Counter>(1), static_cast<Counter>(-1));
	}

	/// Exclusively locks this spin lock, returning immediately on success, otherwise waiting for the lock to become available.
	LEAN_INLINE void lock()
	{
		atomic_increment(m_exclCounter);
		while (!try_lock());
		atomic_decrement(m_exclCounter);
	}

	/// Atomically upgrades shared ownership of this spin lock to exclusive ownership,
	/// returning immediately on success, otherwise waiting for the lock to become available.
	LEAN_INLINE void upgrade_lock()
	{
		atomic_increment(m_exclCounter);
		
		// Unlock required, otherwise multiple upgrade
		// calls at the same time will lead to deadlocks
		unlock_shared();
		while (!try_lock());
		
		atomic_decrement(m_exclCounter);
	}

	/// Atomically releases exclusive ownership and re-acquires shared ownership, permitting waiting threads to continue execution.
	LEAN_INLINE void downgrade_lock()
	{
		atomic_test_and_set(m_counter, static_cast<Counter>(-1), static_cast<Counter>(1));
	}

	/// Unlocks this spin lock, permitting waiting threads to continue execution.
	LEAN_INLINE void unlock()
	{
		atomic_test_and_set(m_counter, static_cast<Counter>(-1), static_cast<Counter>(0));
	}

	/// Tries to obtain shared ownership of this spin lock, returning false if currently locked exclusively by another user.
	LEAN_INLINE bool try_lock_shared()
	{
		while (static_cast<volatile Counter&>(m_exclCounter) == static_cast<Counter>(0))
		{
			Counter counter = static_cast<volatile Counter&>(m_counter);

			if (counter == static_cast<Counter>(-1))
				return false;
			else if (atomic_test_and_set(m_counter, counter, static_cast<Counter>(counter + 1)))
				return true;
		}

		return false;
	}

	/// Obtains shared ownership of this spin lock, returning immediately on success, otherwise waiting for the lock to become available.
	LEAN_INLINE void lock_shared()
	{
		while (!try_lock_shared());
	}

	/// Releases shared ownership of this spin lock, permitting waiting threads to continue execution.
	LEAN_INLINE void unlock_shared()
	{
		Counter counter;

		do
		{
			counter = static_cast<volatile Counter&>(m_counter);
		}
		while (counter != static_cast<Counter>(-1) && counter != static_cast<Counter>(0) &&
			atomic_test_and_set(m_counter, counter, static_cast<Counter>(counter - 1)) );
	}
};

/// Scoped exclusive sharable spin lock.
typedef smart::scoped_lock< shareable_spin_lock<> > scoped_ssl_lock;
/// Scoped shared sharable spin lock.
typedef smart::scoped_lock< shareable_spin_lock<>, shared_lock_policy< shareable_spin_lock<> > > scoped_ssl_lock_shared;
/// Scoped sharable spin lock upgrade.
typedef smart::scoped_lock< shareable_spin_lock<>, upgrade_lock_policy< shareable_spin_lock<> > > scoped_ssl_upgrade_lock;

} // namespace

using concurrent::shareable_spin_lock;

using concurrent::scoped_ssl_lock;
using concurrent::scoped_ssl_lock_shared;
using concurrent::scoped_ssl_upgrade_lock;

} // namespace

#endif