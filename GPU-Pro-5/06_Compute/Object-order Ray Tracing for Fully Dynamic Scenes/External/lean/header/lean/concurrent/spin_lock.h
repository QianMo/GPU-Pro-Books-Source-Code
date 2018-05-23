/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_SPINLOCK
#define LEAN_CONCURRENT_SPINLOCK

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "atomic.h"

// Include automatically to encourage use of scoped_lock
#include "../smart/scoped_lock.h"

namespace lean
{
namespace concurrent
{

/// Implements a simple binary spin lock that is NOT reentrant.
template <class Counter = long>
class spin_lock : public noncopyable
{
private:
	Counter m_counter;

public:
	/// Constructs a binary spin lock.
	spin_lock()
		: m_counter(0) {  }

	/// Tries to lock this spin lock, returning false if currenty locked by another user.
	LEAN_INLINE bool try_lock()
	{
		return atomic_test_and_set(m_counter, static_cast<Counter>(0), static_cast<Counter>(1));
	}

	/// Locks this spin lock, returning immediately on success, otherwise waiting for the lock to become unlocked.
	LEAN_INLINE void lock()
	{
		while (!try_lock());
	}

	/// Unlocks this spin lock, permitting waiting threads to continue execution.
	LEAN_INLINE void unlock()
	{
		atomic_test_and_set(m_counter, static_cast<Counter>(1), static_cast<Counter>(0));
	}
};

/// Scoped exclusive spin lock.
typedef smart::scoped_lock< spin_lock<> > scoped_sl_lock;

} // namespace

using concurrent::spin_lock;

using concurrent::scoped_sl_lock;

} // namespace

#endif