/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_CRITICAL_SECTION
#define LEAN_CONCURRENT_CRITICAL_SECTION

#include "../lean.h"
#include "../tags/noncopyable.h"
#include <windows.h>

// Include automatically to encourage use of scoped_lock
#include "../smart/scoped_lock.h"

namespace lean
{
namespace concurrent
{

/// Implements a light-weight reentrant binary lock.
class critical_section : public noncopyable
{
private:
	CRITICAL_SECTION m_criticalSection;

public:
	/// Constructs a critical section. Throws a runtime_error on failure.
	critical_section(unsigned long spinCount = 4096)
	{
		if (!::InitializeCriticalSectionAndSpinCount(&m_criticalSection, spinCount))
			LEAN_ASSERT_UNREACHABLE();
	}
	/// Destructor.
	~critical_section()
	{
		::DeleteCriticalSection(&m_criticalSection);
	}

	/// Tries to lock this critical section, returning false if currenty locked by another user.
	LEAN_INLINE bool try_lock()
	{
		return (::TryEnterCriticalSection(&m_criticalSection) != FALSE);
	}

	/// Locks this critical section, returning immediately on success, otherwise waiting for the section to become unlocked.
	LEAN_INLINE void lock()
	{
		::EnterCriticalSection(&m_criticalSection);
	}

	/// Unlocks this critical section, permitting waiting threads to continue execution.
	LEAN_INLINE void unlock()
	{
		::LeaveCriticalSection(&m_criticalSection);
	}
};

/// Scoped critical section lock.
typedef smart::scoped_lock<critical_section> scoped_cs_lock;

} // namespace

using concurrent::critical_section;

using concurrent::scoped_cs_lock;

} // namespace

#endif