/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_SEMAPHORE
#define LEAN_CONCURRENT_SEMAPHORE

#include "../lean.h"
#include "../tags/noncopyable.h"
#include <windows.h>

namespace lean
{
namespace concurrent
{

/// Implements a semaphore.
class semaphore : public noncopyable
{
private:
	HANDLE m_semaphore;

public:
	/// Constructs a critical section. Throws a runtime_error on failure.
	explicit semaphore(long initialCount = 1)
		: m_semaphore( ::CreateSemaphoreW(NULL, initialCount, LONG_MAX, NULL) )
	{
		LEAN_ASSERT(m_semaphore != NULL);
	}
	/// Destructor.
	~semaphore()
	{
		::CloseHandle(m_semaphore);
	}

	/// Tries to acquire this semaphore, returning false if currenty unavailable.
	LEAN_INLINE bool try_lock()
	{
		return (::WaitForSingleObject(m_semaphore, 0) == WAIT_OBJECT_0);
	}

	/// Acquires this semaphore, returning immediately on success, otherwise waiting for the semaphore to become available.
	LEAN_INLINE void lock()
	{
		DWORD result = ::WaitForSingleObject(m_semaphore, INFINITE);
		LEAN_ASSERT(result == WAIT_OBJECT_0);
	}

	/// Releases this semaphore, permitting waiting threads to continue execution.
	LEAN_INLINE void unlock()
	{
		BOOL result = ::ReleaseSemaphore(m_semaphore, 1, NULL);
		LEAN_ASSERT(result != FALSE);
	}

	/// Gets the native handle.
	LEAN_INLINE HANDLE native_handle() const
	{
		return m_semaphore;
	}
};

} // namespace

using concurrent::semaphore;

} // namespace

#endif