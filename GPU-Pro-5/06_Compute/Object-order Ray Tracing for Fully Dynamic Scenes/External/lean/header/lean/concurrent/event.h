/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_EVENT
#define LEAN_CONCURRENT_EVENT

#include "../lean.h"
#include "../tags/noncopyable.h"
#include <windows.h>

namespace lean
{
namespace concurrent
{

/// Implements an event.
class event : public noncopyable
{
private:
	HANDLE m_event;

public:
	/// Constructs a critical section. Throws a runtime_error on failure.
	event(bool signaled = false)
		: m_event( ::CreateEventW(NULL, true, signaled, NULL) )
	{
		LEAN_ASSERT(m_event != NULL);
	}
	/// Destructor.
	~event()
	{
		::CloseHandle(m_event);
	}

	/// Waits for the next event notification.
	LEAN_INLINE void wait()
	{
		DWORD result = ::WaitForSingleObject(m_event, INFINITE);
		LEAN_ASSERT(result == WAIT_OBJECT_0);
	}

	/// Sets the event state to signaled.
	LEAN_INLINE void set()
	{
		BOOL result = ::SetEvent(m_event);
		LEAN_ASSERT(result != FALSE);
	}
	/// Resets the event state to non-signaled.
	LEAN_INLINE void reset()
	{
		BOOL result = ::ResetEvent(m_event);
		LEAN_ASSERT(result != FALSE);
	}

	/// Sets the event state.
	LEAN_INLINE void signaled(bool signaled)
	{
		if (signaled)
			set();
		else
			reset();
	}

	/// Gets the native handle.
	LEAN_INLINE HANDLE native_handle() const
	{
		return m_event;
	}
};

} // namespace

using concurrent::event;

} // namespace

#endif