/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_THREAD
#define LEAN_CONCURRENT_THREAD

#include "../lean.h"
#include "../tags/noncopyable.h"
#include <process.h>
#include <windows.h>
#include <lean/logging/win_errors.h>

namespace lean
{
namespace concurrent
{

/// Manages a simple thread.
class thread : public noncopyable
{
private:
	HANDLE m_handle;

	/// Calls the given callable object.
	template <class Callable>
	static unsigned int _stdcall run_thread(void *args)
	{
		unsigned int result = 0;

		Callable *callable = static_cast<Callable*>(args);
		LEAN_ASSERT_NOT_NULL(callable);

		try
		{
			try
			{
				(*callable)();
			}
			catch (const std::exception &exc)
			{
				LEAN_LOG_ERROR_CTX("Unhandled exception during execution of thread", exc.what());
				result = static_cast<unsigned int>(-1);
			}
			catch (...)
			{
				LEAN_LOG_ERROR_MSG("Unhandled exception during execution of thread");
				result = static_cast<unsigned int>(-1);
			}

			delete callable;
		}
		catch (...)
		{
			result = static_cast<unsigned int>(-2);
		}

		return result;
	}

	/// Creates a new thread to run the given callable object.
	template <class Callable>
	static HANDLE run_thread(Callable *callable)
	{
		LEAN_ASSERT_NOT_NULL(callable);

		HANDLE handle = reinterpret_cast<HANDLE>(
				::_beginthreadex(nullptr, 0, &run_thread<Callable>, callable, 0, nullptr)
			);

		if (handle == NULL)
		{
			delete callable;
			LEAN_THROW_WIN_ERROR_MSG("_beginthreadex() failed");
		}

		return handle;
	}

public:
	/// Default constructor.
	thread()
		: m_handle(NULL) { }
	/// Constructs a thread. Throws a runtime_error on failure.
	template <class Callable>
	thread(const Callable &callable)
		: m_handle( run_thread( new Callable(callable) ) ) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a thread. Throws a runtime_error on failure.
	template <class Callable>
	thread(Callable &&callable)
		: m_handle( run_thread( new Callable(std::move(callable)) ) ) { }
	/// Moves the thread managed by the given thread object to this thread object.
	thread(thread &&right) noexcept
		: m_handle(right.m_handle)
	{
		right.m_handle = NULL;
	}
#endif
	/// Destructor.
	~thread()
	{
		if (m_handle != NULL)
			::CloseHandle(m_handle);
	}

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Moves the thread managed by the given thread object to this thread object.
	thread& operator =(thread &&right) noexcept
	{
		if (m_handle != right.m_handle)
		{
			detach();
			m_handle = right.m_handle;
			right.m_handle = NULL;
		}
		return *this;
	}
#endif

	/// Detaches the managed thread from this thread object.
	void detach()
	{
		if (m_handle != NULL)
		{
			::CloseHandle(m_handle);
			m_handle = NULL;
		}
	}

	/// Checks if this thread is valid.
	LEAN_INLINE bool joinable() const
	{
		return (m_handle != NULL);
	}
	/// Waits for the managed thread to exit.
	void join()
	{
		if (m_handle != NULL)
			if (::WaitForSingleObject(m_handle, INFINITE) == WAIT_FAILED)
				LEAN_THROW_WIN_ERROR_MSG("WaitForSingleObject()");
	}

	/// Gets the native handle.
	LEAN_INLINE HANDLE native_handle() const
	{
		return m_handle;
	}
};

} // namespace

using concurrent::thread;

} // namespace

#endif