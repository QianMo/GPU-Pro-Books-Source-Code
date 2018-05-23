/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_HANDLE_GUARD
#define LEAN_SMART_HANDLE_GUARD

#include "../lean.h"
#include "../tags/noncopyable.h"

namespace lean
{
namespace smart
{

/// Default handle guard policy.
template <class Handle>
struct close_handle_policy
{
	/// Returns an invalid handle value.
	static LEAN_INLINE Handle invalid()
	{
		return INVALID_HANDLE_VALUE;
	}
	/// Releases the given handle by calling @code CloseHandle()@endcode.
	static LEAN_INLINE void release(Handle handle)
	{
		if (handle != NULL && handle != INVALID_HANDLE_VALUE)
			::CloseHandle(handle);
	}
};

/// Window handle guard policy.
template <class Handle>
struct destroy_window_policy
{
	/// Returns an invalid handle value.
	static LEAN_INLINE Handle invalid()
	{
		return NULL;
	}
	/// Releases the given handle by calling @code DestroyWindow()@endcode.
	static LEAN_INLINE void release(Handle handle)
	{
		if (handle != NULL)
			::DestroyWindow(handle);
	}
};

/// Dll handle guard policy.
template <class Handle>
struct free_library_policy
{
	/// Returns an invalid handle value.
	static LEAN_INLINE Handle invalid()
	{
		return NULL;
	}
	/// Releases the given handle by calling @code FreeLibrary()@endcode.
	static LEAN_INLINE void release(Handle handle)
	{
		if (handle != NULL)
			::FreeLibrary(handle);
	}
};

/// Handle guard that releases the stored handle on destruction.
template < class Handle, class ReleasePolicy = close_handle_policy<Handle> >
class handle_guard : public noncopyable
{
public:
	/// Type of the handle stored by this guard.
	typedef Handle value_type;

private:
	value_type m_handle;

public:
	/// Releases the given handle on destruction.
	LEAN_INLINE explicit handle_guard(value_type handle = ReleasePolicy::invalid())
		: m_handle(handle) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Releases the given handle on destruction.
	handle_guard(handle_guard<value_type, ReleasePolicy> &&right) noexcept
		: m_handle( right.detach() ) { }
#endif
	/// Releases the stored handle.
	LEAN_INLINE ~handle_guard()
	{
		ReleasePolicy::release(m_handle);
	}

	/// Detaches the stored handle.
	LEAN_INLINE value_type detach()
	{
		value_type handle = m_handle;
		m_handle = ReleasePolicy::invalid();
		return handle;
	}

	/// Replaces the stored handle with the given handle. <b>[ESA]</b>
	handle_guard& operator =(value_type handle)
	{
		// Self-assignment would be wrong
		if (handle != m_handle)
		{
			value_type prevHandle = m_handle;
			m_handle = handle;
			ReleasePolicy::release(prevHandle);
		}
		
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Replaces the stored handle with the one stored by the given r-value guard. <b>[ESA]</b>
	handle_guard& operator =(handle_guard<value_type, ReleasePolicy> &&right) noexcept
	{
		// Self-assignment would be wrong
		if (addressof(right) != this)
		{
			value_type prevHandle = m_handle;
			m_handle = right.detach();
			ReleasePolicy::release(prevHandle);
		}

		return *this;
	}
#endif

	/// Retrieves the stored handle.
	LEAN_INLINE const value_type& get() const { return m_handle; }
	/// Retrieves the stored handle.
	LEAN_INLINE operator value_type() const { return get(); }

	/// Gets a pointer allowing for COM-style handle retrieval. The pointer returned may
	/// only ever be used until the next call to one of this handle's methods.
	LEAN_INLINE value_type* rebind()
	{
		*this = ReleasePolicy::invalid();
		return &m_handle;
	}
};

} // namespace

using smart::handle_guard;

} // namespace

#endif