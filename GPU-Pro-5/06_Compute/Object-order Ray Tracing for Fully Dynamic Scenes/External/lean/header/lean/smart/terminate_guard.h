/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_TERMINATE_GUARD
#define LEAN_SMART_TERMINATE_GUARD

#include "../lean.h"
#include "../tags/noncopyable.h"
#include <exception>

namespace lean
{
namespace smart
{

/// Terminates the application on destruction, if not disarmed.
class terminate_guard : public nonassignable
{
private:
	mutable bool m_armed;

public:
	/// Constructs an armed terminate guard.
	LEAN_INLINE terminate_guard()
		: m_armed(true) { }
	/// Constructs a terminate guard (optionally disarmed).
	LEAN_INLINE explicit terminate_guard(bool arm)
		: m_armed(arm) { }
	/// Copies AND DISARMS the given terminate guard.
	LEAN_INLINE terminate_guard(const terminate_guard &right)
		: m_armed(right.m_armed)
	{
		right.disarm();
	}
	/// Terminates the application, if not disarmed.
	LEAN_INLINE ~terminate_guard()
	{
		if (armed())
		{
			LEAN_ASSERT_DEBUG(false);
			std::terminate();
		}
	}

	/// Sets whether the scope guard is currently armed.
	LEAN_INLINE void armed(bool arm) const { m_armed = arm; }
	/// Gets whether the scope guard is currently armed.
	LEAN_INLINE bool armed() const { return m_armed; }

	/// Disarms this scope guard.
	LEAN_INLINE void disarm() const { armed(false); }
	/// Re-arms this scope guard.
	LEAN_INLINE void arm() const { armed(true); }
};

} // namespace

using smart::terminate_guard;

} // namespace

#endif