/*****************************************************/
/* lean Time                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TIME_TIMER
#define LEAN_TIME_TIMER

#include "../lean.h"
#include <ctime>

namespace lean
{
namespace time
{

/// Default timer class.
class timer
{
private:
	time_t m_time;

public:
	/// Constructs a new timer from the current system time.
	LEAN_INLINE timer()
		: m_time(::time(nullptr)) { }

	/// Updates the reference time stored by this timer.
	LEAN_INLINE void tick() { ::time(&m_time); }

	/// Gets the time that has elapsed since the last tick.
	LEAN_INLINE double seconds() const { return ::difftime(::time(nullptr), m_time); }
	/// Gets the time that has elapsed since the last tick.
	LEAN_INLINE double milliseconds() const { return seconds() * 1000; }
};

// Old C standard definition
#ifndef CLOCKS_PER_SEC
	#define CLOCKS_PER_SEC CLK_TCK
#endif

/// Clock timer class.
class clocktimer
{
private:
	clock_t m_time;

public:
	/// Constructs a new timer from the current process time.
	LEAN_INLINE clocktimer()
		: m_time(::clock()) { }

	/// Updates the reference time stored by this timer.
	LEAN_INLINE void tick() { m_time = ::clock(); }

	/// Gets the time that has elapsed since the last tick.
	LEAN_INLINE double seconds() const { return (double) (::clock() - m_time) * (1.0 / CLOCKS_PER_SEC); }
	/// Gets the time that has elapsed since the last tick.
	LEAN_INLINE double milliseconds() const { return (double) (::clock() - m_time) * (1000.0 / CLOCKS_PER_SEC); }
};

} // namespace

using time::timer;
using time::clocktimer;

} // namespace

#endif