#ifdef LEAN_BUILD_LIB
#include "../../depconfig.h"
#endif

#include "../highres_timer.h"
#include <windows.h>

// Constructs a new timer from the current system time.
LEAN_MAYBE_INLINE lean::time::highres_timer::highres_timer()
{
	::QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&m_frequency));
	::QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&m_time));
}

// Updates the reference time stored by this timer.
LEAN_MAYBE_INLINE void lean::time::highres_timer::tick()
{
	::QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&m_time));
}

// Gets the time that has elapsed since the last tick.
LEAN_MAYBE_INLINE double lean::time::highres_timer::seconds() const
{
	uint8 newTime;
	::QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&newTime));
	return ((newTime - m_time) * 1000000 / m_frequency) * (1.0 / 1000000.0);
}

// Gets the time that has elapsed since the last tick.
LEAN_MAYBE_INLINE double lean::time::highres_timer::milliseconds() const
{
	uint8 newTime;
	::QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&newTime));
	return ((newTime - m_time) * 1000000 / m_frequency) * (1.0 / 1000.0);
}
