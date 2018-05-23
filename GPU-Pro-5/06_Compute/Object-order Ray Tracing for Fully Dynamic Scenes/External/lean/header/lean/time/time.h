/*****************************************************/
/* lean Time                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_TIME_TIME
#define LEAN_TIME_TIME

namespace lean
{
	/// Defines time-related classes such as timers, e.g. to be used in profiling.
	namespace time { }
}

#include "timer.h"
#include "highres_timer.h"

#endif