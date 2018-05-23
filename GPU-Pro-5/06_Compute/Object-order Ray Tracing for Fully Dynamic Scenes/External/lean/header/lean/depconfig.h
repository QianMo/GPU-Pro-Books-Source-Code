/*****************************************************/
/* lean Dependency Config       (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_DEPENDENCY_CONFIG
#define LEAN_DEPENDENCY_CONFIG

#include "config/stdstd.h"
#include "config/windows.h"

#ifdef DOXGEN_READ_THIS
	/// @ingroup GlobalSwitches
	/// Define to enable unchecked fast STL.
	#define LEAN_FAST_STL
	#undef LEAN_FAST_STL
#endif

#ifdef LEAN_FAST_STL
#include "config/faststl.h"
#endif

#endif