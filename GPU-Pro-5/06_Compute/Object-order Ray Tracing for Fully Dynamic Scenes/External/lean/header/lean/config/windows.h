/*****************************************************/
/* lean Dependency Config       (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONFIG_WINDOWS
#define LEAN_CONFIG_WINDOWS

#ifndef STRICT
	#define STRICT
#endif

// Get rid of macros
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN		// Exclude rarely-used stuff from Windows headers

#endif