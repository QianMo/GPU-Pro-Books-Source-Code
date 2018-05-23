/*****************************************************/
/* breeze Framework Launch Lib  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_LAUNCHER
#define BE_LAUNCHER

/// @addtogroup GlobalSwitches Global switches used for configuration
/// @see GlobalMacros
/// @see AssortedSwitches
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this when compiling this library into a DLL.
	#define BE_LAUNCHER_DO_EXPORT
	#undef BE_LAUNCHER_DO_EXPORT
#endif

/// @}

/// @defgroup AssortedSwitches Assorted Switches
/// @see GlobalSwitches

/// @addtogroup GlobalMacros Global macros
/// @see GlobalSwitches
/// @see CPP0X
/// @{

#ifdef BE_LAUNCHER_DO_EXPORT
	
	#ifdef _MSC_VER

		#ifdef BELAUNCHER_EXPORTS
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_LAUNCHER_API __declspec(dllexport)
		#else
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_LAUNCHER_API __declspec(dllimport)
		#endif

	#else
		// TODO: Implement for GCC?
		#error Exporting not yet implemented for this compiler
	#endif

#else
	/// Use this to export classes/methods/functions as part of the public library API.
	#define BE_LAUNCHER_API
#endif

/// @}

#include <beCore/beCore.h>

/// @addtogroup LauncherLibrary Launcher framework library
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Main namespace of the beLauncher library.
namespace beLauncher
{
	/// Opens a message box containing version information.
	BE_LAUNCHER_API void InfoBox();
}

/// @}

#endif