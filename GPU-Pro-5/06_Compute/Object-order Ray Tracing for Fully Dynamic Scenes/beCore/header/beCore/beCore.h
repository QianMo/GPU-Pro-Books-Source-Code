/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE
#define BE_CORE

/// @addtogroup GlobalSwitches Global switches used for configuration
/// @see GlobalMacros
/// @see AssortedSwitches
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this when not compiling this library into a DLL.
	#define BE_CORE_NO_EXPORT
	#undef BE_CORE_NO_EXPORT
#endif

/// @}

/// @defgroup AssortedSwitches Assorted Switches
/// @see GlobalSwitches

/// @addtogroup GlobalMacros Global macros
/// @see GlobalSwitches
/// @see CPP0X
/// @{

#ifndef BE_CORE_NO_EXPORT
	
	#ifdef _MSC_VER

		#ifdef BECORE_EXPORTS
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_CORE_EXPORT __declspec(dllexport)
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_CORE_IMPORT 
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_CORE_API BE_CORE_EXPORT
		#else
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_CORE_EXPORT 
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_CORE_IMPORT __declspec(dllimport)
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_CORE_API BE_CORE_IMPORT
		#endif
		
		#pragma warning(push)
		// Interfaces are declaration-only
		#pragma warning(disable : 4275)

	#else
		// TODO: Implement for GCC?
		#error Exporting not yet implemented for this compiler
	#endif

	/// Defined to only link lean library once.
	#define LEAN_MAYBE_EXPORT BE_CORE_API

#else
	/// Use this to export classes/methods/functions as part of the public library API.
	#define BE_CORE_API
#endif

/// Configures lean to not introduce additional API header dependencies.
#define LEAN_MIN_DEPENDENCY

/// Configures lean to use fast STL
#define LEAN_FAST_STL

/// Configure lean to use the windows heap as default heap.
#define LEAN_DEFAULT_HEAP win_heap

/// @}

#include <lean/depconfig.h>
#include <lean/lean.h>

#include <lean/memory/win_heap.h>
#include <lean/memory/default_heap.h>

#include <lean/strings/types.h>

/// @addtogroup CoreLibray beCore library
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Main namespace of the beCore library.
namespace beCore
{
	/// Opens a message box containing version information.
	BE_CORE_API void InfoBox();

	// Import important types
	using namespace lean::types;
	LEAN_REIMPORT_NUMERIC_TYPES;
	using namespace lean::strings::types;
}

/// Shorthand namespace.
namespace breeze
{
#ifndef DOXYGEN_READ_THIS
	/// beCore namespace alias.
	namespace bec = ::beCore;
#else
	/// beCore namespace alias.
	namespace bec { using namespace ::beCore; }
#endif
}

/// @}

#endif