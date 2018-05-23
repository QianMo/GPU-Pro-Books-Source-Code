/************************************************************/
/* breeze Engine Entity System Module  (c) Tobias Zirr 2011 */
/************************************************************/

#pragma once
#ifndef BE_ENTITYSYSTEM
#define BE_ENTITYSYSTEM

/// @addtogroup GlobalSwitches Global switches used for configuration
/// @see GlobalMacros
/// @see AssortedSwitches
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this when not compiling this library into a DLL.
	#define BE_ENTITYSYSTEM_NO_EXPORT
	#undef BE_ENTITYSYSTEM_NO_EXPORT
#endif

/// @}

/// @defgroup AssortedSwitches Assorted Switches
/// @see GlobalSwitches

/// @addtogroup GlobalMacros Global macros
/// @see GlobalSwitches
/// @see CPP0X
/// @{

#ifndef BE_ENTITYSYSTEM_NO_EXPORT
	
	#ifdef _MSC_VER

		#ifdef BEENTITYSYSTEM_EXPORTS
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_ENTITYSYSTEM_API __declspec(dllexport)
		#else
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_ENTITYSYSTEM_API __declspec(dllimport)
		#endif

	#else
		// TODO: Implement for GCC?
		#error Exporting not yet implemented for this compiler
	#endif

#else
	/// Use this to export classes/methods/functions as part of the public library API.
	#define BE_ENTITYSYSTEM_API
#endif

/// @}

#include <beCore/beCore.h>
#include <beCore/beShared.h>
#include <lean/tags/noncopyable.h>

/// @addtogroup EntitySystemLibray beEntitySystem library
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Main namespace of the beEntitySystem library.
namespace beEntitySystem
{
	// Import important types
	using namespace lean::types;
	LEAN_REIMPORT_NUMERIC_TYPES;
	using namespace lean::strings::types;

	/// Opens a message box containing version information.
	BE_ENTITYSYSTEM_API void InfoBox();
}

/// Shorthand namespace.
namespace breeze
{
#ifndef DOXYGEN_READ_THIS
	/// beEntitySystem namespace alias.
	namespace bees = ::beEntitySystem;
#else
	/// beEntitySystem namespace alias.
	namespace bees { using namespace ::beEntitySystem; }
#endif
}

/// @}

#endif