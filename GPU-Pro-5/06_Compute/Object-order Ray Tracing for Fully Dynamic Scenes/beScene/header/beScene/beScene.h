/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE
#define BE_SCENE

/// @addtogroup GlobalSwitches Global switches used for configuration
/// @see GlobalMacros
/// @see AssortedSwitches
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this when not compiling this library into a DLL.
	#define BE_SCENE_NO_EXPORT
	#undef BE_SCENE_NO_EXPORT
#endif

/// @}

/// @defgroup AssortedSwitches Assorted Switches
/// @see GlobalSwitches

/// @addtogroup GlobalMacros Global macros
/// @see GlobalSwitches
/// @see CPP0X
/// @{

#ifndef BE_SCENE_NO_EXPORT
	
	#ifdef _MSC_VER

		#ifdef BESCENE_EXPORTS
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_SCENE_API __declspec(dllexport)
		#else
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_SCENE_API __declspec(dllimport)
		#endif

	#else
		// TODO: Implement for GCC?
		#error Exporting not yet implemented for this compiler
	#endif

#else
	/// Use this to export classes/methods/functions as part of the public library API.
	#define BE_SCENE_API
#endif

/// @}

#include <beCore/beCore.h>

/// @addtogroup SceneLibray beScene library
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Main namespace of the beScene library.
namespace beScene
{
	// Import important types
	using namespace lean::types;
	LEAN_REIMPORT_NUMERIC_TYPES;
	using namespace lean::strings::types;

	/// Opens a message box containing version information.
	BE_SCENE_API void InfoBox();
}

/// Shorthand namespace.
namespace breeze
{
#ifndef DOXYGEN_READ_THIS
	/// beScene namespace alias.
	namespace besc = ::beScene;
#else DOXYGEN_READ_THIS
	/// beScene namespace alias.
	namespace besc { using namespace ::beScene; }
#endif
}

/// @}

#endif