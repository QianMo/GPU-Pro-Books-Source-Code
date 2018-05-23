/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH
#define BE_MATH

// NOTE: Visual C++ has trouble unrolling loops that use operator != instead of operator < in their conditional expression.
// -> Use operator < whenever possible to allow for optimal code generation: for (...; i < constexpr; ...) { ... }

/// @addtogroup GlobalSwitches Global switches used for configuration
/// @see GlobalMacros
/// @see AssortedSwitches
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this when compiling this library into a DLL.
	#define BE_MATH_EXPORT
	#undef BE_MATH_EXPORT
#endif

/// @}

/// @defgroup AssortedSwitches Assorted Switches
/// @see GlobalSwitches

/// @addtogroup GlobalMacros Global macros
/// @see GlobalSwitches
/// @see CPP0X
/// @{

#ifdef BE_MATH_EXPORT
	
	#ifdef _MSC_VER

		#ifdef BEMATH_EXPORTS
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_MATH_API __declspec(dllexport)
		#else
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_MATH_API __declspec(dllimport)
		#endif

	#else
		// TODO: Implement for GCC?
		#error Exporting not yet implemented for this compiler
	#endif

#else
	/// Use this to export classes/methods/functions as part of the public library API.
	#define BE_MATH_API
#endif

/// Configures lean to not introduce additional API header dependencies.
#define LEAN_MIN_DEPENDENCY

/// Configures lean to use fast STL
#define LEAN_FAST_STL

/// @}

#include <lean/depconfig.h>
#include <lean/lean.h>

/// @addtogroup MathLibray beMath library
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Main namespace of the beMath library.
namespace beMath
{
	// Import important types
	using namespace lean::types;
	LEAN_REIMPORT_NUMERIC_TYPES;

	/// Allows for the creation of uninitialized objects.
	enum uninitialized_t
	{
		uninitialized	///< Allows for the creation of uninitialized objects.
	};

	/// Opens a message box containing version information.
	BE_MATH_API void InfoBox();
}

/// Shorthand namespace.
namespace breeze
{
#ifndef DOXYGEN_READ_THIS
	/// beMath namespace alias.
	namespace bem = ::beMath;
#else
	/// beMath namespace alias.
	namespace bem { using namespace ::beMath; }
#endif
}

/// @}

#include "beConstants.h"

#endif