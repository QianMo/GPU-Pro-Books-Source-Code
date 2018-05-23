/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS
#define BE_GRAPHICS

/// @addtogroup GlobalSwitches Global switches used for configuration
/// @see GlobalMacros
/// @see AssortedSwitches
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this when not compiling this library into a DLL.
	#define BE_GRAPHICS_NO_EXPORT
	#undef BE_GRAPHICS_NO_EXPORT
#endif

/// @}

/// @defgroup AssortedSwitches Assorted Switches
/// @see GlobalSwitches

/// @addtogroup GlobalMacros Global macros
/// @see GlobalSwitches
/// @{

#ifndef BE_GRAPHICS_NO_EXPORT
	
	#ifdef _MSC_VER

		#ifdef BEGRAPHICS_EXPORTS
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_GRAPHICS_API __declspec(dllexport)
		#else
			/// Use this to export classes/methods/functions as part of the public library API.
			#define BE_GRAPHICS_API __declspec(dllimport)
		#endif

	#else
		// TODO: Implement for GCC?
		#error Exporting not yet implemented for this compiler
	#endif

#else
	/// Use this to export classes/methods/functions as part of the public library API.
	#define BE_GRAPHICS_API
#endif

/// @}

#include <beCore/beCore.h>

/// @addtogroup GraphicsLibray beGraphics library
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

#ifdef DOXYGEN_READ_THIS
	/// Define this to choose the DirectX 11 implementation.
	#define BE_GRAPHICS_DIRECTX_11
	#undef BE_GRAPHICS_DIRECTX_11
#endif

// Default to DirectX 11 if nothing else specified
#if !defined(BE_GRAPHICS_DIRECTX_11) && 1 // ... // TODO: Update for every new alternative
	#define BE_GRAPHICS_DIRECTX_11
#endif

/// Main namespace of the beGraphics library.
namespace beGraphics
{
	namespace thisspace = beGraphics;

	// Import important types
	using namespace lean::types;
	LEAN_REIMPORT_NUMERIC_TYPES;
	using namespace lean::strings::types;

	/// Implementation ID enumeration.
	enum ImplementationID
	{
		DX11Implementation = LEAN_MAKE_WORD_4('D', 'X', '1', '1')	///< DirectX 11 implementation ID.
	};

	#ifdef BE_GRAPHICS_DIRECTX_11
		/// Active implementation ID.
		static const ImplementationID ActiveImplementation = DX11Implementation;
	#endif

	/// Graphics implementation interface.
	class Implementation
	{
	public:
		virtual ~Implementation() { }

		/// Gets the implementation identifier.
		virtual ImplementationID GetImplementationID() const = 0;
	};

	/// Checks whether the given object belongs to the given implementation.
	LEAN_INLINE bool Is(const Implementation *pImpl, ImplementationID implID)
	{
		return (pImpl->GetImplementationID() == implID);
	}

	/// Opens a message box containing version information.
	BE_GRAPHICS_API void InfoBox();
}

/// Shorthand namespace.
namespace breeze
{
#ifndef DOXYGEN_READ_THIS
	/// beGraphics namespace alias.
	namespace beg = ::beGraphics;
#else
	/// beGraphics namespace alias.
	namespace beg { using namespace ::beGraphics; }
#endif
}

/// @}

#endif