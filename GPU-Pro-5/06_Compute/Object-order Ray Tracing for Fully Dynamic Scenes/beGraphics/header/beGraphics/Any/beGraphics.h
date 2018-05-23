/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_ANY
#define BE_GRAPHICS_ANY

#include "../beGraphics.h"

namespace beGraphics
{

/// @addtogroup GraphicsLibrayAny Active beGraphics implementation
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Namespace containing the active implementation.
namespace Any { }

/// @}

} // namespace

#ifdef BE_GRAPHICS_DIRECTX_11

	#include "../DX11/beGraphics.h"

	namespace beGraphics
	{
		namespace Any
		{
			using namespace DX11;
		}
	}

#endif

#endif