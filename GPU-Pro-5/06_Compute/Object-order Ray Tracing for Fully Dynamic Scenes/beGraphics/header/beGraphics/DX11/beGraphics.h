/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_DX11
#define BE_GRAPHICS_DX11

#include "../beGraphics.h"
#include <lean/meta/strip.h>
#include <lean/meta/type_traits.h>

/// @addtogroup GlobalSwitches
/// Use this to export classes/methods/functions as part of the public library API.
#define BE_GRAPHICS_DX11_API BE_GRAPHICS_API

namespace beGraphics
{

/// @addtogroup GraphicsLibrayDX11 beGraphics DirectX 11 implementation
/// For a better overview, see <a href="namespaces.html">Namespaces</a>.
/// @see <a href="namespaces.html">Namespaces</a>
/// @{

/// Main namespace of the DX11 implementation.
namespace DX11
{
	namespace parentspace = beGraphics;
	namespace thisspace = DX11;

	/// Checks whether the given object belongs to the DX11 implementation.
	LEAN_INLINE bool IsDX11(const Implementation *pImpl)
	{
		return Is(pImpl, DX11Implementation);
	}

	/// Converts the given abstract type to its DX11 implementation type, if available.
	template <class Abstract>
	struct ToImplementationDX11;

	namespace Impl
	{
		LEAN_DEFINE_HAS_TYPE(Type);

		template <bool, class>
		struct RobustToImplementationDX11Impl { };

		template <class Abstract>
		struct RobustToImplementationDX11Impl<true, Abstract>
		{
			typedef typename lean::strip_modifiers<Abstract>::template undo<
					typename ToImplementationDX11<
						typename lean::strip_modifiers<Abstract>::type
					>::Type
				>::type Type;
		};

		template <class Abstract>
		struct RobustToImplementationDX11
			: public RobustToImplementationDX11Impl<
				has_type_Type< ToImplementationDX11<typename lean::strip_modifiers<Abstract>::type> >::value,
				Abstract > { };

	} // namespace

	/// Casts the given abstract type to its DX11 implementation type, if available.
	template <class Abstract>
	LEAN_INLINE typename Impl::RobustToImplementationDX11<Abstract>::Type* ToImpl(Abstract *pAbstract)
	{
		LEAN_ASSERT(!pAbstract || pAbstract->GetImplementationID() == DX11Implementation);
		return static_cast< typename Impl::RobustToImplementationDX11<Abstract>::Type* >(pAbstract);
	}
	/// Casts the given abstract type to its DX11 implementation type, if available.
	template <class Abstract>
	LEAN_INLINE typename Impl::RobustToImplementationDX11<Abstract>::Type& ToImpl(Abstract &abstr4ct)
	{
		return *ToImpl( lean::addressof(abstr4ct) );
	}

} // namespace

using DX11::ToImpl;

} // namespace

/// @}

#endif