/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_ENUM
#define BE_GRAPHICS_ENUM

#include "beGraphics.h"

namespace beGraphics
{

namespace Impl
{
	/// Converts the given source enumeration value to a corresponding destination value.
	template <class Dest, class Source, class ToDestWrapper, Source SrcCompare, Source End, Dest Unknown>
	struct EnumTo
	{
		/// Recursively finds the destination value corresponding to the given source value.
		/// Will trigger a compiler error if a source value has not been translated.
		LEAN_INLINE static Dest To(Source source)
		{
			return (source == SrcCompare)
				? ToDestWrapper::template Template<SrcCompare>::Value
				: EnumTo<Dest, Source, ToDestWrapper, static_cast<Source>(SrcCompare + 1), End, Unknown>::To(source);
		}
	};
	template <class Dest, class Source, class ToDestWrapper, Source End, Dest Unknown>
	struct EnumTo<Dest, Source, ToDestWrapper, End, End, Unknown>
	{
		LEAN_INLINE static Dest To(Source source) { return Unknown; }
	};

	/// Converts the given destination value to a corresponding source enumeration value.
	template <class Dest, class Source, class ToDestWrapper, Source SrcCompare, Source End, Source Unknown>
	struct EnumFrom
	{
		/// Recursively finds the source value corresponding to the given destination value.
		/// Will trigger a compiler error if a source value has not been translated.
		LEAN_INLINE static Source From(Dest dest)
		{
			return (dest == ToDestWrapper::template Template<SrcCompare>::Value && ToDestWrapper::template Template<SrcCompare>::Exact)
				? SrcCompare
				: EnumFrom<Dest, Source, ToDestWrapper, static_cast<Source>(SrcCompare + 1), End, Unknown>::From(dest);
		}
	};
	template <class Dest, class Source, class ToDestWrapper, Source End, Source Unknown>
	struct EnumFrom<Dest, Source, ToDestWrapper, End, End, Unknown>
	{
		LEAN_INLINE static Source From(Dest dest) { return Unknown; }
	};

	/// Converts the given source enumeration flags to a corresponding destination flags.
	template <class Dest, class Source, class ToDestWrapper, Source SrcCompare, Source End>
	struct FlagsTo
	{
		/// Recursively finds the destination flags corresponding to the given source flags.
		/// Will trigger a compiler error if a source value has not been translated.
		template <class SourceFlags>
		LEAN_INLINE static Dest To(SourceFlags source)
		{
			return ( ((source & SrcCompare) == SrcCompare)
					? SourceFlags(ToDestWrapper::template Template<SrcCompare>::Value)
					: SourceFlags(0)
				) | FlagsTo<Dest, Source, ToDestWrapper, static_cast<Source>(SrcCompare + 1), End>::To(source);
		}
	};
	template <class Dest, class Source, class ToDestWrapper, Source End>
	struct FlagsTo<Dest, Source, ToDestWrapper, End, End>
	{
		template <class SourceFlags>
		LEAN_INLINE static Dest To(SourceFlags source) { return SourceFlags(0); }
	};
	
} // namespace

/// Converts the given source enumeration value to a corresponding destination value, if available, returns the closest match or unknown otherwise.
template <class Dest, class Source, class ToDestWrapper, Source Begin, Source End, Dest Unknown>
LEAN_INLINE Dest EnumTo(Source source)
{
	return Impl::EnumTo<Dest, Source, ToDestWrapper, Begin, End, Unknown>::To(source);
}
/// Converts the given source enumeration flags value to a corresponding destination flags, if available, returns the closest match otherwise.
template <class Dest, class Source, class ToDestWrapper, Source Begin, Source End, class SourceFlags>
LEAN_INLINE Dest FlagsTo(SourceFlags source)
{
	return Impl::FlagsTo<Dest, Source, ToDestWrapper, Begin, End>::To(source);
}
/// Converts the given destination value to a corresponding source enumeration value, if available, returns unknown otherwise.
template <class Dest, class Source, class ToDestWrapper, Source Begin, Source End, Source Unknown>
LEAN_INLINE Source EnumFrom(Dest dest)
{
	return Impl::EnumFrom<Dest, Source, ToDestWrapper, Begin, End, Unknown>::From(dest);
}

} // namespace

/// Gets the name of the template wrapper of the given enumeration template.
#define BE_GRAPHICS_WRAPPED_ENUM_TEMPLATE(EnumTemplate) EnumTemplate##TemplateWrapper

/// Defines a template wrapper for the given enumeration template.
#define BE_GRAPHICS_WRAP_ENUM_TEMPLATE(EnumTemplate, EnumType)	\
	struct BE_GRAPHICS_WRAPPED_ENUM_TEMPLATE(EnumTemplate)		\
	{															\
		template <EnumType From>								\
		struct Template : public EnumTemplate<From> { };		\
	};

#endif