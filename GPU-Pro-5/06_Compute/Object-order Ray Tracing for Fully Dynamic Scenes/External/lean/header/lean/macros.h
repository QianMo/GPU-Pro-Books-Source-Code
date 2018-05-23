/*****************************************************/
/* lean Macros                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MACROS
#define LEAN_MACROS

/// @addtogroup GlobalMacros
/// @{

/// Appends token b to token a.
#define LEAN_JOIN(a, b) a##b
/// Appends token b to token a, evaluating macros first.
#define LEAN_JOIN_VALUES(a, b) LEAN_JOIN(a, b)

/// Quotes the given expression.
#define LEAN_QUOTE(expr) #expr
/// Quotes the given value, evaluating macros first.
#define LEAN_QUOTE_VALUE(value) LEAN_QUOTE(value)

/// String literal that contains the current source file and line
#define LEAN_SOURCE_STRING __FILE__ " (" LEAN_QUOTE_VALUE(__LINE__) ")"

#ifndef LEAN_NO_SOURCE_STRING
	/// String literal that contains the current source file and line
	#define LSS LEAN_SOURCE_STRING
#endif

#ifdef _MSC_VER
	/// String that contains the current function
	#define LEAN_SOURCE_FUNCTION __FUNCTION__
#else
	/// String that contains the current function
	#define LEAN_SOURCE_FUNCTION __func__
#endif

/// Makes a 4-byte word from the given four characters.
#define LEAN_MAKE_WORD_4(a, b, c, d)															\
	static_cast<::lean::uint4>(															\
		static_cast<::lean::uint4>(a) << 3U * ::lean::size_info<lean::uint1>::bits		\
		| static_cast<::lean::uint4>(b) << 2U * ::lean::size_info<lean::uint1>::bits	\
		| static_cast<::lean::uint4>(c) << 1U * ::lean::size_info<lean::uint1>::bits	\
		| static_cast<::lean::uint4>(d) << 0U * ::lean::size_info<lean::uint1>::bits )

/// Makes a 8-byte word from the given eight characters.
#define LEAN_MAKE_WORD_8(a, b, c, d, e, f, g, h)																		\
	static_cast<::lean::uint8>(																					\
		static_cast<::lean::uint8>(LEAN_MAKE_WORD_4(a, b, c, d)) << 1U * ::lean::size_info<lean::uint4>::bits		\
		| static_cast<::lean::uint8>(LEAN_MAKE_WORD_4(e, f, g, h)) << 0U * ::lean::size_info<lean::uint4>::bits )

/// Nothing.
#define LEAN_NOTHING

/// Asserts layout-compatibility of the given (struct) members.
#define LEAN_LAYOUT_COMPATIBLE(s, m, t, n) LEAN_STATIC_ASSERT_MSG( \
	offsetof(s, m) == offsetof(t, n) && sizeof(reinterpret_cast<s*>(0)->m) == sizeof(reinterpret_cast<t*>(0)->n), \
	#s "::" #m " is not layout-compatible with " #t "::" #n)

/// Asserts size-compatibility of the given (struct) types.
#define LEAN_SIZE_COMPATIBLE(s, t) LEAN_STATIC_ASSERT_MSG( \
	sizeof(s) == sizeof(t), \
	#s " is not size-compatible with " #t)

/// @}

#endif