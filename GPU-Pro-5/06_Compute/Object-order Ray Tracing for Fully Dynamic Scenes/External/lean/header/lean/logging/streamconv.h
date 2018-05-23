/*****************************************************/
/* lean Logging                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_LOGGING_STREAMCONV
#define LEAN_LOGGING_STREAMCONV

#include "../lean.h"
#include "../strings/types.h"
#include "../strings/conversions.h"
#include <iosfwd>
#include <ostream>

// Drop-in replacements for missing utf8 streaming operators
template <class Traits, class StringTraits, class StringAlloc>
inline std::basic_ostream<lean::utf8_t, Traits>& operator <<(
	std::basic_ostream<lean::utf8_t, Traits>& stream,
	const std::basic_string<lean::utf16_t, StringTraits, StringAlloc> &string)
{
	return (stream << lean::utf_to_utf8(string));
}
template <class Traits, class StringTraits, class StringAlloc>
inline std::basic_ostream<lean::utf8_t, Traits>& operator <<(
	std::basic_ostream<lean::utf8_t, Traits>& stream,
	const std::basic_string<lean::utf32_t, StringTraits, StringAlloc> &string)
{
	return (stream << lean::utf_to_utf8(string));
}
template <class Traits>
inline std::basic_ostream<lean::utf8_t, Traits>& operator <<(std::basic_ostream<lean::utf8_t, Traits>& stream, const lean::utf16_t *str)
{
	return (stream << lean::utf_to_utf8(str));
}
template <class Traits>
inline std::basic_ostream<lean::utf8_t, Traits>& operator <<(std::basic_ostream<lean::utf8_t, Traits>& stream, const lean::utf32_t *str)
{
	return (stream << lean::utf_to_utf8(str));
}

// Drop-in replacements for missing utf16 streaming operators
template <class Traits, class StringTraits, class StringAlloc>
inline std::basic_ostream<lean::utf16_t, Traits>& operator <<(
	std::basic_ostream<lean::utf16_t, Traits>& stream,
	const std::basic_string<lean::utf32_t, StringTraits, StringAlloc> &string)
{
	return (stream << lean::utf_to_utf16(string));
}
template <class Traits>
inline std::basic_ostream<lean::utf16_t, Traits>& operator <<(std::basic_ostream<lean::utf16_t, Traits>& stream, const lean::utf32_t *str)
{
	return (stream << lean::utf_to_utf16(str));
}

// Drop-in replacements for missing utf32 streaming operators
template <class Traits, class StringTraits, class StringAlloc>
inline std::basic_ostream<lean::utf32_t, Traits>& operator <<(
	std::basic_ostream<lean::utf32_t, Traits>& stream,
	const std::basic_string<lean::utf16_t, StringTraits, StringAlloc> &string)
{
	return (stream << lean::utf_to_utf32(string));
}
template <class Traits>
inline std::basic_ostream<lean::utf32_t, Traits>& operator <<(std::basic_ostream<lean::utf32_t, Traits>& stream, const lean::utf16_t *str)
{
	return (stream << lean::utf_to_utf32(str));
}

#endif