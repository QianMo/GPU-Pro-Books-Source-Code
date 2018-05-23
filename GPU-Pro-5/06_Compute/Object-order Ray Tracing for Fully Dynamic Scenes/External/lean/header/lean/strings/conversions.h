/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_CONVERSIONS
#define LEAN_STRINGS_CONVERSIONS

#include "../lean.h"
#include "types.h"
#include <string>
#include <locale>
#include "../utf8.h"

namespace lean
{
namespace strings
{

/// Gets the current system locale.
static inline const std::locale& system_locale()
{
	static std::locale system("");
	return system;
}


//// UTF / UTF ////

/// Converts the given string from UTF-8 to UTF-16.
template <class String, class Range>
inline String utf8_to_utf16(const Range &wide)
{
	String result;
	result.resize(wide.size());
	result.erase(
		utf8::unchecked::utf8to16(wide.begin(), wide.end(), result.begin()),
		result.end());
	return result;
}
/// Converts the given string from UTF-8 to UTF-16.
template <class String>
LEAN_INLINE String utf_to_utf16(const utf8_ntri &wide)
{
	return utf8_to_utf16<String>(wide);
}
/// Converts the given string from UTF-8 to UTF-16.
LEAN_INLINE utf16_string utf_to_utf16(const utf8_ntri &wide)
{
	return utf8_to_utf16<utf16_string>(wide);
}

/// Converts the given string from UTF-16 to UTF-8.
template <class String, class Range>
inline String utf16_to_utf8(const Range &wide)
{
	String result;
	result.resize(2 * wide.size());
	result.erase(
		utf8::unchecked::utf16to8(wide.begin(), wide.end(), result.begin()),
		result.end());
	return result;
}
/// Converts the given string from UTF-16 to UTF-8.
template <class String>
LEAN_INLINE String utf_to_utf8(const utf16_ntri &wide)
{
	return utf16_to_utf8<String>(wide);
}
/// Converts the given string from UTF-16 to UTF-8.
LEAN_INLINE utf8_string utf_to_utf8(const utf16_ntri &wide)
{
	return utf16_to_utf8<utf8_string>(wide);
}


/// Converts the given string from UTF-8 to UTF-32.
template <class String, class Range>
inline String utf8_to_utf32(const Range &wide)
{
	String result;
	result.resize(wide.size());
	result.erase(
		utf8::unchecked::utf8to32(wide.begin(), wide.end(), result.begin()),
		result.end());
	return result;
}
/// Converts the given string from UTF-8 to UTF-32.
template <class String>
LEAN_INLINE String utf_to_utf32(const utf8_ntri &wide)
{
	return utf8_to_utf32<String>(wide);
}
/// Converts the given string from UTF-8 to UTF-32.
LEAN_INLINE utf32_string utf_to_utf32(const utf8_ntri &wide)
{
	return utf8_to_utf32<utf32_string>(wide);
}

/// Converts the given string from UTF-32 to UTF-8.
template <class String, class Range>
inline String utf32_to_utf8(const Range &wide)
{
	String result;
	result.resize(4 * wide.size());
	result.erase(
		utf8::unchecked::utf32to8(wide.begin(), wide.end(), result.begin()),
		result.end());
	return result;
}
/// Converts the given string from UTF-32 to UTF-8.
template <class String>
LEAN_INLINE String utf_to_utf8(const utf32_ntri &wide)
{
	return utf32_to_utf8<String>(wide);
}
/// Converts the given string from UTF-32 to UTF-8.
LEAN_INLINE utf8_string utf_to_utf8(const utf32_ntri &wide)
{
	return utf32_to_utf8<utf8_string>(wide);
}


/// Converts the given string from UTF-16 to UTF-32.
template <class String>
LEAN_INLINE String utf_to_utf32(const utf16_ntri &wide)
{
	return utf8_to_utf32<String>(utf16_to_utf8<utf8_string>(wide));
}
/// Converts the given string from UTF-16 to UTF-32.
LEAN_INLINE utf32_string utf_to_utf32(const utf16_ntri &wide)
{
	return utf8_to_utf32<utf32_string>(utf16_to_utf8<utf8_string>(wide));
}

/// Converts the given string from UTF-32 to UTF-16.
template <class String>
LEAN_INLINE String utf_to_utf16(const utf32_ntri &wide)
{
	return utf8_to_utf16<String>(utf32_to_utf8<utf8_string>(wide));
}
/// Converts the given string from UTF-32 to UTF-16.
LEAN_INLINE utf16_string utf_to_utf16(const utf32_ntri &wide)
{
	return utf8_to_utf16<utf16_string>(utf32_to_utf8<utf8_string>(wide));
}


/// Converts the given string from UTF-8 to UTF-8.
template <class String>
LEAN_INLINE String utf_to_utf8(const utf8_ntri &wide)
{
	return String(wide.begin(), wide.end());
}
/// Converts the given string from UTF-8 to UTF-8.
LEAN_INLINE utf8_ntri utf_to_utf8(const utf8_ntri &wide)
{
	return wide;
}

/// Converts the given string from UTF-16 to UTF-16.
template <class String>
LEAN_INLINE String utf_to_utf16(const utf16_ntri &wide)
{
	return String(wide.begin(), wide.end());
}
/// Converts the given string from UTF-16 to UTF-16.
LEAN_INLINE utf16_ntri utf_to_utf16(const utf16_ntri &wide)
{
	return wide;
}

/// Converts the given string from UTF-32 to UTF-32.
template <class String>
LEAN_INLINE String utf_to_utf32(const utf32_ntri &wide)
{
	return String(wide.begin(), wide.end());
}
/// Converts the given string from UTF-32 to UTF-32.
LEAN_INLINE utf32_ntri utf_to_utf32(const utf32_ntri &wide)
{
	return wide;
}


namespace impl
{

template <class String, class DestChar>
struct utf_to_utf_helper;

template <class String>
struct utf_to_utf_helper<String, utf8_t>
{
	template <class Range>
	LEAN_INLINE String operator ()(const Range &str) const { return utf_to_utf8<String>(str); }
};
template <class String>
struct utf_to_utf_helper<String, utf16_t>
{
	template <class Range>
	LEAN_INLINE String operator ()(const Range &str) const { return utf_to_utf16<String>(str); }
};
template <class String>
struct utf_to_utf_helper<String, utf32_t>
{
	template <class Range>
	LEAN_INLINE String operator ()(const Range &str) const { return utf_to_utf32<String>(str); }
};

} // namespace

/// Converts the given string from UTF to UTF.
template <class String, class Range>
LEAN_INLINE String utf_to_utf(const Range &str)
{
	return impl::utf_to_utf_helper<String, typename String::value_type>()(str);
}


//// Codepage / UTF-XX ////

/// Widens the given string using either the given locale or the current global locale.
template <class String> // , class Range
inline String char_to_wchar(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	String result;
	result.resize(narrow.size());
	std::use_facet< std::ctype<wchar_t> >(locale).widen(narrow.begin(), narrow.end(), &result[0]);
	return result;
}
/// Widens the given string using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String to_wchar(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return char_to_wchar<String>(narrow, locale);
}
/// Widens the given string using either the given locale or the current global locale.
LEAN_INLINE std::wstring to_wchar(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return char_to_wchar<std::wstring>(narrow, locale);
}

/// Narrows the given string using either the given locale or the current global locale.
template <class String> // , class Range
inline String wchar_to_char(const wchar_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	String result;
	result.resize(wide.size());
	std::use_facet< std::ctype<wchar_t> >(locale).narrow(wide.begin(), wide.end(), invalid, &result[0]);
	return result;
}
/// Narrows the given string using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String to_char(const wchar_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	return wchar_to_char<String>(wide, locale, invalid);
}
/// Narrows the given string using either the given locale or the current global locale.
LEAN_INLINE std::string to_char(const wchar_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	return wchar_to_char<std::string>(wide, locale, invalid);
}


/// Widens the given string to UTF-16 using either the given locale or the current global locale.
template <class String> // , class Range
inline String char_to_utf16(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	String result;
	result.resize(narrow.size());
	std::use_facet< std::ctype<utf16_t> >(locale).widen(narrow.begin(), narrow.end(), &result[0]);
	return result;
}
/// Widens the given string to UTF-16 using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String to_utf16(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return char_to_utf16<String>(narrow, locale);
}
/// Widens the given string to UTF-16 using either the given locale or the current global locale.
LEAN_INLINE utf16_string to_utf16(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return char_to_utf16<utf16_string>(narrow, locale);
}

/// Narrows the given to UTF-16 string using either the given locale or the current global locale.
template <class String> // , class Range
inline String utf16_to_char(const utf16_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	String result;
	result.resize(wide.size());
	std::use_facet< std::ctype<utf16_t> >(locale).narrow(wide.begin(), wide.end(), invalid, &result[0]);
	return result;
}
/// Narrows the given to UTF-16 string using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String utf_to_char(const utf16_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	return utf16_to_char<String>(wide, locale, invalid);
}
/// Narrows the given to UTF-16 string using either the given locale or the current global locale.
LEAN_INLINE std::string utf_to_char(const utf16_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	return utf16_to_char<std::string>(wide, locale, invalid);
}


/// Widens the given string to UTF-32 using either the given locale or the current global locale.
template <class String> // , class Range
inline String char_to_utf32(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	String result;
	result.resize(narrow.size());
	std::use_facet< std::ctype<utf32_t> >(locale).widen(narrow.begin(), narrow.end(), &result[0]);
	return result;
}
/// Widens the given string to UTF-32 using either the given locale or the current global locale.
template <class String>
LEAN_INLINE utf32_string to_utf32(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return char_to_utf32<String>(narrow, locale);
}
/// Widens the given string to UTF-32 using either the given locale or the current global locale.
LEAN_INLINE utf32_string to_utf32(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return char_to_utf32<utf32_string>(narrow, locale);
}

/// Narrows the given UTF-32 string using either the given locale or the current global locale.
template <class String> // , class Range
inline String utf32_to_char(const utf32_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	String result;
	result.resize(wide.size());
	std::use_facet< std::ctype<utf32_t> >(locale).narrow(wide.begin(), wide.end(), invalid, &result[0]);
	return result;
}
/// Narrows the given UTF-32 string using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String utf_to_char(const utf32_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	return utf32_to_char<String>(wide, locale, invalid);
}
/// Narrows the given UTF-32 string using either the given locale or the current global locale.
LEAN_INLINE std::string utf_to_char(const utf32_ntri &wide, const std::locale &locale = std::locale(), char invalid = '?')
{
	return utf32_to_char<std::string>(wide, locale, invalid);
}


//// Codepage / UTF-8 ////

/// Widens the given string to UTF-8 using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String to_utf8(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return utf16_to_utf8<String>(char_to_utf16<utf16_string>(narrow, locale));
}
/// Widens the given string to UTF-8 using either the given locale or the current global locale.
LEAN_INLINE utf8_string to_utf8(const char_ntri &narrow, const std::locale &locale = std::locale())
{
	return utf16_to_utf8<utf8_string>(char_to_utf16<utf16_string>(narrow, locale));
}

/// Narrows the given UTF-8 string using either the given locale or the current global locale.
template <class String>
LEAN_INLINE String utf_to_char(const utf8_ntri &utf8, const std::locale &locale = std::locale(), char invalid = '?')
{
	return utf16_to_char<String>(utf8_to_utf16<utf16_string>(utf8), locale, invalid);
}
/// Narrows the given UTF-8 string using either the given locale or the current global locale.
LEAN_INLINE std::string utf_to_char(const utf8_ntri &utf8, const std::locale &locale = std::locale(), char invalid = '?')
{
	return utf16_to_char<std::string>(utf8_to_utf16<utf16_string>(utf8), locale, invalid);
}

} // namespace

using strings::to_wchar;
using strings::to_char;

using strings::to_utf8;
using strings::to_utf16;
using strings::to_utf32;

using strings::utf_to_utf8;
using strings::utf_to_utf16;
using strings::utf_to_utf32;

using strings::utf_to_char;

} // namespace

#endif