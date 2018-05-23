/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_UTILITY
#define LEAN_STRINGS_UTILITY

#include "../lean.h"
#include <cstring>
#include <cwchar>

namespace lean
{
namespace strings
{
	/// Copies at maximum the given number of characters from source to dest string, always incorporating a terminating null.
	inline size_t strmcpy(char *dest, const char *source, size_t maxChars)
	{
		LEAN_ASSERT(maxChars > 0);

		size_t len = min(::strlen(source), maxChars - 1);
		::memcpy(dest, source, len);
		dest[len] = 0;
		return len;
	}

	/// Copies at maximum the given number of characters from source to dest string, always incorporating a terminating null.
	inline size_t wcsmcpy(wchar_t *dest, const wchar_t *source, size_t maxChars)
	{
		LEAN_ASSERT(maxChars > 0);

		size_t len = min(::wcslen(source), maxChars - 1);
		::memcpy(dest, source, len * sizeof(wchar_t));
		dest[len] = 0;
		return len;
	}

	/// Copies at maximum the given number of characters from source to dest string, always incorporating a terminating null.
	template <class Char, class Range>
	inline Char* strcpy_range(Char *dest, const Range &source)
	{
		size_t len = source.end() - source.begin();
		::memcpy(dest, static_cast<const Char*>(source.begin()), len * sizeof(Char));
		dest[len] = 0;
		return dest;
	}

} // namespace

using strings::strmcpy;
using strings::strcpy_range;

using strings::wcsmcpy;

} // namespace

#endif