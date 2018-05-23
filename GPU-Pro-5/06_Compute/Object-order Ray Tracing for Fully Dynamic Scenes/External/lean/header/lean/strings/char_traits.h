/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_CHAR_TRAITS
#define LEAN_STRINGS_CHAR_TRAITS

#include "../lean.h"
#include <cstring>
#include <cwchar>

namespace lean
{
namespace strings
{

/// Provides common null-terminated character range functionality for the given character type.
/// The default implementation treats characters as arbitrary POD types.
template <class Char>
struct char_traits
{
	/// Character type.
	typedef Char char_type;
	/// Unsigned integer type of range equal to or greater than char_type's.
	typedef typename int_type<sign_class::no_sign, sizeof(char_type)>::type int_type;
	/// Size type.
	typedef size_t size_type;
	
	/// Checks if the given character is null.
	static LEAN_INLINE bool null(const char_type &src)
	{
		return (src == static_cast<char_type>(0));
	}

	/// Checks if the given null-terminated range of characters is empty.
	static LEAN_INLINE bool empty(const char_type *begin)
	{
		return null(*begin);
	}
	/// Gets the length of the given null-terminated range of characters.
	static size_type length(const char_type *begin)
	{
		size_t length = 0;

		while (!null(*begin++))
			++length;

		return length;
	}
	/// Gets the number of code points in the given null-terminated range of characters.
	static LEAN_INLINE size_type count(const char_type *begin)
	{
		return length(begin);
	}
	/// Gets the number of code points in the given null-terminated range of characters.
	static LEAN_INLINE size_type count(const char_type *begin, const char_type *end)
	{
		return end - begin;
	}

	/// Compares the characters in the given null-terminated ranges, returning true if equal.
	static bool equal(const char_type *begin1, const char_type *begin2)
	{
		while (*begin1 == *begin2)
		{
			if (null(*begin1))
				return true;

			++begin1;
			++begin2;
		}

		return false;
	}
	/// Compares the characters in the given null-terminated ranges, returning true if the first is less than the second.
	static bool less(const char_type *begin1, const char_type *begin2)
	{
		while (*begin1 == *begin2)
		{
			if (null(*begin1))
				return false;

			++begin1;
			++begin2;
		}

		// Compare unsigned, correctly handles null (end of string) as smallest number
		return static_cast<int_type>(*begin1) < static_cast<int_type>(*begin2);
	}
};

template <>
struct char_traits<char>
{
	typedef char char_type;
	typedef int_type<sign_class::no_sign, sizeof(char_type)>::type int_type;
	typedef size_t size_type;
	
	static LEAN_INLINE bool null(const char_type &src)
	{
		return (src == static_cast<char_type>(0));
	}
	
	static LEAN_INLINE bool empty(const char_type *begin)
	{
		return null(*begin);
	}
	static LEAN_INLINE size_type length(const char_type *begin)
	{
		using std::strlen;
		return strlen(begin);
	}
	static LEAN_INLINE size_type count(const char_type *begin)
	{
		return length(begin);
	}
	static LEAN_INLINE size_type count(const char_type *begin, const char_type *end)
	{
		return end - begin;
	}

	static bool equal(const char_type *begin1, const char_type *begin2)
	{
		using std::strcmp;
		return (strcmp(begin1, begin2) == 0);
	}
	static bool less(const char_type *begin1, const char_type *begin2)
	{
		using std::strcmp;
		return (strcmp(begin1, begin2) < 0);
	}
};

template <>
struct char_traits<wchar_t>
{
	typedef wchar_t char_type;
	typedef int_type<sign_class::no_sign, sizeof(char_type)>::type int_type;
	typedef size_t size_type;
	
	static LEAN_INLINE bool null(const char_type &src)
	{
		return (src == static_cast<char_type>(0));
	}
	
	static LEAN_INLINE bool empty(const char_type *begin)
	{
		return null(*begin);
	}
	static LEAN_INLINE size_type length(const char_type *begin)
	{
		using std::wcslen;
		return wcslen(begin);
	}
	static LEAN_INLINE size_type count(const char_type *begin)
	{
		return length(begin);
	}
	static LEAN_INLINE size_type count(const char_type *begin, const char_type *end)
	{
		return end - begin;
	}

	static bool equal(const char_type *begin1, const char_type *begin2)
	{
		using std::wcscmp;
		return (wcscmp(begin1, begin2) == 0);
	}
	static bool less(const char_type *begin1, const char_type *begin2)
	{
		using std::wcscmp;
		return (wcscmp(begin1, begin2) < 0);
	}
};

} // namespace

using strings::char_traits;

} // namespace

#endif