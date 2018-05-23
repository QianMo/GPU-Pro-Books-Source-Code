/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_STRING_TRAITS
#define LEAN_STRINGS_STRING_TRAITS

#include "../lean.h"

namespace lean
{
namespace strings
{

/// Provides common string functionality for the given string type.
template <class String>
struct string_traits
{
	/// String type.
	typedef String string_type;
	/// Character type,
	typedef typename string_type::value_type value_type;
	/// Iterator type.
	typedef typename string_type::iterator iterator;
	/// Const iterator type.
	typedef typename string_type::const_iterator const_iterator;
	/// Size type.
	typedef typename string_type::size_type size_type;
	
	/// Constructs a string from the given range.
	template <class Iterator>
	static LEAN_INLINE string_type construct(Iterator begin, Iterator end)
	{
		return string_type(begin, end);
	}
	/// Assigns the given range to the given string.
	template <class Iterator>
	static LEAN_INLINE void assign(string_type &str, Iterator begin, Iterator end)
	{
		str.assign(begin, end);
	}

	/// Resizes the given string.
	static LEAN_INLINE void resize(string_type &str, size_type size)
	{
		str.resize(size);
	}
	/// Reserves the given amount of space.
	static LEAN_INLINE void reserve(string_type &str, size_type size)
	{
		str.reserve(size);
	}

	/// Erases the given range of characters.
	static LEAN_INLINE void erase(string_type &str, iterator begin, iterator end)
	{
		str.erase(begin, end);
	}

	/// Checks if the given string is empty.
	static LEAN_INLINE bool empty(const string_type &str)
	{
		return str.empty();
	}
	/// Gets the size of the given string.
	static LEAN_INLINE size_type size(const string_type &str)
	{
		return str.size();
	}

	/// Gets the beginning of the given range.
	static LEAN_INLINE iterator begin(string_type &str)
	{
		return str.begin();
	}
	/// Gets the beginning of the given range.
	static LEAN_INLINE const_iterator begin(const string_type &str)
	{
		return str.begin();
	}
	/// Gets the end of the given range.
	static LEAN_INLINE iterator end(string_type &str)
	{
		return str.end();
	}
	/// Gets the end of the given range.
	static LEAN_INLINE const_iterator end(const string_type &str)
	{
		return str.end();
	}
};

} // namespace

using strings::string_traits;

} // namespace

#endif