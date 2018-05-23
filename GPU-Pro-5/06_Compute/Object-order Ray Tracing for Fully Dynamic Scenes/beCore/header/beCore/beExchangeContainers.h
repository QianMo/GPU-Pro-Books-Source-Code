/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_EXCHANGE_CONTAINERS
#define BE_CORE_EXCHANGE_CONTAINERS

#include "beCore.h"
#include "beShared.h"

#include <vector>
#include <list>
#include <string>

namespace beCore
{

namespace Exchange
{

/// Defines an STL vector type that may be used across module boundaries.
template <class Element, size_t Alignment = alignof(Element)>
struct vector_t
{
	/// Exchange vector type.
	typedef std::vector<Element, typename exchange_allocator_t<Element, Alignment>::t> t;
};

/// Defines an STL list type that may be used across module boundaries.
template <class Element, size_t Alignment = alignof(Element)>
struct list_t
{
	/// Exchange list type.
	typedef std::list<Element, typename exchange_allocator_t<Element, Alignment>::t> t;
};

/// Defines a string type that may be used across module boundaries.
typedef std::basic_string<char, std::char_traits<char>, exchange_allocator_t<char>::t> string;
/// Defines a string type that may be used across module boundaries.
typedef std::basic_string<wchar_t, std::char_traits<wchar_t>, exchange_allocator_t<wchar_t>::t> wstring;
/// Defines a string type that may be used across module boundaries.
typedef std::basic_string<lean::utf8_t, std::char_traits<lean::utf8_t>, exchange_allocator_t<lean::utf8_t>::t> utf8_string;
/// Defines a string type that may be used across module boundaries.
typedef std::basic_string<lean::utf16_t, std::char_traits<lean::utf16_t>, exchange_allocator_t<lean::utf16_t>::t> utf16_string;
/// Defines a string type that may be used across module boundaries.
typedef std::basic_string<lean::utf32_t, std::char_traits<lean::utf32_t>, exchange_allocator_t<lean::utf32_t>::t> utf32_string;

} // namespace

} // namespace

#endif