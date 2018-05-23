/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_TYPES
#define LEAN_STRINGS_TYPES

#include "../lean.h"
#include "char_traits.h"
#include "nullterminated.h"
#include "nullterminated_range.h"
#include "range.h"

// NOTE: <string> includes loads of cruft
namespace std
{
	template <class T> class allocator;
	template <class C> struct char_traits;
	template <class C, class T, class A> class basic_string;
}

namespace lean
{
namespace strings
{
namespace types
{

/// Implicit nullterminated character half-range.
typedef nullterminated_implicit<char> char_nti;
/// Nullterminated character half-range.
typedef nullterminated<char> char_nt;
/// Implicit nullterminated character range.
typedef nullterminated_range_implicit<char> char_ntri;
/// Nullterminated character range.
typedef nullterminated_range<char> char_ntr;

/// Implicit nullterminated wide-character half-range.
typedef nullterminated_implicit<wchar_t> wchar_nti;
/// Nullterminated wide-character half-range.
typedef nullterminated<wchar_t> wchar_nt;
/// Implicit nullterminated wide-character range.
typedef nullterminated_range_implicit<wchar_t> wchar_ntri;
/// Nullterminated wide-character range.
typedef nullterminated_range<wchar_t> wchar_ntr;


/// Implicit nullterminated 1-byte-character half-range.
typedef nullterminated_implicit<char1> char1_nti;
/// Nullterminated 1-byte-character half-range.
typedef nullterminated<char1> char1_nt;
/// Implicit nullterminated 1-byte-character range.
typedef nullterminated_range_implicit<char1> char1_ntri;
/// Nullterminated 1-byte-character range.
typedef nullterminated_range<char1> char1_ntr;

/// Nullterminated 2-byte-character half-range.
typedef nullterminated_implicit<char2> char2_nti;
/// Nullterminated 2-byte-character half-range.
typedef nullterminated<char2> char2_nt;
/// Nullterminated 2-byte-character range.
typedef nullterminated_range_implicit<char2> char2_ntri;
/// Nullterminated 2-byte-character range.
typedef nullterminated_range<char2> char2_ntr;

/// Implicit nullterminated 4-byte-character half-range.
typedef nullterminated_implicit<char4> char4_nti;
/// Nullterminated 4-byte-character half-range.
typedef nullterminated<char4> char4_nt;
/// Implicit nullterminated 4-byte-character range.
typedef nullterminated_range_implicit<char4> char4_ntri;
/// Nullterminated 4-byte-character range.
typedef nullterminated_range<char4> char4_ntr;

/// 1-byte-character string.
typedef std::basic_string< char1, std::char_traits<char1>, std::allocator<char1> > char1_string;
/// 2-byte-character string.
typedef std::basic_string< char2, std::char_traits<char2>, std::allocator<char2> > char2_string;
/// 4-byte-character string.
typedef std::basic_string< char4, std::char_traits<char4>, std::allocator<char4> > char4_string;

/// Implicit nullterminated utf8-character half-range.
typedef nullterminated_implicit<utf8_t> utf8_nti;
/// Nullterminated utf8-character half-range.
typedef nullterminated<utf8_t> utf8_nt;
/// Implicit nullterminated utf8-character range.
typedef nullterminated_range_implicit<utf8_t> utf8_ntri;
/// Nullterminated utf8-character range.
typedef nullterminated_range<utf8_t> utf8_ntr;

/// Implicit nullterminated utf8-character half-range.
typedef nullterminated_implicit<utf16_t> utf16_nti;
/// Nullterminated utf16-character half-range.
typedef nullterminated<utf16_t> utf16_nt;
/// Implicit nullterminated utf16-character range.
typedef nullterminated_range_implicit<utf16_t> utf16_ntri;
/// Nullterminated utf16-character range.
typedef nullterminated_range<utf16_t> utf16_ntr;

/// Implicit nullterminated utf32-character half-range.
typedef nullterminated_implicit<utf32_t> utf32_nti;
/// Nullterminated utf32-character half-range.
typedef nullterminated<utf32_t> utf32_nt;
/// Implicit nullterminated utf32-character range.
typedef nullterminated_range_implicit<utf32_t> utf32_ntri;
/// Nullterminated utf32-character range.
typedef nullterminated_range<utf32_t> utf32_ntr;

/// UTF-8-character string.
typedef std::basic_string< utf8_t, std::char_traits<utf8_t>, std::allocator<utf8_t> > utf8_string;
/// UTF-16-character string.
typedef std::basic_string< utf16_t, std::char_traits<utf16_t>, std::allocator<utf16_t> > utf16_string;
/// UTF-32-character string.
typedef std::basic_string< utf32_t, std::char_traits<utf32_t>, std::allocator<utf32_t> > utf32_string;

} // namespace

using namespace types;

// STL compatibility
template <class Char, class StringTraits, class StringAllocator, class NullterminatedTraits>
struct nullterminated_compatible< std::basic_string<Char, StringTraits, StringAllocator>, Char, NullterminatedTraits >
{
	typedef std::basic_string<Char, StringTraits, StringAllocator> compatible_type;

	static LEAN_INLINE const Char* from(const compatible_type &from) { return from.c_str(); }
	static LEAN_INLINE const Char* from(const compatible_type &from, const Char *begin) { return begin + from.size(); }
	static compatible_type to(const Char *begin) { return compatible_type(begin); }
	static compatible_type to(const Char *begin, const Char *end) { return compatible_type(begin, end); }
};

/// Returns the length of the given null-terminated array.
template <class Type, size_t Size>
LEAN_INLINE size_t ntarraylen(Type (&)[Size])
{
	return (Size > 0)
		? static_cast<size_t>(Size - 1)
		: 0;
}

} // namespace

using strings::char_nti;
using strings::char_nt;
using strings::char_ntri;
using strings::char_ntr;
using strings::wchar_nti;
using strings::wchar_nt;
using strings::wchar_ntri;
using strings::wchar_ntr;

using strings::char1_nti;
using strings::char1_nt;
using strings::char1_ntri;
using strings::char1_ntr;
using strings::char2_nti;
using strings::char2_nt;
using strings::char2_ntri;
using strings::char2_ntr;
using strings::char4_nti;
using strings::char4_nt;
using strings::char4_ntri;
using strings::char4_ntr;

using strings::char1_string;
using strings::char2_string;
using strings::char4_string;

using strings::utf8_nti;
using strings::utf8_nt;
using strings::utf8_ntri;
using strings::utf8_ntr;

using strings::utf16_nti;
using strings::utf16_nt;
using strings::utf16_ntri;
using strings::utf16_ntr;

using strings::utf32_nti;
using strings::utf32_nt;
using strings::utf32_ntri;
using strings::utf32_ntr;

using strings::utf8_string;
using strings::utf16_string;
using strings::utf32_string;

using strings::ntarraylen;

} // namespace

#endif