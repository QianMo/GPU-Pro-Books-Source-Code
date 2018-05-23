/*****************************************************/
/* lean IO                      (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_IO_WCHARCVT
#define LEAN_IO_WCHARCVT

#include <locale>
#include "../lean.h"
#include "endianness.h"

namespace lean
{
namespace io
{

/// Encodes wchar_t strings as true 16-byte strings.
class wcharcvt : public std::codecvt<wchar_t, char, mbstate_t>
{
public:
	/// Constructs a wchar_t facet object.
	explicit wcharcvt(size_t refs = 0)
		: codecvt(refs) { }

protected:
	/// Copies the given wide-char string to a narrow-char string of double length (~reinterpret_cast).
	virtual result do_out(mbstate_t&,
		const wchar_t* from, const wchar_t* from_end, const wchar_t*& from_next,
		char* to, char* to_end, char*& to_next) const
	{
		byteswap_big(from, from_end, reinterpret_cast<wchar_t*>(to));
		from_next = from_end;
		to_next = to + (reinterpret_cast<const char*>(from_end) - reinterpret_cast<const char*>(from));

		return ok;
	}

	/// Copies the given narrow-char string to a wide-char string of half length (~reinterpret_cast).
	virtual result do_in(mbstate_t&,
		const char* from, const char* from_end, const char*& from_next,
		wchar_t* to, wchar_t* to_end, wchar_t*& to_next) const
	{
		LEAN_STATIC_ASSERT_MSG_ALT(
			(sizeof(wchar_t) & (sizeof(wchar_t) - 1)) == 0,
			"Sizeof(wchar_t) is no power of two.",
			Sizeof_wchar_t_is_no_power_of_two);

		from_next = from + ((from_end - from) & ~(sizeof(wchar_t) - 1));

		byteswap_big(reinterpret_cast<const wchar_t*>(from), reinterpret_cast<const wchar_t*>(from_next), to);
		to_next = to + (reinterpret_cast<const wchar_t*>(from_next) - reinterpret_cast<const wchar_t*>(from));

		return (from_next != from_end) ? partial : ok;
	}

	/// No shifting done.
	virtual result do_unshift(mbstate_t&, char* to, char*, char*& to_next) const
	{
		to_next = to;
		return noconv;
	}

	/// Computes the length of the string to be encoded.
	virtual int do_length(mbstate_t&, const char* from, const char* end, size_t max) const
	{
		LEAN_STATIC_ASSERT_MSG_ALT(
			(sizeof(wchar_t) & (sizeof(wchar_t) - 1)) == 0,
			"Sizeof(wchar_t) is no power of two.",
			Sizeof_wchar_t_is_no_power_of_two);

		return static_cast<int>(
			min(static_cast<size_t>(end - from), max * sizeof(wchar_t)) & ~(sizeof(wchar_t) - 1) );
	}

	/// Gets the (maximum) size of one wide-character.
	virtual int do_max_length() const throw()
	{
		return sizeof(wchar_t);
	}

	/// No real conversion done.
	virtual bool do_always_noconv() const throw()
	{
		return false;
	}

	/// Gets the size of one wide-character.
	virtual int do_encoding() const throw()
	{
		return sizeof(wchar_t);
	}
};

} // namespace

using io::wcharcvt;

} // namespace

#endif