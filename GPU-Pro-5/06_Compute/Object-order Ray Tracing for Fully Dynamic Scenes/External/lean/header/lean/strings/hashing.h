/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_HASHING
#define LEAN_STRINGS_HASHING

#include "../functional/hashing.h"
#include "../functional/bits.h"
#include "nullterminated.h"
#include "nullterminated_range.h"

namespace lean
{
namespace functional
{

template <class Char, class Traits>
struct hash< nullterminated<Char, Traits> > : public std::unary_function<nullterminated<Char, Traits>, size_t>
{
	LEAN_INLINE size_t operator()(const nullterminated<Char, Traits> &element) const
	{
		return compute_hash_nt<size_t>(element.c_str());
	}
};

template <class Char, class Traits>
struct hash< nullterminated_range<Char, Traits> > : public std::unary_function<nullterminated_range<Char, Traits>, size_t>
{
	LEAN_INLINE size_t operator()(const nullterminated_range<Char, Traits> &element) const
	{
		return compute_hash<size_t>(element.data(), element.data_end());
	}
};

} // namespace
} // namespace

#endif