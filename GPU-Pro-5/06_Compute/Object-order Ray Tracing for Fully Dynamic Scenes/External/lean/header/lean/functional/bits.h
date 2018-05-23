/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FUNCTIONAL_BITS
#define LEAN_FUNCTIONAL_BITS

#include "../lean.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace lean
{
namespace functional
{

/// Computes an integer hash value from the given byte array.
template <class Hash, class Iterator>
LEAN_INLINE Hash compute_hash(Iterator begin, Iterator end)
{
	Hash hash = 5381;

	// djb2
	while (begin != end)
		hash = (Hash) (((hash << 5) + hash) ^ *begin++);

	return hash;
}

/// Computes an integer hash value from the given byte array.
template <class Hash, class Iterator>
LEAN_INLINE Hash compute_hash_nt(Iterator begin)
{
	Hash hash = 5381;

	// djb2
	while (*begin)
		hash = (Hash) (((hash << 5) + hash) ^ *begin++);

	return hash;
}

#ifdef _MSC_VER

/// Gets the position of the first set low bit.
LEAN_INLINE unsigned long first_bit_low(unsigned long mask)
{
	unsigned long idx;
	return _BitScanForward(&idx, mask) ? idx : 32;
}

/// Gets the position of the first set high bit.
LEAN_INLINE unsigned long first_bit_high(unsigned long mask)
{
	unsigned long idx;
	return _BitScanReverse(&idx, mask) ? idx : 32;
}

#endif

} // namespace

using functional::compute_hash;

using functional::first_bit_low;
using functional::first_bit_high;

} // namespace

#endif