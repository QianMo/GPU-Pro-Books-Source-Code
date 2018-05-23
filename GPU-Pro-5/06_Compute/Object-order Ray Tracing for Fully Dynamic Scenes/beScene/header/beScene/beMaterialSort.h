/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MATERIALSORT
#define BE_SCENE_MATERIALSORT

#include "beScene.h"

namespace beScene
{

/// Material-based sorting.
namespace MaterialSort
{
	/// Hashes the given pointer address to a number of the given size.
	template <size_t Size>
	LEAN_INLINE typename lean::int_type<lean::sign_class::no_sign, Size>::type HashAddress(uintptr_t ptr)
	{
		uintptr_t result = ptr;

		for (size_t i = Size; i < sizeof(ptr); i += Size)
			result ^= ptr >> (i * lean::bits_per_byte);

		typedef typename lean::int_type<lean::sign_class::no_sign, Size>::type hash_type;
		return static_cast<hash_type>(result);
	}

	/// Hashes the given pointer address to a number of the given size.
	template <size_t Size, class PointerValue>
	LEAN_INLINE typename lean::int_type<lean::sign_class::no_sign, Size>::type HashPointer(PointerValue *ptr)
	{
		return HashAddress<Size>( reinterpret_cast<uintptr_t>(ptr) );
	}

	/// Generates a sort index from the given pass & material.
	template <class Pass, class Material>
	LEAN_INLINE uint4 GenerateSortIndex(Pass *pPass, Material *pMaterial)
	{
		uint4 index;
		index = HashPointer<2>(pPass);
		index <<= 2 * lean::bits_per_byte;
		index |= HashPointer<2>(pMaterial);
		return index;
	}

} // namespace

} // namespace

#endif