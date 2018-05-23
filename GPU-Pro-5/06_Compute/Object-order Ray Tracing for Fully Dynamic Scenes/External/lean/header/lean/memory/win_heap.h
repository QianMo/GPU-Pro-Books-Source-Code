/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_WIN_HEAP
#define LEAN_MEMORY_WIN_HEAP

#include "../lean.h"
#include "alignment.h"

#ifndef LEAN_ASSUME_WIN_ALIGNMENT
	// MONITOR: Windows heap aligns memory to 8 byte (16 on x64) boundaries by default
	// see http://stackoverflow.com/questions/2805896/what-alignment-does-heapalloc-use
	#ifdef LEAN_64_BIT
		/// Specifies alignment requirements that are assumed to always be met by the windows memory management functions
		/// @ingroup MemorySwitches
		#define LEAN_ASSUME_WIN_ALIGNMENT 16
	#else
		/// Specifies alignment requirements that are assumed to always be met by the windows memory management functions
		/// @ingroup MemorySwitches
		#define LEAN_ASSUME_WIN_ALIGNMENT 8
	#endif
#endif

namespace lean
{
namespace memory
{

/// Windows heap.
struct win_heap
{
	/// Size type.
	typedef size_t size_type;
	/// Default alignment.
	static const size_type default_alignment = LEAN_ASSUME_WIN_ALIGNMENT;
	/// Maximum alignment.
	static const size_type max_alignment = static_cast<unsigned char>(-1);

	/// Allocates the given amount of memory.
	LEAN_MAYBE_EXPORT static void* allocate(size_type size);
	/// Frees the given block of memory.
	LEAN_MAYBE_EXPORT static void free(void *memory);

	/// Allocates the given amount of memory respecting the given alignment.
	template <size_t Alignment>
	static LEAN_INLINE void* allocate(size_type size)
	{
		if (Alignment <= default_alignment && is_valid_alignment<Alignment>::value)
			return allocate(size);
		else
		{
			LEAN_STATIC_ASSERT_MSG_ALT(Alignment <= max_alignment,
				"Alignment > max unsigned char unsupported.",
				Alignment_bigger_than_max_unsigned_char_unsupported);

			unsigned char *unaligned = reinterpret_cast<unsigned char*>( allocate(size + Alignment) );
			unsigned char *aligned = upper_align<Alignment>(unaligned);
			aligned[-1] = static_cast<unsigned char>(aligned - unaligned);
			return aligned;
		}
	}
	/// Frees the given aligned block of memory.
	template <size_t Alignment>
	static LEAN_INLINE void free(void *memory)
	{
		if (Alignment <= default_alignment && is_valid_alignment<Alignment>::value)
			free(memory);
		else if (memory)
			free(reinterpret_cast<unsigned char*>(memory) - reinterpret_cast<unsigned char*>(memory)[-1]);
	}
	/// Frees the given aligned block of memory.
	static LEAN_INLINE void free(void *memory, size_t alignment)
	{
		if (alignment <= default_alignment && check_alignment(alignment))
			free(memory);
		else if (memory)
			free(reinterpret_cast<unsigned char*>(memory) - reinterpret_cast<unsigned char*>(memory)[-1]);
	}
};

} // namespace

using memory::win_heap;

} // namespace

#ifdef LEAN_INCLUDE_LINKED
#include "source/win_heap.cpp"
#endif

#endif