/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_CHUNK_HEAP
#define LEAN_MEMORY_CHUNK_HEAP

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "alignment.h"
#include "default_heap.h"

namespace lean
{
namespace memory
{

/// Block of memory that may be zero-sized.
template <size_t Size>
struct optional_mem_block
{
	char memory[Size];

	LEAN_INLINE char* get() { return memory; }
	LEAN_INLINE const char* get() const { return memory; }

	LEAN_INLINE operator char*() { return memory; }
	LEAN_INLINE operator const char*() const { return memory; }
};
template <>
struct optional_mem_block<0>
{
	LEAN_INLINE char* get() { return nullptr; }
	LEAN_INLINE const char* get() const { return nullptr; }

	LEAN_INLINE operator char*() { return nullptr; }
	LEAN_INLINE operator const char*() const { return nullptr; }
};

/// Contiguous chunk allocator heap.
template <size_t ChunkSize, class Heap = default_heap, size_t StaticChunkSize = ChunkSize, size_t DefaultAlignment = sizeof(void*)>
class chunk_heap : public lean::noncopyable
{
public:
	/// Heap type.
	typedef Heap heap_type;
	/// Size type.
	typedef typename heap_type::size_type size_type;
	/// Chunk size.
	static const size_type chunk_size = ChunkSize;
	/// Default alignment.
	static const size_type default_alignment = DefaultAlignment;

private:
	// Optional first static chunk
	optional_mem_block<StaticChunkSize> m_firstChunk;
	
	// Current chunk
	char *m_chunk;
	char *m_chunkOffset;
	char *m_chunkEnd;

	// Next chunk size
	size_type m_nextChunkSize;

	/// Chunk header
	struct chunk_header
	{
		char *prev_chunk;

		/// Constructor.
		chunk_header(char *prev_chunk)
			: prev_chunk(prev_chunk) { }
	};
	// Chunk alignment
	static const size_t chunk_alignment = alignof(chunk_header);

	/// Gets the header of the given dynamically allocated chunk.
	LEAN_INLINE static chunk_header* to_chunk_header(char *chunk)
	{
		return reinterpret_cast<chunk_header*>(chunk) - 1;
	}

	/// Allocates the given amount of memory.
	template <size_t Alignment>
	char* allocate_aligned(size_type size)
	{
		// Get next free memory location
		char *aligned = align<Alignment>(m_chunkOffset);

		// Allocate new chunk, if old chunk too small
		if (size + static_cast<size_type>(aligned - m_chunkOffset) > static_cast<size_type>(m_chunkEnd - m_chunkOffset))
		{
			// Make sure new chunk is large enough for requested amount of memory + alignment
			size_type alignedSize = size + (Alignment - 1);

			size_type nextChunkSize = m_nextChunkSize;
			if (nextChunkSize < alignedSize)
				nextChunkSize = alignedSize;

			nextChunkSize += sizeof(chunk_header);

			char *nextChunkBase = static_cast<char*>( Heap::allocate<chunk_alignment>(nextChunkSize) );
			new( static_cast<void*>(nextChunkBase) ) chunk_header(m_chunk);

			m_chunk = nextChunkBase + sizeof(chunk_header);
			m_chunkOffset = m_chunk;
			m_chunkEnd = nextChunkBase + nextChunkSize;

			// Reset chunk size
			if (chunk_size != 0)
				m_nextChunkSize = chunk_size;

			// Get next free memory location
			aligned = align<Alignment>(m_chunkOffset);
		}

		// Memory now occupied
		m_chunkOffset = aligned + size;

		return aligned;
	}

public:
	/// Constructor.
	LEAN_INLINE chunk_heap(size_type chunkSize = ChunkSize)
		: m_chunk(m_firstChunk),
		m_chunkOffset(m_firstChunk),
		m_chunkEnd(m_firstChunk.get() + StaticChunkSize),
		m_nextChunkSize(chunkSize) { }
	/// Destructor
	LEAN_INLINE ~chunk_heap()
	{
		clear();
	}

	/// Sets the next chunk size.
	LEAN_INLINE void nextChunkSize(size_type nextSize) { m_nextChunkSize = nextSize; }
	/// Gets the next chunk size.
	LEAN_INLINE size_type nextChunkSize() const { return m_nextChunkSize; }

	/// Gets the remaining capacity of the current chunk.
	LEAN_INLINE size_type capacity() const { return m_chunkEnd - m_chunkOffset; }

	/// Tweaks the next chunk size to fit at least the given amount of objects about to be allocated.
	/// Passing 0 for @code minChunkSize@endcode will result in an exact fit of the given amount of objects.
	/// WARNING: When passing 0 for an exact fit, either call @code nextChunkSize()@endcode after
	/// you're done allocating or recall @code reserve()@endcode for sensible reallocation behavior
	/// in subsequent bulk allocations.
	LEAN_INLINE void reserve(size_type newCapacity, size_type minChunkSize = chunk_size)
	{
		size_type currentCapacity = capacity();

		if (newCapacity > currentCapacity) 
			nextChunkSize( max(newCapacity - currentCapacity, minChunkSize) );
	}

	/// Clears and frees all chunks allocated by this allocator.
	void clear()
	{
		// Free as many chunks as possible
		while (m_chunk != m_firstChunk)
		{
			chunk_header *freeChunkBase = to_chunk_header(m_chunk);

			// Immediately store previous chunk in case clear is re-called after exception
			m_chunk = freeChunkBase->prev_chunk;
			m_chunkOffset = m_chunk;
			m_chunkEnd = m_chunk;

			Heap::free<chunk_alignment>(freeChunkBase);
		}

		// Re-initialize with first chunk
		m_chunkEnd = m_chunk + StaticChunkSize;
	}

	/// Clears all chunks and frees all chunks but the first one dynamically allocated by this allocator if it has not been exhausted yet.
	void clearButFirst()
	{
		if (m_chunk != m_firstChunk && to_chunk_header(m_chunk)->prev_chunk == m_firstChunk)
			// Re-initialize first dynamic chunk
			m_chunkOffset = m_chunk;
		else
			clear();
	}

	/// Gets whether the current chunk is static. For advanced clean-up logic only.
	bool currentStatic()
	{
		return (m_chunk == m_firstChunk);
	}
	/// Gets the current chunk offset. For advanced clean-up logic only.
	char* currentOffset()
	{
		return m_chunkOffset;
	}
	/// Clears the current chunk but frees none. For advanced clean-up logic only.
	char* clearCurrent()
	{
		// Re-initialize with current chunk
		m_chunkOffset = m_chunk;
		return m_chunk;
	}
	/// Clears all chunks and frees the current chunk, returning the next. For advanced clean-up logic only.
	char* clearNext()
	{
		if (m_chunk != m_firstChunk)
		{
			chunk_header *freeChunkBase = to_chunk_header(m_chunk);

			// Immediately store previous chunk (exception-safe)
			m_chunk = freeChunkBase->prev_chunk;
			m_chunkOffset = m_chunk;
			m_chunkEnd = (m_chunk == m_firstChunk) ? m_chunk + StaticChunkSize : m_chunk;

			Heap::free<chunk_alignment>(freeChunkBase);
		}

		return m_chunk;
	}

	/// Allocates the given amount of memory.
	LEAN_INLINE void* allocate(size_type size) { return allocate<default_alignment>(size); }
	/// Frees the given block of memory.
	LEAN_INLINE void free(void *memory) { free<default_alignment>(memory); }

	/// Allocates the given amount of memory respecting the given alignment.
	template <size_t Alignment>
	LEAN_INLINE void* allocate(size_type size)
	{
		return allocate_aligned<Alignment>(size);
	}
	/// Frees the given aligned block of memory.
	template <size_t Alignment>
	LEAN_INLINE void free(void *memory)
	{
		// Freeing of individual memory blocks unsupported
	}

	/// Swaps the contents of the given chunk heap with the ones of this chunk heap.
	LEAN_INLINE void swap(chunk_heap &right)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(StaticChunkSize == 0,
			"Swap only supported for purely dynamic chunk heaps.",
			Swap_only_supported_for_purely_dynamic_chunk_heaps);

		using std::swap;

		swap(m_chunk, right.m_chunk);
		swap(m_chunkOffset, right.m_chunkOffset);
		swap(m_chunkEnd, right.m_chunkEnd);
		swap(m_nextChunkSize, right.m_nextChunkSize);
	}
};

/// Swaps the contents of the given chunk heap with the ones of this chunk heap.
template <size_t ChunkSize, class Heap, size_t StaticChunkSize, size_t DefaultAlignment>
LEAN_INLINE void swap(
	chunk_heap<ChunkSize, Heap, StaticChunkSize, DefaultAlignment> &left,
	chunk_heap<ChunkSize, Heap, StaticChunkSize, DefaultAlignment> &right)
{
	left.swap(right);
}

} // namespace

using memory::chunk_heap;

} // namespace

#endif