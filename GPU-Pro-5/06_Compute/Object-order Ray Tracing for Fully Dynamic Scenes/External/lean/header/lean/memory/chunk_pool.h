/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_CHUNK_POOL
#define LEAN_MEMORY_CHUNK_POOL

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "chunk_heap.h"

namespace lean
{
namespace memory
{

/// Contiguous chunk allocator heap.
template <class Element, size_t ChunkSize, class Heap = default_heap, size_t StaticChunkSize = ChunkSize, size_t Alignment = alignof(Element)>
class chunk_pool : public lean::noncopyable
{
public:
	/// Value type.
	typedef Element value_type;
	/// Heap type.
	typedef Heap heap_type;
	/// Size type.
	typedef typename heap_type::size_type size_type;
	/// Chunk size.
	static const size_type chunk_size = ChunkSize;
	/// Alignment.
	static const size_type alignment = Alignment;

private:
	typedef chunk_heap<ChunkSize * sizeof(Element), Heap, StaticChunkSize * sizeof(Element), Alignment> chunk_heap;
	chunk_heap m_heap;

	/// Free element node
	struct free_node
	{
		free_node *next;

		/// Constructor.
		free_node(free_node *next)
			: next(next) { }
	};

	LEAN_STATIC_ASSERT_MSG_ALT(
		sizeof(Element) >= sizeof(free_node),
		"Inline free list requires elements greater or equal to pointers",
		Inline_free_list_requires_elements_greater_or_equal_to_pointers );

	// Free list head
	free_node *m_freeHead;

public:
	/// Constructor.
	LEAN_INLINE chunk_pool(size_type chunkSize = ChunkSize)
		: m_heap( chunkSize * sizeof(Element) ),
		m_freeHead(nullptr) { }

	/// Sets the next chunk size.
	LEAN_INLINE void nextChunkSize(size_type nextSize) { m_heap.nextChunkSize( nextSize * sizeof(Element) ); }
	/// Gets the next chunk size.
	LEAN_INLINE size_type nextChunkSize() const { return m_heap.nextChunkSize() / sizeof(Element); }

	/// Gets the remaining capacity of the current chunk.
	LEAN_INLINE size_type capacity() const { return m_heap.capacity() / sizeof(Element); }

	/// Tweaks the next chunk size to fit at least the given amount of objects about to be allocated.
	/// Passing 0 for @code minChunkSize@endcode will result in an exact fit of the given amount of objects.
	/// WARNING: When passing 0 for an exact fit, either call @code nextChunkSize()@endcode after
	/// you're done allocating or recall @code reserve()@endcode for sensible reallocation behavior
	/// in subsequent bulk allocations.
	LEAN_INLINE void reserve(size_type newCapacity, size_type minChunkSize = chunk_size)
	{
		m_heap.reserve( newCapacity * sizeof(Element), minChunkSize * sizeof(Element) );
	}

	/// Clears and frees all chunks allocated by this allocator.
	void clear()
	{
		m_freeHead = nullptr;
		m_heap.clear();
	}
	
	/// Clears all chunks and frees all chunks but the first one dynamically allocated by this allocator if it has not been exhausted yet.
	void clearButFirst()
	{
		m_freeHead = nullptr;
		m_heap.clearButFirst();
	}

	/// Allocates one element.
	LEAN_INLINE void* allocate()
	{
		void *freeElement;

		if (m_freeHead)
		{
			freeElement = m_freeHead;
			m_freeHead = m_freeHead->next;
		}
		else
			freeElement = m_heap.allocate<alignment>( sizeof(Element) );

		return freeElement;
	}
	/// Frees the given element.
	LEAN_INLINE void free(void *memory)
	{
		m_freeHead = new(memory) free_node(m_freeHead);
	}

	/// Swaps the contents of the given chunk heap with the ones of this chunk heap.
	LEAN_INLINE void swap(chunk_pool &right)
	{
		using std::swap;

		m_heap.swap(right.m_heap);
		swap(m_freeHead, right.m_freeHead);
	}
};

/// Swaps the contents of the given chunk heap with the ones of this chunk heap.
template <class Element, size_t ChunkSize, class Heap, size_t StaticChunkSize, size_t Alignment>
LEAN_INLINE void swap(
	chunk_pool<Element, ChunkSize, Heap, StaticChunkSize, Alignment> &left,
	chunk_pool<Element, ChunkSize, Heap, StaticChunkSize, Alignment> &right)
{
	left.swap(right);
}

} // namespace

using memory::chunk_pool;

} // namespace

#endif