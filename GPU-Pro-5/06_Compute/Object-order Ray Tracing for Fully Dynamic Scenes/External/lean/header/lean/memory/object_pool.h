/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_OBJECT_POOL
#define LEAN_MEMORY_OBJECT_POOL

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "chunk_heap.h"
#include "default_heap.h"

namespace lean
{
namespace memory
{

/// Enhances the chunk_heap by proper object deconstruction.
template <class Element, size_t ChunkSize, class Heap = default_heap, size_t StaticChunkSize = ChunkSize, size_t Alignment = alignof(Element)>
class object_pool : public lean::noncopyable
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
	typedef chunk_heap<0, Heap, StaticChunkSize * sizeof(Element), Alignment> chunk_heap;
	chunk_heap m_heap;

public:
	/// Constructor.
	LEAN_INLINE object_pool(size_type chunkSize = ChunkSize)
		// WARNING: Include alignment, otherwise aligned objects in small chunks might not be destructed
		: m_heap( max(chunkSize * sizeof(Element), sizeof(Element) + (Alignment - 1)) ) { }
	/// Destructs all objects in this pool.
	LEAN_INLINE ~object_pool()
	{
		clear();
	}

	/// Destructs all objects and frees all chunks allocated by this allocator.
	void clear()
	{
		char *chunkEnd = m_heap.currentOffset();
		char *chunkBegin = m_heap.clearCurrent();

		while (chunkBegin)
		{
			const size_t chunkSize = (chunkEnd - chunkBegin);

			char *current = align<Alignment>(chunkBegin);

			while (sizeof(Element) + static_cast<size_t>(current - chunkBegin) <= chunkSize)
			{
				reinterpret_cast<Element*>(current)->~Element();

				current = align<Alignment>(current + sizeof(Element));
			}

			char *nextChunkBegin = m_heap.clearNext();

			if (nextChunkBegin != chunkBegin)
			{
				chunkBegin = nextChunkBegin;
				chunkEnd = chunkBegin
					+ ((m_heap.currentStatic()) ? StaticChunkSize * sizeof(Element) : m_heap.nextChunkSize());
			}
			else
				chunkBegin = nullptr;
		}
	}

	/// Allocates a new element in the object pool. Object MUST BE CONSTRUCTED, WILL BE DESTRUCTED.
	LEAN_INLINE void* allocate()
	{
		return m_heap.allocate<Alignment>( sizeof(Element) );
	}
	/// Places the given value into this object pool. Copy construction MAY NOT THROW, object WILL BE DESTRUCTED.
	LEAN_INLINE Element* place(const Element &value) noexcept
	{
		return new( m_heap.allocate<Alignment>( sizeof(Element) ) ) Element(value);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Places the given value into this object pool. Move construction MAY NOT THROW, object WILL BE DESTRUCTED.
	LEAN_INLINE Element* place(Element &&value) noexcept
	{
		return new( m_heap.allocate<Alignment>( sizeof(Element) ) ) Element( std::move(value) );
	}
#endif
};

} // namespace

using memory::object_pool;

} // namespace

#endif