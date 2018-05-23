//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "D3D11EffectsLite.h"
#include <new>
#include <memory>

namespace D3DEffectsLite
{

struct OutOfMemory { };
struct OutOfBounds : public OutOfMemory { };

static const UINT HeapAlignment = sizeof(uintptr_t);

/// Alignment required to be power of two!
template <class Integer>
inline Integer AlignInteger(Integer integer, Integer alignment = HeapAlignment)
{
	assert((alignment & (alignment - 1)) == 0);
	return (integer + (alignment - 1)) & ~(alignment - 1);
}

/// Alignment required to be power of two!
template <class Value>
inline Value* Align(Value *pointer, size_t alignment = HeapAlignment)
{
	return reinterpret_cast<Value*>( AlignInteger(reinterpret_cast<uintptr_t>(pointer), alignment) );
}

class Heap
{
	BYTE *m_bytes;
	UINT m_byteCount;
	UINT m_top;

	void PreAllocate(size_t size);
	void* PostAllocate(size_t size);

public:
	Heap();
	~Heap();

	void Allocate();
	bool PreAllocation() const { return (m_bytes == nullptr); }

	void* Reserve(size_t size);

	template <class T>
	void* Reserve()
	{
		return Reserve(sizeof(T));
	}
	template <class T>
	void* ReserveMultiple(size_t count)
	{
		return Reserve(sizeof(T) * count);
	}

	template <class T>
	T* NoConstruct()
	{
		return static_cast<T*>( Reserve(sizeof(T)) );
	}
	template <class T>
	T* NoConstructMultiple(size_t count)
	{
		return static_cast<T*>( Reserve(sizeof(T) * count) );
	}

	template <class T>
	T* Construct()
	{
		if (!PreAllocation())
		{
			UINT oldTop = m_top;
			
			try
			{
				return new( PostAllocate(sizeof(T)) ) T();
			}
			catch(...)
			{
				m_top = oldTop;
				throw;
			}
		}
		else
		{
			PreAllocate(sizeof(T));
			return nullptr;
		}
	}
	template <class T>
	T* ConstructMultiple(size_t count)
	{
		if (!PreAllocation())
		{
			UINT oldTop = m_top;

			try
			{
				T *objs = static_cast<T*>( PostAllocate(sizeof(T) * count) );

				for (size_t i = 0; i < count; ++i)
					new( static_cast<void*>(objs + i) ) T();

				return objs;
			}
			catch(...)
			{
				m_top = oldTop;
				throw;
			}
		}
		else
		{
			PreAllocate(sizeof(T) * count);
			return nullptr;
		}
	}
};

inline void* MoveData(Heap &heap, const void *bytes, size_t byteCount)
{
	void* p = heap.NoConstructMultiple<BYTE>(byteCount);
	if (p) memcpy(p, bytes, byteCount);
	return p;
}

template <class T>
inline T* MoveMultiple(Heap &heap, const T *data, size_t count)
{
	T* p = heap.NoConstructMultiple<T>(count);
	if (p) memcpy(p, data, count * sizeof(T));
	return p;
}

template <class T>
inline T* PackData(Heap &heap, const void *bytes, size_t stride, size_t count)
{
	T* p = heap.NoConstructMultiple<T>(count);
	if (p)
		for (size_t i = 0; i < count; ++i)
			new( static_cast<void*>(p + i) ) T( *reinterpret_cast<const T*>(static_cast<const BYTE*>(p) + i * stride) );
	return p;
}

inline char* MoveString(Heap &heap, const char *string)
{
	return static_cast<char*>( MoveData(heap, string, strlen(string) + 1) );
}

} // namespace
