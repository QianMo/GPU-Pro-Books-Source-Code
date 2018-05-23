//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3D11EffectsLite.h"
#include "Heap.h"
#include <cassert>

namespace D3DEffectsLite
{

Heap::Heap()
	: m_bytes(),
	m_byteCount(),
	m_top()
{
}

Heap::~Heap()
{
	GetGlobalAllocator()->Free(m_bytes);
}

namespace
{

inline UINT CheckedUIntSizeAlign(size_t size)
{
	if (size > UINT(-1) - HeapAlignment)
		throw OutOfBounds();

	return AlignInteger(static_cast<UINT>(size), HeapAlignment);
}

} // namespace

void Heap::PreAllocate(size_t size)
{
	assert(m_bytes == nullptr);

	UINT alignedSize = CheckedUIntSizeAlign(size);

	if (m_byteCount > UINT(-1) - alignedSize)
		throw OutOfBounds();

	m_byteCount += alignedSize;
}

void Heap::Allocate()
{
	assert(m_bytes == nullptr);

	m_bytes = static_cast<BYTE*>( GetGlobalAllocator()->Allocate(m_byteCount + HeapAlignment) );

	if (!m_bytes)
		throw OutOfMemory();
}

void* Heap::PostAllocate(size_t size)
{
	assert(m_bytes != nullptr);

	UINT alignedSize = CheckedUIntSizeAlign(size);

	if (alignedSize > m_byteCount || m_top > m_byteCount - alignedSize)
		throw OutOfBounds();

	void *memory = Align(m_bytes, HeapAlignment) + m_top;
	m_top += alignedSize;
	return memory;
}

void* Heap::Reserve(size_t size)
{
	return (!PreAllocation())
		? PostAllocate(size)
		: (PreAllocate(size), nullptr);
}

} // namespace
