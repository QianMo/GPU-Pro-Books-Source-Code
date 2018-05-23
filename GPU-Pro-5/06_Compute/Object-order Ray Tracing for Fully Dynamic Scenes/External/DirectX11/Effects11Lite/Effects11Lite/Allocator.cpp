//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3D11EffectsLite.h"
#include "Allocator.h"

namespace D3DEffectsLite
{

namespace
{

struct DefaultAllocator : public Allocator
{
	HANDLE hProcessHeap;

	DefaultAllocator()
		: hProcessHeap( GetProcessHeap() ) { }

	void* D3DEFFECTSLITE_STDCALL Allocate(UINT size)
	{
		return ::HeapAlloc(hProcessHeap, 0, size);
	}
	void D3DEFFECTSLITE_STDCALL Free(void *data)
	{
		if (data != nullptr)
			::HeapFree(hProcessHeap, 0, data);
	}
};

Allocator*& GlobalAllocatorPointer()
{
	static Allocator *globalAllocatorPointer = GetDefaultAllocator();
	return globalAllocatorPointer;
}

} // namespace

// Gets the default allocator.
Allocator* D3DEFFECTSLITE_STDCALL GetDefaultAllocator()
{
	static DefaultAllocator defaultAllocator;
	return &defaultAllocator;
}

// Sets the global allocator.
void D3DEFFECTSLITE_STDCALL SetGlobalAllocator(Allocator *allocator)
{
	assert(allocator);
	GlobalAllocatorPointer() = allocator;
}

// Gets the global allocator.
Allocator* D3DEFFECTSLITE_STDCALL GetGlobalAllocator()
{
	return GlobalAllocatorPointer();
}

} // namespace

// Gets the default allocator.
D3DEffectsLiteAllocator* D3DEFFECTSLITE_STDCALL D3DELGetDefaultAllocator()
{
	return D3DEffectsLite::GetDefaultAllocator();
}

// Sets the global allocator.
void D3DEFFECTSLITE_STDCALL D3DELSetGlobalAllocator(D3DEffectsLiteAllocator *allocator)
{
	D3DEffectsLite::SetGlobalAllocator(allocator);
}

// Gets the global allocator.
D3DEffectsLiteAllocator* D3DEFFECTSLITE_STDCALL D3DELGetGlobalAllocator()
{
	return D3DEffectsLite::GetGlobalAllocator();
}

// Standard requires these to be non-inline
void* operator new(size_t size, D3DEffectsLite::Allocator &alloc)
{
	assert(size <= UINT(-1));
	return alloc.Allocate( static_cast<UINT>(size) );
}

void operator delete(void *bytes, D3DEffectsLite::Allocator &alloc)
{
	alloc.Free(bytes);
}
