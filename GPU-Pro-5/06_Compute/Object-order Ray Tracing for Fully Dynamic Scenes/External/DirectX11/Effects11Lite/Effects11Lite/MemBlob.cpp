//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr. All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include "D3D11EffectsLite.h"
#include "MemBlob.h"
#include "Allocator.h"

namespace D3DEffectsLite
{

MemBlob::MemBlob(const void *bytes, UINT byteCount)
	: Bytes(GetGlobalAllocator()->Allocate(byteCount)),
	ByteCount(byteCount),
	ReferenceCounter(1)
{
	memcpy(Bytes, bytes, byteCount);
}

MemBlob::~MemBlob()
{
	GetGlobalAllocator()->Free(Bytes);
}

void* D3DEFFECTSLITE_STDCALL MemBlob::Data() const
{
	return Bytes;
}

UINT D3DEFFECTSLITE_STDCALL MemBlob::Size() const
{
	return ByteCount;
}

com_ptr<MemBlob> CreateBlob(const void *bytes, UINT byteCount)
{
	return new(*GetGlobalAllocator()) MemBlob(bytes, byteCount);
}

} // namespace
