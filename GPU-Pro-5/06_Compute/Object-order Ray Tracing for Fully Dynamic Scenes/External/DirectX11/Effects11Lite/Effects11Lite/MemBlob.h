//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "D3D11EffectsLite.h"
#include "RefCounted.h"

namespace D3DEffectsLite
{

struct MemBlob : public RefCounted<Blob>
{
	void *Bytes;
	UINT ByteCount;
	ULONG ReferenceCounter;

	MemBlob(const void *bytes, UINT byteCount);
	~MemBlob();

	void* D3DEFFECTSLITE_STDCALL Data() const;
	UINT D3DEFFECTSLITE_STDCALL Size() const;
};

com_ptr<MemBlob> CreateBlob(const void *bytes, UINT byteCount);

} // namespace
