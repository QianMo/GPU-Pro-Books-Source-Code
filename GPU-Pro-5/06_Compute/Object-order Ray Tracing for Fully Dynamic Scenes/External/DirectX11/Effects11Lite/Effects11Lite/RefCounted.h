//////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) Tobias Zirr.  All Rights Reserved.
//
//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "D3D11EffectsLite.h"
#include "Allocator.h"

namespace D3DEffectsLite
{

template <class Base>
class RefCounted : public Base
{
	ULONG ReferenceCounter;

protected:
	RefCounted() : ReferenceCounter(1) { }
	RefCounted(const RefCounted&) : ReferenceCounter(1) { }
	RefCounted& operator =(const RefCounted&) { return *this; }
	
public:
	virtual ~RefCounted() { }

	HRESULT D3DEFFECTSLITE_STDCALL QueryInterface(REFIID, void **ppvObject)
	{
		assert(ppvObject);
		*ppvObject = this;
		AddRef();
		return S_OK;
	}
	ULONG D3DEFFECTSLITE_STDCALL AddRef()
	{
		D3DEFFECTSLITE_INTERLOCKED_INCREMENT(ReferenceCounter);
		return ReferenceCounter;
	}
	ULONG D3DEFFECTSLITE_STDCALL Release()
	{
		if (D3DEFFECTSLITE_INTERLOCKED_DECREMENT(ReferenceCounter) == 0)
		{
			AllocatorDelete(*GetGlobalAllocator(), this);
			return 0;
		}

		return ReferenceCounter;
	}
};

} // namespace
