/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_BLOB_DX
#define BE_GRAPHICS_BLOB_DX

#include "beGraphics.h"
#include <D3DCommon.h>
#include <lean/concurrent/atomic.h>

namespace beGraphics
{

namespace DX
{

/// Base class for Direct3D blobs.
class Blob : public ID3DBlob
{
private:
	long m_references;

public:
	/// Constructor.
	explicit Blob(long references = 1)
		: m_references(references) { }
	virtual ~Blob() { }

	/// Query interface implementation.
	HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void **ppvObject)
	{
		if (riid == IID_IUnknown)
			*ppvObject = static_cast<IUnknown*>(this);
		// TODO: unresolved?!?
//		if (riid == IID_ID3DBlob)
//			*ppvObject = static_cast<ID3DBlob*>(this);
		else
		{
			*ppvObject = nullptr;
			return E_NOINTERFACE;
		}

		AddRef();
		return S_OK;
	}

	/// (Atomically) increments the reference counter.
	ULONG STDMETHODCALLTYPE AddRef(void)
	{
		return lean::atomic_increment(m_references);
	}

	/// (Atomically) decrements the reference counter.
	ULONG STDMETHODCALLTYPE Release(void)
	{
		if (lean::atomic_decrement(m_references) == 0)
		{
			delete this;
			return 0;
		}
		return m_references;
	}
};

} // namespace

} // namespace

#endif