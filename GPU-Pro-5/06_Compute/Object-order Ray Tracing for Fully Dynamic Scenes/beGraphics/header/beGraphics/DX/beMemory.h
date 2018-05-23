/******************************************************/
/* breeze Engine Graphics Module (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_GRAPHICS_MEMORY_DX
#define BE_GRAPHICS_MEMORY_DX

#include "beGraphics.h"
#include "../beMemory.h"
#include <D3DCommon.h>
#include <lean/smart/com_ptr.h>
#include <lean/concurrent/atomic.h>

namespace beGraphics
{

namespace DX
{

// Memory implementation.
template <class Element>
class Memory : public beCore::Shared, beGraphics::Memory<Element>
{
private:
	lean::com_ptr<ID3DBlob> m_pBlob;
	
	mutable long m_references;
	
public:
	/// Constructor.
	Memory(lean::com_ptr<ID3DBlob> pBlob)
		: m_pBlob(pBlob),
		m_references(1)
	{
		LEAN_ASSERT(m_pBlob != nullptr);
	}
	/// Copy constructor.
	Memory(const Memory &right)
		: m_pBlob(right.m_pBlob),
		m_references(1) { }
	/// Assignment operator.
	Memory& operator =(const Memory &right)
	{
		m_pBlob = right.m_pBlob;
		return *this;
	}
	
	/// Gets a pointer to the data buffer.
	Element* GetData()
	{
		return static_cast<Element*>(m_pBlob->GetBufferPointer());
	}
	/// Gets a pointer to the data buffer.
	const Element* GetData() const
	{
		return static_cast<const Element*>(m_pBlob->GetBufferPointer());
	}
	/// Gets the data buffer size (in elements).
	size_t GetSize() const
	{
		return static_cast<size_t>(m_pBlob->GetBufferSize());
	}
	
	// Increses the reference count of this resource.
	long AddRef() const
	{
		return lean::atomic_increment(m_references);
	}
	// Decreases the reference count of this resource, destroying the resource when the count reaches zero.
	long Release() const
	{
		if (lean::atomic_decrement(m_references) == 0)
		{
			delete this;
			return 0;
		}
		return m_references;
	}

	/// Gets the stored blob.
	ID3DBlob* GetBlob() const { return m_pBlob; }
};

} // namespace

} // namespace

#endif