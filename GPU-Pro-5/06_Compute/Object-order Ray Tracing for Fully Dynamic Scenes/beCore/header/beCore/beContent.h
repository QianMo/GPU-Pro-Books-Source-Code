/******************************************************/
/* breeze Engine Core Module     (c) Tobias Zirr 2011 */
/******************************************************/

#pragma once
#ifndef BE_CORE_CONTENT
#define BE_CORE_CONTENT

#include "beCore.h"
#include "beShared.h"
#include <lean/concurrent/atomic.h>

namespace beCore
{

/// Content interface.
class Content : public Shared
{
private:
	mutable long m_references;

protected:
	const void *m_memory;
	uint8 m_size;

	/// Copy constructor. DOES NOTHING.
	LEAN_INLINE Content(const Content&)
		: m_references(1),
		m_memory(),
		m_size() { }
	/// Assignment operator. DOES NOTHING.
	LEAN_INLINE Content& operator =(const Content&) { return *this; }

	/// Constructor.
	LEAN_INLINE Content()
		: m_references(1),
		m_memory(),
		m_size() { }
	/// Destructor.
	virtual ~Content() throw() { }

public:
	/// Gets the memory.
	LEAN_INLINE const void* Data() const { return m_memory; }
	/// Gets the memory.
	LEAN_INLINE const char* Bytes() const { return reinterpret_cast<const char*>(m_memory); }
	/// Gets the memory.
	LEAN_INLINE uint8 Size() const { return m_size; }

	/// Increments the reference count.
	long AddRef() const
	{
		return lean::atomic_increment(m_references);
	}
	/// Decrements the reference count.
	long Release() const
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

#endif