/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_COM_PTR
#define LEAN_SMART_COM_PTR

#include "../lean.h"
#include "common.h"

namespace lean
{
namespace smart
{

/// Acquires a reference to the given COM object.
template <class COMType>
LEAN_INLINE void acquire_com(COMType &object)
{
	object.AddRef();
}
/// Releases a reference to the given COM object.
template <class COMType>
LEAN_INLINE void release_com(COMType *object)
{
	if (object)
		object->Release();
}

/// Generic com pointer policy.
template <class Type>
struct generic_com_policy
{
	generic_com_policy() { }
	template <class OtherType>
	generic_com_policy(const generic_com_policy<OtherType>&) { }

	/// Calls @code acquire_com@endcode on the given object.
	static LEAN_INLINE void acquire(Type &object)
	{
		acquire_com(object);
	}
	/// Calls @code release_com@endcode on the given object.
	static LEAN_INLINE void release(Type *object)
	{
		release_com(object);
	}
};

/// COM pointer class that performs reference counting on COM objects of the given type.
template < class COMType, bool Critical = false, class Policy = generic_com_policy<COMType> >
class com_ptr
{
public:
	/// Type of the COM object stored by this COM pointer.
	typedef COMType com_type;
	/// Type of the pointer stored by this COM pointer.
	typedef COMType* value_type;

private:
	com_type *m_object;

	/// Acquires the given object.
	static com_type* acquire(com_type *object)
	{
		if (object)
			Policy::acquire(*object);

		return object;
	}

	/// Releases the given object.
	static void release(com_type *object)
	{
		Policy::release(object);
	}

public:
	/// Constructs a COM pointer from the given COM object.
	com_ptr(com_type *object = nullptr)
		: m_object( acquire(object) ) { };
	/// Constructs a COM pointer from the given COM object.
	template <class COMType2>
	com_ptr(COMType2 *object)
		: m_object( acquire(object) ) { };

	/// Constructs a COM pointer from the given COM pointer.
	com_ptr(const com_ptr &right)
		: m_object( acquire(right.m_object) ) { };
	/// Constructs a COM pointer from the given COM pointer.
	template <class COMType2, bool Critical2, class Policy2>
	com_ptr(const com_ptr<COMType2, Critical2, Policy2> &right)
		: m_object( acquire(right.get()) ) { };

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a COM pointer from the given COM pointer.
	template <class COMType2, bool Critical2, class Policy2>
	com_ptr(com_ptr<COMType2, Critical2, Policy2> &&right) noexcept
		: m_object(right.unbind())
	{
		Policy assertSameReleasePolicy((Policy2()));
		(void) assertSameReleasePolicy;
	}
#endif
	
	/// Constructs a COM pointer from the given COM object without incrementing its reference count.
	com_ptr(com_type *object, bind_reference_t) noexcept
		: m_object(object) { };

	/// Destroys the COM pointer.
	~com_ptr()
	{
		release(m_object);
	}

	/// Binds the given COM object reference to this COM pointer.
	static LEAN_INLINE com_ptr<com_type, true> bind(com_type *object)
	{
		// Visual C++ won't inline delegating function calls
		return com_ptr<com_type, true>(object, bind_reference);
	}
	/// Transfers the COM object reference held by this COM pointer to a new COM pointer.
	LEAN_INLINE com_ptr<com_type, true> transfer()
	{
		// Visual C++ won't inline delegating function calls
		return com_ptr<com_type, true>(unbind(), bind_reference);
	}

	/// Replaces the component reference held by this COM pointer by the given reference.
	void rebind(com_type *object)
	{
		com_type *prevObject = m_object;
		m_object = object;
		release(prevObject);
	}
	/// Unbinds the component reference held by this COM pointer.
	LEAN_INLINE com_type* unbind()
	{
		com_type *preObject = m_object;
		m_object = nullptr;
		return preObject;
	}
	/// Replaces the component reference held by this COM pointer by a new reference to the given component.
	LEAN_INLINE void reset(com_type *object)
	{
		// Do not check for redundant assignment
		// -> The code handles redundant assignment just fine
		// -> Checking generates up to twice the code due to unfortunate compiler optimization application order
		rebind(acquire(object));
	}
	/// Releases the component reference held by this pointer.
	LEAN_INLINE void release()
	{
		rebind(nullptr);
	}

	/// Replaces the stored COM object with the given object. <b>[ESA]</b>
	com_ptr& operator =(com_type *object)
	{
		reset(object);
		return *this;
	}
	/// Replaces the stored COM object with one stored by the given COM pointer. <b>[ESA]</b>
	com_ptr& operator =(const com_ptr &right)
	{
		reset(right.m_object);
		return *this;
	}
	/// Replaces the stored COM object with one stored by the given COM pointer. <b>[ESA]</b>
	template <class COMType2, bool Critical2, class Policy2>
	com_ptr& operator =(const com_ptr<COMType2, Critical2, Policy2> &right)
	{
		reset(right.get());
		return *this;
	}

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Replaces the stored COM object with the one stored by the given r-value COM pointer. <b>[ESA]</b>
	template <class COMType2, bool Critical2, class Policy2>
	com_ptr& operator =(com_ptr<COMType2, Critical2, Policy2> &&right) noexcept
	{
		Policy assertSameReleasePolicy((Policy2()));
		(void) assertSameReleasePolicy;

		// Self-assignment would be wrong
		if ((void*) this != (void*) &right)
			rebind(right.unbind());

		return *this;
	}
#endif

	/// Gets the COM object stored by this COM pointer.
	LEAN_INLINE com_type*const& get(void) const { return m_object; };

	/// Gets the COM object stored by this COM pointer.
	LEAN_INLINE com_type& operator *() const { return *m_object; };
	/// Gets the COM object stored by this COM pointer.
	LEAN_INLINE com_type* operator ->() const { return m_object; };
	/// Gets the COM object stored by this COM pointer (getter compatibility).
	LEAN_INLINE com_type* operator ()() const { return m_object; }

	/// Gets the COM object stored by this COM pointer.
	LEAN_INLINE operator com_type*() const
	{
		LEAN_STATIC_ASSERT_MSG_ALT(!Critical,
			"Cannot implicitly cast critical reference, use unbind() for (insecure) storage.",
			Cannot_implicitly_cast_critical_reference__use_unbind_for_insecure_storage);
		return m_object;
	};

	/// Gets a double-pointer allowing for COM-style object retrieval. The pointer returned may
	/// only ever be used until the next call to one of this COM pointer's methods.
	LEAN_INLINE com_type** rebind()
	{
		release();
		return &m_object;
	}

	/// Swaps the given pointers.
	void swap(com_ptr& right)
	{
		value_type prevObject = m_object;
		m_object = right.m_object;
		right.m_object = prevObject;
	}
};

/// Swaps the given pointers.
template <class T, bool C, class P>
LEAN_INLINE void swap(com_ptr<T, C, P> &left, com_ptr<T, C, P> &right)
{
	left.swap(right);
}

/// Binds the given COM reference to a new COM pointer.
template <class COMType>
LEAN_INLINE com_ptr<COMType, true> bind_com(COMType *object)
{
	// Visual C++ won't inline delegating function calls
	return com_ptr<COMType, true>(object, bind_reference);
}

} // namespace

using smart::com_ptr;

using smart::bind_com;

} // namespace

#endif