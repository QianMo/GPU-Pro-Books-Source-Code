/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_SCOPED_PTR
#define LEAN_SMART_SCOPED_PTR

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "../tags/move_ptr.h"
#include "../meta/strip.h"
#include "../functional/variadic.h"
#include "common.h"

namespace lean
{
namespace smart
{

/// Deletes the given object (default policy implementation).
template <class Type>
LEAN_INLINE void release_ptr(const Type *object)
{
	if (sizeof(Type) > 0)
		delete object;
}

/// Deletes the given array (default policy implementation).
template <class Type>
LEAN_INLINE void release_array_ptr(const Type *object)
{
	if (sizeof(Type) > 0)
		delete[] object;
}

/// Generic scoped pointer policy.
template <class Type>
struct generic_ptr_policy
{
	/// Calls @code release_ptr@endcode on the given object.
	static LEAN_INLINE void release(Type *object)
	{
		release_ptr(object);
	}

	generic_ptr_policy() { }
	template <class OtherType>
	generic_ptr_policy(const generic_ptr_policy<OtherType>&) { }

private:
	template <class OtherType>
	generic_ptr_policy(const generic_ptr_policy<OtherType[]>&);
};
template <class Type>
struct generic_ptr_policy<Type[]>
{
	static LEAN_INLINE void release(Type *object)
	{
		release_array_ptr(object);
	}

	generic_ptr_policy() { }
	template <class OtherType>
	generic_ptr_policy(const generic_ptr_policy<OtherType[]>&) { }
};

/// Delete scoped pointer policy.
template <class Type>
struct delete_ptr_policy
{
	/// Deletes the given object.
	static LEAN_INLINE void release(Type *object)
	{
		if (sizeof(Type) > 0)
			delete object;
	}

	delete_ptr_policy() { }
	template <class OtherType>
	delete_ptr_policy(const delete_ptr_policy<OtherType>&) { }

private:
	template <class OtherType>
	delete_ptr_policy(const delete_ptr_policy<OtherType[]>&);
};
template <class Type>
struct delete_ptr_policy<Type[]>
{
	static LEAN_INLINE void release(Type *object)
	{
		if (sizeof(Type) > 0)
			delete[] object;
	}

	delete_ptr_policy() { }
	template <class OtherType>
	delete_ptr_policy(const delete_ptr_policy<OtherType[]>&) { }
};

/// Destroy scoped pointer policy.
template <class Type>
struct destroy_ptr_policy
{
	/// Calls @code object->destroy()@endcode.
	static LEAN_INLINE void release(Type *object)
	{
		if (object)
			object->destroy();
	}

	destroy_ptr_policy() { }
	template <class OtherType>
	destroy_ptr_policy(const destroy_ptr_policy<OtherType>&) { }
};
template <class Type>
struct destroy_ptr_policy<Type[]>;

/// Release scoped pointer policy.
template <class Type>
struct release_ptr_policy
{
	/// Calls @code object->release()@endcode.
	static LEAN_INLINE void release(Type *object)
	{
		if (object)
			object->release();
	}

	release_ptr_policy() { }
	template <class OtherType>
	release_ptr_policy(const release_ptr_policy<OtherType>&) { }
};
template <class Type>
struct release_ptr_policy<Type[]>;

/// Scoped pointer class that releases the object pointed to on destruction.
template < class Type, reference_state_t RefState = stable_ref, class ReleasePolicy = generic_ptr_policy<Type> >
class scoped_ptr : public noncopyable
{
public:
	/// Type of the object pointed to.
	typedef typename lean::strip_array<Type>::type object_type;
	/// Type of the pointer stored by this scoped pointer.
	typedef object_type* value_type;

private:
	value_type m_object;

public:
#ifndef LEAN0X_NO_NULLPTR
	/// Constructs a nullptr.
	LEAN_INLINE scoped_ptr(nullptr_t)
		: m_object( nullptr ) { }
#else
	/// Constructs a nullptr.
	LEAN_INLINE scoped_ptr(int)
		: m_object( nullptr ) { }
#endif
	
	/// Constructs a scoped pointer from the given object pointer.
	LEAN_INLINE explicit scoped_ptr(object_type *object = nullptr)
		: m_object( object ) { }
	/// Constructs a scoped pointer from the given object pointer.
	template <class Type2>
	LEAN_INLINE explicit scoped_ptr(Type2 *object)
		: m_object( object ) { }
	
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a scoped pointer from the given scoped pointer.
	template <class Type2, reference_state_t RefState2, class ReleasePolicy2>
	LEAN_INLINE scoped_ptr(scoped_ptr<Type2, RefState2, ReleasePolicy2> &&right) noexcept
		: m_object( right.detach() )
	{
		ReleasePolicy assertSameReleasePolicy((ReleasePolicy2()));
		(void) assertSameReleasePolicy;
	}
#else
	/// Constructs a scoped pointer by stealing from the given scoped pointer.
	template <class Type2, class ReleasePolicy2>
	LEAN_INLINE scoped_ptr(scoped_ptr<Type2, critical_ref, ReleasePolicy2> &right) noexcept
		: m_object( right.detach() )
	{
		ReleasePolicy assertSameReleasePolicy((ReleasePolicy2()));
		(void) assertSameReleasePolicy;

		LEAN_STATIC_ASSERT_MSG_ALT(RefState == critical_ref,
			"Stealing copy only supported for backwards compatibility, requires critical pointers.",
			Stealing_copy_only_supported_for_backwards_compatibility__requires_critical_pointers);
	}
#endif
	/// Constructs a scoped pointer from the given scoped pointer.
	template <class Type2, reference_state_t RefState2, class ReleasePolicy2>
	LEAN_INLINE scoped_ptr(move_ref< scoped_ptr<Type2, RefState2, ReleasePolicy2> > right) noexcept
		: m_object( right.moved().detach() )
	{
		ReleasePolicy assertSameReleasePolicy((ReleasePolicy2()));
		(void) assertSameReleasePolicy;
	}

	/// Releases the object pointed to.
	LEAN_INLINE ~scoped_ptr()
	{
		ReleasePolicy::release(m_object);
	}

	/// Resets the object pointed to.
	void reset(object_type *object)
	{
		// Self-assignment would be wrong
		if (object != m_object)
		{
			value_type prevObject = m_object;
			m_object = object;
			ReleasePolicy::release(prevObject);
		}
	}

	/// Detaches the object pointed to.
	LEAN_INLINE object_type* detach()
	{
		value_type prevObject = m_object;
		m_object = nullptr;
		return prevObject;
	}

	/// Transfers the stored object.
	LEAN_INLINE move_ref<scoped_ptr> transfer()
	{
		return move_ref<scoped_ptr>(*this);
	}

	// NOTE: Workaround, MSVC++ misses elaborate template type specifiers
	typedef move_ptr<object_type> move_ptr_t;
	/// Gets a move pointer to this pointer.
	LEAN_INLINE move_ptr_t move_ptr()
	{
		return move_ptr_t(m_object);
	}

	/// Destroys the stored object.
	LEAN_INLINE void release()
	{
		reset(nullptr);
	}

	/// Replaces the stored object with the given object. <b>[ESA]</b>
	LEAN_INLINE scoped_ptr& operator =(object_type *object)
	{
		reset(object);
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Replaces the stored object with the one stored by the given r-value scoped pointer. <b>[ESA]</b>
	template <class Type2, reference_state_t RefState2, class ReleasePolicy2>
	LEAN_INLINE scoped_ptr& operator =(scoped_ptr<Type2, RefState2, ReleasePolicy2> &&right) noexcept
	{
		ReleasePolicy assertSameReleasePolicy((ReleasePolicy2()));
		(void) assertSameReleasePolicy;
		reset(right.detach());
		return *this;
	}
#endif
	/// Replaces the stored object with the one stored by the given r-value scoped pointer. <b>[ESA]</b>
	template <class Type2, reference_state_t RefState2, class ReleasePolicy2>
	LEAN_INLINE scoped_ptr& operator =(move_ref< scoped_ptr<Type2, RefState2, ReleasePolicy2> > right) noexcept
	{
		ReleasePolicy assertSameReleasePolicy((ReleasePolicy2()));
		(void) assertSameReleasePolicy;
		reset(right.moved().detach());
		return *this;
	}

	/// Gets the object stored by this scoped pointer.
	LEAN_INLINE object_type*const& get() const { return m_object; }

	/// Gets the object stored by this scoped pointer.
	LEAN_INLINE object_type& operator *() const { return *m_object; }
	/// Gets the object stored by this scoped pointer.
	LEAN_INLINE object_type* operator ->() const { return m_object; }
	/// Gets the object stored by this scoped pointer (getter compatibility).
	LEAN_INLINE object_type* operator ()() const { return m_object; }

	/// Gets the n-th element.
	LEAN_INLINE object_type& operator [](ptrdiff_t n) const { return m_object[n]; }

#ifndef LEAN_SCOPED_PTR_NO_IMPLICIT_CONVERSION
	/// Gets the object stored by this scoped pointer.
	LEAN_INLINE operator object_type*() const
	{
		LEAN_STATIC_ASSERT_MSG_ALT(RefState != critical_ref,
			"Cannot implicitly cast critical pointer, use detach() for (insecure) storage.",
			Cannot_implicitly_cast_critical_pointer__use_detach_for_insecure_storage);
		return m_object;
	}
#else
	/// Gets the object stored by this scoped pointer.
	LEAN_INLINE operator bool() const { return (m_object != nullptr); }
#endif

	/// Gets a double-pointer allowing for COM-style object retrieval. The pointer returned may
	/// only ever be used until the next call to one of this pointer's methods.
	LEAN_INLINE object_type** rebind()
	{
		release();
		return &m_object;
	}

	/// Swaps the given pointers.
	void swap(scoped_ptr& right) noexcept
	{
		value_type prevObject = m_object;
		m_object = right.m_object;
		right.m_object = prevObject;
	}
};

/// Swaps the given pointers.
template <class T, reference_state_t RS, class R>
LEAN_INLINE void swap(scoped_ptr<T, RS, R> &left, scoped_ptr<T, RS, R> &right) noexcept
{
	left.swap(right);
}

/// Constructs a scoped pointer.
template <class T>
LEAN_INLINE scoped_ptr<T, critical_ref> make_scoped_ptr(T *p)
{
	return scoped_ptr<T, critical_ref>(p);
}

#ifdef DOXYGEN_READ_THIS
	/// Creates a new resource using operator new.
	template <class Object>
	scoped_ptr<Object> make_scoped(...);
#else
	#define LEAN_MAKE_SCOPED_FUNCTION_TPARAMS class Object
	#define LEAN_MAKE_SCOPED_FUNCTION_DECL inline scoped_ptr<Object, critical_ref> make_scoped
	#define LEAN_MAKE_SCOPED_FUNCTION_BODY(call) { return scoped_ptr<Object, critical_ref>( new Object##call ); }
	LEAN_VARIADIC_TEMPLATE_T(LEAN_FORWARD, LEAN_MAKE_SCOPED_FUNCTION_DECL, LEAN_MAKE_SCOPED_FUNCTION_TPARAMS, LEAN_NOTHING, LEAN_MAKE_SCOPED_FUNCTION_BODY)
#endif

struct new_scoped_ptr_t { };
struct new_scoped_array_t { };

template <class T>
LEAN_INLINE scoped_ptr<T, critical_ref> operator *(new_scoped_ptr_t, T *p) { return scoped_ptr<T, critical_ref>(p); }
template <class T>
LEAN_INLINE scoped_ptr<T[], critical_ref> operator *(new_scoped_array_t, T *p) { return scoped_ptr<T[], critical_ref>(p); }

} // namespace

using smart::make_scoped;
using smart::make_scoped_ptr;

using smart::scoped_ptr;

} // namespace

#ifndef LEAN_NO_SCOPED_PTR_NEW

/// @addtogroup GlobalMacros
/// @{

/// Modified operator new that returns a scoped_ptr.
#define new_scoped ::lean::smart::new_scoped_ptr_t() * new
/// Modified operator new that returns a scoped_ptr to an array.
#define new_scoped_array ::lean::smart::new_scoped_array_t() * new

/// @}

#endif

#endif