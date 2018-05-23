/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_CLONEABLE_OBJ
#define LEAN_SMART_CLONEABLE_OBJ

#include "../cpp0x.h"
#include "../meta/strip.h"
#include "common.h"

namespace lean
{
namespace smart
{

/// Clones the given cloneable object by calling @code clone(cloneable)@endcode  (default policy implementation).
template <class Cloneable>
LEAN_INLINE typename strip_modref<Cloneable>::type* clone_cloneable(Cloneable LEAN_FW_REF cloneable)
{
	return static_cast<typename strip_modref<Cloneable>::type*>( clone(LEAN_FORWARD(Cloneable, cloneable)) );
}
/// Destroys the given cloneable object by calling @code destroy(cloneable)@endcode  (default policy implementation).
template <class Cloneable>
LEAN_INLINE void destroy_cloneable(Cloneable *cloneable)
{
	destroy(cloneable);
}

namespace impl
{

/// Returns a pointer or a reference to the object pointed to.
template <class Type, bool Pointer>
struct ptr_or_ref_to
{
	template <class T>
	static LEAN_INLINE Type get(T *p) { return p; }
};
template <class Type>
struct ptr_or_ref_to<Type, false>
{
	template <class T>
	static LEAN_INLINE Type get(T *p) { return *p; }
};

} // namespace

/// Cloneable object class that stores an automatic instance of the given cloneable type.
template <class Cloneable, bool PointerSemantics = val_sem>
class cloneable_obj
{
public:
	/// Type of the cloneable value stored by this cloneable object.
	typedef Cloneable value_type;
	/// Const value_type for value semantics, value_type for pointer semantics.
	typedef typename conditional_type<PointerSemantics, value_type, const value_type>::type maybe_const_value_type;
	/// Reference type for value semantics, pointer type for pointer semantics.
	typedef typename conditional_type<PointerSemantics, value_type*, value_type&>::type maybe_pointer;
	/// Const reference type for value semantics, pointer type for pointer semantics.
	typedef typename conditional_type<PointerSemantics, value_type*, const value_type&>::type const_maybe_pointer;

private:
	Cloneable *m_cloneable;

	/// Acquires the given cloneable.
	static Cloneable* acquire(const Cloneable &cloneable)
	{
		return clone_cloneable(cloneable);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Acquires the given cloneable.
	static Cloneable* acquire(Cloneable &&cloneable)
	{
		return clone_cloneable(std::move(cloneable));
	}
#endif
	/// Acquires the given cloneable.
	static Cloneable* acquire(const Cloneable *cloneable)
	{
		return (cloneable)
			? acquire(*cloneable)
			: nullptr;
	}

	/// Releases the given cloneable.
	static void release(const Cloneable *cloneable)
	{
		destroy_cloneable(cloneable);
	}

public:
	/// Constructs a cloneable object by cloning the given cloneable value.
	cloneable_obj(const value_type &cloneable)
		: m_cloneable( acquire(cloneable) ) { };
	/// Constructs a cloneable object by cloning the given cloneable object.
	cloneable_obj(const cloneable_obj &right)
		: m_cloneable( acquire(right.m_cloneable) ) { };
	/// Constructs a cloneable object by cloning the given cloneable object (nullptr allowed).
	cloneable_obj(const value_type *cloneable)
		: m_cloneable( acquire(cloneable) )
	{
		LEAN_STATIC_ASSERT_MSG_ALT(PointerSemantics,
			"Construction from pointer only available for pointer semantics.",
			Construction_from_pointer_only_available_for_pointer_semantics);
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a cloneable object by cloning the given cloneable value.
	cloneable_obj(value_type &&cloneable)
		: m_cloneable( acquire(std::move(cloneable)) ) { };
	/// Constructs a cloneable object by cloning the given cloneable object.
	cloneable_obj(cloneable_obj &&right) noexcept
		: m_cloneable( std::move(right.m_cloneable) )
	{
		// Warning: this "breaks" the other object
		right.m_cloneable = nullptr;
	}
#endif
	/// Destroys the cloneable object.
	~cloneable_obj()
	{
		release(m_cloneable);
	}

	/// Gets a null cloneable object that may only be copied from and assigned to.
	static LEAN_INLINE cloneable_obj null()
	{
		LEAN_STATIC_ASSERT_MSG_ALT(PointerSemantics,
			"Null objects only available for pointer semantics.",
			Null_objects_only_available_for_pointer_semantics);

		// Warning: this object is effectively "broken"
		return cloneable_obj(nullptr);
	}

	/// Replaces the stored cloneable value with a clone of the given cloneable value.
	cloneable_obj& operator =(const value_type &cloneable)
	{
		Cloneable *prevCloneable = m_cloneable;
		m_cloneable = acquire(cloneable);
		release(prevCloneable);
		
		return *this;
	}
	/// Replaces the stored cloneable value with a clone of the given cloneable value (nullptr allowed).
	cloneable_obj& operator =(const value_type *cloneable)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(PointerSemantics,
			"Pointer assignment only available for pointer semantics.",
			Pointer_assignment_only_available_for_pointer_semantics);

		Cloneable *prevCloneable = m_cloneable;
		m_cloneable = acquire(cloneable);
		release(prevCloneable);
		
		return *this;
	}
	/// Replaces the stored cloneable value with a clone of the given cloneable object.
	cloneable_obj& operator =(const cloneable_obj &right)
	{
		if (m_cloneable != right.m_cloneable)
		{
			Cloneable *prevCloneable = m_cloneable;
			m_cloneable = acquire(right.m_cloneable);
			release(prevCloneable);
		}
		
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Replaces the stored cloneable value with a clone of the given cloneable value.
	cloneable_obj& operator =(value_type &&cloneable)
	{
		Cloneable *prevCloneable = m_cloneable;
		m_cloneable = acquire(std::move(cloneable));
		release(prevCloneable);
		
		return *this;
	}
	/// Replaces the stored cloneable value with the value stored by the given cloneable object.
	cloneable_obj& operator =(cloneable_obj &&right) noexcept
	{
		if (m_cloneable != right.m_cloneable)
		{
			Cloneable *prevCloneable = m_cloneable;
			
			m_cloneable = std::move(right.m_cloneable);
			// Warning: this "breaks" the other object
			right.m_cloneable = nullptr;

			release(prevCloneable);
		}
		return *this;
	}
#endif

	/// Gets the value stored by this cloneable object.
	value_type& get(void) { LEAN_ASSERT(m_cloneable); return *m_cloneable; };
	/// Gets the value stored by this cloneable object.
	maybe_const_value_type& get(void) const { LEAN_ASSERT(m_cloneable); return *m_cloneable; };
	/// Gets the value stored by this cloneable object.
	value_type* getptr(void) { return m_cloneable; };
	/// Gets the value stored by this cloneable object.
	maybe_const_value_type* getptr(void) const { return m_cloneable; };
	/// Gets whether this cloneable object is currently storing a value.
	bool valid() const { return (m_cloneable != nullptr); };

	/// Gets the value stored by this cloneable object.
	value_type& operator *() { return get(); };
	/// Gets the value stored by this cloneable object.
	maybe_const_value_type& operator *() const { return get(); };
	/// Gets the value stored by this cloneable object.
	value_type* operator ->() { LEAN_ASSERT(m_cloneable); return m_cloneable; };
	/// Gets the value stored by this cloneable object.
	maybe_const_value_type* operator ->() const { LEAN_ASSERT(m_cloneable); return m_cloneable; };
	/// Gets the value stored by this cloneable object (getter compatibility).
	maybe_pointer operator ()() { return impl::ptr_or_ref_to<maybe_pointer, PointerSemantics>::get( m_cloneable ); }
	/// Gets the value stored by this cloneable object (getter compatibility).
	const_maybe_pointer operator ()() const { return impl::ptr_or_ref_to<const_maybe_pointer, PointerSemantics>::get( m_cloneable ); }

	/// Gets the value stored by this cloneable object.
	operator maybe_pointer() { return impl::ptr_or_ref_to<maybe_pointer, PointerSemantics>::get( m_cloneable ); };
	/// Gets the value stored by this cloneable object.
	operator const_maybe_pointer() const { return impl::ptr_or_ref_to<const_maybe_pointer, PointerSemantics>::get( m_cloneable ); };
};

} // namespace

using smart::cloneable_obj;

} // namespace

#endif