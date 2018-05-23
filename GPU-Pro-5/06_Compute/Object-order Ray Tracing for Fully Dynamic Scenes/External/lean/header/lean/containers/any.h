/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_ANY
#define LEAN_CONTAINERS_ANY

#include "../lean.h"
#include "../meta/strip.h"
#include "../meta/type_traits.h"
#include "../smart/cloneable.h"
#include "../memory/heap_bound.h"
#include <typeinfo>

namespace lean
{
namespace containers
{

/// Any interface.
class any : public cloneable
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(any)

	template <class Value>
	friend Value* any_cast(any*, size_t);

protected:
	/// Gets a pointer to the stored value, if the given type matches the value stored by this object, nullptr otherwise.
	virtual void* get_any_ptr(const std::type_info& type, size_t idx) = 0;

public:
	/// Gets the type of the stored value.
	virtual const std::type_info& type() const = 0;
	/// Gets the number of stored values.
	virtual size_t size() const = 0;
};

template <class BaseType>
struct var_default
{
	typedef BaseType type;
	
	template <class Value>
	type* operator ()(Value *value) const
	{
		return static_cast<type*>(value);
	}
};

template <class UnionType>
struct var_union
{
	typedef UnionType type;
	
	template <class Value>
	type* operator ()(Value *value) const
	{
		return reinterpret_cast<type*>(value);
	}
};

template <class DerefType>
struct var_deref
{
	typedef DerefType type;
	
	template <class Value>
	type* operator ()(Value *value) const
	{
		return *value; // More general, but undefined in null cases: lean::addressof(**value);
	}
};

/// Any value.
template <class Value, class Variance = var_default<Value>, class Heap = default_heap>
class any_value : public heap_bound<Heap>, public any
{
public:
	/// Value type.
	typedef Value value_type;

protected:
	value_type m_value;

	/// Gets a pointer to the stored value, if the given type matches the value stored by this object, nullptr otherwise.
	void* get_any_ptr(const std::type_info& type, size_t idx) LEAN_OVERRIDE
	{
		if (idx == 0)
		{
			typedef typename rec_strip_modifiers<value_type>::type storage_t;
			typedef typename rec_strip_modifiers<typename Variance::type>::type var_t;

			if (typeid(storage_t) == type)
				// NOTE: const_cast also makes sure there is no implicit type conversion
				return const_cast<storage_t*>(&m_value);
			else if (!is_equal<storage_t, var_t>::value && typeid(var_t) == type)
				return const_cast<var_t*>( Variance()(&m_value) );
		}
		
		return nullptr;
	}
	
public:
	/// Constructor.
	LEAN_INLINE any_value()
		: m_value() { }
	/// Constructor.
	LEAN_INLINE any_value(const value_type &value)
		: m_value(value) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructor.
	LEAN_INLINE any_value(value_type &&value)
		: m_value(std::move(value)) { }
#endif
#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Constructor.
	LEAN_INLINE any_value(any_value &&right)
		: m_value(std::move(right.m_value)) { }
#endif

	/// Assignment.
	LEAN_INLINE any_value& operator =(const value_type &value)
	{
		m_value = value;
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Assignment.
	LEAN_INLINE any_value& operator =(value_type &&value)
	{
		m_value = std::move(value);
		return *this;
	}
#endif
#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Assignment.
	LEAN_INLINE any_value& operator =(any_value &&right)
	{
		m_value = std::move(right.m_value);
		return *this;
	}
#endif

	/// Gets the stored value.
	LEAN_INLINE value_type& get() { return m_value; }
	/// Gets the stored value.
	LEAN_INLINE const value_type& get() const { return m_value; }
	/// Gets the stored value.
	LEAN_INLINE volatile value_type& get() volatile { return m_value; }
	/// Gets the stored value.
	LEAN_INLINE const volatile value_type& get() const volatile { return m_value; }

	/// Gets the type of the stored value.
	const std::type_info& type() const LEAN_OVERRIDE
	{
		return typeid(value_type);
	}
	/// Gets the number of stored values.
	size_t size() const LEAN_OVERRIDE { return 1; }

	/// Clones this value.
	LEAN_INLINE any_value* clone() const { return new any_value(*this); }
	/// Moves the contents of this cloneable to a clone.
	LEAN_INLINE any_value* clone_move() { return new any_value(LEAN_MOVE(*this)); }
	/// Destroys a clone.
	LEAN_INLINE void destroy() const { delete this; }
};

/// Any value vector.
template <class Vector, class Variance = var_default<typename Vector::value_type>, class Heap = default_heap>
class any_vector : public any_value<Vector, var_default<Vector>, Heap>
{
public:
	/// Vector type.
	typedef Vector vector_type;
	/// Value type.
	typedef typename vector_type::value_type value_type;

protected:
	/// Gets a pointer to the stored value, if the given type matches the value stored by this object, nullptr otherwise.
	void* get_any_ptr(const std::type_info& type, size_t idx)
	{
		typedef typename rec_strip_modifiers<value_type>::type storage_t;
		typedef typename rec_strip_modifiers<typename Variance::type>::type var_t;

		if (idx < this->m_value.size())
		{
			if (typeid(storage_t) == type)
				// NOTE: const_cast also makes sure there is no implicit type conversion
				return const_cast<storage_t*>(&this->m_value[idx]);
			else if (!is_equal<storage_t, var_t>::value && typeid(var_t) == type)
				return const_cast<var_t*>( Variance()(&this->m_value[idx]) );
		}
		
		return this->any_value::get_any_ptr(type, idx);
	}
	
public:
	/// Constructor.
	LEAN_INLINE any_vector() { }
	/// Constructor.
	LEAN_INLINE any_vector(const vector_type &value)
		: typename any_vector::any_value(value) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructor.
	LEAN_INLINE any_vector(vector_type &&value)
		: typename any_vector::any_value(std::move(value)) { }
#endif
#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Constructor.
	LEAN_INLINE any_vector(any_vector &&right)
		: typename any_vector::any_value(std::move(right)) { }
#endif

	/// Assignment.
	LEAN_INLINE any_vector& operator =(const vector_type &value)
	{
		this->any_value::operator =(value);
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Assignment.
	LEAN_INLINE any_vector& operator =(vector_type &&value)
	{
		this->any_value::operator =(std::move(value));
		return *this;
	}
#endif
#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Assignment.
	LEAN_INLINE any_vector& operator =(any_vector &&right)
	{
		this->any_value::operator =(std::move(right));
		return *this;
	}
#endif

	/// Gets the stored value.
	LEAN_INLINE value_type& get() { return this->m_value[0]; }
	/// Gets the stored value.
	LEAN_INLINE const value_type& get() const { return this->m_value[0]; }
	/// Gets the stored value.
	LEAN_INLINE volatile value_type& get() volatile { return this->m_value[0]; }
	/// Gets the stored value.
	LEAN_INLINE const volatile value_type& get() const volatile { return this->m_value[0]; }

	/// Gets the type of the stored value.
	const std::type_info& type() const LEAN_OVERRIDE
	{
		return typeid(value_type);
	}
	/// Gets the number of stored values.
	size_t size() const LEAN_OVERRIDE { return this->m_value.size(); }

	/// Clones this value.
	LEAN_INLINE any_vector* clone() const { return new any_vector(*this); }
	/// Moves the contents of this cloneable to a clone.
	LEAN_INLINE any_vector* clone_move() { return new any_vector(LEAN_MOVE(*this)); }
	/// Destroys a clone.
	LEAN_INLINE void destroy() const { delete this; }
};

/// Gets a pointer to the value of the given type, if the given value type matches the value stored by the given object, nullptr otherwise.
template <class Value>
LEAN_INLINE Value* any_cast(any *pContainer, size_t idx = 0)
{
	return static_cast<Value*>( 
			(pContainer)
				? pContainer->get_any_ptr( typeid(typename rec_strip_modifiers<Value>::type), idx )
				: nullptr
		);
}
/// Gets a pointer to the value of the given type, if the given value type matches the value stored by the given object, nullptr otherwise.
template <class Value>
LEAN_INLINE const Value* any_cast(const any *pContainer, size_t idx = 0)
{
	return any_cast<const Value>(const_cast<any*>(pContainer), idx);
}
/// Gets a pointer to the value of the given type, if the given value type matches the value stored by the given object, nullptr otherwise.
template <class Value>
LEAN_INLINE volatile Value* any_cast(volatile any *pContainer, size_t idx = 0)
{
	return any_cast<volatile Value>(const_cast<any*>(pContainer), idx);
}
/// Gets a pointer to the value of the given type, if the given value type matches the value stored by the given object, nullptr otherwise.
template <class Value>
LEAN_INLINE const volatile Value* any_cast(const volatile any *pContainer, size_t idx = 0)
{
	return any_cast<const volatile Value>(const_cast<any*>(pContainer), idx);
}

namespace impl
{
	/// Throws a bad_cast exception.
	LEAN_NOINLINE void throw_bad_cast()
	{
		throw std::bad_cast();
	}

} // namespace

/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast_checked(any *container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	nonref_value_type *pValue = any_cast<nonref_value_type>(container, idx);
	
	if (!pValue)
		impl::throw_bad_cast();

	return *pValue;
}
/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast_checked(const any *container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast_checked<const nonref_value_type&>(const_cast<any*>(container), idx);
}
/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast_checked(volatile any *container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast_checked<volatile nonref_value_type&>(const_cast<any*>(container), idx);
}
/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast_checked(const volatile any *container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast_checked<const volatile nonref_value_type&>(const_cast<any*>(container), idx);
}

/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast(any &container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast_checked<nonref_value_type&>(&container, idx);
}
/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast(const any &container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast<const nonref_value_type&>(const_cast<any&>(container), idx);
}
/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast(volatile any &container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast<volatile nonref_value_type&>(const_cast<any&>(container), idx);
}
/// Gets a value of the given type, if the given value type matches the value stored by the given object, throws bad_cast otherwise.
template <class Value>
LEAN_INLINE Value any_cast(const volatile any &container, size_t idx = 0)
{
	typedef typename lean::strip_reference<Value>::type nonref_value_type;
	return any_cast<const volatile nonref_value_type&>(const_cast<any&>(container), idx);
}

/// Gets a value of the given type, if the given value type matches the value stored by the given object, default otherwise.
template <class Value>
LEAN_INLINE Value any_cast_default(const any *container, const Value &defaultValue = Value(), size_t idx = 0)
{
	const Value *pValue = any_cast<Value>(container, idx);
	return (pValue) ? *pValue : defaultValue;
}

/// Gets a value of the given type, if the given value type matches the value stored by the given object, default otherwise.
template <class Value>
LEAN_INLINE Value any_cast_default(const any &container, const Value &defaultValue = Value(), size_t idx = 0)
{
	const Value *pValue = any_cast<Value>(&container, idx);
	return (pValue) ? *pValue : defaultValue;
}

} // namespace

using containers::any;
using containers::any_value;
using containers::any_vector;

using containers::var_default;
using containers::var_union;
using containers::var_deref;

using containers::any_cast;
using containers::any_cast_checked;
using containers::any_cast_default;

} // namespace

#ifdef DOXYGEN_READ_THIS
	/// @ingroup GlobalSwitches
	/// Define this to disable global any_cast and any_cast_checked templates.
	#define LEAN_NO_ANY_CAST
	#undef LEAN_NO_ANY_CAST
#endif

#ifndef LEAN_NO_ANY_CAST
	using lean::any_cast;
	using lean::any_cast_checked;
	using lean::any_cast_default;
#endif

#endif