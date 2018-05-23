/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_FORWARD_VAL
#define LEAN_PIMPL_FORWARD_VAL

#include "../cpp0x.h"

namespace lean
{
namespace pimpl
{

/// Opaque value class that stores values of the given size & forward-declared type.
template <class Value, size_t Size>
class forward_val
{
private:
	char m_value[Size];

	/// Asserts that the specified size matches the size of the stored value.
	LEAN_INLINE static void assert_size()
	{
		LEAN_STATIC_ASSERT_MSG_ALT(sizeof(Value) == Size,
			"specified size does not match value size",
			specified_size_does_not_match_value_size);
	}

public:
	/// Actual type of the value stored by this wrapper, if fully defined.
	typedef Value value_type;

	/// Constructs an opaque value object from the given value.
	forward_val()
	{
		assert_size();
		new(static_cast<void*>(m_value)) Value();
	}
	/// Constructs an opaque value object from the given value.
	forward_val(const value_type &value)
	{
		assert_size();
		new(static_cast<void*>(m_value)) Value(value);
	}
	/// Constructs an opaque value object from the given value.
	forward_val(const forward_val &right)
	{
		assert_size();
		new(static_cast<void*>(m_value)) Value(right.get());
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs an opaque value object from the given r-value.
	forward_val(value_type &&value)
	{
		assert_size();
		new(static_cast<void*>(m_value)) Value(std::move(value));
	}
	/// Constructs an opaque value object from the given r-value.
	forward_val(forward_val &&right)
	{
		assert_size();
		new(static_cast<void*>(m_value)) Value(std::move(right.get()));
	}
#endif
	/// Destructor.
	~forward_val()
	{
		get().~Value();
	}

	/// Replaces the stored value with the given new value.
	forward_val& operator =(const value_type &value)
	{
		get() = value;
		return *this;
	}
	/// Replaces the stored value with the given new value.
	forward_val& operator =(const forward_val &right)
	{
		get() = right.get();
		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Replaces the stored value with the given new r-value.
	forward_val& operator =(value_type &&value)
	{
		get() = std::move(value);
		return *this;
	}
	/// Replaces the stored value with the given new r-value.
	forward_val& operator =(forward_val &&right)
	{
		get() = std::move(right.get());
		return *this;
	}
#endif

	/// Gets a pointer to the value concealed by this opaque wrapper.
	LEAN_INLINE value_type* getptr(void) { assert_size(); return reinterpret_cast<Value*>(m_value); }
	/// Gets a pointer to the value concealed by this opaque wrapper.
	LEAN_INLINE const value_type* getptr(void) const { assert_size(); return reinterpret_cast<const Value*>(m_value); }
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE value_type& get(void) { return *getptr(); }
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE const value_type& get(void) const { return *getptr(); }
	
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE value_type& operator *() { return get(); }
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE const value_type& operator *() const { return get(); }
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE value_type* operator ->() { return getptr(); }
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE const value_type* operator ->() const { return getptr(); }

	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE operator value_type&() { return get(); }
	/// Gets the value concealed by this opaque wrapper.
	LEAN_INLINE operator const value_type&() const { return get(); }
};

} // namespace

using pimpl::forward_val;

} // namespace

#endif