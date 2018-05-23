/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_AUTO_RESTORE
#define LEAN_SMART_AUTO_RESTORE

#include "../lean.h"
#include "../tags/noncopyable.h"

namespace lean
{
namespace smart
{

/// Auto restore class that automatically restores the initial value of the stored object on destruction.
template <class Value>
class auto_restore : public noncopyable
{
private:
	Value &m_value;
	Value m_initialValue;

public:
	/// Type of the value managed by this class.
	typedef Value value_type;

	/// Constructs an auto-restore object that will restore the given object's current value on destruction.
	explicit auto_restore(value_type &value)
		: m_value(value),
		m_initialValue(value) { }
	/// Constructs an auto-restore object that will assign the given value to the given object and restore the given object's current value on destruction.
	auto_restore(value_type &value, const value_type &newValue)
		: m_value(value),
		m_initialValue(value)
	{
		m_value = newValue;
	}

	/// Resets the stored object to its initial value.
	~auto_restore()
	{
		m_value = m_initialValue;
	}
	
	/// Gets the object managed by this class.
	LEAN_INLINE value_type& get(void) { return m_value; };
	/// Gets the object managed by this class.
	LEAN_INLINE const value_type& get(void) const { return m_value; };

	/// Gets the object managed by this class.
	LEAN_INLINE value_type& operator *() { return get(); };
	/// Gets the object managed by this class.
	LEAN_INLINE const value_type& operator *() const { return get(); };
	/// Gets the object managed by this class.
	LEAN_INLINE value_type* operator ->() { return &get(); };
	/// Gets the object managed by this class.
	LEAN_INLINE const value_type* operator ->() const { return &get(); };

	/// Gets the object managed by this class.
	LEAN_INLINE operator value_type&() { return get(); };
	/// Gets the object managed by this class.
	LEAN_INLINE operator const value_type&() const { return get(); };
};

} // namespace

using smart::auto_restore;

} // namespace

#endif