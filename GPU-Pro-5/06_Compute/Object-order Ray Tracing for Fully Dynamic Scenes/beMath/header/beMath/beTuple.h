/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_TUPLE
#define BE_MATH_TUPLE

#include "beMath.h"
#include "beTupleFwd.h"
#include "beComparisonOperators.h"

namespace beMath
{

/// Tuple class.
template <class Class, class Element, size_t Count>
class tuple
{
protected:
	// Statically abstract

#ifdef LEAN0X_NO_DELETE_METHODS
	LEAN_INLINE tuple() throw() { }
	LEAN_INLINE tuple(const tuple&) throw() { }
	// NOTE: Prevent double assignment due to default assignment operators generated in derived classes.
	LEAN_INLINE tuple& operator =(const tuple &right) throw() { return *this; }
#else
	LEAN_INLINE tuple() = default;
	LEAN_INLINE tuple(const tuple&) = default;
	// NOTE: Prevent double assignment due to default assignment operators generated in derived classes.
	LEAN_INLINE tuple& operator =(const tuple &right) = default;
#endif

public:
	/// Value type.
	typedef Element value_type;
	/// Size type.
	typedef size_t size_type;
	/// Number of elements.
	static const size_type count = Count;
	/// Most derived type.
	typedef Class class_type;

	/// Assigns the given value to all tuple elements.
	LEAN_INLINE class_type& operator =(const value_type &element)
	{
		return assign(element);
	}
	/// Assigns the given tuple to this tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& operator =(const tuple<OtherClass, Element, Count> &right)
	{
		return assign(right);
	}

	/// Assigns the given value to all tuple elements.
	LEAN_INLINE class_type& assign(const value_type &element)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] = element;
		return actual();
	}
	/// Assigns the given tuple to this tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& assign(const tuple<OtherClass, Element, Count> &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] = right[i];
		return actual();
	}

	/// Adds the given value to this tuple.
	LEAN_INLINE class_type& operator +=(const value_type &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] += right;
		return actual();
	}
	/// Adds the given tuple to this tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& operator +=(const tuple<OtherClass, Element, Count> &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] += right[i];
		return actual();
	}

	/// Subtracts the given value from this tuple.
	LEAN_INLINE class_type& operator -=(const value_type &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] -= right;
		return actual();
	}
	/// Subtracts the given tuple from this tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& operator -=(const tuple<OtherClass, Element, Count> &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] -= right[i];
		return actual();
	}

	/// Multiplies this tuple by the given value.
	LEAN_INLINE class_type& operator *=(const value_type &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] *= right;
		return actual();
	}
	/// Multiplies this tuple by the given tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& operator *=(const tuple<OtherClass, Element, Count> &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] *= right[i];
		return actual();
	}

	/// Divides this tuple by the given value.
	LEAN_INLINE class_type& operator /=(const value_type &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] /= right;
		return actual();
	}
	/// Divides this tuple by the given tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& operator /=(const tuple<OtherClass, Element, Count> &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] /= right[i];
		return actual();
	}

	/// Assigns the remainder of this tuple divided by the given value.
	LEAN_INLINE class_type& operator %=(const value_type &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] %= right;
		return actual();
	}
	/// Assigns the remainder of this tuple divided by the given tuple.
	template <class OtherClass>
	LEAN_INLINE class_type& operator %=(const tuple<OtherClass, Element, Count> &right)
	{
		for (size_t i = 0; i < Count; ++i)
			(*this)[i] %= right[i];
		return actual();
	}

	/// Gets the actual derived object.
	LEAN_INLINE class_type& actual() { return static_cast<class_type&>(*this); }
	/// Gets the actual derived object.
	LEAN_INLINE const class_type& actual() const { return static_cast<const class_type&>(*this); }

	/// Accesses the n-th element.
	LEAN_INLINE value_type& operator [](size_type n) { return static_cast<class_type&>(*this).element(n); }
	/// Accesses the n-th element.
	LEAN_INLINE const value_type& operator [](size_type n) const { return static_cast<const class_type&>(*this).element(n); }

	/// Gets the number of elements in this tuple.
	LEAN_INLINE size_type size() const { return count; }

	/// Compares this tuple to the given tuple.
	template <class OtherClass>
	bool operator ==(const tuple<OtherClass, Element, Count> &right) const
	{
		for (size_t i = 0; i < Count; ++i)
			if ((*this)[i] != right[i])
				return false;
		return true;
	}
};

/// Negates the given value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator -(const tuple<Operand, Component, Dimension> &operand)
{
	Operand result(uninitialized);

	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = -operand[i];
	
	return result;
}

/// Adds the given values.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator +(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);

	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] + right[i];
	
	return result;
}

/// Subtracts the given right value from the given left value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator -(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);

	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] - right[i];
	
	return result;
}

/// Multiplies the given left value by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator *(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] * right[i];
	return result;
}

/// Divides the given left value by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator /(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] / right[i];
	return result;
}

/// Computes the remainder of the given left value divided by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator %(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] % right[i];
	return result;
}

/// Adds the given values.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator +(const typename Operand::compatible_type &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left + right[i];
	return result;
}

/// Subtracts the given right value from the given left value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator -(const typename Operand::compatible_type &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left - right[i];
	return result;
}

/// Multiplies the given left value by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator *(const typename Operand::compatible_type &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left * right[i];
	return result;
}

/// Divides the given left value by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator /(const typename Operand::compatible_type &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left / right[i];
	return result;
}

/// Computes the remainder of the given left value divided by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator %(const typename Operand::compatible_type &left, const tuple<Operand, Component, Dimension> &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left % right[i];
	return result;
}

/// Returns true iff all components of left are less than the corresponding components of right.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE bool operator <(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	for (size_t i = 0; i < Operand::count; ++i)
		if (left[i] >= right[i])
			return false;
	
	return true;
}

/// Returns true iff all components of the given vectors are approximately equal.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE bool eps_eq(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right, Component delta)
{
	for (size_t i = 0; i < Operand::count; ++i)
		if (abs(left[i] - right[i]) > delta)
			return false;
	
	return true;
}

/// Returns true iff all components of the given vectors are approximately equal.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE bool valid(const tuple<Operand, Component, Dimension> &left)
{
	bool result = true;
	for (size_t i = 0; i < Operand::count; ++i)
		result &= (left[i] == left[i]);
	return result;
}

/// Computes the component-wise minimum of the given two values.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand min_cw(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	using lean::min;

	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = min(left[i], right[i]);
	return result;
}

/// Computes the component-wise maximum of the given two values.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand max_cw(const tuple<Operand, Component, Dimension> &left, const tuple<Operand, Component, Dimension> &right)
{
	using lean::max;

	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = max(left[i], right[i]);
	return result;
}

/// Computes the minimum of all components.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Component min_scan(const tuple<Operand, Component, Dimension> &left)
{
	using lean::min;

	Component result = left[0];
	for (size_t i = 1; i < Operand::count; ++i)
		result = min(result, left[i]);
	return result;
}

/// Computes the maximum of all components.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Component max_scan(const tuple<Operand, Component, Dimension> &left)
{
	using lean::max;

	Component result = left[0];
	for (size_t i = 1; i < Operand::count; ++i)
		result = max(result, left[i]);
	return result;
}

/// Adds the given values.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator +(const tuple<Operand, Component, Dimension> &left, const typename Operand::compatible_type &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] + right;
	return result;
}

/// Subtracts the given right value from the given left value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator -(const tuple<Operand, Component, Dimension> &left, const typename Operand::compatible_type &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] - right;
	return result;
}

/// Multiplies the given left value by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator *(const tuple<Operand, Component, Dimension> &left, const typename Operand::compatible_type &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] * right;
	return result;
}

/// Divides the given left value by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator /(const tuple<Operand, Component, Dimension> &left, const typename Operand::compatible_type &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] / right;
	return result;
}

/// Computes the remainder of the given left value divided by the given right value.
template <class Operand, class Component, size_t Dimension>
LEAN_INLINE Operand operator %(const tuple<Operand, Component, Dimension> &left, const typename Operand::compatible_type &right)
{
	Operand result(uninitialized);
	for (size_t i = 0; i < Operand::count; ++i)
		result.element(i) = left[i] % right;
	return result;
}

} // namespace

#endif