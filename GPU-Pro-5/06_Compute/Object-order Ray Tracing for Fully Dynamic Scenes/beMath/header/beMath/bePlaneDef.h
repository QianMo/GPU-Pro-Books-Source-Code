/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_PLANE_DEF
#define BE_MATH_PLANE_DEF

#include "beMath.h"
#include "bePlaneFwd.h"
#include "beTuple.h"
#include "beVectorDef.h"

namespace beMath
{

/// Plane class.
template <class Component, size_t Dimension>
class plane : private tuple<plane<Component, Dimension>, Component, Dimension + 1>
{
private:
	typedef class tuple<plane<Component, Dimension>, Component, Dimension + 1> base_type;

	vector<Component, Dimension> m_normal;
	Component m_distance;

public:
	/// Component type.
	typedef Component component_type;
	/// Size type.
	typedef typename base_type::size_type size_type;
	/// Element count.
	static const size_type dimension = Dimension;
	/// Compatible scalar type.
	typedef component_type compatible_type;

	/// Normal type.
	typedef vector<component_type, dimension> normal_type;
	/// Distance type.
	typedef component_type distance_type;
	/// Compatible scalar type.
	typedef component_type compatible_type;

	/// Tuple type.
	typedef base_type tuple_type;

	/// Creates a default-initialized plane.
	LEAN_INLINE plane()
		: m_normal(),
		m_distance() { }
	/// Creates an uninitialized plane.
	LEAN_INLINE plane(uninitialized_t)
		: m_normal(uninitialized) { }
	/// Initializes all plane elements from the given tuple.
	template <class TupleClass>
	LEAN_INLINE plane(const class tuple<TupleClass, component_type, dimension + 1> &right)
		: m_normal(right),
		m_distance(right[dimension]) { }
	/// Initializes all plane elements from the given plane normal & distance value.
	template <class TupleClass>
	LEAN_INLINE plane(const class tuple<TupleClass, component_type, dimension> &normal, const distance_type &distance)
		: m_normal(normal),
		m_distance(distance) { }
	/// Initializes all plane elements from the given plane normal & point on plane.
	template <class TupleClass1, class TupleClass2>
	LEAN_INLINE plane(const class tuple<TupleClass1, component_type, dimension> &normal,
		const class tuple<TupleClass2, component_type, dimension> &point)
		: m_normal(normal),
		m_distance( dot(normal, point) ) { }

	/// Scales this plane by the given value.
	LEAN_INLINE plane& operator *=(const compatible_type &right)
	{
		m_normal *= right;
		m_distance *= right;
		return *this;
	}
	/// Scales this plane dividing by the given value.
	LEAN_INLINE plane& operator /=(const compatible_type &right)
	{
		m_normal /= right;
		m_distance /= right;
		return *this;
	}

	/// Gets the normal vector.
	LEAN_INLINE normal_type& n() { return m_normal; }
	/// Gets the normal vector.
	LEAN_INLINE const normal_type& n() const { return m_normal; }
	/// Gets the normal vector.
	LEAN_INLINE distance_type& d() { return m_distance; }
	/// Gets the normal vector.
	LEAN_INLINE const distance_type& d() const { return m_distance; }

	/// Accesses the n-th component.
	LEAN_INLINE component_type& element(size_type n) { return data()[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& element(size_type n) const { return data()[n]; }

	/// Accesses the n-th component.
	LEAN_INLINE component_type& operator [](size_type n) { return data()[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& operator [](size_type n) const { return data()[n]; }

	/// Gets a raw data pointer.
	LEAN_INLINE component_type* data() { return m_normal.data(); }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* data() const { return m_normal.data(); }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* cdata() const { return m_normal.cdata(); }
	
	/// Gets a compatible tuple reference to this plane object.
	LEAN_INLINE tuple_type& tpl() { return static_cast<tuple_type&>(*this); }
	/// Gets a compatible tuple reference to this plane object.
	LEAN_INLINE const tuple_type& tpl() const { return static_cast<const tuple_type&>(*this); }
};

/// Negates the given plane.
template <class Component, size_t Dimension>
LEAN_INLINE plane<Component, Dimension> operator -(const plane<Component, Dimension> &right)
{
	return plane<Component, Dimension>(-right.n(), -right.d());
}

/// Scales the given plane by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE plane<Component, Dimension> operator *(const typename plane<Component, Dimension>::compatible_type &left, const plane<Component, Dimension> &right)
{
	return plane<Component, Dimension>(left * right.n(), left * right.d());
}
/// Scales the given plane by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE plane<Component, Dimension> operator *(const plane<Component, Dimension> &left, const typename plane<Component, Dimension>::compatible_type &right)
{
	return plane<Component, Dimension>(left.n() * right, left.d() * right);
}

/// Scales the given plane dividing by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE plane<Component, Dimension> operator /(const typename plane<Component, Dimension>::compatible_type &left, const plane<Component, Dimension> &right)
{
	return plane<Component, Dimension>(left / right.n(), left / right.d());
}
/// Scales the given plane dividing by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE plane<Component, Dimension> operator /(const plane<Component, Dimension> &left, const typename plane<Component, Dimension>::compatible_type &right)
{
	return plane<Component, Dimension>(left.n() / right, left.d() / right);
}

} // namespace

#endif