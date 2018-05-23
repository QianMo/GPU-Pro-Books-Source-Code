/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_SPHERE_DEF
#define BE_MATH_SPHERE_DEF

#include "beMath.h"
#include "beSphereFwd.h"
#include "beTuple.h"
#include "beVectorDef.h"

namespace beMath
{

/// Sphere class.
template <class Component, size_t Dimension>
class sphere : private tuple<sphere<Component, Dimension>, Component, Dimension + 1>
{
private:
	typedef class tuple<sphere<Component, Dimension>, Component, Dimension + 1> base_type;

public:
	/// Component type.
	typedef Component component_type;
	/// Size type.
	typedef typename base_type::size_type size_type;
	/// Element count.
	static const size_type dimension = Dimension;
	/// Compatible scalar type.
	typedef component_type compatible_type;

	/// Position type.
	typedef vector<component_type, dimension> position_type;
	/// Radius type.
	typedef component_type radius_type;
	/// Compatible scalar type.
	typedef component_type compatible_type;

	/// Tuple type.
	typedef base_type tuple_type;

	position_type center;
	radius_type radius;

	/// Creates a default-initialized sphere.
	LEAN_INLINE sphere()
		: center(),
		radius() { }
	/// Creates an uninitialized sphere.
	LEAN_INLINE sphere(uninitialized_t)
		: center(uninitialized) { }
	/// Initializes all sphere elements from the given tuple.
	template <class TupleClass>
	LEAN_INLINE explicit sphere(const class tuple<TupleClass, component_type, dimension + 1> &right)
		: center(right),
		radius(right[dimension]) { }
	/// Initializes all sphere elements from the given sphere position & radius value.
	template <class TupleClass>
	LEAN_INLINE sphere(const class tuple<TupleClass, component_type, dimension> &position, const radius_type &radius)
		: center(position),
		radius(radius) { }
	/// Initializes all sphere elements from the given sphere position & point on sphere.
	template <class TupleClass1, class TupleClass2>
	LEAN_INLINE sphere(const class tuple<TupleClass1, component_type, dimension> &position,
		const class tuple<TupleClass2, component_type, dimension> &point)
		: center(position),
		radius( dist(position, point) ) { }

	/// Scales this sphere by the given value.
	LEAN_INLINE sphere& operator *=(const compatible_type &right)
	{
		radius *= right;
		return *this;
	}
	/// Scales this sphere dividing by the given value.
	LEAN_INLINE sphere& operator /=(const compatible_type &right)
	{
		radius /= right;
		return *this;
	}

	/// Accesses the n-th component.
	LEAN_INLINE component_type& element(size_type n) { return data()[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& element(size_type n) const { return data()[n]; }

	/// Accesses the n-th component.
	LEAN_INLINE component_type& operator [](size_type n) { return data()[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& operator [](size_type n) const { return data()[n]; }

	/// Gets a raw data pointer.
	LEAN_INLINE component_type* data() { return m_position.data(); }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* data() const { return m_position.data(); }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* cdata() const { return m_position.cdata(); }
	
	/// Gets a compatible tuple reference to this sphere object.
	LEAN_INLINE tuple_type& tpl() { return static_cast<tuple_type&>(*this); }
	/// Gets a compatible tuple reference to this sphere object.
	LEAN_INLINE const tuple_type& tpl() const { return static_cast<const tuple_type&>(*this); }
};

/// Scales the given sphere by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE sphere<Component, Dimension> operator *(const typename sphere<Component, Dimension>::compatible_type &left, const sphere<Component, Dimension> &right)
{
	return sphere<Component, Dimension>(right.center, left * right.radius);
}
/// Scales the given sphere by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE sphere<Component, Dimension> operator *(const sphere<Component, Dimension> &left, const typename sphere<Component, Dimension>::compatible_type &right)
{
	return sphere<Component, Dimension>(left.center, left.radius * right);
}

/// Scales the given sphere dividing by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE sphere<Component, Dimension> operator /(const typename sphere<Component, Dimension>::compatible_type &left, const sphere<Component, Dimension> &right)
{
	return sphere<Component, Dimension>(right.center, left / right.radius);
}
/// Scales the given sphere dividing by the given value.
template <class Component, size_t Dimension>
LEAN_INLINE sphere<Component, Dimension> operator /(const sphere<Component, Dimension> &left, const typename sphere<Component, Dimension>::compatible_type &right)
{
	return sphere<Component, Dimension>(left.center, left.radius / right);
}

} // namespace

#endif