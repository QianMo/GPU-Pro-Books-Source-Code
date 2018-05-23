/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_VECTOR
#define BE_MATH_VECTOR

#include "beMath.h"
#include "beVectorDef.h"
#include <cmath>

namespace beMath
{

/// Computes the dot product of the given two vectors.
template <class LeftClass, class RightClass, class Component, size_t Dimension>
LEAN_INLINE Component dot(const tuple<LeftClass, Component, Dimension> &left, const tuple<RightClass, Component, Dimension> &right)
{
	Component scalar(0);
	for (size_t i = 0; i < Dimension; ++i)
		scalar += left[i] * right[i];
	return scalar;
}

/// Reflects the given vector on the given normalized axis.
template <class Class, class Component, size_t Dimension>
LEAN_INLINE vector<Component, 3> reflect(const tuple<Class, Component, Dimension> &vec,
										 const tuple<Class, Component, Dimension> &axis)
{
	return vec - 2.0f * dot(vec, axis) * axis;
}

/// Flattens the given vector on the given normalized axis.
template <class Class, class Component, size_t Dimension>
LEAN_INLINE vector<Component, 3> flatten(const tuple<Class, Component, Dimension> &vec,
										 const tuple<Class, Component, Dimension> &axis)
{
	return vec - dot(vec, axis) * axis;
}

/// Computes the squared length of the given vector.
template <class Class, class Component, size_t Dimension>
LEAN_INLINE Component lengthSq(const tuple<Class, Component, Dimension> &vector)
{
	return dot(vector, vector);
}

/// Computes the length of the given vector.
template <class Class, class Component, size_t Dimension>
inline Component length(const tuple<Class, Component, Dimension> &vector)
{
	using std::sqrt;
	return Component( sqrt( lengthSq(vector) ) );
}

/// Computes the squared distance between the given two vectors.
template <class LeftClass, class RightClass, class Component, size_t Dimension>
LEAN_INLINE Component distSq(const tuple<LeftClass, Component, Dimension> &left, const tuple<RightClass, Component, Dimension> &right)
{
	return lengthSq(left - right);
}

/// Computes the distance between the given two vectors.
template <class LeftClass, class RightClass, class Component, size_t Dimension>
LEAN_INLINE Component dist(const tuple<LeftClass, Component, Dimension> &left, const tuple<RightClass, Component, Dimension> &right)
{
	return length(left - right);
}

/// Normalizes the given vector.
template <class Class, class Component, size_t Dimension>
inline Class normalize(const tuple<Class, Component, Dimension> &vector)
{
	return vector * (Component(1) / length(vector));
}

/// Computes the cross product of the given two vectors.
template <class LeftClass, class RightClass, class Component>
LEAN_INLINE vector<Component, 3> cross(const tuple<LeftClass, Component, 3> &left, const tuple<RightClass, Component, 3> &right)
{
	vector<Component, 3> result(uninitialized);
	result[0] = left[1] * right[2] - left[2] * right[1];
	result[1] = left[2] * right[0] - left[0] * right[2];
	result[2] = left[0] * right[1] - left[1] * right[0];
	return result;
}

/// Constructs the n-th unit vector.
template <size_t Dimension, class Element>
LEAN_INLINE vector<Element, Dimension> unit(size_t n)
{
	vector<Element, Dimension> result;
	result[n] = Element(1);
	return result;
}

/// Sets the n-th component to the given value and all other components to zero.
template <size_t Dimension, class Element>
LEAN_INLINE vector<Element, Dimension> nvec(size_t n, Element value)
{
	vector<Element, Dimension> result;
	result[n] = value;
	return result;
}

/// Generates a perpendicular vector.
template <class Class, class Component>
inline vector<Component, 3> perpendicular(const tuple<Class, Component, 3> &vector)
{
	return cross( unit<3, float>(abs(vector[0]) > abs(vector[1])) , vector );
}

namespace Types
{
	using beMath::nvec;
	using beMath::unit;

} // namespace

} // namespace

#endif