/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_VECTOR_DEF
#define BE_MATH_VECTOR_DEF

#include "beMath.h"
#include "beVectorFwd.h"
#include "beTuple.h"

namespace beMath
{

/// Vector data.
template <class Component, size_t Dimension>
struct vector_data
{
	Component c[Dimension];
};
template <class Component>
struct vector_data<Component, 1> { union { Component x; Component c[1]; }; };
template <class Component>
struct vector_data<Component, 2> { union { struct { Component x, y; }; Component c[2]; }; };
template <class Component>
struct vector_data<Component, 3> { union { struct { Component x, y, z; }; Component c[3]; }; };
template <class Component>
struct vector_data<Component, 4> { union { struct { Component x, y, z, w; }; Component c[4]; }; };

/// Vector class.
template <class Component, size_t Dimension>
class vector : public tuple< vector<Component, Dimension>, Component, Dimension >, public vector_data<Component, Dimension>
{
private:
	typedef tuple< vector<Component, Dimension>, Component, Dimension > base_type;

public:
	/// Component type.
	typedef Component component_type;
	/// Size type.
	typedef typename base_type::size_type size_type;
	/// Number of components.
	static const size_type dimension = Dimension;
	/// Compatible scalar type.
	typedef component_type compatible_type;

	/// Creates a default-initialized vector.
	LEAN_INLINE vector() { memset(this->c, 0, sizeof(Component) * Dimension); }
	/// Creates an uninitialized vector.
	LEAN_INLINE vector(uninitialized_t) { }
	/// Initializes all components with the given value.
	LEAN_INLINE vector(const component_type &value)
	{
		base_type::assign(value);
	}
	/// Initializes all components with the components of the given tuple.
	template <class Class>
	LEAN_INLINE explicit vector(const tuple<Class, component_type, dimension> &right)
	{
		base_type::assign(right);
	}
	/// Initializes all components with the casted components of the given tuple.
	template <class Class, class Other, size_t OtherDimension>
	LEAN_INLINE explicit vector(const tuple<Class, Other, OtherDimension> &right)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(Dimension <= OtherDimension,
			"Destination tuple type cannot have more elements than source tuple type.",
			Destination_tuple_type_cannot_have_more_elements_than_source_tuple_type);

		for (size_t i = 0; i < Dimension; ++i)
			this->c[i] = static_cast<component_type>(right[i]);
	}
	/// Initializes all components with the casted elements of the given tuple, filling remaining components, if needed.
	template <class Class, class Other, size_t OtherDimension>
	LEAN_INLINE vector(const tuple<Class, Other, OtherDimension> &right, const Other &fill)
	{
		const size_t minDimension = min(Dimension, OtherDimension);
		size_t i = 0;

		for (; i < minDimension; ++i)
			this->c[i] = static_cast<component_type>(right[i]);

		for (; i < Dimension; ++i)
			this->c[i] = static_cast<component_type>(fill);
	}

	/// Assigns the given value to all vector components.
	LEAN_INLINE vector& operator =(const value_type &element)
	{
		return base_type::assign(element);
	}
	/// Assigns the given tuple to this vector.
	template <class OtherClass>
	LEAN_INLINE vector& operator =(const tuple<OtherClass, Component, Dimension> &right)
	{
		return base_type::assign(right);
	}

	/// Accesses the n-th component.
	LEAN_INLINE component_type& element(size_type n) { return this->c[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& element(size_type n) const { return this->c[n]; }

	/// Accesses the n-th component.
	LEAN_INLINE component_type& operator [](size_type n) { return this->c[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& operator [](size_type n) const { return this->c[n]; }

	/// Gets a raw data pointer.
	LEAN_INLINE component_type* data() { return this->c; }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* data() const { return this->c; }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* cdata() const { return this->c; }
};

/// Constructs a vector from the given values.
template <class Element>
LEAN_INLINE vector<Element, 1> vec(const Element &e1)
{
	return vector<Element, 1>(e1);
}
/// Constructs a vector from the given values.
template <class Element>
LEAN_INLINE vector<Element, 2> vec(const Element &e1, const Element &e2)
{
	vector<Element, 2> result(uninitialized);
	result[0] = e1;
	result[1] = e2;
	return result;
}
/// Constructs a vector from the given values.
template <class Element>
LEAN_INLINE vector<Element, 3> vec(const Element &e1, const Element &e2, const Element &e3)
{
	vector<Element, 3> result(uninitialized);
	result[0] = e1;
	result[1] = e2;
	result[2] = e3;
	return result;
}
/// Constructs a vector from the given values.
template <class Element>
LEAN_INLINE vector<Element, 4> vec(const Element &e1, const Element &e2, const Element &e3, const Element &e4)
{
	vector<Element, 4> result(uninitialized);
	result[0] = e1;
	result[1] = e2;
	result[2] = e3;
	result[3] = e4;
	return result;
}

namespace Types
{
	using beMath::vec;

} // namespace

} // namespace

#endif