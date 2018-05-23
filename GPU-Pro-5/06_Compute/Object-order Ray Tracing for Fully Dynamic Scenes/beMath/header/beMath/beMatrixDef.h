/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_MATRIX_DEF
#define BE_MATH_MATRIX_DEF

#include "beMath.h"
#include "beMatrixFwd.h"
#include "beTuple.h"
#include "beComparisonOperators.h"
#include "beVectorDef.h"

namespace beMath
{

// Named debug info.
template <class Component, size_t Dimension>
struct matrix_components { };
template <class Component>
struct matrix_components<Component, 1> { Component _11; };
template <class Component>
struct matrix_components<Component, 2> { Component _11, _12; Component _21, _22; };
template <class Component>
struct matrix_components<Component, 3> { Component _11, _12, _13; Component _21, _22, _23; Component _31, _32, _33; };
template <class Component>
struct matrix_components<Component, 4> { Component _11, _12, _13, _14; Component _21, _22, _23, _24; Component _31, _32, _33, _34; Component _41, _42, _43, _44; };

/// Matrix class.
template <class Component, size_t RowCount, size_t ColumnCount>
class matrix : public tuple<
	matrix<Component, RowCount, ColumnCount>,
	Component,
	RowCount * ColumnCount>
{
private:
	typedef tuple<
		matrix<Component, RowCount, ColumnCount>,
		Component,
		RowCount * ColumnCount> base_type;

public:
	union
	{
		Component c[RowCount * ColumnCount];
		matrix_components<Component, (RowCount < ColumnCount) ? RowCount : ColumnCount> n;
		Component r[RowCount][ColumnCount];
	};

	/// Component type.
	typedef Component component_type;
	/// Size type.
	typedef typename base_type::size_type size_type;
	/// Row count.
	static const size_type rows = RowCount;
	/// Column count.
	static const size_type columns = ColumnCount;
	/// Component count.
	static const size_type dimension = rows * columns;

	/// Identity matrix.
	static const matrix identity;

	/// Row type.
	typedef vector<component_type, columns> row_type;
	/// Compatible scalar type.
	typedef component_type compatible_type;

	/// Creates a default-initialized matrix.
	LEAN_INLINE matrix()
		: c() { }
	/// Creates am un-initialized matrix.
	LEAN_INLINE matrix(uninitialized_t) { }
	/// Initializes all components with the components of given tuple.
	template <class Class>
	LEAN_INLINE explicit matrix(const tuple<Class, component_type, dimension> &right)
	{
		base_type::assign(right);
	}
	/// Initializes all components with the given value.
	LEAN_INLINE explicit matrix(const component_type &element)
	{
		base_type::assign(element);
	}
	/// Initializes all components with the casted components of the given matrix.
	template <class Other, size_t OtherRowCount, size_t OtherColumnCount>
	LEAN_INLINE explicit matrix(const matrix<Other, OtherRowCount, OtherColumnCount> &right)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(RowCount <= OtherRowCount,
			"Destination matrix type cannot have more elements than source matrix type.",
			Destination_matrix_type_cannot_have_more_elements_than_source_matrix_type);

		for (size_t i = 0; i < RowCount; ++i)
			this->row(i) = static_cast<row_type>(right[i]);
	}
	/// Initializes all components with the casted components of the given matrix, filling remaining matrix components, if needed.
	template <class Other, size_t OtherRowCount, size_t OtherColumnCount>
	LEAN_INLINE explicit matrix(const matrix<Other, OtherRowCount, OtherColumnCount> &right, const Other &fill)
	{
		const size_t minRowCount = min(RowCount, OtherRowCount);
		size_t i = 0;

		for (; i < minRowCount; ++i)
			this->row(i) = row_type(right[i], fill);

		for (; i < RowCount; ++i)
			this->row(i) = static_cast<component_type>(fill);
	}

	/// Assigns the given value to all matrix components.
	LEAN_INLINE class_type& operator =(const value_type &element)
	{
		return base_type::assign(element);
	}
	/// Assigns the given tuple to this matrix.
	template <class OtherClass>
	LEAN_INLINE class_type& operator =(const tuple<OtherClass, Component, dimension> &right)
	{
		return base_type::assign(right);
	}

	/// Accesses the n-th component.
	LEAN_INLINE component_type& element(size_type n) { return this->c[n]; }
	/// Accesses the n-th component.
	LEAN_INLINE const component_type& element(size_type n) const { return this->c[n]; }

	/// Accesses the n-th row.
	LEAN_INLINE row_type& row(size_type n) { return *reinterpret_cast<row_type*>(this->r[n]); }
	/// Accesses the n-th row.
	LEAN_INLINE const row_type& row(size_type n) const { return *reinterpret_cast<const row_type*>(this->r[n]); }

	/// Accesses the n-th row.
	LEAN_INLINE row_type& operator [](size_type n) { return row(n); }
	/// Accesses the n-th row.
	LEAN_INLINE const row_type& operator [](size_type n) const { return row(n); }

	/// Gets a raw data pointer.
	LEAN_INLINE component_type* data() { return this->c; }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* data() const { return this->c; }
	/// Gets a raw data pointer.
	LEAN_INLINE const component_type* cdata() const { return this->c; }
};

/// Constructs a diagonal matrix from the given value.
template <size_t RowCount, size_t ColumnCount, class Component>
LEAN_INLINE matrix<Component, RowCount, ColumnCount> mat_diag(const Component &value)
{
	matrix<Component, RowCount, ColumnCount> result;
	const size_t entryCount = min(RowCount, ColumnCount);
	for (size_t i = 0; i < entryCount; ++i)
		result[i][i] = value;
	return result;
}

// Identity matrix.
template <class Component, size_t RowCount, size_t ColumnCount>
const matrix<Component, RowCount, ColumnCount> matrix<Component, RowCount, ColumnCount>::identity
	= mat_diag<RowCount, ColumnCount>( Component(1) );

/// Transposes the given matrix.
template <class Component, size_t RowCount, size_t ColumnCount>
LEAN_INLINE matrix<Component, ColumnCount, RowCount> transpose(const matrix<Component, RowCount, ColumnCount> &operand)
{
	matrix<Component, ColumnCount, RowCount> result(uninitialized);
	for (size_t i = 0; i < RowCount; ++i)
		for (size_t j = 0; j < ColumnCount; ++j)
			result[j][i] = operand[i][j];
	return result;
}

/// Constructs a row matrix from the given rows.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, 1, Dimension> mat_row(const tuple<Tuple, Component, Dimension> &t1)
{
	matrix<Component, 1, Dimension> result(uninitialized);
	result[0] = t1;
	return result;
}
/// Constructs a row matrix from the given rows.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, 2, Dimension> mat_row(const tuple<Tuple, Component, Dimension> &t1,
	const tuple<Tuple, Component, Dimension> &t2)
{
	matrix<Component, 2, Dimension> result(uninitialized);
	result[0] = t1;
	result[1] = t2;
	return result;
}
/// Constructs a row matrix from the given rows.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, 3, Dimension> mat_row(const tuple<Tuple, Component, Dimension> &t1,
	const tuple<Tuple, Component, Dimension> &t2,
	const tuple<Tuple, Component, Dimension> &t3)
{
	matrix<Component, 3, Dimension> result(uninitialized);
	result[0] = t1;
	result[1] = t2;
	result[2] = t3;
	return result;
}
/// Constructs a row matrix from the given rows.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, 4, Dimension> mat_row(const tuple<Tuple, Component, Dimension> &t1,
	const tuple<Tuple, Component, Dimension> &t2,
	const tuple<Tuple, Component, Dimension> &t3,
	const tuple<Tuple, Component, Dimension> &t4)
{
	matrix<Component, 4, Dimension> result(uninitialized);
	result[0] = t1;
	result[1] = t2;
	result[2] = t3;
	result[3] = t4;
	return result;
}

/// Constructs a column matrix from the given columns.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, Dimension, 1> mat_col(const tuple<Tuple, Component, Dimension> &t1)
{
	matrix<Component, Dimension, 1> result(uninitialized);
	for (size_t i = 0; i < Dimension; ++i)
		result[i][0] = t1[i];
	return result;
}
/// Constructs a column matrix from the given columns.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, Dimension, 2> mat_col(const tuple<Tuple, Component, Dimension> &t1,
	const tuple<Tuple, Component, Dimension> &t2)
{
	matrix<Component, Dimension, 2> result(uninitialized);
	for (size_t i = 0; i < Dimension; ++i)
		result[i][0] = t1[i];
	for (size_t i = 0; i < Dimension; ++i)
		result[i][1] = t2[i];
	return result;
}
/// Constructs a column matrix from the given columns.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, Dimension, 3> mat_col(const tuple<Tuple, Component, Dimension> &t1,
	const tuple<Tuple, Component, Dimension> &t2,
	const tuple<Tuple, Component, Dimension> &t3)
{
	matrix<Component, Dimension, 3> result(uninitialized);
	for (size_t i = 0; i < Dimension; ++i)
		result[i][0] = t1[i];
	for (size_t i = 0; i < Dimension; ++i)
		result[i][1] = t2[i];
	for (size_t i = 0; i < Dimension; ++i)
		result[i][2] = t3[i];
	return result;
}
/// Constructs a column matrix from the given columns.
template <class Component, size_t Dimension, class Tuple>
LEAN_INLINE matrix<Component, Dimension, 4> mat_col(const tuple<Tuple, Component, Dimension> &t1,
	const tuple<Tuple, Component, Dimension> &t2,
	const tuple<Tuple, Component, Dimension> &t3,
	const tuple<Tuple, Component, Dimension> &t4)
{
	matrix<Component, Dimension, 4> result(uninitialized);
	for (size_t i = 0; i < Dimension; ++i)
		result[i][0] = t1[i];
	for (size_t i = 0; i < Dimension; ++i)
		result[i][1] = t2[i];
	for (size_t i = 0; i < Dimension; ++i)
		result[i][2] = t3[i];
	for (size_t i = 0; i < Dimension; ++i)
		result[i][3] = t4[i];
	return result;
}

namespace Types
{
	using beMath::mat_diag;
	using beMath::mat_row;
	using beMath::mat_col;

} // namespace

} // namespace

#endif