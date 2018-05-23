/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_MATRIX_FWD
#define BE_MATH_MATRIX_FWD

#include "beMath.h"

namespace beMath
{

/// Matrix class.
template <class Component, size_t RowCount, size_t ColumnCount>
class matrix;

namespace Types
{
	using beMath::matrix;

	/// 2-dimensional float matrix.
	typedef matrix<lean::float4, 2, 2> fmat2;
	/// 3-dimensional float matrix.
	typedef matrix<lean::float4, 3, 3> fmat3;
	/// 4-dimensional float matrix.
	typedef matrix<lean::float4, 4, 4> fmat4;

	/// 2-dimensional double matrix.
	typedef matrix<lean::float8, 2, 2> dmat2;
	/// 3-dimensional double matrix.
	typedef matrix<lean::float8, 3, 3> dmat3;
	/// 4-dimensional double matrix.
	typedef matrix<lean::float8, 4, 4> dmat4;

	/// 2-dimensional int4 matrix.
	typedef matrix<lean::int4, 2, 2> imat2;
	/// 3-dimensional int4 matrix.
	typedef matrix<lean::int4, 3, 3> imat3;
	/// 4-dimensional int4 matrix.
	typedef matrix<lean::int4, 4, 4> imat4;

	/// 2-dimensional int8 matrix.
	typedef matrix<lean::int8, 2, 2> lmat2;
	/// 3-dimensional int8 matrix.
	typedef matrix<lean::int8, 3, 3> lmat3;
	/// 4-dimensional int8 matrix.
	typedef matrix<lean::int8, 4, 4> lmat4;

} // namespace

using namespace Types;

} // namespace

#endif