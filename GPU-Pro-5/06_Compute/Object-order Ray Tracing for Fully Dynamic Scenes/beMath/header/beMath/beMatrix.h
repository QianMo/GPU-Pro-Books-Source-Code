/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_MATRIX
#define BE_MATH_MATRIX

#include "beMath.h"
#include "beMatrixDef.h"
#include "beVector.h"

namespace beMath
{

/// Constructs a projection matrix from the given values.
template <class Component>
LEAN_INLINE matrix<Component, 4, 4> mat_proj(const Component &fov, const Component &aspect,
	const Component &nearPlane, const Component &farPlane) // WARNING: never call these near & far, re-defined by libraries!
{
	matrix<Component, 4, 4> result;

	Component scaleY = Component(1) / tan(fov / Component(2));
	Component scaleX = scaleY / aspect;
	Component depth = farPlane - nearPlane;

	result[0][0] = scaleX;
	result[1][1] = scaleY;
	result[2][2] = farPlane / depth;
	result[2][3] = Component(1);
	result[3][2] = -nearPlane * farPlane / depth;

	return result;
}

/// Constructs a projection matrix from the given values.
template <class Component>
LEAN_INLINE matrix<Component, 4, 4> mat_proj_ortho(const Component &minX, const Component &minY,
	const Component &maxX, const Component &maxY,
	const Component &nearPlane, const Component &farPlane) // WARNING: never call these near & far, re-defined by libraries!
{
	matrix<Component, 4, 4> result;

	Component width = maxX - minX;
	Component height = maxY - minY;
	Component depth = farPlane - nearPlane;

	result[0][0] = Component(2) / width;
	result[1][1] = Component(2) / height;
	result[2][2] = Component(1) / depth;
	result[3][0] = -(minX + maxX) / width;
	result[3][1] = -(minY + maxY) / height;
	result[3][2] = -nearPlane / depth;
	result[3][3] = Component(1);

	return result;
}

/// Constructs a transformation matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3>
LEAN_INLINE matrix<Component, 3, 3> mat_transform3(
	const tuple<Tuple1, Component, 3> &look, const tuple<Tuple2, Component, 3> &up, const tuple<Tuple3, Component, 3> &right)
{
	matrix<Component, 3, 3> result(uninitialized);
	
	result[0] = right;
	result[1] = up;
	result[2] = look;
	
	return result;
}

/// Normalizes the rows.
template <class Component, size_t RowCount, size_t ColumnCount>
LEAN_INLINE matrix<Component, RowCount, ColumnCount> normalize_rows(const matrix<Component, RowCount, ColumnCount> &operand)
{
	matrix<Component, RowCount, ColumnCount> result(uninitialized);

	for (size_t i = 0; i < RowCount; ++i)
		result[i] = normalize(operand[i]);

	return result;
}

/// Constructs a transformation matrix from the given values.
template <class Component, class Tuple1, class Tuple2>
LEAN_INLINE matrix<Component, 3, 3> mat_transform(
	const tuple<Tuple1, Component, 3> &look, const tuple<Tuple2, Component, 3> &up)
{
	return mat_transform3(look, up, cross(up, look));
}

/// Constructs a transformation matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3, class Tuple4>
LEAN_INLINE matrix<Component, 4, 4> mat_transform(const tuple<Tuple1, Component, 3> &pos,
	const tuple<Tuple2, Component, 3> &look, const tuple<Tuple3, Component, 3> &up, const tuple<Tuple4, Component, 3> &right)
{
	matrix<Component, 4, 4> result(uninitialized);
	
	result[0] = vector<Component, 4>(right, 0.0f);
	result[1] = vector<Component, 4>(up, 0.0f);
	result[2] = vector<Component, 4>(look, 0.0f);
	result[3] = vector<Component, 4>(pos, 1.0f);

	return result;
}

/// Constructs a transformation matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3>
LEAN_INLINE matrix<Component, 4, 4> mat_transform4(const tuple<Tuple1, Component, 3> &pos,
	const tuple<Tuple2, Component, 3> &look, const tuple<Tuple3, Component, 3> &up)
{
	return mat_transform(pos, look, up, cross(up, look));
}

/// Constructs an inverse transformation matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3, class Tuple4>
LEAN_INLINE matrix<Component, 4, 4> mat_transform_inverse(const tuple<Tuple1, Component, 3> &pos,
	const tuple<Tuple2, Component, 3> &oneOverLook, const tuple<Tuple3, Component, 3> &oneOverUp, const tuple<Tuple4, Component, 3> &oneOverRight)
{
	matrix<Component, 4, 4> result(uninitialized);
	
	for (size_t i = 0; i < 3; ++i)
	{
		result[i][0] = oneOverRight[i];
		result[i][1] = oneOverUp[i];
		result[i][2] = oneOverLook[i];
		result[i][3] = Component(0);
	}

	result[3][0] = -dot(pos, oneOverRight);
	result[3][1] = -dot(pos, oneOverUp);
	result[3][2] = -dot(pos, oneOverLook);
	result[3][3] = Component(1);

	return result;
}

/// Constructs an inverse transformation matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3, class Tuple4>
LEAN_INLINE matrix<Component, 4, 4> mat_transform_inverse(const tuple<Tuple1, Component, 3> &pos,
	const tuple<Tuple2, Component, 3> &oneOverLook, const tuple<Tuple3, Component, 3> &oneOverUp)
{
	return mat_transform_inverse(pos, oneOverLook, oneOverUp, cross(oneOverUp, oneOverLook));
}

/// Constructs a view matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3, class Tuple4>
LEAN_INLINE matrix<Component, 4, 4> mat_view(const tuple<Tuple1, Component, 3> &pos,
	const tuple<Tuple2, Component, 3> &look, const tuple<Tuple3, Component, 3> &up, const tuple<Tuple4, Component, 3> &right)
{
	matrix<Component, 4, 4> result(uninitialized);
	
	for (size_t i = 0; i < 3; ++i)
	{
		result[i][0] = right[i];
		result[i][1] = up[i];
		result[i][2] = look[i];
		result[i][3] = Component(0);
	}

	result[3][0] = -dot(pos, right);
	result[3][1] = -dot(pos, up);
	result[3][2] = -dot(pos, look);
	result[3][3] = Component(1);

	return result;
}

/// Constructs a view matrix from the given values.
template <class Component, class Tuple1, class Tuple2, class Tuple3>
LEAN_INLINE matrix<Component, 4, 4> mat_view(const tuple<Tuple1, Component, 3> &pos,
	const tuple<Tuple2, Component, 3> &look, const tuple<Tuple3, Component, 3> &up)
{
	return mat_view(pos, look, up, cross(up, look));
}

/// Constructs a rotation matrix from the given x angle.
template <size_t Dimension, class Component>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_rot_x(const Component &angle)
{
	matrix<Component, Dimension, Dimension> result;

	Component cosAngle = cos(angle);
	Component sinAngle = sin(angle);
	
	result[0][0] = Component(1);
	result[1][1] = cosAngle;
	result[1][2] = sinAngle;
	result[2][1] = -sinAngle;
	result[2][2] = cosAngle;

	for (size_t i = 3; i < Dimension; ++i)
		result[i][i] = Component(1);

	return result;
}

/// Constructs a rotation matrix from the given y angle.
template <size_t Dimension, class Component>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_rot_y(const Component &angle)
{
	matrix<Component, Dimension, Dimension> result;

	Component cosAngle = cos(angle);
	Component sinAngle = sin(angle);
	
	result[0][0] = cosAngle;
	result[0][2] = -sinAngle;
	result[1][1] = Component(1);
	result[2][0] = sinAngle;
	result[2][2] = cosAngle;

	for (size_t i = 3; i < Dimension; ++i)
		result[i][i] = Component(1);

	return result;
}

/// Constructs a rotation matrix from the given z angle.
template <size_t Dimension, class Component>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_rot_z(const Component &angle)
{
	matrix<Component, Dimension, Dimension> result;

	Component cosAngle = cos(angle);
	Component sinAngle = sin(angle);
	
	result[0][0] = cosAngle;
	result[0][1] = sinAngle;
	result[1][0] = -sinAngle;
	result[1][1] = cosAngle;
	result[2][2] = Component(1);
	
	for (size_t i = 3; i < Dimension; ++i)
		result[i][i] = Component(1);

	return result;
}

/// Constructs a rotation matrix from the given axis & angle.
template <size_t Dimension, class Component, class TupleClass>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_rot(const tuple<TupleClass, Component, 3> &axis, const Component &angle)
{
	// Taken from CML
	matrix<Component, Dimension, Dimension> result;

    Component sinAngle = sin(angle);
    Component cosAngle = cos(angle);
    Component oneMinusCosAngle = Component(1) - cosAngle;

    Component xomc = axis[0] * oneMinusCosAngle;
    Component yomc = axis[1] * oneMinusCosAngle;
    Component zomc = axis[2] * oneMinusCosAngle;
    
    Component xxomc = axis[0] * xomc;
    Component yyomc = axis[1] * yomc;
    Component zzomc = axis[2] * zomc;
    Component xyomc = axis[0] * yomc;
    Component yzomc = axis[1] * zomc;
    Component zxomc = axis[2] * xomc;

    Component xs = axis[0] * sinAngle;
    Component ys = axis[1] * sinAngle;
    Component zs = axis[2] * sinAngle;

    result[0][0] = xxomc + cosAngle;
    result[0][1] = xyomc + zs;
    result[0][2] = zxomc - ys;
    result[1][0] = xyomc - zs;
    result[1][1] = yyomc + cosAngle;
    result[1][2] = yzomc + xs;
    result[2][0] = zxomc + ys;
    result[2][1] = yzomc - xs;
    result[2][2] = zzomc + cosAngle;

	for (size_t i = 3; i < Dimension; ++i)
		result[i][i] = Component(1);

	return result;
}

/// Constructs a scale matrix from the given scalings.
template <size_t Dimension, class Component>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_scale(const Component &x, const Component &y, const Component &z)
{
	matrix<Component, Dimension, Dimension> result;

	result[0][0] = x;
	result[1][1] = y;
	result[2][2] = z;
	
	for (size_t i = 3; i < Dimension; ++i)
		result[i][i] = Component(1);

	return result;
}

/// Multiplies the given two matrices.
template <class Component, size_t RowCount, size_t ColumnCount, size_t OtherColumnCount>
LEAN_INLINE matrix<Component, RowCount, OtherColumnCount> mul(
	const matrix<Component, RowCount, ColumnCount> &left,
	const matrix<Component, ColumnCount, OtherColumnCount> &right)
{
	matrix<Component, RowCount, OtherColumnCount> result;
	for (size_t i = 0; i < RowCount; ++i)
		for (size_t k = 0; k < ColumnCount; ++k)
			for (size_t j = 0; j < OtherColumnCount; ++j)
				result[i][j] += left[i][k] * right[k][j];
	return result;
}

/// Transforms the given vector.
template <class Component, size_t RowCount, size_t ColumnCount, class TupleClass>
LEAN_INLINE vector<Component, RowCount> mul(
	const matrix<Component, RowCount, ColumnCount> &left,
	const tuple<TupleClass, Component, ColumnCount> &right)
{
	vector<Component, RowCount> result(uninitialized);
	for (size_t i = 0; i < RowCount; ++i)
		result[i] = dot(left[i], right);
	return result;
}

/// Transforms the given vector.
template <class Component, size_t RowCount, size_t ColumnCount, class TupleClass>
LEAN_INLINE vector<Component, ColumnCount> mul(
	const tuple<TupleClass, Component, RowCount> &left,
	const matrix<Component, RowCount, ColumnCount> &right)
{
	vector<Component, ColumnCount> result;
	for (size_t k = 0; k < RowCount; ++k)
		for (size_t i = 0; i < ColumnCount; ++i)
			result[i] += left[k] * right[k][i];
	return result;
}

/// Transforms the given vector, assuming a homogeneous 1.
template <class Component, size_t RowCount, size_t ColumnCount, class TupleClass>
LEAN_INLINE vector<Component, ColumnCount - 1> mulh(
	const tuple<TupleClass, Component, RowCount - 1> &left,
	const matrix<Component, RowCount, ColumnCount> &right)
{
	vector<Component, ColumnCount - 1> result;
	for (size_t k = 0; k < RowCount - 1; ++k)
		for (size_t i = 0; i < ColumnCount - 1; ++i)
			result[i] += left[k] * right[k][i];
	for (size_t i = 0; i < ColumnCount - 1; ++i)
			result[i] += right[RowCount - 1][i];
	return result;
}


/// Transforms the given vector, assuming a homogeneous 1.
template <class Component, size_t RowCount, size_t ColumnCount, class TupleClass>
LEAN_INLINE vector<Component, ColumnCount> mulhx(
	const tuple<TupleClass, Component, RowCount - 1> &left,
	const matrix<Component, RowCount, ColumnCount> &right)
{
	vector<Component, ColumnCount> result;
	for (size_t k = 0; k < RowCount - 1; ++k)
		for (size_t i = 0; i < ColumnCount; ++i)
			result[i] += left[k] * right[k][i];
	for (size_t i = 0; i < ColumnCount; ++i)
			result[i] += right[RowCount - 1][i];
	return result;
}

/// Constructs a rotation matrix from the given angles.
template <size_t Dimension, class Component>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_rot_zxy(const Component &x, const Component &y, const Component &z)
{
	return mul( mul( mat_rot_z<Dimension>(z), mat_rot_x<Dimension>(x) ), mat_rot_y<Dimension>(y) );
}

/// Constructs a rotation matrix from the given angles.
template <size_t Dimension, class Component>
LEAN_INLINE matrix<Component, Dimension, Dimension> mat_rot_yxz(const Component &x, const Component &y, const Component &z)
{
	return mul( mat_rot_y<Dimension>(y), mul( mat_rot_x<Dimension>(x), mat_rot_z<Dimension>(z) ) );
}

/// Gets angles from the given rotation matrix.
template <size_t Dimension, class Component>
LEAN_INLINE vector<Component, 3> angles_rot_yxz(const matrix<Component, Dimension, Dimension> &rot)
{
	vector<Component, 3> angles(uninitialized);

	LEAN_STATIC_ASSERT_MSG_ALT(
		Dimension >= 3,
		"Matrix required to be at least 3x3",
		Matrix_required_to_be_at_least_3x3);

	angles[0] = asin( min(max(rot[1][2], Component(-1)), Component(1))  );

	if (abs(rot[1][2]) < Component(0.999999f))
	{
		angles[1] = atan2(-rot[0][2], rot[2][2]);
		angles[2] = atan2(-rot[1][0], rot[1][1]);
	}
	else
	{
		angles[1] = atan2(rot[2][0], rot[0][0]);
		angles[2] = Component(0);
	}

	return angles;
}

/// Gets angles from the given rotation matrix.
template <size_t Dimension, class Component>
LEAN_INLINE vector<Component, 3> angles_rot_zxy(const matrix<Component, Dimension, Dimension> &rot)
{
	vector<Component, 3> angles(uninitialized);

	LEAN_STATIC_ASSERT_MSG_ALT(
		Dimension >= 3,
		"Matrix required to be at least 3x3",
		Matrix_required_to_be_at_least_3x3);

	angles[0] = asin( min(max(-rot[2][1], Component(-1)), Component(1))  );

	if (abs(rot[2][1]) < Component(0.999999f))
	{
		angles[1] = atan2(rot[2][0], rot[2][2]);
		angles[2] = atan2(rot[0][1], rot[1][1]);
	}
	else
	{
		angles[1] = atan2(-rot[0][2], rot[0][0]);
		angles[2] = Component(0);
	}

	return angles;
}

/// Computes the determinant of the given matrix.
template <class Component>
LEAN_INLINE Component determinant(const matrix<Component, 1, 1> &operand)
{
	return operand[0][0];
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE Component determinant(const matrix<Component, 2, 2> &operand)
{
	return operand[0][0] * operand[1][1] - operand[0][1] * operand[1][0];
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE Component determinant(const matrix<Component, 3, 3> &operand)
{
	// Taken from CML
	return operand[0][0] * (operand[1][1] * operand[2][2] - operand[1][2] * operand[2][1])
		+ operand[0][1] * (operand[1][2] * operand[2][0] - operand[1][0] * operand[2][2])
		+ operand[0][2] * (operand[1][0] * operand[2][1] - operand[1][1] * operand[2][0]);
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE Component determinant(const matrix<Component, 4, 4> &operand)
{
	// Taken from CML
	Component m_22_33_23_32 = operand[2][2] * operand[3][3] - operand[2][3] * operand[3][2];
	Component m_23_30_20_33 = operand[2][3] * operand[3][0] - operand[2][0] * operand[3][3];
	Component m_20_31_21_30 = operand[2][0] * operand[3][1] - operand[2][1] * operand[3][0];
	Component m_21_32_22_31 = operand[2][1] * operand[3][2] - operand[2][2] * operand[3][1];
	Component m_23_31_21_33 = operand[2][3] * operand[3][1] - operand[2][1] * operand[3][3];
	Component m_20_32_22_30 = operand[2][0] * operand[3][2] - operand[2][2] * operand[3][0];

	Component d00 = operand[0][0]
		* (	 
			operand[1][1] * m_22_33_23_32
			+ operand[1][2] * m_23_31_21_33
			+ operand[1][3] * m_21_32_22_31
		  );
	Component d01 = operand[0][1]
		* (
			operand[1][0] * m_22_33_23_32
			+ operand[1][2] * m_23_30_20_33
			+ operand[1][3] * m_20_32_22_30
		  );
	Component d02 = operand[0][2]
		* (
			operand[1][0] * - m_23_31_21_33
			+ operand[1][1] * m_23_30_20_33
			+ operand[1][3] * m_20_31_21_30
		  );
	Component d03 = operand[0][3]
		* (
			operand[1][0] * m_21_32_22_31
			+ operand[1][1] * - m_20_32_22_30
			+ operand[1][2] * m_20_31_21_30
		  );

	return d00 - d01 + d02 - d03;
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE matrix<Component, 1, 1> inverse(const matrix<Component, 1, 1> &operand)
{
	return matrix<Component, 1, 1>(Component(1) / operand[0][0]);
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE matrix<Component, 2, 2> inverse(const matrix<Component, 2, 2> &operand)
{
	matrix<Component, 2, 2> result(uninitialized);

	Component oneOverDet = Component(1) / determinant(operand);
	result[1][1] = operand[0][0] * oneOverDet;
	result[0][1] = -operand[0][1] * oneOverDet;
	result[1][0] = -operand[1][0] * oneOverDet;
	result[0][0] = operand[1][1] * oneOverDet;

	return result;
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE matrix<Component, 3, 3> inverse(const matrix<Component, 3, 3> &operand)
{
	matrix<Component, 3, 3> result(uninitialized);

	// Taken from CML
	Component m_00 = operand[1][1] * operand[2][2] - operand[1][2] * operand[2][1];
	Component m_01 = operand[1][2] * operand[2][0] - operand[1][0] * operand[2][2];
	Component m_02 = operand[1][0] * operand[2][1] - operand[1][1] * operand[2][0];

	Component m_10 = operand[0][2] * operand[2][1] - operand[0][1] * operand[2][2];
	Component m_11 = operand[0][0] * operand[2][2] - operand[0][2] * operand[2][0];
	Component m_12 = operand[0][1] * operand[2][0] - operand[0][0] * operand[2][1];

	Component m_20 = operand[0][1] * operand[1][2] - operand[0][2] * operand[1][1];
	Component m_21 = operand[0][2] * operand[1][0] - operand[0][0] * operand[1][2];
	Component m_22 = operand[0][0] * operand[1][1] - operand[0][1] * operand[1][0];

	Component d = Component(1) / (operand[0][0] * m_00 + operand[0][1] * m_01 + operand[0][2] * m_02);

	result[0][0] = m_00 * d; result[0][1] = m_10 * d; result[0][2] = m_20 * d;
	result[1][0] = m_01 * d; result[1][1] = m_11 * d; result[1][2] = m_21 * d;
	result[2][0] = m_02 * d; result[2][1] = m_12 * d; result[2][2] = m_22 * d;

	return result;
}

/// Inverts the given matrix.
template <class Component>
LEAN_INLINE matrix<Component, 4, 4> inverse(const matrix<Component, 4, 4> &operand)
{
	matrix<Component, 4, 4> result(uninitialized);

	// Taken from CML
	Component m_22_33_23_32 = operand[2][2] * operand[3][3] - operand[2][3] * operand[3][2];
	Component m_23_30_20_33 = operand[2][3] * operand[3][0] - operand[2][0] * operand[3][3];
	Component m_20_31_21_30 = operand[2][0] * operand[3][1] - operand[2][1] * operand[3][0];
	Component m_21_32_22_31 = operand[2][1] * operand[3][2] - operand[2][2] * operand[3][1];
	Component m_23_31_21_33 = operand[2][3] * operand[3][1] - operand[2][1] * operand[3][3];
	Component m_20_32_22_30 = operand[2][0] * operand[3][2] - operand[2][2] * operand[3][0];

	Component d00 = operand[1][1] * m_22_33_23_32 + operand[1][2] * m_23_31_21_33 + operand[1][3] * m_21_32_22_31;
	Component d01 = operand[1][0] * m_22_33_23_32 + operand[1][2] * m_23_30_20_33 + operand[1][3] * m_20_32_22_30;
	Component d02 = operand[1][0] * -m_23_31_21_33 + operand[1][1] * m_23_30_20_33 + operand[1][3] * m_20_31_21_30;
	Component d03 = operand[1][0] * m_21_32_22_31 + operand[1][1] * -m_20_32_22_30 + operand[1][2] * m_20_31_21_30;

	Component d10 = operand[0][1] * m_22_33_23_32 + operand[0][2] * m_23_31_21_33 + operand[0][3] * m_21_32_22_31;
	Component d11 = operand[0][0] * m_22_33_23_32 + operand[0][2] * m_23_30_20_33 + operand[0][3] * m_20_32_22_30;
	Component d12 = operand[0][0] * -m_23_31_21_33 + operand[0][1] * m_23_30_20_33 + operand[0][3] * m_20_31_21_30;
	Component d13 = operand[0][0] * m_21_32_22_31 + operand[0][1] * -m_20_32_22_30 + operand[0][2] * m_20_31_21_30;

	Component m_02_13_03_12 = operand[0][2] * operand[1][3] - operand[0][3] * operand[1][2];
	Component m_03_10_00_13 = operand[0][3] * operand[1][0] - operand[0][0] * operand[1][3];
	Component m_00_11_01_10 = operand[0][0] * operand[1][1] - operand[0][1] * operand[1][0];
	Component m_01_12_02_11 = operand[0][1] * operand[1][2] - operand[0][2] * operand[1][1];
	Component m_03_11_01_13 = operand[0][3] * operand[1][1] - operand[0][1] * operand[1][3];
	Component m_00_12_02_10 = operand[0][0] * operand[1][2] - operand[0][2] * operand[1][0];

	Component d20 = operand[3][1] * m_02_13_03_12 + operand[3][2] * m_03_11_01_13 + operand[3][3] * m_01_12_02_11;
	Component d21 = operand[3][0] * m_02_13_03_12 + operand[3][2] * m_03_10_00_13 + operand[3][3] * m_00_12_02_10;
	Component d22 = operand[3][0] * -m_03_11_01_13 + operand[3][1] * m_03_10_00_13 + operand[3][3] * m_00_11_01_10;
	Component d23 = operand[3][0] * m_01_12_02_11 + operand[3][1] * -m_00_12_02_10 + operand[3][2] * m_00_11_01_10;

	Component d30 = operand[2][1] * m_02_13_03_12 + operand[2][2] * m_03_11_01_13 + operand[2][3] * m_01_12_02_11;
	Component d31 = operand[2][0] * m_02_13_03_12 + operand[2][2] * m_03_10_00_13 + operand[2][3] * m_00_12_02_10;
	Component d32 = operand[2][0] * -m_03_11_01_13 + operand[2][1] * m_03_10_00_13 + operand[2][3] * m_00_11_01_10;
	Component d33 = operand[2][0] * m_01_12_02_11 + operand[2][1] * -m_00_12_02_10 + operand[2][2] * m_00_11_01_10;

	Component d = Component(1) / (operand[0][0] * d00 - operand[0][1] * d01 + operand[0][2] * d02 - operand[0][3] * d03);
	result[0][0] = +d00 * d; result[0][1] = -d10 * d; result[0][2] = +d20 * d; result[0][3] = -d30 * d;
	result[1][0] = -d01 * d; result[1][1] = +d11 * d; result[1][2] = -d21 * d; result[1][3] = +d31 * d;
	result[2][0] = +d02 * d; result[2][1] = -d12 * d; result[2][2] = +d22 * d; result[2][3] = -d32 * d;
	result[3][0] = -d03 * d; result[3][1] = +d13 * d; result[3][2] = -d23 * d; result[3][3] = +d33 * d;

	return result;
}

} // namespace

#endif