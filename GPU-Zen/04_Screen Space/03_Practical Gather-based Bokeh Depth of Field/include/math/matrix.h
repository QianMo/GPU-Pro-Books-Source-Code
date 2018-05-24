#pragma once


#include "types.h"


namespace NMath
{
	Matrix MatrixCustom(
		float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33);
	Matrix MatrixCopy(const Matrix& m);

	Matrix Add(const Matrix& m1, const Matrix& m2);
	Matrix Sub(const Matrix& m1, const Matrix& m2);
	Matrix Mul(const Matrix& m1, const Matrix& m2);
	Matrix Transpose(const Matrix& m);
	Matrix Invert(const Matrix& m);
	Matrix Orthogonalize3x3(const Matrix& m);
	void MulIn(Matrix& m1, const Matrix& m2);
	void TransposeIn(Matrix& m);
	void InvertIn(Matrix& m);
	void Orthogonalize3x3In(Matrix& m);
	Matrix operator * (const Matrix& m1, const Matrix& m2);
	Matrix operator + (const Matrix& m1, const Matrix& m2);
	Matrix operator - (const Matrix& m1, const Matrix& m2);

	void SetZeros(Matrix& m);
	void SetIdentity(Matrix& m);
	void SetTranslate(Matrix& m, float x, float y, float z);
	void SetTranslate(Matrix& m, const Vector3& v);
	void SetRotate(Matrix& m, float x, float y, float z, float angle);
	void SetRotate(Matrix& m, const Vector3& axis, float angle);
	void SetRotateX(Matrix& m, float angle);
	void SetRotateY(Matrix& m, float angle);
	void SetRotateZ(Matrix& m, float angle);
	void SetScale(Matrix& m, float x, float y, float z);
	void SetScale(Matrix& m, const Vector3& s);
	void SetReflect(Matrix& m, const Plane& plane);
	void SetLookAtLH(Matrix& m, const Vector3& eye, const Vector3& at, const Vector3& up);
	void SetLookAtRH(Matrix& m, const Vector3& eye, const Vector3& at, const Vector3& up);
	void SetPerspectiveFovLH(Matrix& m, ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar);
	void SetPerspectiveFovRH(Matrix& m, ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar);
	void SetOrthoOffCenterLH(Matrix& m, ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar);
	void SetOrthoOffCenterRH(Matrix& m, ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar);
	void SetOrthoLH(Matrix& m, ZRange zRange, float width, float height, float zNear, float zFar);
	void SetOrthoRH(Matrix& m, ZRange zRange, float width, float height, float zNear, float zFar);

	Matrix MatrixZeros();
	Matrix MatrixIdentity();
	Matrix MatrixTranslate(float x, float y, float z);
	Matrix MatrixTranslate(const Vector3& v);
	Matrix MatrixRotate(float x, float y, float z, float angle);
	Matrix MatrixRotate(const Vector3& axis, float angle);
	Matrix MatrixRotateX(float angle);
	Matrix MatrixRotateY(float angle);
	Matrix MatrixRotateZ(float angle);
	Matrix MatrixScale(float x, float y, float z);
	Matrix MatrixScale(const Vector3& s);
	Matrix MatrixReflect(const Plane& plane);
	Matrix MatrixLookAtLH(const Vector3& eye, const Vector3& at, const Vector3& up);
	Matrix MatrixLookAtRH(const Vector3& eye, const Vector3& at, const Vector3& up);
	Matrix MatrixPerspectiveFovLH(ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar);
	Matrix MatrixPerspectiveFovRH(ZRange zRange, float fovY, float aspectRatio, float zNear, float zFar);
	Matrix MatrixOrthoOffCenterLH(ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar);
	Matrix MatrixOrthoOffCenterRH(ZRange zRange, float left, float right, float bottom, float top, float zNear, float zFar);
	Matrix MatrixOrthoLH(ZRange zRange, float width, float height, float zNear, float zFar);
	Matrix MatrixOrthoRH(ZRange zRange, float width, float height, float zNear, float zFar);

	float Determinant(
		float _00, float _01, float _02,
		float _10, float _11, float _12,
		float _20, float _21, float _22);

	//

	inline float Determinant(
		float _00, float _01, float _02,
		float _10, float _11, float _12,
		float _20, float _21, float _22)
	{
		return ( (_00*_11*_22 + _10*_21*_02 + _20*_01*_12) - (_02*_11*_20 + _12*_21*_00 + _22*_01*_10) );
	}
}
