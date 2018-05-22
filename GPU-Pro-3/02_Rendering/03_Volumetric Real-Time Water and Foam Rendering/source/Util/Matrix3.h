#ifndef __MATRIX3_H__
#define __MATRIX3_H__

#include "Vector3.h"

// -----------------------------------------------------------------------------
/// Matrix3
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Matrix3
{
public:
	Matrix3();
	Matrix3(float e11, float e12, float e13,
		float e21, float e22, float e23,
		float e31, float e32, float e33);
	Matrix3(const Matrix3& other);

	const float* operator[](int row) const;
	float* operator[](int row);
	float operator()(int row, int col) const;
	float& operator()(int row, int col);

	const Matrix3& operator=(const Matrix3& other);
	bool operator==(const Matrix3& other) const;
	bool operator!=(const Matrix3& other) const;

	const Matrix3 operator+(const Matrix3& other) const;
	const Matrix3& operator+=(const Matrix3& other);

	const Matrix3 operator-(const Matrix3& other) const;
	const Matrix3& operator-=(const Matrix3& other);

	const Matrix3 operator*(float scalar) const;
	const Matrix3& operator*=(float scalar);

	const Matrix3 operator*(const Matrix3& other) const;

	Vector3 operator*(const Vector3& vector3d) const;

	const Matrix3 operator/(float scalar) const;
	const Matrix3& operator/=(float scalar);

	Matrix3 Transposed(void) const;
	Matrix3 Inverse(void) const;

	void ToAxisAngle(Vector3& axis, float& angle) const;
	void FromAxisAngle(const Vector3& axis, float angle);

	Vector3 GetLookVector(void) const;
	Vector3 GetUpVector(void) const;
	Vector3 GetRightVector(void) const;

	void SetLookVector(const Vector3& vec);
	void SetUpVector(const Vector3& vec);
	void SetRightVector(const Vector3& vec);

	void BuildScale(float x, float y, float z);
	void BuildRotationX(float angle);
	void BuildRotationY(float angle);
	void BuildRotationZ(float angle);
	void BuildRotation(float yaw, float pitch, float roll);
	void BuildRotation(const Vector3& axis, float angle);

	static Matrix3 CreateScale(float x, float y, float z);
	static Matrix3 CreateRotation(const Vector3& axis, float angle);
	static Matrix3 CreateRotation(float yaw, float pitch, float roll);
	static Matrix3 CreateRotationX(float angle);
	static Matrix3 CreateRotationY(float angle);
	static Matrix3 CreateRotationZ(float angle);

public:
	float entry[9];

	static const Matrix3 ZERO;
	static const Matrix3 IDENTITY;
};

#endif //__MATRIX3_H__