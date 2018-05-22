
#ifndef __MATRIX4_H__
#define __MATRIX4_H__

#include "Vector4.h"
#include "Matrix3.h"

// -----------------------------------------------------------------------------
/// Matrix4
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Matrix4
{
public:
	Matrix4();
	Matrix4(float e11, float e12, float e13, float e14,
			float e21, float e22, float e23, float e24,
			float e31, float e32, float e33, float e34,
			float e41, float e42, float e43, float e44);
	Matrix4(const Matrix4& other);

	const float* operator[](int row) const;
	float* operator[](int row);
	float operator()(int row, int col) const;
	float& operator()(int row, int col);

	const Matrix4& operator=(const Matrix4& other);
	bool operator==(const Matrix4& other) const;
	bool operator!=(const Matrix4& other) const;

	Matrix4 operator+(const Matrix4& other) const;
	const Matrix4& operator+=(const Matrix4& other);

	Matrix4 operator-(const Matrix4& other) const;
	const Matrix4& operator-=(const Matrix4& other);

	Matrix4 operator*(float scalar) const;
	const Matrix4& operator*=(float scalar);

	Matrix4 operator*(const Matrix4& other) const;
	const Matrix4& operator*=(const Matrix4& other);

	Vector3 operator*(const Vector3& vector3d) const;
	Vector4 operator*(const Vector4& vector4d) const;

	Matrix4 operator/(float scalar) const;
	const Matrix4& operator/=(float scalar);

	const Matrix4 Transposed(void) const;
	const Matrix4 Inverse(void) const;
	void Invert(void);
	float Determinant(void) const;
	bool IsIdentity(void) const;

	void SetTranslation(float x, float y, float z);
	void SetTranslation(const Vector3& trans) { SetTranslation(trans.x, trans.y, trans.z); }
	void Translate(const Vector3& trans);
	Vector3 GetTranslation(void) const;

	void Scale(float x, float y, float z);
	void SetScale(float x, float y, float z);
	Vector3 GetScale(void) const;
	void OrthoNormalize(void);

	Matrix3 GetRotation(void) const;
	void SetRotation(const Matrix3& m);

	Vector3 GetRight(void) const;
	Vector3 GetUp(void) const;
	Vector3 GetDir(void) const;

	void SetRight(const Vector3& right);
	void SetUp(const Vector3& up);
	void SetDir(const Vector3& dir);

	void BuildTranslation(const Vector3& translation);
	void BuildTranslation(float x, float y, float z);
	void BuildScale(float x, float y, float z);	
	void BuildRotationX(float angle);
	void BuildRotationY(float angle);
	void BuildRotationZ(float angle);
	void BuildRotation(float yaw, float pitch, float roll);
	void BuildPerspective(float fieldOfView, float aspectRatio,
						  float nearPlaneZ, float farPlaneZ,
						  bool leftHanded = true);
	void BuildLookDir(const Vector3& position, const Vector3& direction, const Vector3& up);
	void BuildLookAt(const Vector3& position, const Vector3& at, const Vector3& up);
	void BuildPlaneMirrorMatrix(const Vector4& planeEquation);
	void BuildOrthogonal(float width, float height, float zNear, float zFar);

	static Matrix4 Matrix4Translation(const Vector3& translation);
	static Matrix4 Matrix4Translation(float x, float y, float z);
	static Matrix4 Matrix4Scale(float x, float y, float z);
	static Matrix4 Matrix4Scale(const Vector3& scale);
	static Matrix4 Matrix4Rotation(float yaw, float pitch, float roll);
	static Matrix4 Matrix4RotationX(float angle);
	static Matrix4 Matrix4RotationY(float angle);
	static Matrix4 Matrix4RotationZ(float angle);
	static Matrix4 Matrix4Perspective(float fieldOfView, float aspectRatio,
									float nearPlaneZ, float farPlaneZ,
									bool leftHanded = true);
	static Matrix4 Matrix4LookDir(const Vector3& position, const Vector3& direction, const Vector3& up);
	static Matrix4 Matrix4LookAt(const Vector3& position, const Vector3& at, const Vector3& up);
	static Matrix4 Matrix4PlaneMirrorMatrix(const Vector4& planeEquation);
	static Matrix4 Matrix4Orthogonal(float width, float height, float zNear, float zFar);

public:
	static const Matrix4 ZERO;
	static const Matrix4 IDENTITY;

	union
	{
		struct
		{
			// entries are stored in column-major format
			float entry[16];
		};

		struct
		{
			// first number is the row, second number is the column
			float m11; float m21; float m31; float m41;
			float m12; float m22; float m32; float m42;
			float m13; float m23; float m33; float m43;
			float m14; float m24; float m34; float m44;
		};

	};
};

#endif //__MATRIX4_H__
