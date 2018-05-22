#ifndef __QUATERNION_H__
#define __QUATERNION_H__

#include "Matrix3.h"

// -----------------------------------------------------------------------------
/// Quaternion
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Quaternion
{
public:
	// default constructor, no initialization
	Quaternion();

	// constructors with initialization
	Quaternion(float x, float y, float z, float w);
	Quaternion(const Quaternion& other);
    Quaternion(const Matrix3& other);

	// constructor for building a quaternion from an axis and an angle
	Quaternion(const Vector3& axis, float angle);

	// assignment operator
	const Quaternion& operator=(const Quaternion& other);

	// comparison operators
	bool operator==(const Quaternion& other) const;
	bool operator!=(const Quaternion& other) const;

	// negation operator
	Quaternion operator-() const;

	// addition
	Quaternion operator+(const Quaternion& other) const;
	const Quaternion& operator+=(const Quaternion& other);

	// subtraction
	Quaternion operator-(const Quaternion& other) const;
	const Quaternion& operator-=(const Quaternion& other);

	// multiplication with another quaternion
	Quaternion operator*(const Quaternion& other) const;

	// multiplication with scalar
	Quaternion operator*(const float scalar) const;
	const Quaternion& operator*=(const float scalar);    

	// division by scalar
	Quaternion operator/(const float scalar) const;
	const Quaternion& operator/=(const float scalar);

    // multiplication with vector3
    Vector3 operator*(const Vector3& vec) const;

    Quaternion Inverse(void) const;     

	// methods
	float GetLength() const;
	float GetSquaredLength() const;
	float DotProduct(const Quaternion& other) const;
	void Normalize();
	void Negate();

	// rotation of a vector by a quaternion
	Vector3 Rotate(const Vector3& vector3d) const;

	//Get rotation matrix
	Matrix3 BuildMatrix(void) const;

	void ToEulerAngles(float& heading, float& bank, float &attitude) const;

	void ToAxisAngle(Vector3& axis, float& angle) const;
	void FromAxisAngle(const Vector3& axis, const float& angle);

	//normalized lerp
	Quaternion Lerp(const Quaternion& other, const float factor) const;
	//slerp
	Quaternion Slerp(const Quaternion& other, const float factor) const;

	void Print(const char* name = "Quaternion") const;

public:
	static const Quaternion ZERO;
	static const Quaternion IDENTITY;

	union
	{
		struct
		{
			float tuple[4];
		};

		struct
		{
			float x;
			float y;
			float z;
			float w;
		};
	};
};
#endif //__QUATERNION_H__