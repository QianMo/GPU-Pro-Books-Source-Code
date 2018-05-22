
#ifndef __VECTOR4_H__
#define __VECTOR4_H__

#include "Vector3.h"


// -----------------------------------------------------------------------------
/// Vector4
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Vector4
{
public:
	Vector4();
	Vector4(const float x, const float y, const float z, const float w);
	Vector4(const Vector4& other);

	const Vector4& operator=(const Vector4& other);
	bool operator==(const Vector4& other) const;
	bool operator!=(const Vector4& other) const;
	bool operator<(const Vector4& other) const;

	const Vector4 operator-() const;

	const Vector4 operator+(const Vector4& other) const;
	const Vector4& operator+=(const Vector4& other);

	const Vector4 operator-(const Vector4& other) const;
	const Vector4& operator-=(const Vector4& other);

	const Vector4 operator*(const Vector4& other) const;
	const Vector4& operator*=(const Vector4& other);

	const Vector4 operator*(const float scalar) const;
	const Vector4& operator*=(const float scalar);

	const Vector4 operator/(const Vector4& other) const;
	const Vector4& operator/=(const Vector4& other);

	const Vector4 operator/(const float scalar) const;
	const Vector4& operator/=(const float scalar);

	float Length() const;
	float SquaredLength() const;
	float DotProduct(const Vector4& other) const;

public:
	static const Vector4 ZERO;
	static const Vector4 X_AXIS;
	static const Vector4 Y_AXIS;
	static const Vector4 Z_AXIS;
	static const Vector4 W_AXIS;

	union
	{
		struct
		{
			float comp[4];
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

#endif //__VECTOR4_H__
