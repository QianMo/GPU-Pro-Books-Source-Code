#include "Vector3.h"
#include "../Util//Math.h"

const Vector3 Vector3::ZERO(0.0f, 0.0f, 0.0f);
const Vector3 Vector3::ONE(1.0f, 1.0f, 1.0f);
const Vector3 Vector3::X_AXIS(1.0f, 0.0f, 0.0f);
const Vector3 Vector3::Y_AXIS(0.0f, 1.0f, 0.0f);
const Vector3 Vector3::Z_AXIS(0.0f, 0.0f, 1.0f);


// -----------------------------------------------------------------------------
// ------------------------------ Vector3::Length ------------------------------
// -----------------------------------------------------------------------------
float Vector3::Length() const
{
	return Math::Sqrt(SquaredLength());
}


// -----------------------------------------------------------------------------
// ----------------------------- Vector3::operator= ----------------------------
// -----------------------------------------------------------------------------
const Vector3& Vector3::operator=(const Vector3& other)
{
	x = other.x;
	y = other.y;
	z = other.z;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Vector3::operator/ ----------------------------
// -----------------------------------------------------------------------------
Vector3 Vector3::operator/(float scalar) const
{
	float reciprocal = 1 / scalar;
	return Vector3(x * reciprocal, y * reciprocal, z * reciprocal);
}


// -----------------------------------------------------------------------------
// ---------------------------- Vector3::operator/= ----------------------------
// -----------------------------------------------------------------------------
const Vector3& Vector3::operator/=(float scalar)
{
	float reciprocal = 1 / scalar;
	x *= reciprocal;
	y *= reciprocal;
	z *= reciprocal;
	return (*this);
}


// -----------------------------------------------------------------------------
// ----------------------------- Vector3::Normalize ----------------------------
// -----------------------------------------------------------------------------
void Vector3::Normalize()
{
	float len = Length();
	if (Math::IsNotEqual(len, 0.0f))
	{
		float invLen = 1.0f / len;
		x *= invLen;
		y *= invLen;
		z *= invLen;
	}
}


// -----------------------------------------------------------------------------
// ------------------------------ Vector3::Unitize -----------------------------
// -----------------------------------------------------------------------------
float Vector3::Unitize(float fTolerance /* = 1e-06f */)
{
	float length = Length();

	if ( length > fTolerance )
	{
		float invLength = 1.0f/length;
		x *= invLength;
		y *= invLength;
		z *= invLength;
	}
	else
	{
		length = 0.0f;
	}

	return length;
}


// -----------------------------------------------------------------------------
// ------------------------- Vector3::UnitCrossProduct -------------------------
// -----------------------------------------------------------------------------
Vector3 Vector3::UnitCrossProduct(const Vector3& other) const
{
	Vector3 cross(y * other.z - z * other.y,
							   z * other.x - x * other.z,
							   x * other.y - y * other.x);

	cross.Normalize();
	return cross;
}
