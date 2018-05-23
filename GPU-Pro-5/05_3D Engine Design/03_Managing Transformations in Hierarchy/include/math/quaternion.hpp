#ifndef MATH_QUATERNION_HPP
#define MATH_QUATERNION_HPP

#include <cmath>

#include <math/common.hpp>
#include <math/vector.hpp>
#include <math/matrix.hpp>



class Quaternion
{
public:
	static Quaternion Identity()
	{
		Quaternion temp;
		temp.LoadIdentity();
		return temp;
	}

	static Quaternion Rotate(const Vector3& axis, float angle)
	{
		Quaternion temp;
		temp.LoadRotate(axis.x, axis.y, axis.z, angle);
		return temp;
	}

	static Quaternion RotateX(float angle)
	{
		Quaternion temp;
		temp.LoadRotateX(angle);
		return temp;
	}

	static Quaternion RotateY(float angle)
	{
		Quaternion temp;
		temp.LoadRotateY(angle);
		return temp;
	}

	static Quaternion RotateZ(float angle)
	{
		Quaternion temp;
		temp.LoadRotateZ(angle);
		return temp;
	}

	static Quaternion Slerp(const Quaternion& q1, const Quaternion q2, float t)
	{
		float omega = acosf(q1 % q2);

		float s1 = sinf((1.0f - t)*omega);
		float s2 = sinf(t*omega);
		float s3 = sinf(omega);
		float r1 = s1 / s3;
		float r2 = s2 / s3;

		Quaternion temp;

		temp.w = r1 * q1.w + r2 * q2.w;
		temp.x = r1 * q1.x + r2 * q2.x;
		temp.y = r1 * q1.y + r2 * q2.y;
		temp.z = r1 * q1.z + r2 * q2.z;

		return temp;
	}

public:
	Quaternion() {}

	Quaternion(float x, float y, float z, float w)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	Quaternion(const Vector3& axis, float angle)
	{
		LoadRotate(axis, angle);
	}

	float GetAngle() const
	{
		return 2.0f * ArcCos_Clamped(w);
	}

	Vector3 GetAxis() const
	{
		float squaredSinAngleOver2 = 1.0f - w*w;

		if (squaredSinAngleOver2 <= 0.0f)
		{
			return Vector3(1.0f, 0.0f, 0.0f);
		}
		else
		{
			float oneOverSinAngleOver2 = 1.0f / (sqrtf(squaredSinAngleOver2));
			return Vector3(oneOverSinAngleOver2*x, oneOverSinAngleOver2*y, oneOverSinAngleOver2*z);
		}
	}

	float GetLength() const
	{
		return sqrtf(*this % *this);
	}

	void Normalize()
	{
		float oneOverLength = 1.0f / GetLength();

		x *= oneOverLength;
		y *= oneOverLength;
		z *= oneOverLength;
		w *= oneOverLength;
	}

	Quaternion GetNormalized() const
	{
		Quaternion temp(x, y, z, w);
		temp.Normalize();
		return temp;
	}

	Quaternion GetConjugate() const
	{
		Quaternion temp;

		temp.x = -x;
		temp.y = -y;
		temp.z = -z;
		temp.w = w;

		return temp;
	}

	// it's simply a multiplication of angle by exponent
	void Pow(float exponent)
	{
		float angleOver2 = acosf(w);
		float newAngleOver2 = angleOver2 * exponent;
		float ratio = sinf(newAngleOver2) / sinf(angleOver2);

		x *= ratio;
		y *= ratio;
		z *= ratio;
		w = cosf(newAngleOver2);
	}

	Quaternion GetPowered(float exponent) const
	{
		Quaternion temp(x, y, z, w);
		temp.Pow(exponent);
		return temp;
	}

	Vector3 GetRotatedVector(const Vector3& v) const
	{
		Quaternion temp = (GetConjugate()) * (Quaternion(v.x, v.y, v.z, 0.0f)) * (*this);
		return Vector3(temp.x, temp.y, temp.z);
	}

	Vector3 ToEulerAngles() const
	{
		Vector3 temp;

		temp.x = atan2f(2.0f*(w*x + y*z), 1.0f - 2.0f*(x*x + y*y));
		temp.y = asinf(2.0f*(w*y - z*x));
		temp.z = atan2f(2.0f*(w*z + x*y), 1.0f - 2.0f*(y*y + z*z));

		return temp;
	}

	Matrix GetMatrix() const
	{
		return Matrix::Rotate(GetAxis(), GetAngle());
	}

	void LoadIdentity()
	{
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
		w = 1.0f;
	}

	void LoadRotate(float x, float y, float z, float angle)
	{
		angle = 0.5f * angle;
		float c = cosf(angle);
		float s = sinf(angle);

		this->x = x * s;
		this->y = y * s;
		this->z = z * s;
		this->w = c;
	}

	void LoadRotate(const Vector3& axis, float angle)
	{
		LoadRotate(axis.x, axis.y, axis.z, angle);
	}

	void LoadRotateX(float angle)
	{
		angle = 0.5f * angle;

		this->x = sinf(angle);
		this->y = 0.0f;
		this->z = 0.0f;
		this->w = cosf(angle);
	}

	void LoadRotateY(float angle)
	{
		angle = 0.5f * angle;

		this->x = 0.0f;
		this->y = sinf(angle);
		this->z = 0.0f;
		this->w = cosf(angle);
	}

	void LoadRotateZ(float angle)
	{
		angle = 0.5f * angle;

		this->x = 0.0f;
		this->y = 0.0f;
		this->z = sinf(angle);
		this->w = cosf(angle);
	}

	Quaternion& operator = (const Quaternion& q)
	{
		x = q.x;
		y = q.y;
		z = q.z;
		w = q.w;

		return *this;
	}

	bool operator == (const Quaternion& q) const
	{
		return (
			(fabs(x - q.x) < epsilon4) &&
			(fabs(y - q.y) < epsilon4) &&
			(fabs(z - q.z) < epsilon4) &&
			(fabs(w - q.w) < epsilon4)
			);
	}

	bool operator != (const Quaternion& q) const
	{
		return !(*this == q);
	}

	// dot product
	float operator % (const Quaternion& q) const
	{
		return x*q.x + y*q.y + z*q.z + w*q.w;
	}

	Quaternion operator * (const Quaternion& q) const
	{
		Quaternion temp;

		temp.x = w*q.x + x*q.w + z*q.y - y*q.z;
		temp.y = w*q.y + y*q.w + x*q.z - z*q.x;
		temp.z = w*q.z + z*q.w + y*q.x - x*q.y;
		temp.w = w*q.w - x*q.x - y*q.y - z*q.z;

		return temp;
	}

	Quaternion& operator *= (const Quaternion& q)
	{
		*this = *this * q;
		return *this;
	}

public:
	float x, y, z;
	float w;
};



inline ostream & operator << (ostream &stream, const Quaternion& q)
{
	stream << "[" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "]";
	return stream;
}



typedef Quaternion quat;



#endif
