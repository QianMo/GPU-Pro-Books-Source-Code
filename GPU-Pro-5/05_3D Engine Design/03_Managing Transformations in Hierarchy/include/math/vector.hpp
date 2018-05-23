#ifndef MATH_VECTOR_HPP
#define MATH_VECTOR_HPP

#include <cmath>
#include <iostream>

#include <math/constants.hpp>

using namespace std;



class Vector2;
class Vector3;
class Vector4;

class Matrix;



// ----------------------------------------------------------------------------



class Vector2
{
public:
	static float GetDistance(const Vector2& v1, const Vector2& v2)
	{
		float dx = v1.x - v2.x;
		float dy = v1.y - v2.y;

		return sqrtf(dx*dx + dy*dy);
	}

	static float GetAngle(const Vector2& v1, const Vector2& v2)
	{
		return acosf(v1 % v2);
	}

	static Vector2 GetReflectedVector(const Vector2& input, const Vector2& normal)
	{
		return (input - 2.0f*normal * (input % normal));
	}

	static Vector2 Slerp(const Vector2& v1, const Vector2& v2, float t)
	{
		float omega = GetAngle(v1, v2);

		float s1 = sinf((1.0f - t)*omega);
		float s2 = sinf(t*omega);
		float s3 = sinf(omega);

		return s1/s3 * v1 + s2/s3 * v2;
	}

public:
	Vector2() {}

	Vector2(float xy)
	{
		x = xy;
		y = xy;
	}

	Vector2(float x, float y)
	{
		this->x = x;
		this->y = y;
	}

	Vector2(const Vector2& v)
	{
		x = v.x;
		y = v.y;
	}

	explicit Vector2(const Vector3& v);

	explicit Vector2(const Vector4& v);

	void SetLength(float length)
	{
		Vector2 temp(x, y);
		temp.Normalize();

		x = temp.x * length;
		y = temp.y * length;
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
	}

	Vector2 GetNormalized() const
	{
		Vector2 temp(x, y);
		temp.Normalize();
		return temp;
	}

	Vector2& operator = (const Vector2& v)
	{
		x = v.x;
		y = v.y;

		return *this;
	}

	bool operator == (const Vector2& v) const
	{
		return (
			(fabs(x - v.x) < epsilon4) &&
			(fabs(y - v.y) < epsilon4)
			);
	}

	bool operator != (const Vector2& v) const
	{
		return !(*this == v);
	}

	// dot product
	float operator % (const Vector2& v) const
	{
		return (x*v.x + y*v.y);
	}

	Vector2& operator += (const Vector2& v)
	{
		x += v.x;
		y += v.y;

		return *this;
	}

	Vector2 operator + (const Vector2& v) const
	{
		Vector2 temp(*this);
		temp += v;
		return temp;
	}

	Vector2 operator - () const
	{
		return Vector2(-x, -y);
	}

	Vector2& operator -= (const Vector2& v)
	{
		x -= v.x;
		y -= v.y;

		return *this;
	}

	Vector2 operator - (const Vector2& v) const
	{
		Vector2 temp(*this);
		temp -= v;
		return temp;
	}

	Vector2& operator *= (const Vector2& v)
	{
		x *= v.x;
		y *= v.y;

		return *this;
	}

	Vector2 operator * (const Vector2& v) const
	{
		Vector2 temp(*this);
		temp *= v;
		return temp;
	}

	Vector2& operator *= (float s)
	{
		x *= s;
		y *= s;

		return *this;
	}

	Vector2 operator * (float s) const
	{
		Vector2 temp(*this);
		temp *= s;
		return temp;
	}

	friend Vector2 operator * (float s, const Vector2& v)
	{
		return v * s;
	}

public:
	union
	{
		struct
		{
			float x, y;
		};

		float _[2];
	};
};



inline ostream & operator << (ostream &stream, const Vector2& v)
{
	stream << "[" << v.x << ", " << v.y << "]";
	return stream;
}



typedef Vector2 vec2;



// ----------------------------------------------------------------------------



class Vector3
{
public:
	static float GetDistance(const Vector3& v1, const Vector3& v2)
	{
		float dx = v1.x - v2.x;
		float dy = v1.y - v2.y;
		float dz = v1.z - v2.z;

		return sqrtf(dx*dx + dy*dy + dz*dz);
	}

	static float GetAngle(const Vector3& v1, const Vector3& v2)
	{
		return acosf(v1 % v2);
	}

	static Vector3 GetReflectedVector(const Vector3& input, const Vector3& normal)
	{
		return (input - 2.0f*normal * (input % normal));
	}

	static Vector3 Slerp(const Vector3& v1, const Vector3& v2, float t)
	{
		float omega = GetAngle(v1, v2);

		float s1 = sinf((1.0f - t)*omega);
		float s2 = sinf(t*omega);
		float s3 = sinf(omega);

		return s1/s3 * v1 + s2/s3 * v2;
	}

public:
	static const Vector3 Zero;
	static const Vector3 One;
	static const Vector3 AxisX;
	static const Vector3 AxisY;
	static const Vector3 AxisZ;

public:
	Vector3() {}

	Vector3(float xyz)
	{
		x = xyz;
		y = xyz;
		z = xyz;
	}

	Vector3(float x, float y, float z)
	{
		this->x = x;
		this->y = y;
		this->z = z;
	}

	Vector3(const Vector3& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
	}

	explicit Vector3(const Vector2& v);

	explicit Vector3(const Vector4& v);

	void SetLength(float length)
	{
		Vector3 temp(x, y, z);
		temp.Normalize();

		x = temp.x * length;
		y = temp.y * length;
		z = temp.z * length;
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
	}

	Vector3 GetNormalized() const
	{
		Vector3 temp(x, y, z);
		temp.Normalize();
		return temp;
	}

	Vector3& operator = (const Vector3& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;

		return *this;
	}

	bool operator == (const Vector3& v) const
	{
		return (
			(fabs(x - v.x) < epsilon4) &&
			(fabs(y - v.y) < epsilon4) &&
			(fabs(z - v.z) < epsilon4)
			);
	}

	bool operator != (const Vector3& v) const
	{
		return !(*this == v);
	}

	// dot product
	float operator % (const Vector3& v) const
	{
		return (x*v.x + y*v.y + z*v.z);
	}

	// cross product
	Vector3 operator ^ (const Vector3& v) const
	{
		return Vector3(y*v.z - z*v.y, z*v.x - x*v.z, x*v.y - y*v.x);
	}

	Vector3& operator += (const Vector3& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;

		return *this;
	}

	Vector3 operator + (const Vector3& v) const
	{
		Vector3 temp(*this);
		temp += v;
		return temp;
	}

	Vector3 operator - () const
	{
		return Vector3(-x, -y, -z);
	}

	Vector3& operator -= (const Vector3& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;

		return *this;
	}

	Vector3 operator - (const Vector3& v) const
	{
		Vector3 temp(*this);
		temp -= v;
		return temp;
	}

	Vector3& operator *= (const Vector3& v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;

		return *this;
	}

	Vector3 operator * (const Vector3& v) const
	{
		Vector3 temp(*this);
		temp *= v;
		return temp;
	}

	Vector3& operator *= (float s)
	{
		x *= s;
		y *= s;
		z *= s;

		return *this;
	}

	Vector3 operator * (float s) const
	{
		Vector3 temp(*this);
		temp *= s;
		return temp;
	}

	friend Vector3 operator * (float s, const Vector3& v)
	{
		return v * s;
	}

	Vector3 operator * (const Matrix& m) const;
	Vector3& operator *= (const Matrix& m);

public:
	union
	{
		struct
		{
			float x, y, z;
		};

		float _[3];
	};
};



inline ostream & operator << (ostream &stream, const Vector3& v)
{
	stream << "[" << v.x << ", " << v.y << ", " << v.z << "]";
	return stream;
}



typedef Vector3 vec3;



// ----------------------------------------------------------------------------



class Vector4
{
public:
	Vector4() {}

	Vector4(float xyzw)
	{
		x = xyzw;
		y = xyzw;
		z = xyzw;
		w = xyzw;
	}

	Vector4(float xyz, float w)
	{
		this->x = xyz;
		this->y = xyz;
		this->z = xyz;
		this->w = w;
	}

	Vector4(float x, float y, float z, float w = 1.0f)
	{
		this->x = x;
		this->y = y;
		this->z = z;
		this->w = w;
	}

	Vector4(const Vector4& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;
	}

	explicit Vector4(const Vector2& v);

	explicit Vector4(const Vector3& v);

	void DivideByW()
	{
		float oneOverW = 1.0f / w;
		x *= oneOverW;
		y *= oneOverW;
		z *= oneOverW;
		w = 1.0f;
	}

	Vector4 GetDividedByW() const
	{
		float oneOverW = 1.0f / w;
		return Vector4(x*oneOverW, y*oneOverW, z*oneOverW, 1.0f);
	}

	Vector4& operator = (const Vector4& v)
	{
		x = v.x;
		y = v.y;
		z = v.z;
		w = v.w;

		return *this;
	}

	bool operator == (const Vector4& v) const
	{
		return (
			(fabs(x - v.x) < epsilon4) &&
			(fabs(y - v.y) < epsilon4) &&
			(fabs(z - v.z) < epsilon4) &&
			(fabs(w - v.w) < epsilon4)
			);
	}

	bool operator != (const Vector4& v) const
	{
		return !(*this == v);
	}

	// dot product
	float operator % (const Vector4& v) const
	{
		return (x*v.x + y*v.y + z*v.z + w*v.w);
	}

	Vector4& operator += (const Vector4& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		w += v.w;

		return *this;
	}

	Vector4 operator + (const Vector4& v) const
	{
		Vector4 temp(*this);
		temp += v;
		return temp;
	}

	Vector4 operator - () const
	{
		return Vector4(-x, -y, -z, -w);
	}

	Vector4& operator -= (const Vector4& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		w -= v.w;

		return *this;
	}

	Vector4 operator - (const Vector4& v) const
	{
		Vector4 temp(*this);
		temp -= v;
		return temp;
	}

	Vector4& operator *= (const Vector4& v)
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;
		w *= v.w;

		return *this;
	}

	Vector4 operator * (const Vector4& v) const
	{
		Vector4 temp(*this);
		temp *= v;
		return temp;
	}

	Vector4& operator *= (float s)
	{
		x *= s;
		y *= s;
		z *= s;
		w *= s;

		return *this;
	}

	Vector4 operator * (float s) const
	{
		Vector4 temp(*this);
		temp *= s;
		return temp;
	}

	friend Vector4 operator * (float s, const Vector4& v)
	{
		return v * s;
	}

	Vector4 operator * (const Matrix& m) const;
	Vector4& operator *= (const Matrix& m);

public:
	union
	{
		struct
		{
			float x, y, z, w;
		};

		float _[4];
	};
};



inline ostream & operator << (ostream &stream, const Vector4& v)
{
	stream << "[" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << "]";
	return stream;
}



typedef Vector4 vec4;



// ----------------------------------------------------------------------------



#endif
