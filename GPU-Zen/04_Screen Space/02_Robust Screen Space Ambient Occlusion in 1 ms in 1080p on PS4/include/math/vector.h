#pragma once


#include "types.h"
#include "common.h"


namespace NMath
{
	// Vector2

	Vector2 VectorCustom(float x, float y);
	Vector2 VectorCopy(const Vector2& v);
	Vector2 Add(const Vector2& v1, const Vector2& v2);
	Vector2 Sub(const Vector2& v1, const Vector2& v2);
	Vector2 Mul(const Vector2& v1, const Vector2& v2);
	Vector2 Mul(const Vector2& v, float f);
	Vector2 Div(const Vector2& v, float f);
	void AddIn(Vector2& v1, const Vector2& v2);
	void SubIn(Vector2& v1, const Vector2& v2);
	void MulIn(Vector2& v1, const Vector2& v2);
	void MulIn(Vector2& v, float f);
	void DivIn(Vector2& v, float f);
	bool Equal(const Vector2& v1, const Vector2& v2);
	float Dot(const Vector2& v1, const Vector2& v2);
	float LengthSquared(const Vector2& v);
	float Length(const Vector2& v);
	void SetLength(Vector2& v, float newLength);
	Vector2 Normalize(const Vector2& v);
	void NormalizeIn(Vector2& v);
	Vector2 Pow(const Vector2& v, float f);
	void PowIn(Vector2& v, float f);
	float Angle(const Vector2& v1, const Vector2& v2);
	float MaxComponent(const Vector2& v);
	Vector2 Reflect(const Vector2& input, const Vector2& normal);
	Vector2 Refract(const Vector2& input, const Vector2& normal, float eta);
	Vector2 Slerp(const Vector2& v1, const Vector2& v2, float t);
	Vector2 Abs(const Vector2& v);
	Vector2 Clamp(const Vector2& v, const Vector2& min, const Vector2& max);
	Vector2 Average(const Vector2* points, int pointsCount);
	Vector2 operator + (const Vector2& v1, const Vector2& v2);
	Vector2 operator - (const Vector2& v1, const Vector2& v2);
	Vector2 operator - (const Vector2& v);
	Vector2 operator * (const Vector2& v1, const Vector2& v2);
	Vector2 operator * (const Vector2& v, float f);
	Vector2 operator * (float f, const Vector2& v);
	Vector2 operator / (const Vector2& v, float f);
	Vector2& operator += (Vector2& v1, const Vector2& v2);
	Vector2& operator -= (Vector2& v1, const Vector2& v2);
	Vector2& operator *= (Vector2& v1, const Vector2& v2);
	Vector2& operator *= (Vector2& v, float f);
	Vector2& operator /= (Vector2& v, float f);
	bool operator == (const Vector2& v1, const Vector2& v2);

	// Vector3

	Vector3 VectorCustom(float x, float y, float z);
	Vector3 VectorCopy(const Vector3& v);
	Vector3 Add(const Vector3& v1, const Vector3& v2);
	Vector3 Sub(const Vector3& v1, const Vector3& v2);
	Vector3 Mul(const Vector3& v1, const Vector3& v2);
	Vector3 Mul(const Vector3& v, float f);
	Vector3 Div(const Vector3& v, float f);
	void AddIn(Vector3& v1, const Vector3& v2);
	void SubIn(Vector3& v1, const Vector3& v2);
	void MulIn(Vector3& v1, const Vector3& v2);
	void MulIn(Vector3& v, float f);
	void DivIn(Vector3& v, float f);
	bool Equal(const Vector3& v1, const Vector3& v2);
	float Dot(const Vector3& v1, const Vector3& v2);
	Vector3 Cross(const Vector3& v1, const Vector3& v2);
	float LengthSquared(const Vector3& v);
	float Length(const Vector3& v);
	void SetLength(Vector3& v, float newLength);
	Vector3 Normalize(const Vector3& v);
	void NormalizeIn(Vector3& v);
	Vector3 Pow(const Vector3& v, float f);
	void PowIn(Vector3& v, float f);
	float Angle(const Vector3& v1, const Vector3& v2);
	float MaxComponent(const Vector3& v);
	Vector3 Reflect(const Vector3& input, const Vector3& normal);
	Vector3 Refract(const Vector3& input, const Vector3& normal, float eta);
	Vector3 Slerp(const Vector3& v1, const Vector3& v2, float t);
	Vector3 Abs(const Vector3& v);
	Vector3 Clamp(const Vector3& v, const Vector3& min, const Vector3& max);
	Vector3 Average(const Vector3* points, int pointsCount);
	Vector3 operator + (const Vector3& v1, const Vector3& v2);
	Vector3 operator - (const Vector3& v1, const Vector3& v2);
	Vector3 operator - (const Vector3& v);
	Vector3 operator * (const Vector3& v1, const Vector3& v2);
	Vector3 operator * (const Vector3& v, float f);
	Vector3 operator * (float f, const Vector3& v);
	Vector3 operator / (const Vector3& v, float f);
	Vector3& operator += (Vector3& v1, const Vector3& v2);
	Vector3& operator -= (Vector3& v1, const Vector3& v2);
	Vector3& operator *= (Vector3& v, float f);
	Vector3& operator *= (Vector3& v1, const Vector3& v2);
	Vector3& operator /= (Vector3& v, float f);
	bool operator == (const Vector2& v1, const Vector2& v2);

	// Vector4

	Vector4 VectorCustom(float x, float y, float z, float w);
	Vector4 VectorCopy(const Vector4& v);
	Vector4 Add(const Vector4& v1, const Vector4& v2);
	Vector4 Sub(const Vector4& v1, const Vector4& v2);
	Vector4 Mul(const Vector4& v1, const Vector4& v2);
	Vector4 Mul(const Vector4& v, float f);
	Vector4 Div(const Vector4& v, float f);
	void AddIn(Vector4& v1, const Vector4& v2);
	void SubIn(Vector4& v1, const Vector4& v2);
	void MulIn(Vector4& v1, const Vector4& v2);
	void MulIn(Vector4& v, float f);
	void DivIn(Vector4& v, float f);
	bool Equal(const Vector4& v1, const Vector4& v2);
	Vector4 Pow(const Vector4& v, float f);
	void PowIn(Vector4& v, float f);
	float MaxComponent(const Vector4& v);
	Vector4 Abs(const Vector4& v);
	Vector4 Clamp(const Vector4& v, const Vector4& min, const Vector4& max);
	Vector4 DivideByW(const Vector4& v);
	void DivideByWIn(Vector4& v);
	Vector4 operator + (const Vector4& v1, const Vector4& v2);
	Vector4 operator - (const Vector4& v1, const Vector4& v2);
	Vector4 operator - (const Vector4& v);
	Vector4 operator * (const Vector4& v1, const Vector4& v2);
	Vector4 operator * (const Vector4& v, float f);
	Vector4 operator * (float f, const Vector4& v);
	Vector4 operator / (const Vector4& v, float f);
	Vector4& operator += (Vector4& v1, const Vector4& v2);
	Vector4& operator -= (Vector4& v1, const Vector4& v2);
	Vector4& operator *= (Vector4& v, float f);
	Vector4& operator *= (Vector4& v1, const Vector4& v2);
	Vector4& operator /= (Vector4& v, float f);
	bool operator == (const Vector2& v1, const Vector2& v2);

	//

	const Vector2 Vector2Zero = VectorCustom(0.0f, 0.0f);
	const Vector3 Vector3Zero = VectorCustom(0.0f, 0.0f, 0.0f);
	const Vector4 Vector4Zero = VectorCustom(0.0f, 0.0f, 0.0f, 0.0f);

	const Vector2 Vector2One = VectorCustom(1.0f, 1.0f);
	const Vector3 Vector3One = VectorCustom(1.0f, 1.0f, 1.0f);
	const Vector4 Vector4One = VectorCustom(1.0f, 1.0f, 1.0f, 1.0f);

	const Vector2 Vector2EX = VectorCustom(1.0f, 0.0f);
	const Vector2 Vector2EY = VectorCustom(0.0f, 1.0f);
	const Vector3 Vector3EX = VectorCustom(1.0f, 0.0f, 0.0f);
	const Vector3 Vector3EY = VectorCustom(0.0f, 1.0f, 0.0f);
	const Vector3 Vector3EZ = VectorCustom(0.0f, 0.0f, 1.0f);

	// Vector2

	inline Vector2 VectorCustom(float x, float y)
	{
		Vector2 temp;

		temp.x = x;
		temp.y = y;

		return temp;
	}

	inline Vector2 VectorCopy(const Vector2& v)
	{
		return VectorCustom(v.x, v.y);
	}

	inline Vector2 Add(const Vector2& v1, const Vector2& v2)
	{
		return VectorCustom(v1.x + v2.x, v1.y + v2.y);
	}

	inline Vector2 Sub(const Vector2& v1, const Vector2& v2)
	{
		return VectorCustom(v1.x - v2.x, v1.y - v2.y);
	}

	inline Vector2 Mul(const Vector2& v1, const Vector2& v2)
	{
		return VectorCustom(v1.x*v2.x, v1.y*v2.y);
	}

	inline Vector2 Mul(const Vector2& v, float f)
	{
		return VectorCustom(v.x*f, v.y*f);
	}

	inline Vector2 Div(const Vector2& v, float f)
	{
		f = 1.0f / f;
		return VectorCustom(v.x*f, v.y*f);
	}

	inline void AddIn(Vector2& v1, const Vector2& v2)
	{
		v1.x += v2.x;
		v1.y += v2.y;
	}

	inline void SubIn(Vector2& v1, const Vector2& v2)
	{
		v1.x -= v2.x;
		v1.y -= v2.y;
	}

	inline void MulIn(Vector2& v1, const Vector2& v2)
	{
		v1.x *= v2.x;
		v1.y *= v2.y;
	}

	inline void MulIn(Vector2& v, float f)
	{
		v.x *= f;
		v.y *= f;
	}

	inline void DivIn(Vector2& v, float f)
	{
		f = 1.0f / f;
		v.x *= f;
		v.y *= f;
	}

	inline bool Equal(const Vector2& v1, const Vector2& v2)
	{
		if (Abs(v1.x - v2.x) < 0.0001f &&
			Abs(v1.y - v2.y) < 0.0001f)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline float Dot(const Vector2& v1, const Vector2& v2)
	{
		return v1.x*v2.x + v1.y*v2.y;
	}

	inline float LengthSquared(const Vector2& v)
	{
		return Dot(v, v);
	}

	inline float Length(const Vector2& v)
	{
		return Sqrt(LengthSquared(v));
	}

	inline void SetLength(Vector2& v, float newLength)
	{
		float scale = newLength / Length(v);

		v.x *= scale;
		v.y *= scale;
	}

	inline Vector2 Normalize(const Vector2& v)
	{
		Vector2 temp = VectorCopy(v);
		NormalizeIn(temp);
		return temp;
	}

	inline void NormalizeIn(Vector2& v)
	{
		float oneOverLength = 1.0f / Length(v);

		v.x *= oneOverLength;
		v.y *= oneOverLength;
	}

	inline Vector2 Pow(const Vector2& v, float f)
	{
		return VectorCustom(Pow(v.x, f), Pow(v.y, f));
	}

	inline void PowIn(Vector2& v, float f)
	{
		v.x = Pow(v.x, f);
		v.y = Pow(v.y, f);
	}

	inline float Angle(const Vector2& v1, const Vector2& v2)
	{
		return Cos(Dot(v1, v2));
	}

	inline float MaxComponent(const Vector2& v)
	{
		return Max(v.x, v.y);
	}

	inline Vector2 Reflect(const Vector2& input, const Vector2& normal)
	{
		return input - 2.0f * Dot(input, normal) * normal;
	}

	inline Vector2 Refract(const Vector2& input, const Vector2& normal, float eta)
	{
		float IdotN = Dot(input, normal);
		float k = 1.0f - Sqr(eta) * (1.0f - Sqr(IdotN));
		if (k < 0.0f)
			return Vector2Zero;
		else
			return eta * input - (eta * IdotN + Sqrt(k)) * normal;
	}

	inline Vector2 Slerp(const Vector2& v1, const Vector2& v2, float t)
	{
		float omega = Angle(v1, v2);

		float s1 = Sin((1.0f - t)*omega);
		float s2 = Sin(t*omega);
		float s3 = Sin(omega);

		return v1*(s1/s3) + v2*(s2/s3);
	}

	inline Vector2 Abs(const Vector2& v)
	{
		return VectorCustom(Abs(v.x), Abs(v.y));
	}

	inline Vector2 Clamp(const Vector2& v, const Vector2& min, const Vector2& max)
	{
		return VectorCustom(Clamp(v.x, min.x, max.x), Clamp(v.y, min.y, max.y));
	}

	inline Vector2 Average(const Vector2* points, int pointsCount)
	{
		Vector2 avg = points[0];

		for (int i = 1; i < pointsCount; i++)
			avg += points[i];

		return avg / (float)pointsCount;
	}

	inline Vector2 operator + (const Vector2& v1, const Vector2& v2)
	{
		return Add(v1, v2);
	}

	inline Vector2 operator - (const Vector2& v1, const Vector2& v2)
	{
		return Sub(v1, v2);
	}

	inline Vector2 operator - (const Vector2& v)
	{
		return VectorCustom(-v.x, -v.y);
	}

	inline Vector2 operator * (const Vector2& v1, const Vector2& v2)
	{
		return Mul(v1, v2);
	}

	inline Vector2 operator * (const Vector2& v, float f)
	{
		return Mul(v, f);
	}

	inline Vector2 operator * (float f, const Vector2& v)
	{
		return Mul(v, f);
	}

	inline Vector2 operator / (const Vector2& v, float f)
	{
		return Div(v, f);
	}

	inline Vector2& operator += (Vector2& v1, const Vector2& v2)
	{
		AddIn(v1, v2);
		return v1;
	}

	inline Vector2& operator -= (Vector2& v1, const Vector2& v2)
	{
		SubIn(v1, v2);
		return v1;
	}

	inline Vector2& operator *= (Vector2& v, float f)
	{
		MulIn(v, f);
		return v;
	}

	inline Vector2& operator *= (Vector2& v1, const Vector2& v2)
	{
		MulIn(v1, v2);
		return v1;
	}

	inline Vector2& operator /= (Vector2& v, float f)
	{
		DivIn(v, f);
		return v;
	}

	inline bool operator == (const Vector2& v1, const Vector2& v2)
	{
		return Equal(v1, v2);
	}

	// Vector3

	inline Vector3 VectorCustom(float x, float y, float z)
	{
		Vector3 temp;

		temp.x = x;
		temp.y = y;
		temp.z = z;

		return temp;
	}

	inline Vector3 VectorCopy(const Vector3& v)
	{
		return VectorCustom(v.x, v.y, v.z);
	}

	inline Vector3 Add(const Vector3& v1, const Vector3& v2)
	{
		return VectorCustom(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}

	inline Vector3 Sub(const Vector3& v1, const Vector3& v2)
	{
		return VectorCustom(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}

	inline Vector3 Mul(const Vector3& v1, const Vector3& v2)
	{
		return VectorCustom(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z);
	}

	inline Vector3 Mul(const Vector3& v, float f)
	{
		return VectorCustom(v.x*f, v.y*f, v.z*f);
	}

	inline Vector3 Div(const Vector3& v, float f)
	{
		f = 1.0f / f;
		return VectorCustom(v.x*f, v.y*f, v.z*f);
	}

	inline void AddIn(Vector3& v1, const Vector3& v2)
	{
		v1.x += v2.x;
		v1.y += v2.y;
		v1.z += v2.z;
	}

	inline void SubIn(Vector3& v1, const Vector3& v2)
	{
		v1.x -= v2.x;
		v1.y -= v2.y;
		v1.z -= v2.z;
	}

	inline void MulIn(Vector3& v, float f)
	{
		v.x *= f;
		v.y *= f;
		v.z *= f;
	}

	inline void MulIn(Vector3& v1, const Vector3& v2)
	{
		v1.x *= v2.x;
		v1.y *= v2.y;
		v1.z *= v2.z;
	}

	inline void DivIn(Vector3& v, float f)
	{
		f = 1.0f / f;
		v.x *= f;
		v.y *= f;
		v.z *= f;
	}

	inline bool Equal(const Vector3& v1, const Vector3& v2)
	{
		if (Abs(v1.x - v2.x) < 0.0001f &&
			Abs(v1.y - v2.y) < 0.0001f &&
			Abs(v1.z - v2.z) < 0.0001f)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline float Dot(const Vector3& v1, const Vector3& v2)
	{
		return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
	}

	inline Vector3 Cross(const Vector3& v1, const Vector3& v2)
	{
		Vector3 temp;

		temp.x = v1.y*v2.z - v1.z*v2.y;
		temp.y = v1.z*v2.x - v1.x*v2.z;
		temp.z = v1.x*v2.y - v1.y*v2.x;

		return temp;
	}

	inline float LengthSquared(const Vector3& v)
	{
		return Dot(v, v);
	}

	inline float Length(const Vector3& v)
	{
		return Sqrt(LengthSquared(v));
	}

	inline void SetLength(Vector3& v, float newLength)
	{
		float scale = newLength / Length(v);

		v.x *= scale;
		v.y *= scale;
		v.z *= scale;
	}

	inline Vector3 Normalize(const Vector3& v)
	{
		Vector3 temp = VectorCopy(v);
		NormalizeIn(temp);
		return temp;
	}

	inline void NormalizeIn(Vector3& v)
	{
		float oneOverLength = 1.0f / Length(v);

		v.x *= oneOverLength;
		v.y *= oneOverLength;
		v.z *= oneOverLength;
	}

	inline Vector3 Pow(const Vector3& v, float f)
	{
		return VectorCustom(Pow(v.x, f), Pow(v.y, f), Pow(v.z, f));
	}

	inline void PowIn(Vector3& v, float f)
	{
		v.x = Pow(v.x, f);
		v.y = Pow(v.y, f);
		v.z = Pow(v.z, f);
	}

	inline float Angle(const Vector3& v1, const Vector3& v2)
	{
		return Cos(Dot(v1, v2));
	}

	inline float MaxComponent(const Vector3& v)
	{
		return Max(Max(v.x, v.y), v.z);
	}

	inline Vector3 Reflect(const Vector3& input, const Vector3& normal)
	{
		return input - 2.0f * Dot(input, normal) * normal;
	}

	inline Vector3 Refract(const Vector3& input, const Vector3& normal, float eta)
	{
		float IdotN = Dot(input, normal);
		float k = 1.0f - Sqr(eta) * (1.0f - Sqr(IdotN));
		if (k < 0.0f)
			return Vector3Zero;
		else
			return eta * input - (eta * IdotN + Sqrt(k)) * normal;
	}

	inline Vector3 Slerp(const Vector3& v1, const Vector3& v2, float t)
	{
		float omega = Angle(v1, v2);

		float s1 = Sin((1.0f - t) * omega);
		float s2 = Sin(t * omega);
		float s3 = Sin(omega);

		return v1*(s1/s3) + v2*(s2/s3);
	}

	inline Vector3 Abs(const Vector3& v)
	{
		return VectorCustom(Abs(v.x), Abs(v.y), Abs(v.z));
	}

	inline Vector3 Clamp(const Vector3& v, const Vector3& min, const Vector3& max)
	{
		return VectorCustom(Clamp(v.x, min.x, max.x), Clamp(v.y, min.y, max.y), Clamp(v.z, min.z, max.z));
	}

	inline Vector3 Average(const Vector3* points, int pointsCount)
	{
		Vector3 avg = points[0];

		for (int i = 1; i < pointsCount; i++)
			avg += points[i];

		return avg / (float)pointsCount;
	}

	inline Vector3 operator + (const Vector3& v1, const Vector3& v2)
	{
		return Add(v1, v2);
	}

	inline Vector3 operator - (const Vector3& v1, const Vector3& v2)
	{
		return Sub(v1, v2);
	}

	inline Vector3 operator - (const Vector3& v)
	{
		return VectorCustom(-v.x, -v.y, -v.z);
	}

	inline Vector3 operator * (const Vector3& v1, const Vector3& v2)
	{
		return Mul(v1, v2);
	}

	inline Vector3 operator * (const Vector3& v, float f)
	{
		return Mul(v, f);
	}

	inline Vector3 operator * (float f, const Vector3& v)
	{
		return Mul(v, f);
	}

	inline Vector3 operator / (const Vector3& v, float f)
	{
		return Div(v, f);
	}

	inline Vector3& operator += (Vector3& v1, const Vector3& v2)
	{
		AddIn(v1, v2);
		return v1;
	}

	inline Vector3& operator -= (Vector3& v1, const Vector3& v2)
	{
		SubIn(v1, v2);
		return v1;
	}

	inline Vector3& operator *= (Vector3& v1, const Vector3& v2)
	{
		MulIn(v1, v2);
		return v1;
	}

	inline Vector3& operator *= (Vector3& v, float f)
	{
		MulIn(v, f);
		return v;
	}

	inline Vector3& operator /= (Vector3& v, float f)
	{
		DivIn(v, f);
		return v;
	}

	inline bool operator == (const Vector3& v1, const Vector3& v2)
	{
		return Equal(v1, v2);
	}

	// Vector4

	inline Vector4 VectorCustom(float x, float y, float z, float w)
	{
		Vector4 temp;

		temp.x = x;
		temp.y = y;
		temp.z = z;
		temp.w = w;

		return temp;
	}

	inline Vector4 VectorCopy(const Vector4& v)
	{
		return VectorCustom(v.x, v.y, v.z, v.w);
	}

	inline Vector4 Add(const Vector4& v1, const Vector4& v2)
	{
		return VectorCustom(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
	}

	inline Vector4 Sub(const Vector4& v1, const Vector4& v2)
	{
		return VectorCustom(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w);
	}

	inline Vector4 Mul(const Vector4& v1, const Vector4& v2)
	{
		return VectorCustom(v1.x*v2.x, v1.y*v2.y, v1.z*v2.z, v1.w*v2.w);
	}

	inline Vector4 Mul(const Vector4& v, float f)
	{
		return VectorCustom(v.x*f, v.y*f, v.z*f, v.w*f);
	}

	inline Vector4 Div(const Vector4& v, float f)
	{
		f = 1.0f / f;
		return VectorCustom(v.x*f, v.y*f, v.z*f, v.w*f);
	}

	inline void AddIn(Vector4& v1, const Vector4& v2)
	{
		v1.x += v2.x;
		v1.y += v2.y;
		v1.z += v2.z;
		v1.w += v2.w;
	}

	inline void SubIn(Vector4& v1, const Vector4& v2)
	{
		v1.x -= v2.x;
		v1.y -= v2.y;
		v1.z -= v2.z;
		v1.w -= v2.w;
	}

	inline void MulIn(Vector4& v1, const Vector4& v2)
	{
		v1.x *= v2.x;
		v1.y *= v2.y;
		v1.z *= v2.z;
		v1.w *= v2.w;
	}

	inline void MulIn(Vector4& v, float f)
	{
		v.x *= f;
		v.y *= f;
		v.z *= f;
		v.w *= f;
	}

	inline void DivIn(Vector4& v, float f)
	{
		f = 1.0f / f;
		v.x *= f;
		v.y *= f;
		v.z *= f;
		v.w *= f;
	}

	inline bool Equal(const Vector4& v1, const Vector4& v2)
	{
		if (Abs(v1.x - v2.x) < 0.0001f &&
			Abs(v1.y - v2.y) < 0.0001f &&
			Abs(v1.z - v2.z) < 0.0001f &&
			Abs(v1.w - v2.w) < 0.0001f)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	inline Vector4 Pow(const Vector4& v, float f)
	{
		return VectorCustom(Pow(v.x, f), Pow(v.y, f), Pow(v.z, f), Pow(v.w, f));
	}

	inline void PowIn(Vector4& v, float f)
	{
		v.x = Pow(v.x, f);
		v.y = Pow(v.y, f);
		v.z = Pow(v.z, f);
		v.w = Pow(v.w, f);
	}

	inline float MaxComponent(const Vector4& v)
	{
		return Max(Max(v.x, v.y), Max(v.z, v.w));
	}

	inline Vector4 Abs(const Vector4& v)
	{
		return VectorCustom(Abs(v.x), Abs(v.y), Abs(v.z), Abs(v.w));
	}

	inline Vector4 Clamp(const Vector4& v, const Vector4& min, const Vector4& max)
	{
		return VectorCustom(Clamp(v.x, min.x, max.x), Clamp(v.y, min.y, max.y), Clamp(v.z, min.z, max.z), Clamp(v.w, min.w, max.w));
	}

	inline Vector4 DivideByW(const Vector4& v)
	{
		float oneOverW = 1.0f / v.w;
		return VectorCustom(v.x*oneOverW, v.y*oneOverW, v.z*oneOverW, 1.0f);
	}

	inline void DivideByWIn(Vector4& v)
	{
		float oneOverW = 1.0f / v.w;

		v.x *= oneOverW;
		v.y *= oneOverW;
		v.z *= oneOverW;
		v.w = 1.0f;
	}

	inline Vector4 operator + (const Vector4& v1, const Vector4& v2)
	{
		return Add(v1, v2);
	}

	inline Vector4 operator - (const Vector4& v1, const Vector4& v2)
	{
		return Sub(v1, v2);
	}

	inline Vector4 operator - (const Vector4& v)
	{
		return VectorCustom(-v.x, -v.y, -v.z, -v.w);
	}

	inline Vector4 operator * (const Vector4& v1, const Vector4& v2)
	{
		return Mul(v1, v2);
	}

	inline Vector4 operator * (const Vector4& v, float f)
	{
		return Mul(v, f);
	}

	inline Vector4 operator * (float f, const Vector4& v)
	{
		return Mul(v, f);
	}

	inline Vector4 operator / (const Vector4& v, float f)
	{
		return Div(v, f);
	}

	inline Vector4& operator += (Vector4& v1, const Vector4& v2)
	{
		AddIn(v1, v2);
		return v1;
	}

	inline Vector4& operator -= (Vector4& v1, const Vector4& v2)
	{
		SubIn(v1, v2);
		return v1;
	}

	inline Vector4& operator *= (Vector4& v1, const Vector4& v2)
	{
		MulIn(v1, v2);
		return v1;
	}

	inline Vector4& operator *= (Vector4& v, float f)
	{
		MulIn(v, f);
		return v;
	}

	inline Vector4& operator /= (Vector4& v, float f)
	{
		DivIn(v, f);
		return v;
	}

	inline bool operator == (const Vector4& v1, const Vector4& v2)
	{
		return Equal(v1, v2);
	}
}
