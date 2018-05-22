#ifndef __VECTOR2__H__
#define __VECTOR2__H__

#include <math.h>

// -----------------------------------------------------------------------------
/// Vector2
// -----------------------------------------------------------------------------
/// \ingroup 
/// 
/// 
// -----------------------------------------------------------------------------
class Vector2
{
public:
	union
	{
		struct
		{
			float x;
			float y;
		};

		struct
		{
			float u;
			float v;
		};

		float		comp[2];
	};

	inline Vector2() {}
	inline Vector2(const float _x, const float _y) : x(_x), y(_y) {}
	
	inline operator float* ()		{return (float*)(comp);}

	inline Vector2 operator + (const Vector2& v) const					{return Vector2(x + v.x, y + v.y);}
	inline Vector2 operator - (const Vector2& v) const					{return Vector2(x - v.x, y - v.y);}
	inline Vector2 operator - () const									{return Vector2(-x, -y);}
	inline Vector2 operator * (const Vector2& v) const					{return Vector2(x * v.x, y * v.y);}
	inline Vector2 operator * (const float f) const						{return Vector2(x * f, y * f);}
	inline Vector2 operator / (const Vector2& v) const					{return Vector2(x / v.x, y / v.y);}
	inline Vector2 operator / (const float f) const						{return Vector2(x / f, y / f);}
	inline friend Vector2 operator * (const float f, const Vector2& v)	{return Vector2(v.x * f, v.y * f);}

	inline Vector2 operator = (const Vector2& v)	{x = v.x; y = v.y; return *this;}
	inline Vector2 operator += (const Vector2& v)	{x += v.x; y += v.y; return *this;}
	inline Vector2 operator -= (const Vector2& v)	{x -= v.x; y -= v.y; return *this;}
	inline Vector2 operator *= (const Vector2& v)	{x *= v.x; y *= v.y; return *this;}
	inline Vector2 operator *= (const float f)		{x *= f; y *= f; return *this;}
	inline Vector2 operator /= (const Vector2& v)	{x /= v.x; y /= v.y; return *this;}
	inline Vector2 operator /= (const float f)		{x /= f; y /= f; return *this;}

	inline bool operator == (const Vector2& v) const	{if(x != v.x) return false; return y == v.y;}
	inline bool operator != (const Vector2& v) const	{if(x != v.x) return true; return y != v.y;}

	static inline float		Vector2Length(const Vector2& v) {return sqrtf(v.x * v.x + v.y * v.y);}
	static inline float		Vector2LengthSq(const Vector2& v) {return v.x * v.x + v.y * v.y;}
	static inline Vector2	Vector2Normalize(const Vector2& v) {return v / sqrtf(v.x * v.x + v.y * v.y);}
	static inline Vector2	Vector2NormalizeEx(const Vector2& v) {return v / (sqrtf(v.x * v.x + v.y * v.y) + 0.0001f);}
};

#endif