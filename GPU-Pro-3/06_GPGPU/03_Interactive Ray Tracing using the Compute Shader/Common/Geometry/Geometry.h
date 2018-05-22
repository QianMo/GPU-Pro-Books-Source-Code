/*
 * pbrt source code Copyright(c) 1998-2004 Matt Pharr and Greg Humphreys
 *
 * All Rights Reserved.
 * For educational use only; commercial use expressly forbidden.
 * NO WARRANTY, express or implied, for this software.
 * (See file License.txt for complete license)
 */

// --------------------------------------------------------------------	//
// This code was modified by the authors of the demo. The original		//
// PBRT code is available at https://github.com/mmp/pbrt-v2. Basically, //
// we removed all STL-based implementation and it was merged with		//
// our current framework.												//
// --------------------------------------------------------------------	//

// ------------------------------------------------
// Geometry.h
// ------------------------------------------------
// Math Lib for Ray Tracing.

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <assert.h>
#include <math.h>

class Ray;
class RayDifferential;
class BBox;
class Transform;
#include <float.h>
#include <algorithm>
using std::min;
using std::max;
using std::swap;

// Platform-specific definitions
#ifdef WINDOWS
#define isnan _isnan
#endif

// Global Constants
#ifdef M_PI
#undef M_PI
#endif
#define M_PI       3.14159265358979323846f
#ifndef INFINITY
#define INFINITY FLT_MAX
#endif

#ifdef NDEBUG
#define Assert(expr) ((void)0)
#else
#define Assert(expr) ((void)0)
#endif // NDEBUG

struct MortonCode
{
	MortonCode() :m_iCode(0), m_iId(0) { }
	MortonCode(int a_iCode, int a_iId) : m_iCode(a_iCode), m_iId(a_iId) {}
	unsigned __int64 m_iCode;
	int m_iId;	

	bool operator< ( const MortonCode obj)
	{
		return m_iCode < obj.m_iCode;
	}
};

// Global Inline Functions
inline float Lerp(float t, float v1, float v2) 
{
    return (1.f - t) * v1 + t * v2;
}

inline float Clamp(float val, float low, float high) 
{
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
}


inline int Clamp(int val, int low, int high) 
{
    if (val < low) return low;
    else if (val > high) return high;
    else return val;
}

struct TIntersection
{
	int IDTr;
	float u, v, t;
	TIntersection(int IDTr = -1, float t = 0) { this->IDTr = IDTr; this->t = t;}
};

class Vector3 
{
public:
	// Vector3 Public Data
    float x, y, z;

    // Vector3 Public Methods
    Vector3() { x = y = z = 0.f; }
    Vector3(float xx, float yy, float zz) : x(xx), y(yy), z(zz) { Assert(!HasNaNs()); }
    bool HasNaNs() const { return isnan(x) || isnan(y) || isnan(z); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    Vector3(const Vector3 &v) {
        Assert(!v.HasNaNs());
        x = v.x; y = v.y; z = v.z;
    }
    
    Vector3 &operator=(const Vector3 &v) {
        Assert(!v.HasNaNs());
        x = v.x; y = v.y; z = v.z;
        return *this;
    }

#endif // !NDEBUG
    Vector3 operator+(const Vector3 &v) const 
	{
        Assert(!v.HasNaNs());
        return Vector3(x + v.x, y + v.y, z + v.z);
    }
    Vector3& operator+=(const Vector3 &v) 
	{
        Assert(!v.HasNaNs());
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    /*Vector3 operator-(const Vector3 &v) const 
	{
        Assert(!v.HasNaNs());
        return Vector3(x - v.x, y - v.y, z - v.z);
    }*/
	Vector3 operator-(const Vector3* v) const 
	{
        Assert(!v.HasNaNs());
        return Vector3(x - v->x, y - v->y, z - v->z);
    }
    Vector3& operator-=(const Vector3 &v) 
	{
        Assert(!v.HasNaNs());
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    bool operator==(const Vector3 &v) const 
	{
        return x == v.x && y == v.y && z == v.z;
    }
    Vector3 operator*(float f) const 
	{ 
		return Vector3(f*x, f*y, f*z); 
	}
	Vector3 operator*(Vector3 v) const 
	{ 
		return Vector3(v.x*x, v.y*y, v.z*z); 
	}
    Vector3 &operator*=(float f) {
        //Assert(!isnan(f));
        x *= f; y *= f; z *= f;
        return *this;
    }
    Vector3 operator/(float f) const {
        Assert(f != 0);
        float inv = 1.f / f;
        return Vector3(x * inv, y * inv, z * inv);
    }
	Vector3 operator/(Vector3 v) const 
	{ 
		return Vector3(v.x/x, v.y/y, v.z/z); 
	}
    Vector3 &operator/=(float f) {
        Assert(f != 0);
        float inv = 1.f / f;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }
    Vector3 operator-() const 
	{ 
		return Vector3(-x, -y, -z); 
	}
    float operator[](int i) const 
	{
        Assert(i >= 0 && i <= 2);
        return (&x)[i];
    }
    float &operator[](int i) {
        Assert(i >= 0 && i <= 2);
        return (&x)[i];
    }

    float LengthSquared() const { return x*x + y*y + z*z; }
    float Length() const { return sqrtf(LengthSquared()); }
};

typedef Vector3 Vector;
typedef Vector3 Normal;
typedef Vector3 Point;
typedef Vector3 Color;
typedef unsigned int Pixel;

inline Vector operator*(float f, const Vector &v) { return v*f; }

inline Vector3 operator-(const Vector3 &A,const Vector3 &B)
{
	Assert(!v.HasNaNs());
	return Vector3(A.x - B.x, A.y - B.y, A.z - B.z);
}

inline void Dot(float &res, const Vector &v1, const Vector &v2) 
{
    Assert(!v1.HasNaNs() && !v2.HasNaNs());
    res = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

inline void AbsDot(float &res, const Vector &v1, const Vector &v2) {
    Assert(!v1.HasNaNs() && !v2.HasNaNs());
	Dot(res, v1, v2);
    res = fabsf(res);
}

inline void Cross(Vector &v, const Vector &v1, const Vector &v2) {
    Assert(!v1.HasNaNs() && !v2.HasNaNs());
    v.x = (v1.y * v2.z) - (v1.z * v2.y);
    v.y = (v1.z * v2.x) - (v1.x * v2.z);
    v.z = (v1.x * v2.y) - (v1.y * v2.x);
}
inline void Length(float &res, const Vector &v)
{
	Dot(res,v,v);
	res = sqrtf(res);
}
inline void Normalize(Vector &v) 
{ 
	float length;
	Length(length, v);
	v.x /= length; 
	v.y /= length; 
	v.z /= length; 
	//return v / v.Length(); 
}
inline void CoordinateSystem(const Vector &v1, Vector *v2, Vector *v3) 
{
    if (fabsf(v1.x) > fabsf(v1.y)) {
        float invLen = 1.f / sqrtf(v1.x*v1.x + v1.z*v1.z);
        *v2 = Vector(-v1.z * invLen, 0.f, v1.x * invLen);
    }
    else {
        float invLen = 1.f / sqrtf(v1.y*v1.y + v1.z*v1.z);
        *v2 = Vector(0.f, v1.z * invLen, -v1.y * invLen);
    }
    Cross(*v3, v1, *v2);
}

inline void BaryCentric(Point &Out,const Point &V1,const Point &V2,const Point &V3, float u, float v)
{
	float t = 1.0f - (u +v);
	Out.x = V1.x*t+V2.x*u+V3.x*v;
	Out.y = V1.y*t+V2.y*u+V3.y*v;
	Out.z = V1.z*t+V2.z*u+V3.z*v;
}
inline float Distance(const Point &p1, const Point &p2) 
{
    return (p1 - p2).Length();
}
inline float DistanceSquared(const Point &p1, const Point &p2) 
{
    return (p1 - p2).LengthSquared();
}
inline Vector Faceforward(float &res, const Vector &v, const Vector &v2)
{
	Dot(res, v, v2);
    (res < 0.f) ? -v : v;
}
inline Vector SphericalDirection(float sintheta,float costheta, float phi) 
{
    return Vector(sintheta * cosf(phi),sintheta * sinf(phi), costheta);
}

inline Vector SphericalDirection(float sintheta, float costheta,
                                 float phi, const Vector &x,
                                 const Vector &y, const Vector &z) 
{
    return sintheta * cosf(phi) * x +
           sintheta * sinf(phi) * y + costheta * z;
}

inline float SphericalTheta(const Vector &v) 
{
    return acosf(Clamp(v.z, -1.f, 1.f));
}

inline float SphericalPhi(const Vector &v) 
{
    float p = atan2f(v.y, v.x);
    return (p < 0.f) ? p + 2.f*M_PI : p;
}

#endif /*GEOMETRY_H*/