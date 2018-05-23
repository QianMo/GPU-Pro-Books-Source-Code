//--------------------------------------------------------------------------------------
// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.
//--------------------------------------------------------------------------------------

#ifndef __CPUTMath_h__
#define __CPUTMath_h__
#include <math.h>

/*
 * Constants
 */
static const float kEpsilon  = 0.00001f;
static const float kPi       = 3.14159265358979323846f;
static const float k2Pi      = kPi*2.0f;
static const float kPiDiv2   = kPi/2.0f;
static const float kInvPi    = 1.0f/kPi;
static const float kDegToRad = (kPi/180.0f);
static const float kRadToDeg = (180.0f/kPi);
static inline float DegToRad( float fDeg ) { return fDeg * kDegToRad; }
static inline float RadToDeg( float fRad ) { return fRad * kRadToDeg; }

template<typename T>
static inline void Swap(T &a, T &b)
{
    T t = b;
    b = a;
    a = t;
}
struct float2;
struct float3;
struct float4;

/**************************************\
float2
\**************************************/
struct float2
{
    union
    {
        struct
        {
            float x;
            float y;
        };
        float f[2];
    };

    /***************************************\
    |   Constructors                        |
    \***************************************/
    float2() {}
    explicit float2(float f) : x(f), y(f) { }
    explicit float2(float _x, float _y) : x(_x), y(_y) { }
    explicit float2(float* f) : x(f[0]), y(f[1]) { }
    float2(const float2 &v) : x(v.x), y(v.y) { }
    const float2 &operator=(const float2 &v)
    {
        x = v.x;
        y = v.y;        
        return *this;
    }

    /***************************************\
    |   Basic math operations               |
    \***************************************/
    float2 operator+(const float2 &r) const
    {
        return float2(x+r.x, y+r.y);
    }
    const float2 &operator+=(const float2 &r)
    {
        x += r.x;
        y += r.y;
        return *this;
    }
    float2 operator-(const float2 &r) const
    {
        return float2(x-r.x, y-r.y);
    }
    const float2 &operator-=(const float2 &r)
    {
        x -= r.x;
        y -= r.y;
        return *this;
    }
    /***************************************\
    |   Basic math operations with scalars  |
    \***************************************/
    float2 operator+(float f) const
    {
        return float2(x+f, y+f);
    }
    const float2 &operator+=(float f)
    {
        x += f;
        y += f;
        return *this;
    }
    float2 operator-(float f) const
    {
        return float2(x-f, y-f);
    }
    const float2 &operator-=(float f)
    {
        x -= f;
        y -= f;
        return *this;
    }
    float2 operator*(float f) const
    {
        return float2(x*f, y*f);
    }
    const float2 &operator*=(float f)
    {
        x *= f;
        y *= f;
        return *this;
    }
    float2 operator/(float f) const
    {
        return float2(x/f, y/f);
    }
    const float2 &operator/=(float f)
    {
        x /= f;
        y /= f;
        return *this;
    }

    /***************************************\
    |   Other math                          |
    \***************************************/
    // Equality
    bool operator==(const float2 &r) const
    {
        return (x==r.x && y == r.y);
    }
    bool operator!=(const float2 &r) const
    {
        return !(*this == r);
    }

    // Hadd
    float hadd(void) const
    {
        return x + y;
    }

    // Length
    float lengthSq(void) const
    {
        return x*x + y*y;
    }
    float length(void) const
    {
        return sqrtf(lengthSq());
    }
    void normalize(void)
    {
        *this /= length();
    }
};

inline float dot2(const float2 &l, const float2 &r)
{
    return l.x*r.x + l.y*r.y;
}

inline float2 normalize(const float2 &v)
{
    float length = v.length();
    return v / length;
}








/**************************************\
float3
\**************************************/
struct float3
{
    union
    {
        struct
        {
            float x;
            float y;
            float z;
        };
        float f[3];
    };

    /***************************************\
    |   Constructors                        |
    \***************************************/
    float3() {}
    explicit float3(float f) : x(f), y(f), z(f) { }
    explicit float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { }
    explicit float3(float* f) : x(f[0]), y(f[1]), z(f[2]) { }
    float3(const float3 &v) : x(v.x), y(v.y), z(v.z) { }
    float3(const float4 &v);
    const float3 &operator=(const float3 &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }

    /***************************************\
    |   Basic math operations               |
    \***************************************/
    float3 operator+(const float3 &r) const
    {
        return float3(x+r.x, y+r.y, z+r.z);
    }
    const float3 &operator+=(const float3 &r)
    {
        x += r.x;
        y += r.y;
        z += r.z;
        return *this;
    }
    float3 operator-(const float3 &r) const
    {
        return float3(x-r.x, y-r.y, z-r.z);
    }
    const float3 &operator-=(const float3 &r)
    {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        return *this;
    }
    float3 operator*(const float3 &r) const
    {
        return float3(x*r.x, y*r.y, z*r.z);
    }
    const float3 &operator*=(const float3 &r)
    {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        return *this;
    }
    float3 operator/(const float3 &r) const
    {
        return float3(x/r.x, y/r.y, z/r.z);
    }
    const float3 &operator/=(const float3 &r)
    {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        return *this;
    }

    /***************************************\
    |   Basic math operations with scalars  |
    \***************************************/
    float3 operator+(float f) const
    {
        return float3(x+f, y+f, z+f);
    }
    const float3 &operator+=(float f)
    {
        x += f;
        y += f;
        z += f;
        return *this;
    }
    float3 operator-(float f) const
    {
        return float3(x-f, y-f, z-f);
    }
    const float3 &operator-=(float f)
    {
        x -= f;
        y -= f;
        z -= f;
        return *this;
    }
    float3 operator*(float f) const
    {
        return float3(x*f, y*f, z*f);
    }
    const float3 &operator*=(float f)
    {
        x *= f;
        y *= f;
        z *= f;
        return *this;
    }
    float3 operator/(float f) const
    {
        return float3(x/f, y/f, z/f);
    }
    const float3 &operator/=(float f)
    {
        x /= f;
        y /= f;
        z /= f;
        return *this;
    }

    /***************************************\
    |   Other math                          |
    \***************************************/
    // Equality
    bool operator==(const float3 &r) const
    {
        return x==r.x &&y == r.y &&z == r.z;
    }
    bool operator!=(const float3 &r) const
    {
        return !(*this == r);
    }

    // Hadd
    float hadd(void) const
    {
        return x + y + z;
    }

    // Length
    float lengthSq(void) const
    {
        return x*x + y*y + z*z;
    }
    float length(void) const
    {
        return sqrtf(lengthSq());
    }
    float3 normalize(void)
    {
        return (*this /= length());
    }
};

inline float dot3(const float3 &l, const float3 &r)
{
    return l.x*r.x + l.y*r.y + l.z*r.z;
}
inline float3 cross3(const float3 &l, const float3 &r)
{
    return float3(  l.y*r.z-l.z*r.y,
                    l.z*r.x-l.x*r.z,
                    l.x*r.y-l.y*r.x);
}
inline float3 normalize(const float3 &v)
{
    float length = v.length();
    return v / length;
}


/**************************************\
float4
\**************************************/
__declspec(align(16)) struct float4
{
    union
    {
        struct
        {
            float x;
            float y;
            float z;
            float w;
        };
        float f[4];
    };

    /***************************************\
    |   Constructors                        |
    \***************************************/
    float4() {}
    explicit float4(float f) : x(f), y(f), z(f), w(f) { }
    explicit float4(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) { }
    explicit float4(float* f) : x(f[0]), y(f[1]), z(f[2]), w(f[3]) { }
    float4(const float4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) { }
    float4(const float3 &v, float _w) : x(v.x), y(v.y), z(v.z), w(_w) { }
    const float4 &operator=(const float4 &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    /***************************************\
    |   Basic math operations               |
    \***************************************/
    float4 operator+(const float4 &r) const
    {
        return float4(x+r.x, y+r.y, z+r.z, w+r.w);
    }
    const float4 &operator+=(const float4 &r)
    {
        x += r.x;
        y += r.y;
        z += r.z;
        w += r.w;
        return *this;
    }
    float4 operator-(const float4 &r) const
    {
        return float4(x-r.x, y-r.y, z-r.z, w-r.w);
    }
    const float4 &operator-=(const float4 &r)
    {
        x -= r.x;
        y -= r.y;
        z -= r.z;
        w -= r.w;
        return *this;
    }
    float4 operator*(const float4 &r) const
    {
        return float4(x*r.x, y*r.y, z*r.z, w*r.w);
    }
    const float4 &operator*=(const float4 &r)
    {
        x *= r.x;
        y *= r.y;
        z *= r.z;
        w *= r.w;
        return *this;
    }
    float4 operator/(const float4 &r) const
    {
        return float4(x/r.x, y/r.y, z/r.z, w/r.w);
    }
    const float4 &operator/=(const float4 &r)
    {
        x /= r.x;
        y /= r.y;
        z /= r.z;
        w /= r.w;
        return *this;
    }

    /***************************************\
    |   Basic math operations with scalars  |
    \***************************************/
    float4 operator+(float f) const
    {
        return float4(x+f, y+f, z+f, w+f);
    }
    const float4 &operator+=(float f)
    {
        x += f;
        y += f;
        z += f;
        return *this;
    }
    float4 operator-(float f) const
    {
        return float4(x-f, y-f, z-f, w-f);
    }
    const float4 &operator-=(float f)
    {
        x -= f;
        y -= f;
        z -= f;
        w -= f;
        return *this;
    }
    float4 operator*(float f) const
    {
        return float4(x*f, y*f, z*f, w*f);
    }
    const float4 &operator*=(float f)
    {
        x *= f;
        y *= f;
        z *= f;
        w *= f;
        return *this;
    }
    float4 operator/(float f) const
    {
        return float4(x/f, y/f, z/f, w/f);
    }
    const float4 &operator/=(float f)
    {
        x /= f;
        y /= f;
        z /= f;
        w /= f;
        return *this;
    }

    /***************************************\
    |   Other math                          |
    \***************************************/
    // Equality
    bool operator==(const float4 &r) const
    {
        return x==r.x && y == r.y && z == r.z && w == r.w;
    }
    bool operator!=(const float4 &r) const
    {
        return !(*this == r);
    }

    // Hadd
    float hadd(void) const
    {
        return x + y + z + w;
    }

    // Length
    float lengthSq(void) const
    {
        return x*x + y*y + z*z + w*w;
    }
    float length(void) const
    {
        return sqrtf(lengthSq());
    }
    void normalize(void)
    {
        *this /= length();
    }
};

inline float dot4(const float4 &l, const float4 &r)
{
    return l.x*r.x + l.y*r.y + l.z*r.z + l.w*r.w;
}
inline float4 normalize(const float4 &v)
{
    float length = v.length();
    return v / length;
}

inline float3::float3(const float4 &v) : x(v.x), y(v.y), z(v.z) { }


/**************************************\
float3x3
\**************************************/
struct float4x4;
struct float3x3
{
    struct
    {
        float3 r0;
        float3 r1;
        float3 r2;
    };

    /***************************************\
    |   Constructors                        |
    \***************************************/
    float3x3() {}
    explicit float3x3(float f) : r0(f), r1(f), r2(f) { }
    explicit float3x3(float* f) : r0(f+0), r1(f+3), r2(f+6) { }
    float3x3(const float3 &_r0, const float3 &_r1, const float3 &_r2) : r0(_r0), r1(_r1), r2(_r2) { }
    float3x3(float _m00, float _m01, float _m02,
             float _m10, float _m11, float _m12,
             float _m20, float _m21, float _m22)
             : r0(_m00,_m01,_m02)
             , r1(_m10,_m11,_m12)
             , r2(_m20,_m21,_m22)
    {
    }
    float3x3(const float4x4 &m);
    const float3x3 &operator=(const float3x3 &m)
    {
        r0 = m.r0;
        r1 = m.r1;
        r2 = m.r2;
        return *this;
    }

    /***************************************\
    |   Basic math operations               |
    \***************************************/

    #define MTX3_INDEX(f,r,c) ((f)[(r*3)+c])
    inline float3x3 operator*(const float3x3 &r) const
    {
        float3x3 m(1,0,0,0,1,0,0,0,1);

        const float* left     = (const float*)&this->r0;
        const float* right    = (const float*)&r.r0;
        float* result   = (float*)&m;

        int ii, jj, kk;
        for(ii=0; ii<3; ++ii) /* row */
        {
            for(jj=0; jj<3; ++jj) /* column */
            {
                float sum = MTX3_INDEX(left,ii,0)*MTX3_INDEX(right,0,jj);
                for(kk=1; kk<3; ++kk)
                {
                    sum += (MTX3_INDEX(left,ii,kk)*MTX3_INDEX(right,kk,jj));
                }
                MTX3_INDEX(result,ii,jj) = sum;
            }
        }
        return m;
    }
    #undef MTX3_INDEX

    inline float3 operator*(const float3 &v) const
    {
        float3 a;

        a.x = (r0*v).hadd();
        a.y = (r1*v).hadd();
        a.z = (r2*v).hadd();

        return a;
    }

    /***************************************\
    |   Basic math operations with scalars  |
    \***************************************/
    float3x3 operator+(float f) const
    {
        return float3x3(r0+f, r1+f, r2+f);
    }
    const float3x3 &operator+=(float f)
    {
        r0 += f;
        r1 += f;
        r2 += f;
        return *this;
    }
    float3x3 operator-(float f) const
    {
        return float3x3(r0-f, r1-f, r2-f);
    }
    const float3x3 &operator-=(float f)
    {
        r0 -= f;
        r1 -= f;
        r2 -= f;
        return *this;
    }
    float3x3 operator*(float f) const
    {
        return float3x3(r0*f, r1*f, r2*f);
    }
    const float3x3 &operator*=(float f)
    {
        r0 *= f;
        r1 *= f;
        r2 *= f;
        return *this;
    }
    float3x3 operator/(float f) const
    {
        return float3x3(r0/f, r1/f, r2/f);
    }
    const float3x3 &operator/=(float f)
    {
        r0 /= f;
        r1 /= f;
        r2 /= f;
        return *this;
    }

    /***************************************\
    |   Other math                          |
    \***************************************/
    // Equality
    bool operator==(const float3x3 &r) const
    {
        return r0 == r.r0 && r1 == r.r1 && r2 == r.r2;
    }
    bool operator!=(const float3x3 &r) const
    {
        return !(*this == r);
    }

    float determinant() const
    {
        float f0 = r0.x *  (r1.y*r2.z-r2.y*r1.z);
        float f1 = r0.y * -(r1.x*r2.z-r2.x*r1.z);
        float f2 = r0.z *  (r1.x*r2.y-r2.x*r1.y);

        return f0 + f1 + f2;
    }

    void transpose(void)
    {
        Swap(r0.y, r1.x);
        Swap(r0.z, r2.x);
        Swap(r1.z, r2.y);
    }

    void invert(void)
    {
        float det = determinant();
        float3x3 inv;

        inv.r0.x =   (r1.y*r2.z) - (r1.z*r2.y);
        inv.r0.y = -((r1.x*r2.z) - (r1.z*r2.x));
        inv.r0.z =   (r1.x*r2.y) - (r1.y*r2.x);

        inv.r1.x = -((r0.y*r2.z) - (r0.z*r2.y));
        inv.r1.y =   (r0.x*r2.z) - (r0.z*r2.x);
        inv.r1.z = -((r0.x*r2.y) - (r0.y*r2.x));

        inv.r2.x =   (r0.y*r1.z) - (r0.z*r1.y);
        inv.r2.y = -((r0.x*r1.z) - (r0.z*r1.x));
        inv.r2.z =   (r0.x*r1.y) - (r0.y*r1.x);

        inv.transpose();
        inv /= det;

        *this = inv;
    }
};
inline float3x3 float3x3Identity(void)
{
    return float3x3(1,0,0,
                    0,1,0,
                    0,0,1);
}
inline float determinant(const float3x3&m)
{
    return m.determinant();
}

inline float3x3 transpose(const float3x3 &m)
{
    float3x3 ret = m;
    ret.transpose();
    return ret;
}

inline float3x3 inverse(const float3x3 &m)
{
    float3x3 ret = m;
    ret.invert();
    return ret;
}

inline float3x3 float3x3RotationX(float rad)
{
    float       c = cosf( rad );
    float       s = sinf( rad );
    float3x3     m( 1.0f, 0.0f, 0.0f,
                    0.0f,    c,    s,
                    0.0f,   -s,    c );
    return m;
}
inline float3x3 float3x3RotationY(float rad)
{
    float       c = cosf( rad );
    float       s = sinf( rad );
    float3x3     m(    c, 0.0f,   -s,
                    0.0f, 1.0f, 0.0f,
                       s, 0.0f,    c );
    return m;
}
inline float3x3 float3x3RotationZ(float rad)
{

    float       c = cosf( rad );
    float       s = sinf( rad );
    float3x3     m(    c,    s, 0.0f,
                      -s,    c, 0.0f,
                    0.0f, 0.0f, 1.0f );
    return m;
}

inline float3x3 float3x3RotationAxis(const float3 &axis, float rad )
{
    float3 normAxis = normalize(axis);
    float c = cosf(rad);
    float s = sinf(rad);
    float t = 1 - c;

    float x = normAxis.x;
    float y = normAxis.y;
    float z = normAxis.z;

    float3x3 m;

    m.r0.x = (t * x * x) + c;
    m.r0.y = (t * x * y) + s * z;
    m.r0.z = (t * x * z) - s * y;

    m.r1.x = (t * x * y) - (s * z);
    m.r1.y = (t * y * y) + c;
    m.r1.z = (t * y * z) + (s * x);

    m.r2.x = (t * x * z) + (s * y);
    m.r2.y = (t * y * z) - (s * x);
    m.r2.z = (t * z * z) + c;

    return m;
}

inline float3x3 float3x3Scale(float x, float y, float z)
{
    return float3x3(x,0,0,
                    0,y,0,
                    0,0,z);
}

/**************************************\
float4x4
\**************************************/
struct float4x4
{
    struct
    {
        float4 r0;
        float4 r1;
        float4 r2;
        float4 r3;
    };

    /***************************************\
    |   Constructors                        |
    \***************************************/
    float4x4() {}
    explicit float4x4(float f) : r0(f), r1(f), r2(f), r3(f) { }
    explicit float4x4(float* f) : r0(f+0), r1(f+4), r2(f+8), r3(f+12) { }
    float4x4(const float4 &_r0, const float4 &_r1, const float4 &_r2, const float4 &_r3) : r0(_r0), r1(_r1), r2(_r2), r3(_r3) { }
    float4x4(float _m00, float _m01, float _m02, float _m03,
             float _m10, float _m11, float _m12, float _m13,
             float _m20, float _m21, float _m22, float _m23,
             float _m30, float _m31, float _m32, float _m33)
             : r0(_m00,_m01,_m02,_m03)
             , r1(_m10,_m11,_m12,_m13)
             , r2(_m20,_m21,_m22,_m23)
             , r3(_m30,_m31,_m32,_m33)
    {
    }
    float4x4(const float3x3 &m)
        : r0(m.r0, 0.0f)
        , r1(m.r1, 0.0f)
        , r2(m.r2, 0.0f)
        , r3(0.0f, 0.0f, 0.0f, 1.0f)
    {
    }
    const float4x4 &operator=(const float4x4 &m)
    {
        r0 = m.r0;
        r1 = m.r1;
        r2 = m.r2;
        r3 = m.r3;
        return *this;
    }

    /***************************************\
    |   Basic math operations               |
    \***************************************/

    #define MTX4_INDEX(f,r,c) ((f)[(r*4)+c])
    inline float4x4 operator*(const float4x4 &r) const
    {
        float4x4 m(1,0,0,0,
                   0,1,0,0,
                   0,0,1,0,
                   0,0,0,1);

        const float* left     = (const float*)&this->r0;
        const float* right    = (const float*)&r.r0;
        float* result   = (float*)&m;

        int ii, jj, kk;
        for(ii=0; ii<4; ++ii) /* row */
        {
            for(jj=0; jj<4; ++jj) /* column */
            {
                float sum = MTX4_INDEX(left,ii,0)*MTX4_INDEX(right,0,jj);
                for(kk=1; kk<4; ++kk)
                {
                    sum += (MTX4_INDEX(left,ii,kk)*MTX4_INDEX(right,kk,jj));
                }
                MTX4_INDEX(result,ii,jj) = sum;
            }
        }
        return m;
    }
    #undef MTX4_INDEX

    inline float4 operator*(const float4 &v) const
    {
        float4 a;

        a.x = (r0*v).hadd();
        a.y = (r1*v).hadd();
        a.z = (r2*v).hadd();
        a.w = (r3*v).hadd();

        return a;
    }

    /***************************************\
    |   Basic math operations with scalars  |
    \***************************************/
    float4x4 operator+(float f) const
    {
        return float4x4(r0+f, r1+f, r2+f, r3+f);
    }
    const float4x4 &operator+=(float f)
    {
        r0 += f;
        r1 += f;
        r2 += f;
        r3 += f;
        return *this;
    }
    float4x4 operator-(float f) const
    {
        return float4x4(r0-f, r1-f, r2-f, r3-f);
    }
    const float4x4 &operator-=(float f)
    {
        r0 -= f;
        r1 -= f;
        r2 -= f;
        r3 -= f;
        return *this;
    }
    float4x4 operator*(float f) const
    {
        return float4x4(r0*f, r1*f, r2*f, r3*f);
    }
    const float4x4 &operator*=(float f)
    {
        r0 *= f;
        r1 *= f;
        r2 *= f;
        r3 *= f;
        return *this;
    }
    float4x4 operator/(float f) const
    {
        return float4x4(r0/f, r1/f, r2/f, r3/f);
    }
    const float4x4 &operator/=(float f)
    {
        r0 /= f;
        r1 /= f;
        r2 /= f;
        r3 /= f;
        return *this;
    }

    /***************************************\
    |   Other math                          |
    \***************************************/
    // Equality
    bool operator==(const float4x4 &r) const
    {
        return r0 == r.r0 && r1 == r.r1 && r2 == r.r2 && r3 == r.r3;
    }
    bool operator!=(const float4x4 &r) const
    {
        return !(*this == r);
    }

    float determinant() const
    {
        float det = 0.0f;

        float3x3 a( r1.y,r1.z,r1.w,
                    r2.y,r2.z,r2.w,
                    r3.y,r3.z,r3.w);

        float3x3 b( r1.x,r1.z,r1.w,
                    r2.x,r2.z,r2.w,
                    r3.x,r3.z,r3.w);

        float3x3 c( r1.x,r1.y,r1.w,
                    r2.x,r2.y,r2.w,
                    r3.x,r3.y,r3.w);

        float3x3 d( r1.x,r1.y,r1.z,
                    r2.x,r2.y,r2.z,
                    r3.x,r3.y,r3.z);


        det += r0.x * a.determinant();

        det -= r0.y * b.determinant();

        det += r0.z * c.determinant();

        det -= r0.w * d.determinant();

        return det;
    }

    void transpose(void)
    {
        Swap(r0.y, r1.x);
        Swap(r0.z, r2.x);
        Swap(r0.w, r3.x);
        Swap(r1.z, r2.y);
        Swap(r1.w, r3.y);
        Swap(r2.w, r3.z);
    }

    void invert(void)
    {
        float4x4 ret;
        float recip;

        /* temp matrices */

        /* row 1 */
        float3x3 a( r1.y,r1.z,r1.w,
                    r2.y,r2.z,r2.w,
                    r3.y,r3.z,r3.w);

        float3x3 b( r1.x,r1.z,r1.w,
                    r2.x,r2.z,r2.w,
                    r3.x,r3.z,r3.w);

        float3x3 c( r1.x,r1.y,r1.w,
                    r2.x,r2.y,r2.w,
                    r3.x,r3.y,r3.w);

        float3x3 d( r1.x,r1.y,r1.z,
                    r2.x,r2.y,r2.z,
                    r3.x,r3.y,r3.z);

        /* row 2 */
        float3x3 e( r0.y,r0.z,r0.w,
                    r2.y,r2.z,r2.w,
                    r3.y,r3.z,r3.w);

        float3x3 f( r0.x,r0.z,r0.w,
                    r2.x,r2.z,r2.w,
                    r3.x,r3.z,r3.w);

        float3x3 g( r0.x,r0.y,r0.w,
                    r2.x,r2.y,r2.w,
                    r3.x,r3.y,r3.w);

        float3x3 h( r0.x,r0.y,r0.z,
                    r2.x,r2.y,r2.z,
                    r3.x,r3.y,r3.z);


        /* row 3 */
        float3x3 i( r0.y,r0.z,r0.w,
                    r1.y,r1.z,r1.w,
                    r3.y,r3.z,r3.w);

        float3x3 j( r0.x,r0.z,r0.w,
                    r1.x,r1.z,r1.w,
                    r3.x,r3.z,r3.w);

        float3x3 k( r0.x,r0.y,r0.w,
                    r1.x,r1.y,r1.w,
                    r3.x,r3.y,r3.w);

        float3x3 l( r0.x,r0.y,r0.z,
                    r1.x,r1.y,r1.z,
                    r3.x,r3.y,r3.z);


        /* row 4 */
        float3x3 m( r0.y, r0.z, r0.w,
                    r1.y, r1.z, r1.w,
                    r2.y, r2.z, r2.w);

        float3x3 n( r0.x, r0.z, r0.w,
                    r1.x, r1.z, r1.w,
                    r2.x, r2.z, r2.w);

        float3x3 o( r0.x,r0.y,r0.w,
                    r1.x,r1.y,r1.w,
                    r2.x,r2.y,r2.w);

        float3x3 p( r0.x,r0.y,r0.z,
                    r1.x,r1.y,r1.z,
                    r2.x,r2.y,r2.z);

        /* row 1 */
        ret.r0.x = a.determinant();

        ret.r0.y = -b.determinant();

        ret.r0.z = c.determinant();

        ret.r0.w = -d.determinant();

        /* row 2 */
        ret.r1.x = -e.determinant();

        ret.r1.y = f.determinant();

        ret.r1.z = -g.determinant();

        ret.r1.w = h.determinant();

        /* row 3 */
        ret.r2.x = i.determinant();

        ret.r2.y = -j.determinant();

        ret.r2.z = k.determinant();

        ret.r2.w = -l.determinant();

        /* row 4 */
        ret.r3.x = -m.determinant();

        ret.r3.y = n.determinant();

        ret.r3.z = -o.determinant();

        ret.r3.w = p.determinant();

        ret.transpose();
        recip = 1.0f/determinant();
        ret *= recip;

        *this = ret;
    }

    // Axis access
    float3 getXAxis(void) const
    {
        return r0;
    }
    float3 getYAxis(void) const
    {
        return r1;
    }
    float3 getZAxis(void) const
    {
        return r2;
    }
    float3 getPosition(void) const
    {
        return float3(r3);
    }

    void orthonormalize(void)
    {
        float3 x = getXAxis();
        float3 y = getYAxis();
        float3 z;

        x.normalize();
        z = normalize(cross3(x, y));
        y = normalize(cross3(z, x));

        r0 = float4(x, 0.0f);
        r1 = float4(y, 0.0f);
        r2 = float4(z, 0.0f);
    }
};

inline float3x3::float3x3(const float4x4 &m)
    : r0(m.r0)
    , r1(m.r1)
    , r2(m.r2)
{
}
inline float determinant(const float4x4&m)
{
    return m.determinant();
}
inline float4x4 float4x4Identity(void)
{
    return float4x4(1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1);
}
inline float4x4 transpose(const float4x4 &m)
{
    float4x4 ret = m;
    ret.transpose();
    return ret;
}

inline float4x4 inverse(const float4x4 &m)
{
    float4x4 ret = m;
    ret.invert();
    return ret;
}

inline float4x4 float4x4RotationX(float rad)
{
    float       c = cosf( rad );
    float       s = sinf( rad );
    float4x4     m( 1.0f, 0.0f, 0.0f, 0.0f,
                    0.0f,    c,    s, 0.0f,
                    0.0f,   -s,    c, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}
inline float4x4 float4x4RotationY(float rad)
{
    float       c = cosf( rad );
    float       s = sinf( rad );
    float4x4     m(    c, 0.0f,   -s, 0.0f,
                    0.0f, 1.0f, 0.0f, 0.0f,
                       s, 0.0f,    c, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}
inline float4x4 float4x4RotationZ(float rad)
{

    float       c = cosf( rad );
    float       s = sinf( rad );
    float4x4     m(    c,    s, 0.0f, 0.0f,
                      -s,    c, 0.0f, 0.0f,
                    0.0f, 0.0f, 1.0f, 0.0f,
                    0.0f, 0.0f, 0.0f, 1.0f);
    return m;
}

inline float4x4 float4x4RotationAxis(const float3 &axis, float rad )
{
    float3 normAxis = normalize(axis);
    float c = cosf(rad);
    float s = sinf(rad);
    float t = 1 - c;

    float x = normAxis.x;
    float y = normAxis.y;
    float z = normAxis.z;

    float4x4 m = float4x4Identity();

    m.r0.x = (t * x * x) + c;
    m.r0.y = (t * x * y) + s * z;
    m.r0.z = (t * x * z) - s * y;

    m.r1.x = (t * x * y) - (s * z);
    m.r1.y = (t * y * y) + c;
    m.r1.z = (t * y * z) + (s * x);

    m.r2.x = (t * x * z) + (s * y);
    m.r2.y = (t * y * z) - (s * x);
    m.r2.z = (t * z * z) + c;

    return m;
}

inline float4x4 float4x4Scale(float x, float y, float z)
{
    return float4x4(x,0,0,0,
                    0,y,0,0,
                    0,0,z,0,
                    0,0,0,1);
}

inline float4x4 float4x4Translation(float x, float y, float z)
{
    float4x4 m = float4x4Identity();
    m.r3.x = x;
    m.r3.y = y;
    m.r3.z = z;
    return m;
}
inline float4x4 float4x4Translation(const float3 &v)
{
    return float4x4Translation(v.x, v.y, v.z);
}

inline float4x4 float4x4PerspectiveFovLH(float fov, float aspect, float nearPlane, float farPlane)
{
    float height = tanf(fov * 0.5f);
    float width  = height * aspect;
    float diff   = farPlane-nearPlane;
    float div    = farPlane / diff;

    float4x4 m(
        1.0f/width,          0.0f,            0.0f,    0.0f,
              0.0f,   1.0f/height,            0.0f,    0.0f,
              0.0f,          0.0f,            div,     1.0f,
              0.0f,          0.0f, -nearPlane*div,     0.0f
    );
    return m;
}
inline float4x4 float4x4PerspectiveFovRH(float fov, float aspect, float nearPlane, float farPlane)
{
    float4x4 m = float4x4Identity();

    float f = tanf(kPiDiv2 - (fov/2.0f));
    float diff = nearPlane-farPlane;
    float div = farPlane / diff;

    m.r0.x = f/aspect;
    m.r1.y = f;
    m.r2.z = div;
    m.r2.w = -1;
    m.r3.z = nearPlane * div;
    m.r3.w = 0;
    return m;
}
inline float4x4 float4x4PerspectiveLH(float width, float height, float nearPlane, float farPlane)
{
    float4x4 m = float4x4Identity();

    m.r0.x = 2*nearPlane/width;
    m.r1.y = 2*nearPlane/height;
    m.r2.z = farPlane/(farPlane-nearPlane);
    m.r2.w = 1;
    m.r3.z = nearPlane*farPlane/(nearPlane-farPlane);
    m.r3.w = 0;
    return m;
}
inline float4x4 float4x4PerspectiveRH(float width, float height, float nearPlane, float farPlane)
{
    float4x4 m = float4x4Identity();

    m.r0.x = 2*nearPlane/width;
    m.r1.y = 2*nearPlane/height;
    m.r2.z = farPlane/(nearPlane-farPlane);
    m.r2.w = -1;
    m.r3.z = nearPlane*farPlane/(nearPlane-farPlane);
    m.r3.w = 0;
    return m;
}
inline float4x4 float4x4OrthographicOffCenterLH(float left, float right, float top, float bottom, float nearPlane, float farPlane)
{
    float4x4 m = float4x4Identity();

    float diff = farPlane-nearPlane;

    m.r0.x = 2.0f/(right-left);
    m.r1.y = 2.0f/(top-bottom);
    m.r2.z = 1.0f/diff;
    m.r3.x = -((left+right)/(right-left));
    m.r3.y = -((top+bottom)/(top-bottom));
    m.r3.z = -nearPlane/diff;

    return m;
}
inline float4x4 float4x4OrthographicOffCenterRH(float left, float right, float top, float bottom, float nearPlane, float farPlane)
{
    float4x4 m = float4x4Identity();
    float diff = nearPlane-farPlane;

    m.r0.x = 2.0f/(right-left);
    m.r1.y = 2.0f/(top-bottom);
    m.r2.z = 1.0f/diff;
    m.r3.x = -((left+right)/(right-left));
    m.r3.y = -((top+bottom)/(top-bottom));
    m.r3.z = nearPlane/diff;

    return m;
}
inline float4x4 float4x4OrthographicLH(float width, float height, float nearPlane, float farPlane)
{
    float halfWidth = width/2.0f;
    float halfHeight = height/2.0f;

    return float4x4OrthographicOffCenterLH(-halfWidth, halfWidth, halfHeight, -halfHeight, nearPlane, farPlane);
}
inline float4x4 float4x4OrthographicRH(float width, float height, float nearPlane, float farPlane)
{
    float halfWidth = width/2.0f;
    float halfHeight = height/2.0f;

    return float4x4OrthographicOffCenterRH(-halfWidth, halfWidth, halfHeight, -halfHeight, nearPlane, farPlane);
}

/**************************************\
Quaternion
\**************************************/
struct quaternion : public float4
{

    /***************************************\
    |   Constructors                        |
    \***************************************/
    quaternion() {}
    explicit quaternion(float f) : float4(f) { }
    explicit quaternion(float _x, float _y, float _z, float _w) : float4(_x,_y,_z,_w) { }
    explicit quaternion(float* f) : float4(f) { }
    quaternion(const quaternion &v) : float4(v) { }
    quaternion(const float3 &v, float _w)
    {
        float3  norm = ::normalize(v);
        float   a = _w*0.5f;
        float   s = sinf(a);
        x = norm.x*s;
        y = norm.y*s;
        z = norm.z*s;
        w = cosf(a);
    }
    const quaternion &operator=(const quaternion &v)
    {
        x = v.x;
        y = v.y;
        z = v.z;
        w = v.w;
        return *this;
    }

    /* Methods */
    float3 getXAxis(void) const
    {
        float3 ret( 1-2*(y*y+z*z),
                      2*(x*y+w*z),
                      2*(x*z-y*w));
        return ret;
    }
    float3 getYAxis(void) const
    {
        float3 ret(   2*(x*y-z*w),
                    1-2*(x*x+z*z),
                      2*(y*z+x*w));
        return ret;
    }
    float3 getZAxis(void) const
    {
        float3 ret(   2*(x*z+y*w),
                      2*(y*z-x*w),
                    1-2*(x*x+y*y));
        return ret;
    }

    float3x3 getMatrix(void) const
    {
        quaternion q = *this;
        q.normalize();
        float xx = q.x * q.x;
        float yy = q.y * q.y;
        float zz = q.z * q.z;

        float xy = q.x * q.y;
        float zw = q.z * q.w;
        float xz = q.x * q.z;
        float yw = q.y * q.w;
        float yz = q.y * q.z;
        float xw = q.x * q.w;

        float3x3 ret(1 - 2*(yy+zz),     2*(xy+zw),     2*(xz-yw),
                         2*(xy-zw), 1 - 2*(xx+zz),     2*(yz+xw),
                         2*(xz+yw),     2*(yz-xw), 1 - 2*(xx+yy) );
        return ret;
    }

    quaternion conjugate(void) const
    {
        return quaternion(-x, -y, -z, w);
    }
    quaternion inverse(void) const
    {
        // Note: Only normalized quaternions are supportted at the moment
        quaternion q = *this;
        q.normalize();
        return q.conjugate();
    }
};

inline quaternion quaternionMultiply(const quaternion &l, const quaternion &r)
{
    quaternion q(   r.w*l.x + r.x*l.w + r.y*l.z - r.z*l.y,
                    r.w*l.y + r.y*l.w + r.z*l.x - r.x*l.z,
                    r.w*l.z + r.z*l.w + r.x*l.y - r.y*l.x,
                    r.w*l.w - r.x*l.x - r.y*l.y - r.z*l.z );
    return q;
}

inline quaternion quaternionIdentity(void)
{
    return quaternion(0.0f, 0.0f, 0.0f, 1.0f);
}

inline float3 Min( float3 &v0, float3 &v1 )
{
    float3 result;
    result.x = v0.x < v1.x ? v0.x : v1.x;
    result.y = v0.y < v1.y ? v0.y : v1.y;
    result.z = v0.z < v1.z ? v0.z : v1.z;
    return result;
}

inline float3 Max( float3 &v0, float3 &v1 )
{
    float3 result;
    result.x = v0.x > v1.x ? v0.x : v1.x;
    result.y = v0.y > v1.y ? v0.y : v1.y;
    result.z = v0.z > v1.z ? v0.z : v1.z;
    return result;
}

inline float4 Min( float4 &v0, float4 &v1 )
{
    float4 result;
    result.x = v0.x < v1.x ? v0.x : v1.x;
    result.y = v0.y < v1.y ? v0.y : v1.y;
    result.z = v0.z < v1.z ? v0.z : v1.z;
    result.w = v0.w < v1.w ? v0.w : v1.w;
    return result;
}

inline float4 Max( float4 &v0, float4 &v1 )
{
    float4 result;
    result.x = v0.x > v1.x ? v0.x : v1.x;
    result.y = v0.y > v1.y ? v0.y : v1.y;
    result.z = v0.z > v1.z ? v0.z : v1.z;
    result.w = v0.w > v1.w ? v0.w : v1.w;
    return result;
}

inline float4 operator*(const float4 &v, const float4x4 &m)
{
    float4 result;

    result  = m.r0 * v.x;
    result += m.r1 * v.y;
    result += m.r2 * v.z;
    result += m.r3 * v.w;

    return result;
}

#endif // #ifndef __CPUTMath_h__
