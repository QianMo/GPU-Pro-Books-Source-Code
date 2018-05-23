#ifndef BE_UTILITY_MATH_H
#define BE_UTILITY_MATH_H

#define BE_UTILITY_MATH_TYPE float
#define BE_UTILITY_MATH_TYPE_FLOAT
#include "Utility/Math.fx"
#undef BE_UTILITY_MATH_TYPE
#undef BE_UTILITY_MATH_TYPE_FLOAT

#define BE_UTILITY_MATH_TYPE uint
#define BE_UTILITY_MATH_TYPE_UINT
#include "Utility/Math.fx"
#undef BE_UTILITY_MATH_TYPE
#undef BE_UTILITY_MATH_TYPE_UINT

#define BE_UTILITY_MATH_TYPE int
#define BE_UTILITY_MATH_TYPE_INT
#include "Utility/Math.fx"
#undef BE_UTILITY_MATH_TYPE
#undef BE_UTILITY_MATH_TYPE_INT

/// PI.
static const float PI = 3.141592653589793f;

/// Computes the median of the given vector.
float median(float3 values)
{
	float m = dot(values, 1.0f);
	m -= min3(values);
	m -= max3(values);
	return m;
}
/// Computes the median of the given vector.
float median(float4 values)
{
	float m = dot(values, 1.0f);
	m -= min4(values);
	m -= max4(values);
	return m * 0.5f;
}

/// Interpolates the given set of three values.
float interpolate(float v0, float v1, float v2, float3 b)
{
	return b.x * v0 + b.y * v1 + b.z * v2;
}
/// Interpolates the given set of three vectors.
float2 interpolate(float2 v0, float2 v1, float2 v2, float3 b)
{
	return b.x * v0 + b.y * v1 + b.z * v2;
}
/// Interpolates the given set of three vectors.
float3 interpolate(float3 v0, float3 v1, float3 v2, float3 b)
{
	return b.x * v0 + b.y * v1 + b.z * v2;
}
/// Interpolates the given set of three vectors.
float4 interpolate(float4 v0, float4 v1, float4 v2, float3 b)
{
	return b.x * v0 + b.y * v1 + b.z * v2;
}

/// sqrt(1 - x^2).
float pyt1(float x)
{
	return sqrt( saturate(1.0f - x * x) );
}

/// ceil(x / d).
uint ceil_div(uint x, uint d)
{
	return (x + d - 1u) / d;
}
uint2 ceil_div(uint2 x, uint2 d)
{
	return (x + d - 1u) / d;
}
uint3 ceil_div(uint3 x, uint3 d)
{
	return (x + d - 1u) / d;
}
uint4 ceil_div(uint4 x, uint4 d)
{
	return (x + d - 1u) / d;
}

/// Replaces NaNs.
float repnan(float v, float r)
{
	return isnan(v) ? r : v;
}
float2 repnan(float2 v, float2 r)
{
	return float2( repnan(v.x, r.x), repnan(v.y, r.y) );
}
float3 repnan(float3 v, float3 r)
{
	return float3( repnan(v.x, r.x), repnan(v.y, r.y), repnan(v.z, r.z) );
}
float4 repnan(float4 v, float4 r)
{
	return float4( repnan(v.x, r.x), repnan(v.y, r.y), repnan(v.z, r.z), repnan(v.w, r.w) );
}

/// Replaces according to the given mask.
float repmask(bool b, float v, float r)
{
	return b ? r : v;
}
float2 repmask(bool2 b, float2 v, float2 r)
{
	return float2( repmask(b.x, v.x, r.x), repmask(b.y, v.y, r.y) );
}
float3 repmask(bool3 b, float3 v, float3 r)
{
	return float3( repmask(b.x, v.x, r.x), repmask(b.y, v.y, r.y), repmask(b.z, v.z, r.z) );
}
float4 repmask(bool4 b, float4 v, float4 r)
{
	return float4( repmask(b.x, v.x, r.x), repmask(b.y, v.y, r.y), repmask(b.z, v.z, r.z), repmask(b.w, v.w, r.w) );
}

#endif

#ifdef BE_UTILITY_MATH_TYPE

#define JOIN(a, b) a##b
#define TYPE BE_UTILITY_MATH_TYPE
#define VTYPE(d) JOIN(TYPE, d)


/// Computes the minimum component of the given vector.
TYPE min2(VTYPE(2) values)
{
	return min(values.x, values.y);
}
/// Computes the maximum component of the given vector.
TYPE max2(VTYPE(2) values)
{
	return max(values.x, values.y);
}

/// Computes the minimum component of the given vector.
TYPE min3(VTYPE(3) values)
{
	return min( min2(values.xy), values.z );
}
/// Computes the maximum component of the given vector.
TYPE max3(VTYPE(3) values)
{
	return max( max2(values.xy), values.z );
}

/// Computes the minimum component of the given vector.
TYPE min4(VTYPE(4) values)
{
	return min2( min(values.xy, values.zw) );
}
/// Computes the maximum component of the given vector.
TYPE max4(VTYPE(4) values)
{
	return max2( max(values.xy, values.zw) );
}


#ifndef BE_UTILITY_MATH_TYPE_UINT

/// Either 1 or -1.
TYPE sign1(TYPE x)
{
	return (TYPE(0) <= x) ? TYPE(1) : TYPE(-1);
}
VTYPE(2) sign1(VTYPE(2) x)
{
	return VTYPE(2)(sign1(x.x), sign1(x.y));
}
VTYPE(3) sign1(VTYPE(3) x)
{
	return VTYPE(3)(sign1(x.x), sign1(x.y), sign1(x.z));
}
VTYPE(4) sign1(VTYPE(4) x)
{
	return VTYPE(4)(sign1(x.x), sign1(x.y), sign1(x.z), sign1(x.w));
}

#endif


/// Square.
TYPE sq(TYPE x)
{
	return x * x;
}
VTYPE(2) sq(VTYPE(2) x)
{
	return x * x;
}
VTYPE(3) sq(VTYPE(3) x)
{
	return x * x;
}
VTYPE(4) sq(VTYPE(4) x)
{
	return x * x;
}

/// Length squared.
TYPE lengthsq(TYPE x)
{
	return x * x;
}
TYPE lengthsq(VTYPE(2) x)
{
	return dot(x, x);
}
TYPE lengthsq(VTYPE(3) x)
{
	return dot(x, x);
}
TYPE lengthsq(VTYPE(4) x)
{
	return dot(x, x);
}


#if defined(BE_UTILITY_MATH_TYPE_INT) || defined(BE_UTILITY_MATH_TYPE_UINT)

/// Returns the given value mod 3.
TYPE mod3(TYPE i)
{
	return i % TYPE(3);
}

#endif


#undef JOIN
#undef TYPE
#undef VTYPE

#endif