#ifndef MATH_COMMON_HPP
#define MATH_COMMON_HPP

#include <essentials/types.hpp>

#include <math/constants.hpp>
#include <math/vector.hpp>
#include <math/plane.hpp>



inline void Unpack_64_To_2x32(uint64 _64, uint32& _1_32, uint32& _2_32)
{
	_1_32 = (uint32)(_64 >> 32);
	_2_32 = (uint32)_64;
}

inline uint64 Pack_2x32_To_64(uint32 _1_32, uint32 _2_32)
{
	return (((uint64)_1_32) << 32) | ((uint64)_2_32);
}

template <class TYPE>
inline TYPE Max(TYPE a, TYPE b)
{
	return a > b ? a : b;
}

template <class TYPE>
inline TYPE Min(TYPE a, TYPE b)
{
	return a < b ? a : b;
}

inline bool IsPowerOfTwo(int x)
{
	return !(x & (x-1));
}

inline int Round(float x)
{
	return (int)(x + 0.5f);
}

inline float Frac(float x)
{
	return x - (int)x;
}

template <class TYPE>
inline TYPE Clamp(TYPE x, TYPE a, TYPE b)
{
	return Max(Min(x, b), a);
}

inline float Saturate(float x)
{
	return Clamp(x, 0.0f, 1.0f);
}

template <class TYPE>
inline TYPE Sqr(TYPE x)
{
	return x * x;
}

inline float Log2(float x)
{
	return logf(x) / logf(2.0f);
}

inline float DegToRad(float degrees)
{
	return (degrees * (pi / 180.0f));
}

inline float RadToDeg(float radians)
{
	return (radians * (180.0f / pi));
}

void Randomize();

uint32 Rand();

uint32 Rand(uint32 from, uint32 to);

float RandFloat();

float RandFloat(float from, float to);

inline float ArcCos_Clamped(float x)
{
	if (x <= -1.0f)
		return pi;
	else if (x >= 1.0f)
		return 0.0f;
	else
		return acosf(x);
}



inline Vector2 Solve2x2(const Vector2& p1, const Vector2& p2)
{
	float a = (p1.y - p2.y) / (p1.x - p2.x);
	float b = p1.y - a*p1.x;

	return Vector2(a, b);
}



inline float GetDeterminant(float a, float b, float c,
							float d, float e, float f,
							float g, float h, float i)
{
	return ( (a*e*i + d*h*c + g*b*f) - (c*e*g + f*h*a + i*b*d) );
}

inline float GetDeterminant(const Vector3& row0, const Vector3& row1, const Vector3& row2)
{
	return GetDeterminant(row0.x, row0.y, row0.z, row1.x, row1.y, row1.z, row2.x, row2.y, row2.z);
}



inline byte GetMaxPossibleMipmapsCount(ushort width, ushort height)
{
	byte mipmapsCount = 1;
	ushort size = Max(width, height);

	if (size == 1)
		return 1;

	do {
		size >>= 1;
		mipmapsCount++;
	} while (size != 1);

	return mipmapsCount;
}



inline Vector3 GetNormalForTriangle(
	const Vector3& v1,
	const Vector3& v2,
	const Vector3& v3)
{
	return ( (v2 - v1) ^ (v3 - v1) ).GetNormalized();
}

void ComputeTangentBasisForTriangle(
	const Vector3& v1, const Vector2& uv1,
	const Vector3& v2, const Vector2& uv2,
	const Vector3& v3, const Vector2& uv3,
	Vector3& tangent, Vector3& bitangent, Vector3& normal);

// checks wheter a point is bounded by three planes forming "triangle"
bool IsPointInsideTriangle(
	const Vector3& point,
	const Vector3& v1,
	const Vector3& v2,
	const Vector3& v3);

bool DoesTriangleOverlapTriangle(
	const Vector3& t1_v1,
	const Vector3& t1_v2,
	const Vector3& t1_v3,
	const Vector3& t2_v1,
	const Vector3& t2_v2,
	const Vector3& t2_v3);

bool DoesQuadOverlapQuad(
	const Vector3& q1_v1,
	const Vector3& q1_v2,
	const Vector3& q1_v3,
	const Vector3& q1_v4,
	const Vector3& q2_v1,
	const Vector3& q2_v2,
	const Vector3& q2_v3,
	const Vector3& q2_v4);

inline void ComputeIntersectionPointBetweenLineAndPlane(
	const Vector3& startPoint,
	const Vector3& direction,
	const Plane& plane,
	Vector3& intersectionPoint,
	float& t)
{
	t = - (plane.a*startPoint.x + plane.b*startPoint.y + plane.c*startPoint.z + plane.d) /
		  (plane.a*direction.x + plane.b*direction.y + plane.c*direction.z);
	intersectionPoint = startPoint + t*direction;
}

void ComputeBarycentricWeightsForPointWithRespectToTriangle(
	const Vector3& point,
	const Vector3& v1,
	const Vector3& v2,
	const Vector3& v3,
	float& w1, float& w2, float& w3);

// direction
//
//     ^
//     |
//     |
//     - - - >  rightVector
//
// imagine you're looking along the direction vector
//
void ComputeCoordinateFrameForDirectionVector(
	Vector3 direction,
	Vector3& rightVector,
	Vector3& upVector,
	bool rightHanded = true);

//               orientation
//
//                    ^
//                    |
//                    |
// rightVector  < - - -
//
// imagine you're looking against the orientation vector
//
inline void ComputeCoordinateFrameForOrientationVector(
	Vector3 orientation,
	Vector3& rightVector,
	Vector3& upVector,
	bool rightHanded = true)
{
	ComputeCoordinateFrameForDirectionVector(-orientation, rightVector, upVector, rightHanded);
}



#endif
