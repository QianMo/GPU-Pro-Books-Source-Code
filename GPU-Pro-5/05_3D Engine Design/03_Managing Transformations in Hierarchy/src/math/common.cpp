#include <cstdlib>
#include <ctime>

#include <essentials/types.hpp>

#include <math/common.hpp>



void Randomize()
{
	srand((uint)time(NULL));
}



uint32 Rand()
{
	uint32 r1 = rand() % 2048; // 2^11
	uint32 r2 = rand() % 2048; // 2^11
	uint32 r3 = rand() % 1024; // 2^10

	return (r1) | (r2 << 11) | (r3 << 22);
}



uint32 Rand(uint32 from, uint32 to)
{
	return from + (Rand() % (to - from + 1));
}



float RandFloat()
{
	return (float)rand() / (float)RAND_MAX;
}



float RandFloat(float from, float to)
{
	return from + (RandFloat() * (to - from));
}



// algorithm comes from Fernando's and Kilgard's "The Cg Tutorial"
void ComputeTangentBasisForTriangle(
	const Vector3& v1, const Vector2& uv1,
	const Vector3& v2, const Vector2& uv2,
	const Vector3& v3, const Vector2& uv3,
	Vector3& tangent, Vector3& bitangent, Vector3& normal)
{
	Vector3 delta1_xuv = Vector3(v2.x - v1.x, uv2.x - uv1.x, uv2.y - uv1.y);
	Vector3 delta2_xuv = Vector3(v3.x - v1.x, uv3.x - uv1.x, uv3.y - uv1.y);
	Vector3 cross_xuv = delta1_xuv ^ delta2_xuv;

	Vector3 delta1_yuv = Vector3(v2.y - v1.y, uv2.x - uv1.x, uv2.y - uv1.y);
	Vector3 delta2_yuv = Vector3(v3.y - v1.y, uv3.x - uv1.x, uv3.y - uv1.y);
	Vector3 cross_yuv = delta1_yuv ^ delta2_yuv;

	Vector3 delta1_zuv = Vector3(v2.z - v1.z, uv2.x - uv1.x, uv2.y - uv1.y);
	Vector3 delta2_zuv = Vector3(v3.z - v1.z, uv3.x - uv1.x, uv3.y - uv1.y);
	Vector3 cross_zuv = delta1_zuv ^ delta2_zuv;

	tangent.x = - cross_xuv.y / cross_xuv.x;
	tangent.y = - cross_yuv.y / cross_yuv.x;
	tangent.z = - cross_zuv.y / cross_zuv.x;

	bitangent.x = - cross_xuv.z / cross_xuv.x;
	bitangent.y = - cross_yuv.z / cross_yuv.x;
	bitangent.z = - cross_zuv.z / cross_zuv.x;

	normal = GetNormalForTriangle(v1, v2, v3);
}



bool IsPointInsideTriangle(
	const Vector3& point,
	const Vector3& v1,
	const Vector3& v2,
	const Vector3& v3)
{
	Vector3 triangleNormal = GetNormalForTriangle(v1, v2, v3);

	Plane plane1(v1, v2, v2 + triangleNormal);
	Plane plane2(v2, v3, v3 + triangleNormal);
	Plane plane3(v3, v1, v1 + triangleNormal);

	if (plane1.GetSignedDistanceFromPoint(point) <= 0.0f &&
		plane2.GetSignedDistanceFromPoint(point) <= 0.0f &&
		plane3.GetSignedDistanceFromPoint(point) <= 0.0f)
	{
		return true;
	}
	else
	{
		return false;
	}
}



bool DoesTriangleOverlapTriangle(
	const Vector3& t1_v1,
	const Vector3& t1_v2,
	const Vector3& t1_v3,
	const Vector3& t2_v1,
	const Vector3& t2_v2,
	const Vector3& t2_v3)
{
	if (IsPointInsideTriangle(t1_v1, t2_v1, t2_v2, t2_v3) ||
		IsPointInsideTriangle(t1_v2, t2_v1, t2_v2, t2_v3) ||
		IsPointInsideTriangle(t1_v3, t2_v1, t2_v2, t2_v3) ||
		IsPointInsideTriangle(t2_v1, t1_v1, t1_v2, t1_v3) ||
		IsPointInsideTriangle(t2_v2, t1_v1, t1_v2, t1_v3) ||
		IsPointInsideTriangle(t2_v3, t1_v1, t1_v2, t1_v3))
	{
		return true;
	}
	else
	{
		return false;
	}
}



bool DoesQuadOverlapQuad(
	const Vector3& q1_v1,
	const Vector3& q1_v2,
	const Vector3& q1_v3,
	const Vector3& q1_v4,
	const Vector3& q2_v1,
	const Vector3& q2_v2,
	const Vector3& q2_v3,
	const Vector3& q2_v4)
{
	if (DoesTriangleOverlapTriangle(q1_v1, q1_v2, q1_v3, q2_v1, q2_v2, q2_v3) ||
		DoesTriangleOverlapTriangle(q1_v1, q1_v2, q1_v3, q2_v1, q2_v3, q2_v4) ||
		DoesTriangleOverlapTriangle(q1_v1, q1_v3, q1_v4, q2_v1, q2_v2, q2_v3) ||
		DoesTriangleOverlapTriangle(q1_v1, q1_v3, q1_v4, q2_v1, q2_v3, q2_v4))
	{
		return true;
	}
	else
	{
		return false;
	}
}



// algorithm comes from "3D Math Primer for Graphics and Game Development", chapter 12.
void ComputeBarycentricWeightsForPointWithRespectToTriangle(
	const Vector3& point,
	const Vector3& v1,
	const Vector3& v2,
	const Vector3& v3,
	float& w1, float& w2, float& w3)
{
	Vector3 e1 = v3 - v2;
	Vector3 e2 = v1 - v3;
	Vector3 e3 = v2 - v1;

	Vector3 d1 = point - v1;
	Vector3 d2 = point - v2;
	Vector3 d3 = point - v3;

	Vector3 normal = GetNormalForTriangle(v1, v2, v3);

	float A_T = 0.5f * (e1 ^ e2) % normal;
	float A_T1 = 0.5f * (e1 ^ d3) % normal;
	float A_T2 = 0.5f * (e2 ^ d1) % normal;
	float A_T3 = 0.5f * (e3 ^ d2) % normal;

	w1 = A_T1 / A_T;
	w2 = A_T2 / A_T;
	w3 = A_T3 / A_T;
}



void ComputeCoordinateFrameForDirectionVector(
	Vector3 direction,
	Vector3& rightVector,
	Vector3& upVector,
	bool rightHanded)
{
	direction.Normalize();

	Vector3 up(0.0f, 1.0f, 0.0f);
	if (fabs(direction % up) >= 1.0f - epsilon4) // if direction and up are almost colinear then direction must be tilted a bit along Z axis
		direction.z = 0.01f;

	direction.Normalize();

	if (rightHanded)
	{
		rightVector = (direction ^ up).GetNormalized();
		upVector = (rightVector ^ direction).GetNormalized();
	}
	else
	{
		rightVector = (up ^ direction).GetNormalized();
		upVector = (direction ^ rightVector).GetNormalized();
	}
}
