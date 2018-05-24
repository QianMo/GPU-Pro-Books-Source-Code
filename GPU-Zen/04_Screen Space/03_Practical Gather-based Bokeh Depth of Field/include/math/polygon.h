#pragma once


#include "types.h"
#include "vector.h"


namespace NMath
{
	Vector3 TriangleNormal(const Vector3& v1, const Vector3& v2, const Vector3& v3);
	void TriangleTangentBasis(
		const Vector3& v1, const Vector2& uv1,
		const Vector3& v2, const Vector2& uv2,
		const Vector3& v3, const Vector2& uv3,
		Vector3& tangent, Vector3& bitangent, Vector3& normal);
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
	void TriangleBarycentricWeightsForPoint(
		const Vector3& point,
		const Vector3& v1,
		const Vector3& v2,
		const Vector3& v3,
		float& w1, float& w2, float& w3);

	bool DoesQuadOverlapQuad(
		const Vector3& q1_v1,
		const Vector3& q1_v2,
		const Vector3& q1_v3,
		const Vector3& q1_v4,
		const Vector3& q2_v1,
		const Vector3& q2_v2,
		const Vector3& q2_v3,
		const Vector3& q2_v4);

	//

	inline Vector3 TriangleNormal(const Vector3& v1, const Vector3& v2, const Vector3& v3)
	{
		return Normalize(Cross((v2 - v1), (v3 - v1)));
	}
}
