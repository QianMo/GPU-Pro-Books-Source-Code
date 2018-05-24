#pragma once


#include "types.h"
#include "vector.h"
#include "matrix.h"


namespace NMath
{
	Vector2 SolveLineCoeffs(const Vector2& p1, const Vector2& p2);
	bool SolveQuadraticRoots(float a, float b, float c, float& t1, float &t2);

	Vector2 SolveLeastSquaresLineDirection(const Vector2& linePoint, const Vector2* points, int pointsCount, int iterationsCount);
	Vector3 SolveLeastSquaresLineDirection(const Vector3& linePoint, const Vector3* points, int pointsCount, int iterationsCount);

	//

	inline Vector2 SolveLineCoeffs(const Vector2& p1, const Vector2& p2)
	{
		float a = (p1.y - p2.y) / (p1.x - p2.x);
		float b = p1.y - a*p1.x;

		return VectorCustom(a, b);
	}

	inline bool SolveQuadraticRoots(float a, float b, float c, float& x1, float &x2)
	{
		float delta = b*b - 4.0f*a*c;

		if (delta < 0.0f)
			return false;

		float deltaSqrt = Sqrt(delta);

		float q;
		if (b < 0.0f)
			q = -0.5f * (b - deltaSqrt);
		else
			q = -0.5f * (b + deltaSqrt);

		x1 = q / a;
		x2 = c / q;

		if (x1 > x2)
			Swap(x1, x2);

		return true;
	}

	inline Vector2 SolveLeastSquaresLineDirection(const Vector2& linePoint, const Vector2* points, int pointsCount, int iterationsCount)
	{
		Matrix matrix;
		SetZeros(matrix);
		for (int i = 0; i < pointsCount; i++)
		{
			Vector2 diff = points[i] - linePoint;

			matrix.m[0][0] += diff.x * diff.x;
			matrix.m[0][1] += diff.x * diff.y;
			matrix.m[1][0] += diff.y * diff.x;
			matrix.m[1][1] += diff.y * diff.y;
		}

		Vector2 eigenVector = Vector2EX;
		for (int i = 0; i < iterationsCount; i++)
		{
			eigenVector = eigenVector * matrix;
			eigenVector = eigenVector / MaxComponent(eigenVector);
		}

		return Normalize(eigenVector);
	}

	inline Vector3 SolveLeastSquaresLineDirection(const Vector3& linePoint, const Vector3* points, int pointsCount, int iterationsCount)
	{
		Matrix matrix;
		SetZeros(matrix);
		for (int i = 0; i < pointsCount; i++)
		{
			Vector3 diff = points[i] - linePoint;

			matrix.m[0][0] += diff.x * diff.x;
			matrix.m[0][1] += diff.x * diff.y;
			matrix.m[0][2] += diff.x * diff.z;
			matrix.m[1][0] += diff.y * diff.x;
			matrix.m[1][1] += diff.y * diff.y;
			matrix.m[1][2] += diff.y * diff.z;
			matrix.m[2][0] += diff.z * diff.x;
			matrix.m[2][1] += diff.z * diff.y;
			matrix.m[2][2] += diff.z * diff.z;
		}

		Vector3 eigenVector = Vector3EX;
		for (int i = 0; i < iterationsCount; i++)
		{
			eigenVector = eigenVector * matrix;
			eigenVector = eigenVector / MaxComponent(eigenVector);
		}

		return Normalize(eigenVector);
	}
}
