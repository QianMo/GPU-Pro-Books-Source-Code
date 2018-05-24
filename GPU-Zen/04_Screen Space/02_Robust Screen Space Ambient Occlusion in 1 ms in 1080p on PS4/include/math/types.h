#pragma once


namespace NMath
{
	struct Vector2
	{
		float x, y;
	};

	struct Vector3
	{
		float x, y, z;
	};

	struct Vector4
	{
		float x, y, z, w;
	};

	struct Quaternion
	{
		float x, y, z, w;
	};

	struct Plane
	{
		float a, b, c, d;
	};

	struct Matrix
	{
		union
		{
			float m[4][4];
			struct
			{
				Vector4 x, y, z, w;
			};
		};
	};

	struct Spherical
	{
		float theta, phi; // theta in [0..Pi], phi in [0..2*Pi]
	};

	enum class ZRange { ZeroToOne, MinusOneToPlusOne };
}
