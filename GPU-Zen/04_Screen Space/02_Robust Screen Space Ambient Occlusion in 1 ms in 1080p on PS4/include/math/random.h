#pragma once


#include "vector.h"
#include <essentials/types.h>

#include <cstdlib>
#include <ctime>


using namespace NEssentials;


namespace NMath
{
	void Randomize();

	uint16 Random16();
	uint16 Random16(uint16 from, uint16 to);

	uint32 Random32();
	uint32 Random32(uint32 from, uint32 to);

	float RandomFloat(); // [0..1]
	float RandomFloat(float from, float to);

	Vector2 RandomUnitVector2();
	Vector3 RandomUnitVector3();

	//

	inline void Randomize()
	{
		srand((uint)time(nullptr));
	}

	inline uint16 Random16()
	{
		uint16 r1 = rand() % 256; // 2^8
		uint16 r2 = rand() % 256; // 2^8

		return (r1) | (r2 << 8);
	}

	inline uint16 Random16(uint16 from, uint16 to)
	{
		return from + (Random16() % (to - from + 1));
	}

	inline uint32 Random32()
	{
		uint32 r1 = rand() % 2048; // 2^11
		uint32 r2 = rand() % 2048; // 2^11
		uint32 r3 = rand() % 1024; // 2^10

		return (r1) | (r2 << 11) | (r3 << 22);
	}

	inline uint32 Random32(uint32 from, uint32 to)
	{
		return from + (Random32() % (to - from + 1));
	}

	inline float RandomFloat()
	{
		return (float)rand() / (float)RAND_MAX;
	}

	inline float RandomFloat(float from, float to)
	{
		return from + (RandomFloat() * (to - from));
	}

	inline Vector2 RandomUnitVector2()
	{
		Vector2 v;

		v.x = RandomFloat() - 0.5f;
		v.y = RandomFloat() - 0.5f;
		NormalizeIn(v);

		return v;
	}

	inline Vector3 RandomUnitVector3()
	{
		Vector3 v;

		v.x = RandomFloat() - 0.5f;
		v.y = RandomFloat() - 0.5f;
		v.z = RandomFloat() - 0.5f;
		NormalizeIn(v);

		return v;
	}
}
