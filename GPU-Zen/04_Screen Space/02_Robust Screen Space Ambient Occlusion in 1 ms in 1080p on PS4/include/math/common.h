#pragma once


#include "constants.h"
#include <essentials/types.h>

#include <cmath>


using namespace NEssentials;


namespace NMath
{
	float Pow(float x, float y);
	float Sqrt(float x);
	float Sin(float x);
	float Cos(float x);
	float Tan(float x);
	float ASin(float x);
	float ACos(float x);
	float ACos_Clamped(float x);
	float ATan2(float y, float x);
	template <class TYPE> TYPE Abs(TYPE x);
	template <class TYPE> TYPE Sqr(TYPE x);
	template <class TYPE> TYPE Max(TYPE x, TYPE y);
	template <class TYPE> TYPE Min(TYPE x, TYPE y);
	template <class TYPE> TYPE Clamp(TYPE x, TYPE min, TYPE max);
	template <class TYPE> bool IsPowerOfTwo(TYPE x);
	template <class TYPE> void Swap(TYPE& x, TYPE& y);
	int Round(float x);
	float Frac(float x);
	float Saturate(float x);
	float Log2(float x);

	uint32 Idx(uint32 x, uint32 y, uint32 width);
	uint32 Idx(uint32 x, uint32 y, uint32 z, uint32 width, uint32 height);

	//

	inline float Pow(float x, float y)
	{
		return powf(x, y);
	}

	inline float Sqrt(float x)
	{
		return sqrtf(x);
	}

	inline float Sin(float x)
	{
		return sinf(x);
	}

	inline float Cos(float x)
	{
		return cosf(x);
	}

	inline float Tan(float x)
	{
		return tanf(x);
	}

	inline float ASin(float x)
	{
		return asinf(x);
	}

	inline float ACos(float x)
	{
		return acosf(x);
	}

	inline float ACos_Clamped(float x)
	{
		if (x <= -1.0f)
			return Pi;
		else if (x >= 1.0f)
			return 0.0f;
		else
			return acosf(x);
	}

	inline float ATan2(float y, float x)
	{
		return atan2f(y, x);
	}

	template <class TYPE>
	inline TYPE Abs(TYPE x)
	{
		if (x < 0)
			return -x;
		else
			return x;
	}

	template <class TYPE>
	inline TYPE Sqr(TYPE x)
	{
		return x * x;
	}

	template <class TYPE>
	inline TYPE Max(TYPE x, TYPE y)
	{
		return x > y ? x : y;
	}

	template <class TYPE>
	inline TYPE Min(TYPE x, TYPE y)
	{
		return x < y ? x : y;
	}

	template <class TYPE>
	inline TYPE Clamp(TYPE x, TYPE min, TYPE max)
	{
		return Max(Min(x, max), min);
	}

	template <class TYPE>
	inline bool IsPowerOfTwo(TYPE x)
	{
		return !(x & (x-1));
	}

	template <class TYPE>
	inline void Swap(TYPE& x, TYPE& y)
	{
		TYPE z = x;
		x = y;
		y = z;
	}

	inline int Round(float x)
	{
		return (int)(x + 0.5f);
	}

	inline float Frac(float x)
	{
		return x - (int)x;
	}

	inline float Saturate(float x)
	{
		return Clamp(x, 0.0f, 1.0f);
	}

	inline float Log2(float x)
	{
		return logf(x) / logf(2.0f);
	}

	inline uint32 Idx(uint32 x, uint32 y, uint32 width)
	{
		return y*width + x;
	}

	inline uint32 Idx(uint32 x, uint32 y, uint32 z, uint32 width, uint32 height)
	{
		return z*width*height + y*height + x;
	}
}
