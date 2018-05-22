#ifndef __MATH_H__
#define __MATH_H__

#include <stdlib.h>
#include <limits>
#include <math.h>
#include <assert.h>
#include "Vector3.h"
#include "Matrix3.h"

class Ray;

// -----------------------------------------------------------------------------
/// Math
// -----------------------------------------------------------------------------
/// \ingroup 
///
// -----------------------------------------------------------------------------
class Math
{
public:
	static const float MAX_FLOAT;
	static const float EPSILON_FLOAT;
	static const float PI;
	static const float HALF_PI;
	static const float DOUBLE_PI;
	static const float DEG_TO_RAD;
	static const float RAD_TO_DEG;

	struct FrustumPlane
	{
		union
		{
			struct
			{
				Vector3 normal;
				float d;
			};

			float plane[4];
		};
	};

	static int NextPowerOf2Value(int val);
	static bool IsPowerOf2(int val);

	static float GetAngleY(const Vector3& direction);
	static Vector3 GetDirectionFromAngleY(float angle);

	static float Trunc(float val);
	static float Floor(float val) { return floorf(val); }
	static float Ceil(float val) { return ceilf(val); }
	static float Frac(float val);
	static float Clamp(float val, float min, float max);
	static int Clamp(int val, int min, int max);

	static bool IsSmaller(float x, float y) { return x < (y-EPSILON_FLOAT); }
	static bool IsBigger(float x, float y) { return x > (y+EPSILON_FLOAT); }
	static bool IsEqual(float x, float y) { return !IsSmaller(x,y) && !IsBigger(x,y); }
	static bool IsNotEqual(float x, float y) { return IsSmaller(x,y) || IsBigger(x,y); }    

	static float Sin(float degree) { return sinf(degree); }
	static float ASin(float degree)
	{
		assert((degree >= -1.0f) && (degree <= +1.0f));
		return asinf(degree);
	}
	static float Cos(float degree) { return cosf(degree); }
	static float ACos(float c)
	{
		assert((c >= -1.0f) && (c <= +1.0f));
		return acosf(c);
	}
	static float Tan(float degree) { return tanf(degree); }
	static float ATan(float degree) { return atanf(degree); }
	static float ATan2(float x, float y) { return atan2f(x, y); }
	static float Cot(float degree) { return 1.0f/Tan(degree); }

	static float Sqrt(float value) { return sqrtf(value); }
	static float Log(float value) { return logf(value); }
	static float InvSqrt(float value);
	static float Abs(float value) { return abs(value); }
	static int Abs(int value) { return abs(value); }
	static float Pow(float base, float exponent) { return powf(base, exponent); }
	static int Pow(int base, int exponent) { return (int)pow((float)base, (float)exponent); }

	static int Max(int x, int y) { return((x)>(y)?(x):(y)); }
	static float Max(float x, float y) { return((x)>(y)?(x):(y)); }
	static int Min(int x, int y) { return((x)<(y)?(x):(y)); }
	static float Min(float x, float y) { return((x)<(y)?(x):(y)); }

	static float RandomFloat(const float fMin, const float fMax) {return fMin + (fMax - fMin) * ((float)(rand() % 10001) / 10000.0f);}

	static void DirectionToPitchHeading(const Vector3& dir, float& pitch, float& heading);
};

#define RAD2DEG(a) (((a)*180.0f)/(Math::PI))
#define DEG2RAD(a) (((a)*Math::PI)/180.0f)

#endif //__MATH_H__