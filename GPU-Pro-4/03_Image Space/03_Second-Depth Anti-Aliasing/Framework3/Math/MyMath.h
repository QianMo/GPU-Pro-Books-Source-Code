
/* * * * * * * * * * * * * Author's note * * * * * * * * * * * *\
*   _       _   _       _   _       _   _       _     _ _ _ _   *
*  |_|     |_| |_|     |_| |_|_   _|_| |_|     |_|  _|_|_|_|_|  *
*  |_|_ _ _|_| |_|     |_| |_|_|_|_|_| |_|     |_| |_|_ _ _     *
*  |_|_|_|_|_| |_|     |_| |_| |_| |_| |_|     |_|   |_|_|_|_   *
*  |_|     |_| |_|_ _ _|_| |_|     |_| |_|_ _ _|_|  _ _ _ _|_|  *
*  |_|     |_|   |_|_|_|   |_|     |_|   |_|_|_|   |_|_|_|_|    *
*                                                               *
*                     http://www.humus.name                     *
*                                                                *
* This file is a part of the work done by Humus. You are free to   *
* use the code in any way you like, modified, unmodified or copied   *
* into your own work. However, I expect you to respect these points:  *
*  - If you use this file and its contents unmodified, or use a major *
*    part of this file, please credit the author and leave this note. *
*  - For use in anything commercial, please request my approval.     *
*  - Share your work and ideas too as much as you can.             *
*                                                                *
\* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _MYMATH_H_
#define _MYMATH_H_

#include <math.h>

#define PI 3.14159265358979323846f

// This code supposedly originates from Id-software
// It computes a fast 1 / sqrtf(v) approximation
/*
inline float rsqrtf(float v){
    float v_half = v * 0.5f;
    int i = *(int *) &v;
    i = 0x5f3759df - (i >> 1);
    v = *(float *) &i;
    return v * (1.5f - v_half * v * v);
}
*/
inline float rsqrtf(const float v){
	union {
		float vh;
		int i0;
	};

	union {
		float vr;
		int i1;
	};

	vh = v * 0.5f;
	i1 = 0x5f3759df - (i0 >> 1);
	return vr * (1.5f - vh * vr * vr);
}


inline float sqrf(const float x){
    return x * x;
}

inline float sincf(const float x){
	return (x == 0)? 1 : sinf(x) / x;
}

#define roundf(x) floorf((x) + 0.5f)

#ifndef min
#define min(x, y) ((x < y)? x : y)
#endif

#ifndef max
#define max(x, y) ((x > y)? x : y)
#endif

inline float intAdjustf(const float x, const float diff = 0.01f){
	float f = roundf(x);

	return (fabsf(f - x) < diff)? f : x;
}

inline unsigned int getClosestPowerOfTwo(const unsigned int x){
	unsigned int i = 1;
	while (i < x) i += i;

	if (4 * x < 3 * i) i >>= 1;
	return i;
}

inline unsigned int getUpperPowerOfTwo(const unsigned int x){
	unsigned int i = 1;
	while (i < x) i += i;

	return i;
}

inline unsigned int getLowerPowerOfTwo(const unsigned int x){
	unsigned int i = 1;
	while (i <= x) i += i;

	return i >> 1;
}

inline int round(float x){
	if (x > 0){
		return int(x + 0.5f);
	} else {
		return int(x - 0.5f);
	}
}

#endif // _MYMATH_H_
