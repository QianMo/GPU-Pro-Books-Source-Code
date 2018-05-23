/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_BITS
#define BE_MATH_BITS

#include "beMath.h"
#include "beVectorDef.h"

namespace beMath
{

/// Puts a zero bit in-between each of the lower 16 bits of the given value.
inline uint4 bitsep1(uint4 x)
{
	// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	x &= 0xffff;                     // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}

/// Puts two zero bits in-between each of the lower 10 bits of the given value.
inline uint4 bitsep2(uint4 x)
{
	// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

/// Inverse of bitsep1.
inline uint4 bitcomp1(uint4 x)
{
	// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	x &= 0x55555555;                  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	x = (x ^ (x >> 1)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x >> 2)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x >> 4)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x >> 8)) & 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
	return x;
}

/// Inverse of bitsep2.
inline uint4 bitcomp2(uint4 x)
{
	// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
	return x;
}

/// Morton code for 2 dimensions.
inline uint4 bitzip(const vector<uint4, 2> &v)
{
	return (bitsep1(v[1]) << 1) + bitsep1(v[0]);
}

/// Morton code for 3 dimensions.
inline uint4 bitzip(const vector<uint4, 3> &v)
{
	return (bitsep2(v[2]) << 2) + (bitsep2(v[1]) << 1) + bitsep2(v[0]);
}

/// 2 dimensions from morton code.
inline vector<uint4, 2> bitunzip2(uint4 c)
{
	return vec( bitcomp1(c), bitcomp1(c >> 1) );
}

/// 3 dimensions from morton code.
inline vector<uint4, 3> bitunzip3(uint4 c)
{
	return vec( bitcomp2(c), bitcomp2(c >> 1), bitcomp2(c >> 2) );
}

} // namespace

#endif