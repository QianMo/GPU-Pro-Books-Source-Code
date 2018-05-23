#ifndef BE_UTILITY_BITS_H
#define BE_UTILITY_BITS_H

/// Converts the given float into an integer bit representation maintaing relative order.
int asint_ordered(float f)
{
	int i = asint(f);
	if (i < 0)
		i = (1 << 31) - i;
	return i;
}
/// Converts the given float into an integer bit representation maintaing relative order.
int2 asint_ordered(float2 f)
{
	return int2(asint_ordered(f.x), asint_ordered(f.y));
}
/// Converts the given float into an integer bit representation maintaing relative order.
int3 asint_ordered(float3 f)
{
	return int3(asint_ordered(f.x), asint_ordered(f.y), asint_ordered(f.z));
}
/// Converts the given float into an integer bit representation maintaing relative order.
int4 asint_ordered(float4 f)
{
	return int4(asint_ordered(f.x), asint_ordered(f.y), asint_ordered(f.z), asint_ordered(f.w));
}

/// Converts the given ordered integer bit representation into the corresponding float.
float asfloat_ordered(int i)
{
	if (i < 0)
		i = (1 << 31) - i;
	return asfloat(i);
}
/// Converts the given ordered integer bit representation into the corresponding float.
float2 asfloat_ordered(int2 i)
{
	return float2(asfloat_ordered(i.x), asfloat_ordered(i.y));
}
/// Converts the given ordered integer bit representation into the corresponding float.
float3 asfloat_ordered(int3 i)
{
	return float3(asfloat_ordered(i.x), asfloat_ordered(i.y), asfloat_ordered(i.z));
}
/// Converts the given ordered integer bit representation into the corresponding float.
float4 asfloat_ordered(int4 i)
{
	return float4(asfloat_ordered(i.x), asfloat_ordered(i.y), asfloat_ordered(i.z), asfloat_ordered(i.w));
}

/// Vector interlocked min.
#define InterlockedMin2(a, b) (InterlockedMin(a.x, b.x), InterlockedMin(a.y, b.y))
/// Vector interlocked max.
#define InterlockedMax2(a, b) (InterlockedMax(a.x, b.x), InterlockedMax(a.y, b.y))
/// Vector interlocked min.
#define InterlockedMin3(a, b) (InterlockedMin(a.x, b.x), InterlockedMin(a.y, b.y), InterlockedMin(a.z, b.z))
/// Vector interlocked max.
#define InterlockedMax3(a, b) (InterlockedMax(a.x, b.x), InterlockedMax(a.y, b.y), InterlockedMax(a.z, b.z))
/// Vector interlocked min.
#define InterlockedMin4(a, b) (InterlockedMin(a.x, b.x), InterlockedMin(a.y, b.y), InterlockedMin(a.z, b.z), InterlockedMin(a.w, b.w))
/// Vector interlocked max.
#define InterlockedMax4(a, b) (InterlockedMax(a.x, b.x), InterlockedMax(a.y, b.y), InterlockedMax(a.z, b.z), InterlockedMax(a.w, b.w))

/// Packs the given vector into 16-bit float values.
uint packf16(float2 v)
{
	uint2 h = f32tof16(v);
	return h.x | (h.y << 16);
}

/// Packs the given vector into 16-bit float values.
uint2 packf16(float3 v)
{
	uint3 h = f32tof16(v);
	h.x |= h.y << 16;
	return h.xz;
}

/// Packs the given vector into 16-bit float values.
uint2 packf16(float4 v)
{
	uint4 h = f32tof16(v);
	return h.xz | (h.yw << 16);
}

/// Unpacks the given 16-bit float vector.
float2 unpackf16(uint v)
{
	uint2 h = uint2(v, v >> 16);
	return f16tof32(h);
}

/// Unpacks the given 16-bit float vector.
float4 unpackf16(uint2 v)
{
	uint4 h = uint4(v, v >> 16).xzyw;
	return f16tof32(h);
}

/// Converts the given color vector into a 32-bit color value.
uint packu8(float4 color)
{
	uint4 b = D3DCOLORtoUBYTE4(color);
	b = b << uint4(0, 8, 16, 24);
	return b.x | b.y | b.z | b.w;
}

/// Converts the given 32-bit color value into a color vector.
float4 unpacku8(uint color)
{
	return ( (color >> uint4(16, 8, 0, 24)) & 0xff ) / 255.0f;
}

/// Puts a zero bit in-between each of the lower 16 bits of the given value.
uint bitsep1(uint x)
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
uint bitsep2(uint x)
{
	// http://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

/// Puts four zero bits in-between each of the lower 6 bits of the given value.
uint bitsep4l(uint x)
{
	x &= 0x0000003f;                  // x = ---- ---- ---- ---- ---- ---- --54 3210
	x = (x ^ (x << 16)) & 0x00f000ff; // x = ---- ---- 7654 ---- ---- ---- ---- 3210
	x = (x ^ (x <<  8)) & 0xc0300c03; // x = 76-- ---- --54 ---- ---- 32-- ---- --10
	x = (x ^ (x <<  4)) & 0x42108421; // x = -6-- --5- ---4 ---- 3--- -2-- --1- ---0
	return x;
}

/// Puts four zero bits in-between bits 6 and 7 of the given value.
uint bitsep4h(uint x)
{
	x = (x >> 6) & 0x00000003;        // x = ---- ---- ---- ---- ---- ---- ---- --76
	x = (x ^ (x << 4)) & 0x00000021;  // x = ---- ---- ---- ---- ---- ---- --7- ---6
	return x;
}

/// Puts four zero bits in-between each of the lower 8 bits of the given value.
uint2 bitsep4(uint x)
{
	return uint2( bitsep4l(x), bitsep4h(x) );
}

/// Inverse of bitsep1.
uint bitcomp1(uint x)
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
uint bitcomp2(uint x)
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
uint bitzip(uint2 v)
{
	return (bitsep1(v.y) << 1) + bitsep1(v.x);
}

/// Morton code for 3 dimensions.
uint bitzip(uint3 v)
{
	return (bitsep2(v.z) << 2) + (bitsep2(v.y) << 1) + bitsep2(v.x);
}

/// 2 dimensions from morton code.
uint2 bitunzip2(uint c)
{
	return uint2( bitcomp1(c), bitcomp1(c >> 1) );
}

/// 3 dimensions from morton code.
uint3 bitunzip3(uint c)
{
	return uint3( bitcomp2(c), bitcomp2(c >> 1), bitcomp2(c >> 2) );
}

uint bitpack(uint2 v)
{
	return v.x + (v.y << 16);
}

uint bitpack(uint3 v)
{
	return v.x + (v.y << 10) + (v.z << 20);
}

uint2 bitunpack2(uint c)
{
	return uint2(c & 0xFFFF, c >> 16);
}

uint3 bitunpack3(uint c)
{
	return uint3(c & 0x3FF, (c >> 10) & 0x3FF, c >> 20);
}

/// Generates a bit mask with bits minIdx to maxIdx set (inclusive).
uint4 rangeMask128(int minIdx, int maxIdx)
{
	int4 maxIdxShift = int4(31, 63, 95, 127) - maxIdx;
	int4 minIdxShift = minIdx - int4(0, 32, 64, 96);

	uint4 rangeMask = ((uint4) -(maxIdxShift < 32)) >> (uint4) max(maxIdxShift, 0);
	rangeMask &= ((uint4) -(minIdxShift < 32)) << (uint4) max(minIdxShift, 0);

	return rangeMask;
}

/// Generates a bit mask with bit idx set.
uint4 mask128(int idx)
{
	uint cidx = (uint) clamp(idx, 0, 127);
	return ((uint4) ((cidx >> 5) == uint4(0, 1, 2, 3))) << cidx;
}

#endif