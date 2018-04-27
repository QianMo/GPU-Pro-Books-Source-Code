/*!  \file itype.h
 */

#ifndef _FBXSDK_KLIB_ITYPE_H_
#define _FBXSDK_KLIB_ITYPE_H_

/**************************************************************************************

 Copyright © 2001 - 2008 Autodesk, Inc. and/or its licensors.
 All Rights Reserved.

 The coded instructions, statements, computer programs, and/or related material 
 (collectively the "Data") in these files contain unpublished information 
 proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 Canada and United States of America federal copyright law and by international 
 treaties. 
 
 The Data may not be disclosed or distributed to third parties, in whole or in
 part, without the prior written consent of Autodesk, Inc. ("Autodesk").

 THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 
 IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 OR DAMAGE. 

**************************************************************************************/
#include <karch/types.h>

#define K_INTERNAL

#define K_BIT0		       (0x1)
#define K_BIT1		       (0x2)
#define K_BIT2		       (0x4)
#define K_BIT3		       (0x8)
#define K_BIT4		      (0x10)
#define K_BIT5		      (0x20)
#define K_BIT6		      (0x40)
#define K_BIT7		      (0x80)
#define K_BIT8		     (0x100)
#define K_BIT9		     (0x200)
#define K_BIT10		     (0x400)
#define K_BIT11		     (0x800)
#define K_BIT12		    (0x1000)
#define K_BIT13		    (0x2000)
#define K_BIT14		    (0x4000)
#define K_BIT15		    (0x8000)
#define K_BIT16		   (0x10000)
#define K_BIT17		   (0x20000)
#define K_BIT18		   (0x40000)
#define K_BIT19		   (0x80000)
#define K_BIT20		  (0x100000)
#define K_BIT21		  (0x200000)
#define K_BIT22		  (0x400000)
#define K_BIT23		  (0x800000)
#define K_BIT24		 (0x1000000)
#define K_BIT25		 (0x2000000)
#define K_BIT26		 (0x4000000)
#define K_BIT27		 (0x8000000)
#define K_BIT28		(0x10000000)
#define K_BIT29		(0x20000000)
#define K_BIT30		(0x40000000)
#define K_BIT31		(0x80000000)

#define K_BITS8(b7, b6, b5, b4, b3, b2, b1, b0) \
	((kUInt8(b7) << 7) | (kUInt8(b6) << 6) | (kUInt8(b5) << 5) | (kUInt8(b4) << 4) | \
	 (kUInt8(b3) << 3) | (kUInt8(b2) << 2) | (kUInt8(b1) << 1) | kUInt8(b0))

#define K_BITS16( \
	b15, b14, b13, b12, b11, b10, b9, b8, \
	b7, b6, b5, b4, b3, b2, b1, b0) \
\
	((kUInt16(b15) << 15) | (kUInt16(b14) << 14) | (kUInt16(b13) << 13) | (kUInt16(b12) << 12) | \
	 (kUInt16(b11) << 11) | (kUInt16(b10) << 10) | (kUInt16(b9) << 9) | (kUInt16(b8) << 8) | \
	 (kUInt16(b7) << 7) | (kUInt16(b6) << 6) | (kUInt16(b5) << 5) | (kUInt16(b4) << 4) | \
	 (kUInt16(b3) << 3) | (kUInt16(b2) << 2) | (kUInt16(b1) << 1) | kUInt16(b0))

#define K_BITS32( \
	b31, b30, b29, b28, b27, b26, b25, b24, \
	b23, b22, b21, b20, b19, b18, b17, b16, \
	b15, b14, b13, b12, b11, b10, b9, b8, \
	b7, b6, b5, b4, b3, b2, b1, b0) \
\
	((kUInt32(b31) << 31) | (kUInt32(b30) << 30) | (kUInt32(b29) << 29) | (kUInt32(b28) << 28) | \
	 (kUInt32(b27) << 27) | (kUInt32(b26) << 26) | (kUInt32(b25) << 25) | (kUInt32(b24) << 24) | \
	 (kUInt32(b23) << 23) | (kUInt32(b22) << 22) | (kUInt32(b21) << 21) | (kUInt32(b20) << 20) | \
	 (kUInt32(b19) << 19) | (kUInt32(b18) << 18) | (kUInt32(b17) << 17) | (kUInt32(b16) << 16) | \
	 (kUInt32(b15) << 15) | (kUInt32(b14) << 14) | (kUInt32(b13) << 13) | (kUInt32(b12) << 12) | \
	 (kUInt32(b11) << 11) | (kUInt32(b10) << 10) | (kUInt32(b9) << 9) | (kUInt32(b8) << 8) | \
	 (kUInt32(b7) << 7) | (kUInt32(b6) << 6) | (kUInt32(b5) << 5) | (kUInt32(b4) << 4) | \
	 (kUInt32(b3) << 3) | (kUInt32(b2) << 2) | (kUInt32(b1) << 1) | kUInt32(b0))

#define K_BITS64( \
	b63, b62, b61, b60, b59, b58, b57, b56, \
	b55, b54, b53, b52, b51, b50, b49, b48, \
	b47, b46, b45, b44, b43, b42, b41, b40, \
	b39, b38, b37, b36, b35, b34, b33, b32, \
	b31, b30, b29, b28, b27, b26, b25, b24, \
	b23, b22, b21, b20, b19, b18, b17, b16, \
	b15, b14, b13, b12, b11, b10, b9, b8, \
	b7, b6, b5, b4, b3, b2, b1, b0) \
\
	((kUInt64(b63) << 63) | (kUInt64(b62) << 62) | (kUInt64(b61) << 61) | (kUInt64(b60) << 60) | \
	 (kUInt64(b59) << 59) | (kUInt64(b58) << 58) | (kUInt64(b57) << 57) | (kUInt64(b56) << 56) | \
	 (kUInt64(b55) << 55) | (kUInt64(b54) << 54) | (kUInt64(b53) << 53) | (kUInt64(b52) << 52) | \
	 (kUInt64(b51) << 51) | (kUInt64(b50) << 50) | (kUInt64(b49) << 49) | (kUInt64(b48) << 48) | \
	 (kUInt64(b47) << 47) | (kUInt64(b46) << 46) | (kUInt64(b45) << 45) | (kUInt64(b44) << 44) | \
	 (kUInt64(b43) << 43) | (kUInt64(b42) << 42) | (kUInt64(b41) << 41) | (kUInt64(b40) << 40) | \
	 (kUInt64(b39) << 39) | (kUInt64(b38) << 38) | (kUInt64(b37) << 37) | (kUInt64(b36) << 36) | \
	 (kUInt64(b35) << 35) | (kUInt64(b34) << 34) | (kUInt64(b33) << 33) | (kUInt64(b32) << 32) | \
	 (kUInt64(b31) << 31) | (kUInt64(b30) << 30) | (kUInt64(b29) << 29) | (kUInt64(b28) << 28) | \
	 (kUInt64(b27) << 27) | (kUInt64(b26) << 26) | (kUInt64(b25) << 25) | (kUInt64(b24) << 24) | \
	 (kUInt64(b23) << 23) | (kUInt64(b22) << 22) | (kUInt64(b21) << 21) | (kUInt64(b20) << 20) | \
	 (kUInt64(b19) << 19) | (kUInt64(b18) << 18) | (kUInt64(b17) << 17) | (kUInt64(b16) << 16) | \
	 (kUInt64(b15) << 15) | (kUInt64(b14) << 14) | (kUInt64(b13) << 13) | (kUInt64(b12) << 12) | \
	 (kUInt64(b11) << 11) | (kUInt64(b10) << 10) | (kUInt64(b9) << 9) | (kUInt64(b8) << 8) | \
	 (kUInt64(b7) << 7) | (kUInt64(b6) << 6) | (kUInt64(b5) << 5) | (kUInt64(b4) << 4) | \
	 (kUInt64(b3) << 3) | (kUInt64(b2) << 2) | (kUInt64(b1) << 1) | kUInt64(b0))

#define K_BYTES2(b1, b0) \
	((kUInt16(b1) << 8) | kUInt16(b0))

#define K_BYTES4(b3, b2, b1, b0) \
	((kUInt32(b3) << 24) | (kUInt32(b2) << 16) | (kUInt32(b1) << 8) | kUInt32(b0))

#define K_BYTES8(b7, b6, b5, b4, b3, b2, b1, b0) \
	((kUInt64(b7) << 56) | (kUInt64(b6) << 48) | (kUInt64(b5) << 40) | (kUInt64(b4) << 32) | \
	 (kUInt64(b3) << 24) | (kUInt64(b2) << 16) | (kUInt64(b1) << 8) | kUInt64(b0))

// member pad, example: int K_MEMPAD(var, sizeof(int));

#define K_MEMPAD(var, n) \
	var; \
	kInt8 _##var##_pad[n] \

// get this pointer from a member address
#define K_THIS(ptr, type, member)	\
	((type *) (kOffset(ptr) - kOffset(&((type *) 0)->member)))

#define K_LENGTHOF(array)		int(sizeof(array) / sizeof(array[0]))

#define kSwap(x, y, t)			t = x; x = y; y = t

//******************************************************************************
//
//	pointer manipulation
//
//******************************************************************************

// void pointer increment
K_INLINE void *kInc(void *p, size_t i)
{
	return (void *) (((size_t *) p) + i);
}

//******************************************************************************
//
//	basic type information
//
//******************************************************************************

// upward saturation value
template<class T> K_INLINE T kMin(T);
// common specializations
K_TEMPLATE_SPEC(kSChar) K_INLINE kSChar kMin(kSChar)
{
	return K_SCHAR_MIN;
}
K_TEMPLATE_SPEC(kUChar) K_INLINE kUChar kMin(kUChar)
{
	return K_UCHAR_MIN;
}
K_TEMPLATE_SPEC(kShort) K_INLINE kShort kMin(kShort)
{
	return K_SHORT_MIN;
}
K_TEMPLATE_SPEC(kUShort) K_INLINE kUShort kMin(kUShort)
{
	return K_USHORT_MIN;
}
K_TEMPLATE_SPEC(int) K_INLINE int kMin(int)
{
	return K_INT_MIN;
}
K_TEMPLATE_SPEC(kUInt) K_INLINE kUInt kMin(kUInt)
{
	return K_UINT_MIN;
}

K_TEMPLATE_SPEC(kLongLong) K_INLINE kLongLong kMin(kLongLong)
{
	return K_LONGLONG_MIN;
}
K_TEMPLATE_SPEC(kULongLong) K_INLINE kULongLong kMin(kULongLong)
{
	return K_ULONGLONG_MIN;
}

// downward saturation value
template<class T> K_INLINE T kMax(T);
// common specializations
K_TEMPLATE_SPEC(kSChar) K_INLINE kSChar kMax(kSChar)
{
	return K_SCHAR_MAX;
}
K_TEMPLATE_SPEC(kUChar) K_INLINE kUChar kMax(kUChar)
{
	return K_UCHAR_MAX;
}
K_TEMPLATE_SPEC(kShort) K_INLINE kShort kMax(kShort)
{
	return K_SHORT_MAX;
}
K_TEMPLATE_SPEC(kUShort) K_INLINE kUShort kMax(kUShort)
{
	return K_USHORT_MAX;
}
K_TEMPLATE_SPEC(int) K_INLINE int kMax(int)
{
	return K_INT_MAX;
}
K_TEMPLATE_SPEC(kUInt) K_INLINE kUInt kMax(kUInt)
{
	return K_UINT_MAX;
}

K_TEMPLATE_SPEC(kLongLong) K_INLINE kLongLong kMax(kLongLong)
{
	return K_LONGLONG_MAX;
}
K_TEMPLATE_SPEC(kULongLong) K_INLINE kULongLong kMax(kULongLong)
{
	return K_ULONGLONG_MAX;
}

//******************************************************************************
//
//	bit manipulation routines
//
//******************************************************************************

// logical shift left
template<class T> K_INLINE T kShl(T x)
{
	return x << 1;
}
template<class T> K_INLINE T kShl(T x, int y)
{
	return x << y;
}

// logical shift right
template<class T> K_INLINE T kShr(T x)
{
	return x >> 1;
}
// common specializations
K_TEMPLATE_SPEC(kSChar) K_INLINE kSChar kShr(kSChar x)
{
	return kSChar(kUChar(x) >> 1);
}
K_TEMPLATE_SPEC(kShort) K_INLINE kShort kShr(kShort x)
{
	return kShort(kUShort(x) >> 1);
}
K_TEMPLATE_SPEC(int) K_INLINE int kShr(int x)
{
	return int(kUInt(x) >> 1);
}

K_TEMPLATE_SPEC(kLongLong) K_INLINE kLongLong kShr(kLongLong x)
{
	return kLongLong(kULongLong(x) >> 1);
}

template<class T> K_INLINE T kShr(T x, int y)
{
	return x >> y;
}
// common specializations
K_TEMPLATE_SPEC(kSChar) K_INLINE kSChar kShr(kSChar x, int y)
{
	return kSChar(kUChar(x) >> y);
}
K_TEMPLATE_SPEC(kShort) K_INLINE kShort kShr(kShort x, int y)
{
	return kShort(kUShort(x) >> y);
}
K_TEMPLATE_SPEC(int) K_INLINE int kShr(int x, int y)
{
	return int(kUInt(x) >> y);
}

K_TEMPLATE_SPEC(kLongLong) K_INLINE kLongLong kShr(kLongLong x, int y)
{
	return kLongLong(kULongLong(x) >> y);
}

// logical roll left
template<class T> K_INLINE T kRotl(T x)
{
	return kShl(x) | kShr(x, (sizeof(x) << 3) - 1);
}
template<class T> K_INLINE T kRotl(T x, int y)
{
	y &= ((sizeof(x) << 3) - 1);

	return kShl(x, y) | kShr(x, (sizeof(x) << 3) - y);
}

// logical roll right
template<class T> K_INLINE T kRotr(T x)
{
	return kShr(x) | kShl(x, (sizeof(x) << 3) - 1);
}
template<class T> K_INLINE T kRotr(T x, int y)
{
	y &= ((sizeof(x) << 3) - 1);

	return kShr(x, y) | kShl(x, (sizeof(x) << 3) - y);
}

// bit flip
template<class T> K_INLINE T kFlip(T x, T y)
{ 
	T z = 0; 

	while(y) { z = (z << 1) | (x & 1); x = kShr(x); y = kShr(y); } 
	
	return z;
}

// most significant bit (-1 for none)
template<class T> K_INLINE int kMSB(T x)
{
	int n = -1;

	while(x) { x = kShr(x); n++; }
	
	return n;
}

// least significant bit (-1 for none)
template<class T> K_INLINE int kLSB(T x)
{
	int n = sizeof(x) << 3;
	
	while(x) { x = kShl(x); n--; } 
	
	return (n == (sizeof(x) << 3)) ? -1 : n;
}

// bit count
template<class T> K_INLINE int kBCount(T x) 
{
	int n = 0; 
	
	while(x) { n += int(x & 1); x = kShr(x); } 
	
	return n;
}

// swap bytes
template<class T> K_INLINE T kSwab(T x)
{
	switch(sizeof(x)) {
	case 2:
		{
			kUInt8 t[2];

			t[0] = ((kUInt8 *) &x)[1];
			t[1] = ((kUInt8 *) &x)[0];

			return *(T *) &t;
		}
	case 4:
		{
			kUInt8 t[4];

			t[0] = ((kUInt8 *) &x)[3];
			t[1] = ((kUInt8 *) &x)[2];
			t[2] = ((kUInt8 *) &x)[1];
			t[3] = ((kUInt8 *) &x)[0];

			return *(T *) &t;
		}
	case 8:
		{
			kUInt8 t[8];

			t[0] = ((kUInt8 *) &x)[7];
			t[1] = ((kUInt8 *) &x)[6];
			t[2] = ((kUInt8 *) &x)[5];
			t[3] = ((kUInt8 *) &x)[4];
			t[4] = ((kUInt8 *) &x)[3];
			t[5] = ((kUInt8 *) &x)[2];
			t[6] = ((kUInt8 *) &x)[1];
			t[7] = ((kUInt8 *) &x)[0];

			return *(T *) &t;
		}
	default:
		return x;
	}
}
// common specializations
K_TEMPLATE_SPEC(kSChar) K_INLINE kSChar kSwab(kSChar x)
{
	return x;
}
K_TEMPLATE_SPEC(kUChar) K_INLINE kUChar kSwab(kUChar x)
{
	return x;
}

//******************************************************************************
//
//	basic routines
//
//******************************************************************************

// modulo
template<class T> K_INLINE T kMod(T x, T y)
{
	return x % y;
}

// two's complement negation
template<class T> K_INLINE T kNeg(T x)
{
	return -x;
}
// common specializations
K_TEMPLATE_SPEC(kUChar) K_INLINE kUChar kNeg(kUChar x)
{
	return kUChar(-kSChar(x));
}
K_TEMPLATE_SPEC(kUShort) K_INLINE kUShort kNeg(kUShort x)
{
	return kUShort(-kShort(x));
}
K_TEMPLATE_SPEC(kUInt) K_INLINE kUInt kNeg(kUInt x)
{
	return kUInt(-int(x));
}

K_TEMPLATE_SPEC(kULongLong) K_INLINE kULongLong kNeg(kULongLong x)
{
	return kULongLong(-kLongLong(x));
}

// absolute value
template<class T> K_INLINE T kAbs(T x)
{
	return (x >= 0) ? x : ((x > kMin(x)) ? -x : kMax(x));
}
// common specializations
K_TEMPLATE_SPEC(kUChar) K_INLINE kUChar kAbs(kUChar x)
{
	return x;
}
K_TEMPLATE_SPEC(kUShort) K_INLINE kUShort kAbs(kUShort x)
{
	return x;
}
K_TEMPLATE_SPEC(kUInt) K_INLINE kUInt kAbs(kUInt x)
{
	return x;
}

K_TEMPLATE_SPEC(kULongLong) K_INLINE kULongLong kAbs(kULongLong x)
{
	return x;
}

// minimum value
template<class T> K_INLINE T kMin(T x, T y)
{
	return (x < y) ? x : y;
}

// maximum value
template<class T> K_INLINE T kMax(T x,  T y)
{
	return (x > y) ? x : y;
}

// minimum and maximum value
template<class T> K_INLINE T kMinMax( T x,  T y, T *z)
{
	return (*z = kMax(x, y)), kMin(x, y);
}

// clamp value
template<class T> K_INLINE T kClamp( T value,  T min,  T max)
{
	return (value<min)?min:((value>max)?max:value);
}

// range check
template<class T> K_INLINE bool kIsIncl( T x,  T y,  T z)
{
	return (x >= y) && (x <= z);
}
template<class T> K_INLINE bool kIsExcl( T x,  T y,  T z)
{
	return (x < y) || (x > z);
}

//******************************************************************************
//
//	alignment
//
//******************************************************************************

// arbitrary floor
template<class T> K_INLINE T kFloor(T x)
{
	return x;
}
template<class T> K_INLINE T kFloor(T x, T y)
{
	return y * (kFloor(x / y));
}

// power of two floor
template<class T> K_INLINE T kFloor2(T x)
{
	return 1 << kMSB(x);
}
template<class T> K_INLINE T kFloor2(T x, T y)
{
	return x & ~(y - 1);
}

// arbitrary ceil
template<class T> K_INLINE T kCeil(T x)
{
	return x;
}
template<class T> K_INLINE T kCeil(T x, T y)
{
	return kFloor(T(x + y - 1), y);
}

// power of two ceil
template<class T> K_INLINE T kCeil2(T x)
{
	return kFloor2(2 * x - 1);
}
template<class T> K_INLINE T kCeil2(T x, T y)
{
	return x + (kNeg(x) & (y - 1));
}

// upward alignment
template<class T> K_INLINE T kAlign(T x, T y)
{
	return kCeil(x, y);
}

// power of two alignment
template<class T> K_INLINE T kAlign2(T x)
{
	return kCeil2(x);
}
template<class T> K_INLINE T kAlign2(T x, T y)
{
	return kCeil2(x, y);
}

//******************************************************************************
//
//	bit set type
//
//******************************************************************************

typedef kEnum	kBSet;

// bit set union
K_INLINE kBSet kBSetUnion(kBSet x, kBSet y)
{
	return x | y;
}

// bit set intersection
K_INLINE kBSet kBSetInter(kBSet x, kBSet y)
{
	return x & y;
}

// bit set difference
K_INLINE kBSet kBSetDiff(kBSet x, kBSet y)
{
	return x & ~y;
}

// bit set subset
K_INLINE bool kBSetIsSub(kBSet x, kBSet y)
{
	return (x & y) == x;
}

// conditional bit set
K_INLINE kBSet kBSetCond(kBSet x, int c)
{
	return kBSet(!c - 1) & x;
}

// bit subset intersection
K_INLINE kBSet kBSetAnd(kBSet x, kBSet y, kBSet z)
{
	return (x & ~z) | (x & y & z);
}

// bit subset union
K_INLINE kBSet kBSetOr(kBSet x, kBSet y, kBSet z)
{
	return x | ((x | y) & z);
}

// bit set last
K_INLINE int kBSetLast(kBSet x)
{
	return kMSB(x);
}

// bit set first
K_INLINE int kBSetFirst(kBSet x)
{
	return kLSB(x);
}

// bit set population
K_INLINE int kBSetSize(kBSet x) 
{
	return kBCount(x);
}

#if defined(KARCH_ARCH_IA32)
#include <klib/itype-ia32.h>
#elif defined(KARCH_ARCH_IA64) || defined(KARCH_ARCH_X64)
#include <klib/itype-generic.h>
#elif defined(KARCH_ARCH_AMD64)
#include <klib/itype-generic.h>
#elif defined(KARCH_ARCH_MIPS)
#include <klib/itype-mips.h>
#elif defined(KARCH_ARCH_EE)
#include <klib/itype-generic.h>
#elif defined(KARCH_ARCH_POWERPC)
#include <klib/itype-generic.h>
#else
#error architecture not supported
#endif

// BigEndian to LittleEndian conversions
#ifdef KARCH_LITTLE_ENDIAN
	#define kBigEndianToNative( x ) x = kSwab( x )
	#define kLittleEndianToNative( x )
#else
	#define kBigEndianToNative( x ) 
	#define kLittleEndianToNative( x ) x = kSwab( x )
#endif

#endif /* _FBXSDK_KLIB_ITYPE_H_ */
