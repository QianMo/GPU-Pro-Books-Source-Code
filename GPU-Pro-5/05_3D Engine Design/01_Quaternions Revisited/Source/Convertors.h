/****************************************************************************

  GPU Pro 5 : Quaternions revisited - sample code
  All sample code written from scratch by Sergey Makeev specially for article.

  Copyright (c) 2013, Sergey Makeev

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software.

  2. If you use this software in a non-commercial product, an acknowledgment
     in the product documentation would be appreciated but is not required.

  3. If you use this software in a commercial product you requried to credit
     the author.

  4. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.

  5. This notice may not be removed or altered from any source distribution.
    

   Please let me know if you use the code in your products or have any questions or suggestions.

   e-mail: sergeymakeev@inbox.ru
   http://linkedin.com/in/sergeymakeev/

****************************************************************************/
#pragma once


#include "Math.h"


// Conversion routines from FBX coordinate system to D3D coordinate system
namespace Convert
{

// Convert a right-handed Y-up matrix to a left-handed Y-up matrix.
inline Matrix4x3 RightHanded_To_LeftHanded(const Matrix4x3 & rightHandedMatrix)
{
	Matrix4x3 leftHanded = rightHandedMatrix;

	leftHanded.m[0][2] = -leftHanded.m[0][2];
	leftHanded.m[1][2] = -leftHanded.m[1][2];
	leftHanded.m[2][2] = -leftHanded.m[2][2];
	leftHanded.m[3][2] = -leftHanded.m[3][2];

	return leftHanded;
}

//Convert unit (0..1 range) single precision float to byte
inline unsigned char UnitFp32ToUInt8( float value )
{
	if (value < 0.0f)
		value = 0.0f;

	if (value > 1.0f)
		value = 1.0f;

	int result = (int)floorf( value * 255.0f + 0.5f );

	if ( result > 255 )
		result = 255;

	return ( unsigned char )( result );
}

//Convert unit (0..1 range) single precision float to ushort
inline unsigned short UnitFp32ToUInt16(float value)
{
	if (value < 0.0f)
		value = 0.0f;

	if (value > 1.0f)
		value = 1.0f;

	int result = (int)floorf( value * 65535.0f + 0.5f );

	if ( result > 65535 )
		result = 65535;

	return ( unsigned short )( result );
}


//Convert centered unit (-1..1 range) single precision float to short
inline short CenteredUnitFp32ToInt16(float value)
{
	if (value < -1.0f)
		value = -1.0f;

	if (value > 1.0f)
		value = 1.0f;

	int result = (int)floorf( (value * 32767.0f) + (value < 0.0f ? -0.5f : 0.5f) );

	if ( result < -32767 )
		result = -32767;

	if ( result > 32767 )
		result = 32767;

	return ( short )( result );
}


//Quantize uv -8..8 to short
inline short QuantizeTexCoord(float texCoord)
{
	return CenteredUnitFp32ToInt16(texCoord / 8.0f);
}

//Quantize skin weight 0..1 to short
inline short QuantizeSkinWeight(float weight)
{
	return CenteredUnitFp32ToInt16(weight * 2.0f - 1.0f);
}



//Quantize normalized vector to dword
inline unsigned long QuantizeNormalizedVector(const Vector3 & vec, unsigned char alpha = 0)
{
	unsigned long r = alpha << 24 | UnitFp32ToUInt8(vec.x * 0.5f + 0.5f) << 16 | UnitFp32ToUInt8(vec.y * 0.5f + 0.5f) << 8 | UnitFp32ToUInt8(vec.z * 0.5f + 0.5f);
	return r;
}

//Quantize and pack normalized quaternion and handedness bit to dword
inline unsigned long QuantizeNormalizedQuaternionWithHandedness(const Quaternion & q, bool invertedHandedness)
{
	unsigned char handednessMask = invertedHandedness ? 0x0 : 0x80;
	unsigned char packedHandednessQuatW = handednessMask | (UnitFp32ToUInt8(q.w * 0.5f + 0.5f) >> 1);
	unsigned long r = packedHandednessQuatW << 24 | UnitFp32ToUInt8(q.x * 0.5f + 0.5f) << 16 | UnitFp32ToUInt8(q.y * 0.5f + 0.5f) << 8 | UnitFp32ToUInt8(q.z * 0.5f + 0.5f);
	return r;
}


//Quantize normalized quaternion to dword
inline unsigned long QuantizeNormalizedQuaternion(const Quaternion & q)
{
	unsigned long r = UnitFp32ToUInt8(q.w * 0.5f + 0.5f) << 24 | UnitFp32ToUInt8(q.x * 0.5f + 0.5f) << 16 | UnitFp32ToUInt8(q.y * 0.5f + 0.5f) << 8 | UnitFp32ToUInt8(q.z * 0.5f + 0.5f);
	return r;
}

//Convert single precision float to half precision float
inline unsigned short Fp32ToFp16( float value )
{
	unsigned long dwFloat = *((unsigned long *)&value);
	unsigned long dwMantissa = dwFloat & 0x7fffff;
	int iExp = (int)((dwFloat >> 23) & 0xff) - (int)0x70;
	unsigned long dwSign = dwFloat >> 31;

	int result = ( (dwSign << 15) | (((unsigned long)(Utils::Max(iExp, 0))) << 10) | (dwMantissa >> 13) );
	result = result & 0xFFFF;
	return (unsigned short)result;
}

//Convert half precision float to single precision float
inline float Fp16ToFp32( unsigned short value )
{
	unsigned long mantissa;
	unsigned long exponent;
	unsigned long result;

	mantissa = (unsigned long)(value & 0x03FF);

	if ((value & 0x7C00) != 0)
	{
		exponent = (unsigned long)((value >> 10) & 0x1F);
	}
	else if (mantissa != 0)
	{
		exponent = 1;

		do
		{
			exponent--;
			mantissa <<= 1;
		} while ((mantissa & 0x0400) == 0);

		mantissa &= 0x03FF;
	}
	else
	{
		exponent = (unsigned long) - 112;
	}

	result = ((value & 0x8000) << 16) | ((exponent + 112) << 23) | (mantissa << 13);
	return *(float*)&result;
}

} // end of namespace

