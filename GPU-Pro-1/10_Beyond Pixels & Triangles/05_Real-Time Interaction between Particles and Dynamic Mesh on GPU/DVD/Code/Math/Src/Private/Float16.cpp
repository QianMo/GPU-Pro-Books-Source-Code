#include "Precompiled.h"
#include "Float32.h"
#include "Float16.h"

namespace Mod
{
	namespace Math
	{
		void
		Float16::FromFloat( float fp32val )
		{
			Float32 FP32;
			FP32.value = fp32val;

			// Copy sign-bit
			sign = FP32.sign;

			// Check for zero or denormal.
			if ( !FP32.exponent )			// Minimum exponent?
			{
				// Set to 0.
				exponent = 0;
				mantissa = 0;
			}
			// Check for INF or NaN.
			else
			if ( FP32.exponent == 255 )		// Maximum exponent? (2^8 - 1)
			{
				exponent = 31;
				mantissa = UINT16(FP32.mantissa >> 13);
			}
			// Handle normal number.
			else
			{				
				if( FP32.exponent > 127 + 15 )
				{
					// make INF in this case
					exponent = 31;
					mantissa = 0;
				}
				else
				if( FP32.exponent <= 127 - 15 )
				{
					// make zero in this case
					exponent = 0;
					mantissa = 0;
				}
				else
				{
					// we're in range
					exponent = INT32(FP32.exponent) - 127 + 15;
					mantissa = UINT16(FP32.mantissa >> 13);
				}
			}
		}

		//------------------------------------------------------------------------

		float Float16::AsFloat() const
		{
			Float32	result;

			result.sign = sign;
			if ( !exponent )
			{
				// Zero or denormal. Just clamp to zero...
				result.exponent = 0;
				result.mantissa = 0;
			}
			else
			if (exponent == 31)		// 2^5 - 1
			{
				// Infinity or NaN.
				result.exponent = 255;
				result.mantissa = UINT32(mantissa) << 13;
			}
			else
			{
				// Normal number.
				result.exponent = INT32(exponent) - 15 + 127; // Stored exponents are biased by half their range.
				result.mantissa = UINT32(mantissa) << 13;
			}

			return result.value;
		}
	}
}