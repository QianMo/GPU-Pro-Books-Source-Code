#ifndef MATH_FLOAT32_H_INCLUDED
#define MATH_FLOAT32_H_INCLUDED

namespace Mod
{

	namespace Math
	{
		union Float32
		{
			struct
			{
				DWORD	mantissa : 23;
				DWORD	exponent : 8;
				DWORD	sign : 1;			
			};
			float	value;
		};
	}
}

#endif