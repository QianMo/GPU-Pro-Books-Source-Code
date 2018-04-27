#ifndef MATH_FLOAT16_H_INCLUDED
#define MATH_FLOAT16_H_INCLUDED

namespace Mod
{

	namespace Math
	{

		/*
			IEEE FLOAT16:

			M: 10-bit mantissa 
			E: 5-bit exponent 
			S: 1-bit sign 

			special cases:

			E=0, M=0			== 0.0
			E=0, M!=0			== Denormalized value (M / 2^10) * 2^-14
			0<E<31, M=any		== (1 + M / 2^10) * 2^(E-15)
			E=31, M=0			== Infinity
			E=31, M!=0			== NAN
		*/

		struct Float16
		{
			union
			{
				struct
				{
					UINT16	mantissa	: 10;
					UINT16	exponent	: 5;
					UINT16	sign		: 1;
				};

				UINT16	encoded;
			};
	
			// manipulation / access
		public:
			void  FromFloat( float fp32val );
			float AsFloat() const;
		};

	}
}

#endif