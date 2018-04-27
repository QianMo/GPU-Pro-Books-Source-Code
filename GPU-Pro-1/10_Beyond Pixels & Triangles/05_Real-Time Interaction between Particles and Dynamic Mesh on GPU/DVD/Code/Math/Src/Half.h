#ifndef MATH_HALF_H_INCLUDED
#define MATH_HALF_H_INCLUDED

#include "Float16.h"

namespace Mod
{
	namespace Math
	{

		class half
		{

			// manipulation/ access
		public:
			void FromFloat( float fp32val );
			float AsFloat() const;

			bool operator == ( half rhs ) const;
			half& operator += ( half rhs );
			half& operator -= ( half rhs );
			half& operator *= ( half rhs );
			half& operator /= ( half rhs );

			half operator -() const;

			// data
		private:
			Float16 mVal;
		};

		//------------------------------------------------------------------------

		half operator +( half lhs, half rhs );
		half operator -( half lhs, half rhs );
		half operator *( half lhs, half rhs );
		half operator /( half lhs, half rhs );

		//------------------------------------------------------------------------

	}
}

#endif