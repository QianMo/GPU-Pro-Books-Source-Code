#include "Precompiled.h"
#include "Half.h"

namespace Mod
{
	namespace Math
	{
		void
		half::FromFloat( float fp32val )
		{
			MD_STATIC_ASSERT( sizeof(half)==2);
			mVal.FromFloat( fp32val );
		}

		//------------------------------------------------------------------------
		
		float
		half::AsFloat() const
		{
			return mVal.AsFloat();
		}

		//------------------------------------------------------------------------

		bool
		half::operator == ( half rhs ) const
		{
			return mVal.encoded == rhs.mVal.encoded;
		}

		//------------------------------------------------------------------------

		half&
		half::operator += ( half rhs )
		{
			float v1 = rhs.mVal.AsFloat();
			float v2 = mVal.AsFloat();

			mVal.FromFloat( v1 + v2 );

			return *this;
		}

		//------------------------------------------------------------------------

		half&
		half::operator -= ( half rhs )
		{
			return *this += -rhs;
		}

		//------------------------------------------------------------------------

		half&
		half::operator *= ( half rhs )
		{
			float v1 = rhs.mVal.AsFloat();
			float v2 = mVal.AsFloat();

			mVal.FromFloat( v1 * v2 );

			return *this;
		}

		//------------------------------------------------------------------------

		half&
		half::operator /= ( half rhs )
		{
			float v1 = rhs.mVal.AsFloat();
			float v2 = mVal.AsFloat();

			mVal.FromFloat( v1 / v2 );

			return *this;
		}

		//------------------------------------------------------------------------

		half
		half::operator -() const
		{
			half tmp( *this );
			tmp.mVal.sign = ~mVal.sign;
			return tmp;
		}

		//------------------------------------------------------------------------

		half operator +( half lhs, half rhs )
		{
			return lhs += rhs;
		}

		//------------------------------------------------------------------------

		half operator -( half lhs, half rhs )
		{
			return lhs -= rhs;
		}

		//------------------------------------------------------------------------

		half operator *( half lhs, half rhs )
		{
			return lhs *= rhs;
		}

		//------------------------------------------------------------------------

		half operator /( half lhs, half rhs )
		{
			return lhs /= rhs;
		}

		//------------------------------------------------------------------------

	}
}