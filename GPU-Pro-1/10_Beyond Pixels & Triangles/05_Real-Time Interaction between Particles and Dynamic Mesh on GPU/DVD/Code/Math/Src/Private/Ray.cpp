#include "Precompiled.h"
#include "Ray.h"

#include "Operations.h"

namespace Mod
{
	namespace Math
	{
		//------------------------------------------------------------------------

		Ray::Ray(const float3& start, const float3& dir):
		mStart(start),
		mDir( normalize( dir ) )
		{
		}

		//------------------------------------------------------------------------

		Ray::~Ray()
		{

		}

		//------------------------------------------------------------------------

		const float3&
		Ray::GetStart() const
		{
			return mStart;
		}

		//------------------------------------------------------------------------

		const float3&
		Ray::GetDir() const
		{
			return mDir;
		}

		//------------------------------------------------------------------------

		float3
		Ray::GetAt(float t) const
		{
			return mStart + mDir * t;
		}

		//------------------------------------------------------------------------

		float
		Ray::GetXAt (float t) const
		{
			return mStart.x + mDir.x * t;
		}

		//------------------------------------------------------------------------

		float
		Ray::GetYAt(float t) const
		{
			return mStart.y + mDir.y * t;
		}

		//------------------------------------------------------------------------

		float
		Ray::GetZAt(float t) const
		{
			return mStart.z + mDir.z * t;
		}

		//------------------------------------------------------------------------

		void
		Ray::Transform(const float3x4& matrix)
		{
			mStart	= mul( float4( mStart, 1 ), matrix );
			mDir	= mul( mDir, matrix );
		}

		//------------------------------------------------------------------------

		void
		Ray::Normalize()
		{
			mDir = normalize( mDir );
		}
	}
}
