#include "Precompiled.h"

#include "Ray.h"
#include "Frustum.h"

#include "Operations.h"

#include "BBox.h"

namespace Mod
{

	namespace Math
	{

		BBox::BBox(const float3& min, const float3& max):
		mMin(min),
		mMax(max)
		{

		}

		//------------------------------------------------------------------------

		BBox::~BBox()
		{

		}

		//------------------------------------------------------------------------

		void
		BBox::Grow(const float3& vec )
		{
			// NOTE: these conditions may not exclude each other
			// because bbox that is initialized for growing from 0 has
			// max < min

			if(vec.x < mMin.x)	mMin.x = vec.x;

			if(vec.x > mMax.x)	mMax.x = vec.x;

			if(vec.y < mMin.y)	mMin.y = vec.y;

			if(vec.y > mMax.y)	mMax.y = vec.y;

			if(vec.z < mMin.z)	mMin.z = vec.z;

			if(vec.z > mMax.z)	mMax.z = vec.z;
		}

		//------------------------------------------------------------------------

		void
		BBox::Grow(const BBox& box)
		{
			Grow(box.GetMin());
			Grow(box.GetMax());
		}

		//------------------------------------------------------------------------

		bool
		BBox::Intersects(const Ray& ray) const
		{
			const float3& start		= ray.GetStart();
			const float3& dir		= ray.GetDir();

		#define TEST_PLANE(VAL,C1,C2,C3,getC2,getC3)														\
			{																								\
				float t##C1	= VAL;																			\
				float t		= (t##C1-start.C1)/dir.C1;														\
				if(t < 0)																					\
					return false;																			\
																											\
				float t##C2	= ray.getC2( t );																\
																											\
				if( t##C2 >= mMin.C2 && t##C2 <= mMax.C2 )													\
				{																							\
					float t##C3 = ray.getC3( t );															\
					if( t##C3 >= mMin.C3 && t##C3 <= mMax.C3 )												\
						return true;																		\
				}																							\
			}


			if(dir.z > 0)
				TEST_PLANE(mMin.z,z,x,y,GetXAt,GetYAt)
			else
			if(dir.z < 0)
				TEST_PLANE(mMax.z,z,x,y,GetXAt,GetYAt)

			if(dir.x > 0)
				TEST_PLANE(mMin.x,x,y,z,GetYAt,GetZAt)
			else
			if(dir.x < 0)
				TEST_PLANE(mMax.x,x,y,z,GetYAt,GetZAt)

			if(dir.y > 0)
				TEST_PLANE(mMin.y,y,x,z,GetXAt,GetZAt)
			else
			if(dir.y < 0)
				TEST_PLANE(mMax.y,y,x,z,GetXAt,GetZAt)
		#undef TEST_PLANE
			return false;	
		}

		//------------------------------------------------------------------------

		void
		BBox::Transform(const float3x4& matrix)
		{
			mMin = mul( float4(mMin,1), matrix );
			mMax = mul( float4(mMax,1), matrix );
		}

		//------------------------------------------------------------------------

		float
		BBox::GetIntersectionDistance(const Ray& ray) const
		{
			const float3& start		= ray.GetStart();
			const float3& dir		= ray.GetDir();

		#define TEST_PLANE(VAL,C1,C2,C3,getC2,getC3)														\
			{																								\
				float t##C1	= VAL;																			\
				float t		= (t##C1-start.C1)/dir.C1;														\
				if(t >= 0)																					\
				{																							\
					float t##C2	= ray.getC2( t );															\
																											\
					if( t##C2 >= mMin.C2 && t##C2 <= mMax.C2 )												\
					{																						\
						float t##C3 = ray.getC3( t );														\
						if( t##C3 >= mMin.C3 && t##C3 <= mMax.C3 )											\
							return t;																		\
					}																						\
				}																							\
			}


			if(dir.z > 0)
				TEST_PLANE(mMin.z,z,x,y,GetXAt,GetYAt)
			else
			if(dir.z < 0)
				TEST_PLANE(mMax.z,z,x,y,GetXAt,GetYAt)

			if(dir.x > 0)
				TEST_PLANE(mMin.x,x,y,z,GetYAt,GetZAt)
			else
			if(dir.x < 0)
				TEST_PLANE(mMax.x,x,y,z,GetYAt,GetZAt)

			if(dir.y > 0)
				TEST_PLANE(mMin.y,y,x,z,GetXAt,GetZAt)
			else
			if(dir.y < 0)
				TEST_PLANE(mMax.y,y,x,z,GetXAt,GetZAt)

			// negative signals of no intersection
			return -1.0f;

		#undef TEST_PLANE

		}

		//------------------------------------------------------------------------

		const float3&
		BBox::GetMin() const
		{
			return mMin;
		}

		//------------------------------------------------------------------------

		const float3&
		BBox::GetMax() const
		{
			return mMax;
		}

		//------------------------------------------------------------------------

		float3 get_center( const BBox& box )
		{
			return 0.5f * ( box.GetMax() + box.GetMin() );
		}

		//------------------------------------------------------------------------

		float3 get_extents( const BBox& box )
		{
			return 0.5f * ( box.GetMax() - box.GetMin() );
		}

		//------------------------------------------------------------------------

		float get_radius( const BBox& box )
		{
			return length( get_extents( box ) );
		}

		//------------------------------------------------------------------------

		bool intersects( const BBox& bbox, const float3x4& transform, const Frustum& frustum )
		{
			// to bring frustum into box space we must do 
			// 1. T  = inverse( transform ) ( to box space )
			// 2. T' = transpose( inverse( T ) ) ( plane tranform specifics )
			// So T = transpose( inverse( inverse( T ) ) ) = transpose( T )
			float4x4 ftrans = transpose( to4x4(transform) );

			Frustum tfrust;

			for( UINT32 i = 0; i < Frustum::NUM_PLANES; i ++ )
			{
				tfrust.planes[ i ] = mul( frustum.planes[ i ], ftrans );
			}

			const float3& mi = bbox.GetMin();
			const float3& ma = bbox.GetMax();

			const float4& centre	= float4( 0.5f * ( mi + ma ),							1 );
			const float4& ext		= float4( ma - float3( centre.x, centre.y, centre.z ),	1 );

			for( UINT32 i = 0; i < Frustum::NUM_PLANES; i ++ )
			{	
				const float4& p = tfrust.planes[i];

				float Q = dot( centre, p );
				float R = fabs( p.x * ext.x ) + fabs( p.y * ext.y ) + fabs( p.z * ext.z );

				if( Q < -R )
					return false;
			}

			return true;
		}

		//------------------------------------------------------------------------
	}
}