#ifndef MATH_BBOX_H_INCLUDED
#define MATH_BBOX_H_INCLUDED

#include "Forw.h"
#include "Types.h"

namespace Mod
{
	namespace Math
	{
		class BBox
		{
			// construction/ destruction
		public:
			BBox(const float3& min, const float3& max);
			~BBox();

			// manipulation/ access
		public:	
			void			Grow(const float3& vec );
			void			Grow(const BBox& box);
			void			Transform(const float3x4& matrix);
			bool			Intersects(const Ray& ray) const;

			// returns negative value if no intersection
			float			GetIntersectionDistance(const Ray& ray) const;

			const float3& GetMin() const;
			const float3& GetMax() const;
			// data
		private:
			float3 mMin;
			float3 mMax;
		};

		float3	get_center( const BBox& box );
		float3	get_extents( const BBox& box );
		float	get_radius( const BBox& box );

		// transform should bring bbox to view space
		bool	intersects( const BBox& bbox, const float3x4& transform, const Frustum& frustum );
	}
}

#endif