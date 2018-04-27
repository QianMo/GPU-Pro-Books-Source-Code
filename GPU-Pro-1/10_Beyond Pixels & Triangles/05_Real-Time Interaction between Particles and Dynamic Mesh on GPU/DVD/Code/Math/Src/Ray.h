#ifndef MATH_RAY_H_INCLUDED
#define MATH_RAY_H_INCLUDED

#include "Types.h"

namespace Mod
{
	namespace Math
	{
		class Ray
		{
			// construction/ destruction
		public:
			Ray(const float3& start, const float3& dir);
			~Ray();

			// manipulation/ access
		public:
			const float3& GetStart() const;
			const float3& GetDir() const;

			float3		GetAt(float t) const;
			float		GetXAt (float t) const;
			float		GetYAt(float t) const;
			float		GetZAt(float t) const;

			void		Transform(const float3x4& matrix);
			void		Normalize();

			// data
		private:
			float3 mStart;
			float3 mDir;

		};
	}
}

#endif