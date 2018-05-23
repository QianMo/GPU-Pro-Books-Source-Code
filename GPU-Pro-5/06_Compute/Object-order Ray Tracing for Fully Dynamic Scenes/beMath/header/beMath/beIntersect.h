/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_INTERSECT
#define BE_MATH_INTERSECT

#include "beMath.h"
#include "beVector.h"
#include "bePlane.h"
#include "beSphere.h"
#include "beAAB.h"
#include "beConstants.h"

namespace beMath
{

/// Computes the signed distance of the given box from the given plane.
template <class Component, size_t Dimension>
inline Component sdist(const plane<Component, Dimension> &p, const aab<Component, Dimension> &box)
{
	Component minDist = Component(0), maxDist = Component(0);

	for (size_t i = 0; i < Dimension; ++i)
		if(p[i] >= 0.0f)
		{
			minDist += p[i] * box.min[i];
			maxDist += p[i] * box.max[i];
		}
		else
		{
			minDist += p[i] * box.max[i];
			maxDist += p[i] * box.min[i];
		}

	if (minDist > p.d())
		return minDist - p.d();
	else if (maxDist < p.d())
		return maxDist - p.d();
	else
		return 0.0f;
}

/// Computes the distance of the given box from the given plane.
template <class Component, size_t Dimension>
LEAN_INLINE Component dist(const plane<Component, Dimension> &p, const aab<Component, Dimension> &box)
{
	return abs( sdist(p, box) );
}

/// Intersects the given ray with the given plane.
template <class Component, size_t Dimension>
LEAN_INLINE Component ray_intersect(const vector<Component, Dimension> &orig, const vector<Component, Dimension> &dir,
									const plane<Component, Dimension> &plane)
{
	return -sdist(plane, orig) / dot(plane.n(), dir);
}

/// Computes the point of intersection for the given normalized ray and sphere (intersection at o + d * t; t = return value, -1 on failure).
template <class Component, size_t Dimension>
inline Component ray_intersect(const vector<Component, Dimension> &o, const vector<Component, Dimension> &d, const sphere<Component, Dimension> &s)
{
	vector<Component, Dimension> oToC = s.center - o;
	Component distSqDelta = lengthSq(oToC) - s.radius * s.radius;

	// Inside
	if(distSqDelta <= Component(0))
		return Component(0);

	Component deltaC = dot(d, oToC);
	Component deltaTSq = deltaC * deltaC - distSqDelta;

	// Miss: sqrt yields NaN
//	if(deltaTSq < Component(0))
//		return ieee<Component>::NaN;

	return deltaC - sqrt(deltaTSq);
}

/// Computes the point of intersection for the given normalized ray and aab (intersection at o + d * t; t = return value, -1 on failure).
template <class Component, size_t Dimension>
inline Component ray_intersect(const vector<Component, Dimension> &o, const vector<Component, Dimension> &d, const aab<Component, Dimension> &box)
{
	const Component noIntersection = ieee<Component>::NaN;

	Component mainT = Component(-1);
	size_t mainAxis = 0;
	bool bInside = true;

	for (size_t i = 0; i < Dimension; ++i)
	{
		Component t = Component(-1);

		if (o[i] < box.min[i])
		{
			t = (box.min[i] - o[i]) / d[i];
			if (t < 0.0f)
				return noIntersection;
			bInside = false;
		}
		else if (box.max[i] < o[i])
		{
			t = (box.max[i] - o[i]) / d[i];
			if (t < 0.0f)
				return noIntersection;
			bInside = false;
		}
		
		if (t > mainT)
		{
			mainT = t;
			mainAxis = i;
		}
	}

	if (bInside)
		return Component(0);

	for (size_t i = 0; i < Dimension; ++i)
		if (i != mainAxis)
		{
			float x = o[i] + d[i] * mainT;
			if (x < box.min[i]  || box.max[i] < x)
				return noIntersection;
		}

	return mainT;
}

} // namespace

#endif