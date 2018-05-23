/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_PROJECTION
#define BE_MATH_PROJECTION

#include "beMath.h"
#include "beMatrix.h"
#include "bePlane.h"
#include "beUtility.h"

namespace beMath
{

/// Extracts the left frustum plane from the given view-projection matrix.
template <class Component>
LEAN_INLINE plane<Component, 3> frustum_left(const matrix<Component, 4, 4> &viewProj)
{
	return plane<Component, 3>(
			vec(-(viewProj[0][3] + viewProj[0][0]),
				-(viewProj[1][3] + viewProj[1][0]),
				-(viewProj[2][3] + viewProj[2][0]) ),
			viewProj[3][3] + viewProj[3][0]
		);
}

/// Extracts the right frustum plane from the given view-projection matrix.
template <class Component>
LEAN_INLINE plane<Component, 3> frustum_right(const matrix<Component, 4, 4> &viewProj)
{
	return plane<Component, 3>(
			vec(-(viewProj[0][3] - viewProj[0][0]),
				-(viewProj[1][3] - viewProj[1][0]),
				-(viewProj[2][3] - viewProj[2][0]) ),
			viewProj[3][3] - viewProj[3][0]
		);
}

/// Extracts the top frustum plane from the given view-projection matrix.
template <class Component>
LEAN_INLINE plane<Component, 3> frustum_top(const matrix<Component, 4, 4> &viewProj)
{
	return plane<Component, 3>(
			vec(-(viewProj[0][3] - viewProj[0][1]),
				-(viewProj[1][3] - viewProj[1][1]),
				-(viewProj[2][3] - viewProj[2][1]) ),
			viewProj[3][3] - viewProj[3][1]
		);
}

/// Extracts the bottom frustum plane from the given view-projection matrix.
template <class Component>
LEAN_INLINE plane<Component, 3> frustum_bottom(const matrix<Component, 4, 4> &viewProj)
{
	return plane<Component, 3>(
			vec(-(viewProj[0][3] + viewProj[0][1]),
				-(viewProj[1][3] + viewProj[1][1]),
				-(viewProj[2][3] + viewProj[2][1]) ),
			viewProj[3][3] + viewProj[3][1]
		);
}

/// Extracts the near frustum plane from the given view-projection matrix.
template <class Component>
LEAN_INLINE plane<Component, 3> frustum_near(const matrix<Component, 4, 4> &viewProj)
{
	return plane<Component, 3>(
			vec(-viewProj[0][2],
				-viewProj[1][2],
				-viewProj[2][2] ),
			viewProj[3][2]
		);
}

/// Extracts the far frustum plane from the given view-projection matrix.
template <class Component>
LEAN_INLINE plane<Component, 3> frustum_far(const matrix<Component, 4, 4> &viewProj)
{
	return plane<Component, 3>(
			vec(-(viewProj[0][3] - viewProj[0][2]),
				-(viewProj[1][3] - viewProj[1][2]),
				-(viewProj[2][3] - viewProj[2][2]) ),
			viewProj[3][3] - viewProj[3][2]
		);
}

/// Extracts the six view frustum planes from the given view-projection matrix.
template <class Component>
void extract_frustum(const matrix<Component, 4, 4> &viewProj, plane<Component, 3> *planes)
{
	planes[0] = normalize( frustum_near(viewProj) );
	planes[1] = normalize( frustum_left(viewProj) );
	planes[2] = normalize( frustum_right(viewProj) );
	planes[3] = normalize( frustum_far(viewProj) );
	planes[4] = normalize( frustum_bottom(viewProj) );
	planes[5] = normalize( frustum_top(viewProj) );
}

/// Replaces the near plane by the given plane.
template <class Component>
matrix<Component, 4, 4> replace_near_plane(const matrix<Component, 4, 4> &proj, const plane<Component, 3> &clipPlane)
{
	matrix<Component, 4, 4> result(proj);

	if (clipPlane.d() < 0.0f)
	{
		vector<Component, 4> farCorner = vec(
				(sign0(clipPlane[0]) + proj[2][0]) / proj[0][0],
				(sign0(clipPlane[1]) + proj[2][1]) / proj[1][1],
				-1.0f,
				(1.0f + proj[2][2]) / proj[3][2]
			);

		vector<Component, 4> viewPlane = vec(clipPlane[0], clipPlane[1], clipPlane[2], -clipPlane[3]);
		plane<Component, 3> projPlane = viewPlane * (1.0f / dot(viewPlane, farCorner));

		result[0][2] = projPlane[0];
		result[1][2] = projPlane[1];
		result[2][2] = projPlane[2];
		result[3][2] = projPlane[3];
	}

	return result;
}

} // namespace

#endif