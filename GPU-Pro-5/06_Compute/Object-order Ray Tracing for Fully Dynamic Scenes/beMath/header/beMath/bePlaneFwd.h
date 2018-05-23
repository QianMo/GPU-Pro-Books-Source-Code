/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_PLANE_FWD
#define BE_MATH_PLANE_FWD

#include "beMath.h"

namespace beMath
{

/// Plane class.
template <class Component, size_t Dimension>
class plane;

namespace Types
{
	using beMath::plane;

	/// 2-dimensional float plane.
	typedef plane<lean::float4, 2> fplane2;
	/// 3-dimensional float matplaneix.
	typedef plane<lean::float4, 3> fplane3;
	/// 4-dimensional float plane.
	typedef plane<lean::float4, 4> fplane4;

	/// 2-dimensional double plane.
	typedef plane<lean::float8, 2> dplane2;
	/// 3-dimensional double plane.
	typedef plane<lean::float8, 3> dplane3;
	/// 4-dimensional double plane.
	typedef plane<lean::float8, 4> dplane4;

	/// 2-dimensional int4 plane.
	typedef plane<lean::int4, 2> iplane2;
	/// 3-dimensional int4 plane.
	typedef plane<lean::int4, 3> iplane3;
	/// 4-dimensional int4 plane.
	typedef plane<lean::int4, 4> iplane4;

	/// 2-dimensional int8 plane.
	typedef plane<lean::int8, 2> lplane2;
	/// 3-dimensional int8 plane.
	typedef plane<lean::int8, 3> lplane3;
	/// 4-dimensional int8 plane.
	typedef plane<lean::int8, 4> lplane4;

} // namespace

using namespace Types;

} // namespace

#endif