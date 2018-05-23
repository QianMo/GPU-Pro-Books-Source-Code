/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_SPHERE_FWD
#define BE_MATH_SPHERE_FWD

#include "beMath.h"

namespace beMath
{

/// Sphere class.
template <class Component, size_t Dimension>
class sphere;

namespace Types
{
	using beMath::sphere;

	/// 2-dimensional float sphere.
	typedef sphere<lean::float4, 2> fsphere2;
	/// 3-dimensional float matsphereix.
	typedef sphere<lean::float4, 3> fsphere3;
	/// 4-dimensional float sphere.
	typedef sphere<lean::float4, 4> fsphere4;

	/// 2-dimensional double sphere.
	typedef sphere<lean::float8, 2> dsphere2;
	/// 3-dimensional double sphere.
	typedef sphere<lean::float8, 3> dsphere3;
	/// 4-dimensional double sphere.
	typedef sphere<lean::float8, 4> dsphere4;

	/// 2-dimensional int4 sphere.
	typedef sphere<lean::int4, 2> isphere2;
	/// 3-dimensional int4 sphere.
	typedef sphere<lean::int4, 3> isphere3;
	/// 4-dimensional int4 sphere.
	typedef sphere<lean::int4, 4> isphere4;

	/// 2-dimensional int8 sphere.
	typedef sphere<lean::int8, 2> lsphere2;
	/// 3-dimensional int8 sphere.
	typedef sphere<lean::int8, 3> lsphere3;
	/// 4-dimensional int8 sphere.
	typedef sphere<lean::int8, 4> lsphere4;

} // namespace

using namespace Types;

} // namespace

#endif