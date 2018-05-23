/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_AAB_FWD
#define BE_MATH_AAB_FWD

#include "beMath.h"

namespace beMath
{

/// Sphere class.
template <class Component, size_t Dimension>
class aab;

namespace Types
{
	using beMath::aab;

	/// 2-dimensional float box.
	typedef aab<lean::float4, 2> faab2;
	/// 3-dimensional float box.
	typedef aab<lean::float4, 3> faab3;
	/// 4-dimensional float box.
	typedef aab<lean::float4, 4> faab4;

	/// 2-dimensional double box.
	typedef aab<lean::float8, 2> daab2;
	/// 3-dimensional double box.
	typedef aab<lean::float8, 3> daab3;
	/// 4-dimensional double box.
	typedef aab<lean::float8, 4> daab4;

	/// 2-dimensional int4 box.
	typedef aab<lean::int4, 2> iaab2;
	/// 3-dimensional int4 box.
	typedef aab<lean::int4, 3> iaab3;
	/// 4-dimensional int4 box.
	typedef aab<lean::int4, 4> iaab4;

	/// 2-dimensional int8 box.
	typedef aab<lean::int8, 2> laab2;
	/// 3-dimensional int8 box.
	typedef aab<lean::int8, 3> laab3;
	/// 4-dimensional int8 box.
	typedef aab<lean::int8, 4> laab4;

} // namespace

using namespace Types;

} // namespace

#endif