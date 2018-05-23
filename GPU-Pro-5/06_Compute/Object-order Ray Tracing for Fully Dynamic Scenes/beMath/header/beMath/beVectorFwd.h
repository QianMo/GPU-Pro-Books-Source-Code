/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_VECTOR_FWD
#define BE_MATH_VECTOR_FWD

#include "beMath.h"

namespace beMath
{

/// Vector class.
template <class Component, size_t Dimension>
class vector;

namespace Types
{
	using beMath::vector;

	/// 2-dimensional float vector.
	typedef vector<lean::float4, 2> fvec2;
	/// 3-dimensional float vector.
	typedef vector<lean::float4, 3> fvec3;
	/// 4-dimensional float vector.
	typedef vector<lean::float4, 4> fvec4;

	/// 2-dimensional double vector.
	typedef vector<lean::float8, 2> dvec2;
	/// 3-dimensional double vector.
	typedef vector<lean::float8, 3> dvec3;
	/// 4-dimensional double vector.
	typedef vector<lean::float8, 4> dvec4;

	/// 2-dimensional int4 vector.
	typedef vector<lean::int4, 2> ivec2;
	/// 3-dimensional int4 vector.
	typedef vector<lean::int4, 3> ivec3;
	/// 4-dimensional int4 vector.
	typedef vector<lean::int4, 4> ivec4;

	/// 2-dimensional int8 vector.
	typedef vector<lean::int8, 2> lvec2;
	/// 3-dimensional int8 vector.
	typedef vector<lean::int8, 3> lvec3;
	/// 4-dimensional int8 vector.
	typedef vector<lean::int8, 4> lvec4;

} // namespace

using namespace Types;

} // namespace

#endif