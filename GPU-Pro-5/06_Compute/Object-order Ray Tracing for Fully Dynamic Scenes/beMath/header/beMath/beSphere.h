/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_SPHERE
#define BE_MATH_SPHERE

#include "beMath.h"
#include "beSphereDef.h"
#include "beVector.h"
#include "beMatrix.h"

namespace beMath
{

/// Transforms the given sphere.
template <class Component, size_t Dimension>
LEAN_INLINE sphere<Component, Dimension> mulh(const sphere<Component, Dimension> &left, const matrix<Component, Dimension + 1, Dimension + 1> &right)
{
	float maxScaleSq = 0.0f;

	for (size_t i = 0; i < Dimension; ++i)
		maxScaleSq = max( lengthSq( vector<Component, Dimension>(right[i]) ), maxScaleSq );


	return sphere<Component, Dimension>( mulh(left.center, right), left.radius * sqrt(maxScaleSq) );
}

} // namespace

#endif