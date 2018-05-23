/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_TUPLE_FWD
#define BE_MATH_TUPLE_FWD

#include "beMath.h"

namespace beMath
{

/// Tuple class.
template <class Class, class Element, size_t Count>
class tuple;

namespace Types
{
	using beMath::tuple;

} // namespace

using namespace Types;

} // namespace

#endif