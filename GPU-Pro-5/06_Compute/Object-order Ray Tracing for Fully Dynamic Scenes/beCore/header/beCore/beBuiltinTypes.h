/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_BUILTIN_TYPES
#define BE_CORE_BUILTIN_TYPES

#include "beCore.h"
#include "beValueType.h"

namespace beCore
{

/// Gets the value type description for the given built-in type. Linker error, if unavailable.
template <class Type>
BE_CORE_API const ValueTypeDesc& GetBuiltinType(ValueTypes &valueTypes = GetValueTypes());

} // namespace

#endif