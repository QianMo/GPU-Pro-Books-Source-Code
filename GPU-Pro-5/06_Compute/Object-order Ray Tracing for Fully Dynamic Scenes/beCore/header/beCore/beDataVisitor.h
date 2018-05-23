/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_DATA_VISITOR
#define BE_CORE_DATA_VISITOR

#include "beCore.h"
#include <lean/type_info.h>

namespace beCore
{

struct ValueTypeDesc;

/// Data visitor.
class LEAN_INTERFACE DataVisitor
{
	LEAN_INTERFACE_BEHAVIOR(DataVisitor)

public:
	/// Visits the given values.
	BE_CORE_API virtual bool Visit(const ValueTypeDesc &typeDesc, void *values, size_t count)
	{
		Visit(typeDesc, const_cast<const void*>(values), count);
		return false;
	}
	/// Visits the given values.
	BE_CORE_API virtual void Visit(const ValueTypeDesc &typeDesc, const void *values, size_t count) { }
};

} // namespace

#endif