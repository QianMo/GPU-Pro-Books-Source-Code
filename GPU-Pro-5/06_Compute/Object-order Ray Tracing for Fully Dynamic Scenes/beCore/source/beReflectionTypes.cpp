/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beReflectionTypes.h"

namespace beCore
{

const ComponentType ReflectionTypes[ReflectionType::Count] = {
		{ "Bool" },
		{ "Integer" },
		{ "Float" },
		{ "String" },
		{ "File" }
	};

// Gets an address to the reflection type.
const ComponentType* GetReflectionType(ReflectionType::T type)
{
	LEAN_ASSERT(type < ReflectionType::Count);
	return &ReflectionTypes[type];
}

// Gets the reflection type.
ReflectionType::T GetReflectionType(const ComponentType *type)
{
	size_t typeIdx = type - ReflectionTypes;
	return (typeIdx < ReflectionType::Count)
		? ReflectionType::T(typeIdx)
		: ReflectionType::None;
}

} // namespace
