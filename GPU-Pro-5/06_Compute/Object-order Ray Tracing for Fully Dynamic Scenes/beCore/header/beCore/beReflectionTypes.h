/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_REFLECTION_TYPES
#define BE_CORE_REFLECTION_TYPES

#include "beCore.h"
#include "beComponent.h"
#include "beExchangeContainers.h"

namespace beCore
{

/// Common reflection types.
struct ReflectionType
{
	/// Enumeration.
	enum T
	{
		Boolean,	///< Boolean (bool)
		Integer,	///< Number (int)
		Float,		///< Number (float)
		String,		///< String (Exchange::utf8_string)
		File,		///< Path string (Exchange::utf8_string)

		Count,

		None = 0x7fffffff	///< No reflection type.
	};

	LEAN_MAKE_ENUM_STRUCT(ReflectionType)
};

/// Defines the type corresponding to the given reflection type.
template <ReflectionType::T Type>
struct reflection_type;
template <> struct reflection_type<ReflectionType::Boolean> { typedef bool type; };
template <> struct reflection_type<ReflectionType::Integer> { typedef int type; };
template <> struct reflection_type<ReflectionType::Float> { typedef float type; };
template <> struct reflection_type<ReflectionType::String> { typedef Exchange::utf8_string type; };
template <> struct reflection_type<ReflectionType::File> { typedef Exchange::utf8_string type; };

/// Gets an address to the reflection type.
BE_CORE_API const ComponentType* GetReflectionType(ReflectionType::T type);
/// Gets the reflection type.
BE_CORE_API ReflectionType::T GetReflectionType(const ComponentType *type);

} // namespace

#endif