/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_VALUE_TYPE
#define BE_CORE_VALUE_TYPE

#include "beCore.h"
#include <lean/properties/property_type.h>

namespace beCore
{

class TextSerializer;
	
/// Value type descriptor.
struct ValueTypeDesc
{
	const lean::property_type_info Info;	///< Type info.
	const utf8_ntr Name;					///< Type name.
	const TextSerializer *Text;				///< Type reflector.

	/// Initializing constructor.
	explicit ValueTypeDesc(const lean::property_type_info &info,
			const TextSerializer *pText = nullptr)
		: Info( info ),
		Name( info.type.name() ),
		Text( pText ) { }
};

class ValueTypes;

/// Gets the value type register.
BE_CORE_API ValueTypes& GetValueTypes();

/// Adds a component type.
BE_CORE_API const ValueTypeDesc& AddValueType(const lean::property_type_info &type, ValueTypes &types = GetValueTypes());
/// Adds a component type.
BE_CORE_API void RemoveValueType(const lean::property_type_info &type, ValueTypes &types = GetValueTypes());

} // namespace

#endif