/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beBuiltinTypes.h"
#include "beCore/beValueTypes.h"

#include <lean/properties/property_types.h>
#include "beCore/beGenericTextSerializer.h"

namespace beCore
{

namespace
{

template <class Type, class TextSerializerType>
const ValueTypeDesc& RegisterBuiltinType(ValueTypes &valueTypes)
{
	static const GenericTextSerializer<TextSerializerType> textSerializer;
	const ValueTypeDesc &desc = valueTypes.AddType( lean::get_property_type_info<Type>() );
	valueTypes.SetSerializer(&textSerializer);
	return desc;
}

template <class Type>
const ValueTypeDesc& RegisterBoolType(ValueTypes &valueTypes)
{
	return RegisterBuiltinType< Type, lean::io::bool_serialization<Type> >(valueTypes);
}

template <class Type>
const ValueTypeDesc& RegisterIntType(ValueTypes &valueTypes)
{
	return RegisterBuiltinType< Type, lean::io::int_serialization<Type> >(valueTypes);
}

template <class Type>
const ValueTypeDesc& RegisterFloatType(ValueTypes &valueTypes)
{
	return RegisterBuiltinType< Type, lean::io::float_serialization<Type> >(valueTypes);
}

} // namespace

template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<bool>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterBoolType<bool>(valueTypes); return desc; }

template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<unsigned char>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<unsigned char>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<char>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<char>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<unsigned short>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<unsigned short>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<short>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<short>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<unsigned int>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<unsigned int>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<int>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<int>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<unsigned long>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<unsigned long>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<long>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<long>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<unsigned long long>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<unsigned long long>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<long long>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterIntType<long long>(valueTypes); return desc; }

template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<float>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterFloatType<float>(valueTypes); return desc; }
template <>
BE_CORE_API const ValueTypeDesc& GetBuiltinType<double>(ValueTypes &valueTypes) { static const ValueTypeDesc &desc = RegisterFloatType<double>(valueTypes); return desc; }

} // namespace
