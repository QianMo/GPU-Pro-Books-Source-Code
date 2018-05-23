/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beValueTypes.h"
#include "beCore/beTextSerializer.h"

#include <lean/logging/errors.h>

namespace beCore
{

// Constructor.
ValueTypes::ValueTypes()
{
}

// Destructor.
ValueTypes::~ValueTypes()
{
}

// Adds the given component type, returning a unique address for this type.
const ValueTypeDesc& ValueTypes::AddType(const lean::property_type_info &typeInfo)
{
	utf8_nt typeName( typeInfo.type.name() );
	std::pair<typename_map::iterator, bool> it = m_typeNames.insert(
			std::make_pair( typeName, ValueTypeDesc(typeInfo) )
		);

	if (!it.second)
		LEAN_THROW_ERROR_CTX("Value type name collision", typeInfo.type.name());

	return it.first->second;
}

// Removes the given component type.
void ValueTypes::RemoveType(const lean::property_type_info &typeInfo)
{
	typename_map::iterator it = m_typeNames.find( utf8_nt(typeInfo.type.name()) );

	if (it != m_typeNames.end() && it->second.Info.type == typeInfo.type)
		m_typeNames.erase(it);
}

// Sets the given serializer for the given type.
void ValueTypes::SetSerializer(const TextSerializer *text)
{
	LEAN_ASSERT(text);

	utf8_nt typeName = text->GetType();
	typename_map::iterator it = m_typeNames.find(typeName);

	if (it != m_typeNames.end())
		it->second.Text = text;
	else
		LEAN_THROW_ERROR_CTX("Serializer type not registered", typeName.c_str());
}

// Unsets the given serializer for the given type.
void ValueTypes::UnsetSerializer(const TextSerializer *text)
{
	LEAN_ASSERT(text);

	utf8_nt typeName = text->GetType();
	typename_map::iterator it = m_typeNames.find(typeName);

	if (it != m_typeNames.end() && it->second.Text == text)
		it->second.Text = nullptr;
}

// Gets a component type address for the given component type name.
const ValueTypeDesc* ValueTypes::GetDesc(const utf8_ntri &name) const
{
	typename_map::const_iterator it = m_typeNames.find(utf8_nt(name));

	return (it != m_typeNames.end())
		? &it->second
		: nullptr;
}

// Gets all reflectors.
ValueTypes::TypeDescs ValueTypes::GetDescs() const
{
	TypeDescs descs;
	descs.reserve(m_typeNames.size());

	for (typename_map::const_iterator it = m_typeNames.begin(); it != m_typeNames.end(); ++it)
		descs.push_back(&it->second);

	return descs;
}

// Gets the component type register.
ValueTypes& GetValueTypes()
{
	static ValueTypes valueTypes;
	return valueTypes;
}

// Adds a value type.
const ValueTypeDesc& AddValueType(const lean::property_type_info &type, ValueTypes &types)
{
	return types.AddType(type);
}

// Adds a value type.
void RemoveValueType(const lean::property_type_info &type, ValueTypes &types)
{
	types.RemoveType(type);
}

} // namespace
