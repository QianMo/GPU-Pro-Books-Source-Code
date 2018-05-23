/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#include "beCoreInternal/stdafx.h"
#include "beCore/beReflectionProperties.h"
#include "beCore/beReflectionPropertyProvider.h"

#include "beCore/bePropertyVisitor.h"

namespace beCore
{

// Gets the ID of the given property.
uint4 GetPropertyID(uint4 baseOffset, PropertyRange range, const utf8_ntri &name)
{
	return lean::find_property(range, name, ReflectionPropertyProvider::InvalidPropertyID, baseOffset);
}

// Gets the name of the given property.
utf8_ntr GetPropertyName(uint4 baseOffset, PropertyRange range, uint4 id)
{
	uint4 rangeID = id - baseOffset;
	return (rangeID < range.size()) ? utf8_ntr(range[rangeID].name) : utf8_ntr("");
}

// Gets the type of the given property.
PropertyDesc GetPropertyDesc(uint4 baseOffset, PropertyRange range, uint4 id)
{
	PropertyDesc desc;

	uint4 rangeID = id - baseOffset;
	if (rangeID < range.size())
	{
		const ReflectionProperty &property = range[rangeID];
		desc = PropertyDesc(*property.type_info, property.count, property.widget);
	}

	return desc;
}

// Sets the given (raw) values.
bool SetProperty(uint4 baseOffset, PropertyRange range, PropertyProvider &provider, uint4 id, const std::type_info &type, const void *values, size_t count)
{
	uint4 rangeID = id - baseOffset;
	if (rangeID < range.size())
	{
		const ReflectionProperty &desc = range[rangeID];
		if (desc.setter.valid())
			return (*desc.setter)(provider, type, values, desc.count);
	}
	return false;
}

// Gets the given number of (raw) values.
bool GetProperty(uint4 baseOffset, PropertyRange range, const PropertyProvider &provider, uint4 id, const std::type_info &type, void *values, size_t count)
{
	uint4 rangeID = id - baseOffset;
	if (rangeID < range.size())
	{
		const ReflectionProperty &desc = range[rangeID];
		if (desc.getter.valid())
			return (*desc.getter)(provider, type, values, desc.count);
	}
	return false;
}

// Visits a property for modification.
bool WriteProperty(uint4 baseOffset, PropertyRange range, PropertyProvider &provider, uint4 id, PropertyVisitor &visitor, uint4 flags)
{
	uint4 rangeID = id - baseOffset;
	if (rangeID < range.size())
	{
		const ReflectionProperty &desc = range[rangeID];
		bool bPartialWrite = (flags & PropertyVisitFlags::PartialWrite) != 0;

		if ( desc.setter.valid() &&
			(!bPartialWrite || desc.getter.valid()) &&
			(!(flags & PropertyVisitFlags::PersistentOnly) || desc.persistence & PropertyPersistence::Read) )
		{
			const lean::property_type &propertyType = *desc.type_info->Info.property_type;
			size_t size = propertyType.size(desc.count);

			static const size_t StackBufferSize = 16 * 8;
			char stackBuffer[StackBufferSize];

			lean::scoped_property_data<lean::deallocate_property_data_policy> bufferGuard(
				propertyType, (size <= StackBufferSize) ? nullptr : propertyType.allocate(desc.count), desc.count );
			void *values = (bufferGuard.data()) ? bufferGuard.data() : stackBuffer;

			propertyType.construct(values, desc.count);
			lean::scoped_property_data<lean::destruct_property_data_policy> valueGuard(propertyType, values, desc.count);

			if (!bPartialWrite || (*desc.getter)(provider, desc.type_info->Info.type, values, desc.count))
			{
				bool bModified = visitor.Visit(provider, id, PropertyDesc(*desc.type_info, desc.count, desc.widget), values);

				if (bModified)
					return (*desc.setter)(provider, desc.type_info->Info.type, values, desc.count);
			}
		}
	}
	return false;
}

// Visits a property for reading.
bool ReadProperty(uint4 baseOffset, PropertyRange range, const PropertyProvider &provider, uint4 id, PropertyVisitor &visitor, uint4 flags)
{
	uint4 rangeID = id - baseOffset;
	if (rangeID < range.size())
	{
		const ReflectionProperty &desc = range[rangeID];

		if ( desc.getter.valid() &&
			(!(flags & PropertyVisitFlags::PersistentOnly) || desc.persistence & PropertyPersistence::Write) )
		{
			const lean::property_type &propertyType = *desc.type_info->Info.property_type;
			size_t size = propertyType.size(desc.count);

			static const size_t StackBufferSize = 16 * 8;
			char stackBuffer[StackBufferSize];

			lean::scoped_property_data<lean::deallocate_property_data_policy> bufferGuard(
				propertyType, (size <= StackBufferSize) ? nullptr : propertyType.allocate(desc.count), desc.count );
			void *values = (bufferGuard.data()) ? bufferGuard.data() : stackBuffer;

			propertyType.construct(values, desc.count);
			lean::scoped_property_data<lean::destruct_property_data_policy> valueGuard(propertyType, values, desc.count);

			if ((*desc.getter)(provider, desc.type_info->Info.type, values,  desc.count))
			{
				// WARNING: Call read-only overload!
				visitor.Visit(provider, id, PropertyDesc(*desc.type_info, desc.count, desc.widget), const_cast<const void*>(values));
				return true;
			}
		}
	}
	return false;
}

// Gets the number of properties.
uint4 ReflectionPropertyProvider::GetPropertyCount() const
{
	return static_cast<uint4>( GetReflectionProperties().size() );
}

// Gets the ID of the given property.
uint4 ReflectionPropertyProvider::GetPropertyID(const utf8_ntri &name) const
{
	return bec::GetPropertyID(0, GetReflectionProperties(), name);
}

// Gets the name of the given property.
utf8_ntr ReflectionPropertyProvider::GetPropertyName(uint4 id) const
{
	return bec::GetPropertyName(0, GetReflectionProperties(), id);
}

// Gets the type of the given property.
PropertyDesc ReflectionPropertyProvider::GetPropertyDesc(uint4 id) const
{
	return bec::GetPropertyDesc(0, GetReflectionProperties(), id);
}

// Sets the given (raw) values.
bool ReflectionPropertyProvider::SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count)
{
	bool bModified = bec::SetProperty(0, GetReflectionProperties(), *this, id, type, values, count);
	if (bModified)
		EmitPropertyChanged();
	return bModified;
}

// Gets the given number of (raw) values.
bool ReflectionPropertyProvider::GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const
{
	return bec::GetProperty(0, GetReflectionProperties(), *this, id, type, values, count);
}

// Visits a property for modification.
bool ReflectionPropertyProvider::WriteProperty(uint4 id, PropertyVisitor &visitor, uint4 flags)
{
	bool bModified = bec::WriteProperty(0, GetReflectionProperties(), *this, id, visitor, flags);
	if (bModified)
		EmitPropertyChanged();
	return bModified;
}

// Visits a property for reading.
bool ReflectionPropertyProvider::ReadProperty(uint4 id, PropertyVisitor &visitor, uint4 flags) const
{
	return bec::ReadProperty(0, GetReflectionProperties(), *this, id, visitor, flags);
}

// Overwrite to emit a property changed signal.
void ReflectionPropertyProvider::EmitPropertyChanged() const
{
}

} // namespace
