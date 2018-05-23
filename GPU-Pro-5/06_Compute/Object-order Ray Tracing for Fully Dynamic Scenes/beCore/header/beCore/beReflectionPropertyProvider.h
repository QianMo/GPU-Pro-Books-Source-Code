/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_REFLECTION_PROPERTY_PROVIDER
#define BE_CORE_REFLECTION_PROPERTY_PROVIDER

#include "beCore.h"
#include "beReflectedComponent.h"

namespace beCore
{

struct ReflectionProperty;	

/// Generic property provider base class.
class LEAN_INTERFACE ReflectionPropertyProvider : public ReflectedPropertyProvider<>
{
	LEAN_INTERFACE_BEHAVIOR(ReflectionPropertyProvider)

public:
	/// Property range type.
	typedef lean::range<const ReflectionProperty*> Properties;

protected:
	/// Override to emit a property changed signal.
	BE_CORE_API virtual void EmitPropertyChanged() const;

public:
	/// Invalid property ID.
	static const uint4 InvalidPropertyID = static_cast<uint4>(-1);

	/// Gets the reflection properties.
	virtual Properties GetReflectionProperties() const = 0;
#ifdef DOXYGEN_READ_THIS
	/// Gets the reflection properties.
	static Properties GetOwnProperties() = 0;
#endif
	
	/// Gets the number of properties.
	BE_CORE_API uint4 GetPropertyCount() const;
	/// Gets the ID of the given property.
	BE_CORE_API uint4 GetPropertyID(const utf8_ntri &name) const;
	/// Gets the name of the given property.
	BE_CORE_API utf8_ntr GetPropertyName(uint4 id) const;
	/// Gets the type of the given property.
	BE_CORE_API PropertyDesc GetPropertyDesc(uint4 id) const;

	/// Sets the given (raw) values.
	BE_CORE_API bool SetProperty(uint4 id, const std::type_info &type, const void *values, size_t count);
	/// Gets the given number of (raw) values.
	BE_CORE_API bool GetProperty(uint4 id, const std::type_info &type, void *values, size_t count) const;

	/// Visits a property for modification.
	BE_CORE_API bool WriteProperty(uint4 id, PropertyVisitor &visitor, uint4 flags = PropertyVisitFlags::None);
	/// Visits a property for reading.
	BE_CORE_API bool ReadProperty(uint4 id, PropertyVisitor &visitor, uint4 flags = PropertyVisitFlags::None) const;
};

} // namespace

#endif