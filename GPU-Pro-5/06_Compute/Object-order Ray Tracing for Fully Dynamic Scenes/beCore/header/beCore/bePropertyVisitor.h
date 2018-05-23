/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_PROPERTY_VISITOR
#define BE_CORE_PROPERTY_VISITOR

#include "beCore.h"
#include "bePropertyProvider.h"
#include "beDataVisitor.h"

namespace beCore
{

/// Property visitor.
class LEAN_INTERFACE PropertyVisitor
{
	LEAN_INTERFACE_BEHAVIOR(PropertyVisitor)

public:
	/// Visits the given values.
	BE_CORE_API virtual bool Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, void *values)
	{
		Visit(provider, propertyID, desc, const_cast<const void*>(values));
		return false;
	}
	/// Visits the given values.
	BE_CORE_API virtual void Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, const void *values) { }
};

/// Property data visitor adapter.
class PropertyDataVisitor : public PropertyVisitor
{
	DataVisitor *m_pVisitor;

public:
	/// Constructor.
	PropertyDataVisitor(DataVisitor &visitor)
		: m_pVisitor(&visitor) { }

	/// Visits the given values.
	BE_CORE_API bool Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, void *values) LEAN_OVERRIDE
	{
		return m_pVisitor->Visit(*desc.TypeDesc, values, desc.Count);
	}
	/// Visits the given values.
	BE_CORE_API void Visit(const PropertyProvider &provider, uint4 propertyID, const PropertyDesc &desc, const void *values) LEAN_OVERRIDE
	{
		m_pVisitor->Visit(*desc.TypeDesc, values, desc.Count);
	}
};

} // namespace

#endif