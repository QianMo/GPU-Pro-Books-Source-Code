/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_COMPONENT_OBSERVATION
#define BE_CORE_COMPONENT_OBSERVATION

#include "beCore.h"

namespace beCore
{

class Component;
class PropertyProvider;
class ReflectedComponent;

/// Property listener.
class LEAN_INTERFACE ComponentObserver
{
	LEAN_INTERFACE_BEHAVIOR(ComponentObserver)

public:
	/// Called when properties in the given provider might have changed.
	BE_CORE_API virtual void PropertyChanged(const PropertyProvider &provider) { }
	/// Called when child components in the given provider might have changed.
	BE_CORE_API virtual void ChildChanged(const ReflectedComponent &provider) { }
	/// Called when the structure of the given component has changed.
	BE_CORE_API virtual void StructureChanged(const Component &provider) { }
	/// Called when the given component has been replaced.
	BE_CORE_API virtual void ComponentReplaced(const Component &previous) { }
};

} // namespace

#endif