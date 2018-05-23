/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_REFLECTED_COMPONENT
#define BE_CORE_REFLECTED_COMPONENT

#include "beCore.h"
#include "bePropertyProvider.h"

#include <lean/containers/any.h>
#include <lean/smart/cloneable_obj.h>
#include <lean/smart/com_ptr.h>

namespace beCore
{

/// Reflected component base class.
class LEAN_INTERFACE ReflectedComponent : public PropertyProvider
{
	LEAN_INTERFACE_BEHAVIOR(ReflectedComponent)

public:
	/// Gets the number of child components.
	virtual uint4 GetComponentCount() const = 0;
	/// Gets the name of the n-th child component.
	BE_CORE_API virtual Exchange::utf8_string GetComponentName(uint4 idx) const
	{
		const ComponentType *pType = GetComponentType(idx);
		return (pType) ? pType->Name : "";
	}
	/// Returns true, if the n-th component is issential.
	BE_CORE_API virtual bool IsComponentEssential(uint4 idx) const { return true; }
	/// Gets the n-th child property provider, nullptr if not a property provider.
	BE_CORE_API virtual lean::com_ptr<PropertyProvider, lean::critical_ref> GetPropertyProvider(uint4 idx)
	{
		return lean::bind_com( const_cast<PropertyProvider*>( const_cast<const ReflectedComponent*>(this)->GetPropertyProvider(idx).unbind() ) );
	}
	/// Gets the n-th child property provider, nullptr if not a property provider.
	BE_CORE_API virtual lean::com_ptr<const PropertyProvider, lean::critical_ref> GetPropertyProvider(uint4 idx) const { return GetReflectedComponent(idx); }
	
	/// Gets the n-th reflected child component, nullptr if not reflected.
	BE_CORE_API virtual lean::com_ptr<ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx)
	{
		return lean::bind_com( const_cast<ReflectedComponent*>( const_cast<const ReflectedComponent*>(this)->GetReflectedComponent(idx).unbind() ) );
	}
	/// Gets the n-th reflected child component, nullptr if not reflected.
	virtual lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const = 0;

	/// Gets the type of the n-th child component.
	virtual const ComponentType* GetComponentType(uint4 idx) const = 0;
	/// Gets the n-th component.
	virtual lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const = 0;

	/// Returns true, if the n-th component can be replaced.
	virtual bool IsComponentReplaceable(uint4 idx) const = 0;
	/// Sets the n-th component.
	virtual void SetComponent(uint4 idx, const lean::any &pComponent) = 0;

	/// Gets a pointer to the reflected component interface.
	friend LEAN_INLINE ReflectedComponent* Reflect(ReflectedComponent *pReflected) { return pReflected; }
	/// Gets a pointer to the reflected component interface.
	friend LEAN_INLINE const ReflectedComponent* Reflect(const ReflectedComponent *pReflected) { return pReflected; }
};

/// Reflected component base class.
template <class Interface = ReflectedComponent>
class LEAN_INTERFACE RigidReflectedComponent : public Interface
{
	LEAN_BASE_BEHAVIOR(RigidReflectedComponent)

protected:
	LEAN_BASE_DELEGATE(RigidReflectedComponent, Interface)

public:
	/// Gets the name of the n-th child component.
	virtual Exchange::utf8_string GetComponentName(uint4 idx) const LEAN_OVERRIDE = 0;

	/// Gets the type of the n-th child component.
	virtual const ComponentType* GetComponentType(uint4 idx) const LEAN_OVERRIDE { return nullptr; }
	/// Gets the n-th component.
	virtual lean::cloneable_obj<lean::any, true> GetComponent(uint4 idx) const LEAN_OVERRIDE { return nullptr; }
	
	/// Returns true, if the n-th component can be replaced.
	virtual bool IsComponentReplaceable(uint4 idx) const LEAN_OVERRIDE { return false; }
	/// Sets the n-th component.
	virtual void SetComponent(uint4 idx, const lean::any &pComponent) LEAN_OVERRIDE { }
};

/// Reflected component base class.
template <class Interface = ReflectedComponent>
class LEAN_INTERFACE ReflectedPropertyProvider : public RigidReflectedComponent<Interface>
{
	LEAN_BASE_BEHAVIOR(ReflectedPropertyProvider)

protected:
	LEAN_BASE_DELEGATE(ReflectedPropertyProvider, RigidReflectedComponent<Interface>)

public:
	/// Gets the number of child components.
	virtual uint4 GetComponentCount() const LEAN_OVERRIDE { return 0; }
	/// Gets the name of the n-th child component.
	virtual Exchange::utf8_string GetComponentName(uint4 idx) const LEAN_OVERRIDE
	{
		const ComponentType *pType = GetComponentType(idx);
		return (pType) ? pType->Name : "";
	}
	/// Gets the n-th reflected child component, nullptr if not reflected.
	virtual lean::com_ptr<const ReflectedComponent, lean::critical_ref> GetReflectedComponent(uint4 idx) const LEAN_OVERRIDE { return nullptr; }
};

} // namespace

#endif