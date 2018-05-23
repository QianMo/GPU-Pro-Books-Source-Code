/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_RESOURCE_MANAGER_IMPL
#define BE_CORE_RESOURCE_MANAGER_IMPL

#include "beCore.h"
#include "beResourceManager.h"

namespace beCore
{

/// Resource manager implementation for named resources.
template < class ResourceT, class DerivedT, class BaseT = ResourceManager<ResourceT> >
class ResourceManagerImpl : public BaseT
{
public:
	/// Value type.
	typedef ResourceT Resource;
	/// Most-derived class.
	typedef DerivedT Derived;

	/// Gets information on the given resource.
	ComponentInfo GetInfo(const Resource *resource) const LEAN_OVERRIDE;
	/// Gets information on the resources currently available.
	ComponentInfoVector GetInfo() const LEAN_OVERRIDE;

	/// Sets the given name for the given resource.
	void SetName(Resource *resource, const utf8_ntri &name, bool bKeepOldName = false) LEAN_OVERRIDE;
	/// Gets the name of the given resource.
	utf8_ntr GetName(const Resource *resource) const LEAN_OVERRIDE;
	/// Gets the resource by name.
	Resource* GetByName(const utf8_ntri &name, bool bThrow = false) const LEAN_OVERRIDE;

	/// Gets a unique resource name from the given name.
	Exchange::utf8_string GetUniqueName(const utf8_ntri &name) const LEAN_OVERRIDE;

	/// Replaces the given old resource by the given new resource.
	void Replace(Resource *oldResource, Resource *newResource) LEAN_OVERRIDE;
};

/// Resource manager implementation for filed resources.
template < class ResourceT, class DerivedT, class BaseT = FiledResourceManager<ResourceT> >
class FiledResourceManagerImpl : public ResourceManagerImpl<ResourceT, DerivedT, BaseT>
{
public:
	/// Value type.
	typedef ResourceT Resource;
	/// Most-derived class.
	typedef DerivedT Derived;

	/// Gets information on the given resource.
	ComponentInfo GetInfo(const Resource *resource) const LEAN_OVERRIDE;
	/// Gets information on the resources currently available.
	ComponentInfoVector GetInfo() const LEAN_OVERRIDE;

	/// Changes the file of the given resource.
	void SetFile(Resource *resource, const utf8_ntri &file) LEAN_OVERRIDE;
	/// Unsets the file of the given resource.
	void Unfile(Resource *resource) LEAN_OVERRIDE;
	/// Gets the file of the given resource.
	utf8_ntr GetFile(const Resource *resource) const LEAN_OVERRIDE;
};

} // namespace

#endif