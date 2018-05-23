/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_RESOURCE_MANAGER
#define BE_CORE_RESOURCE_MANAGER

#include "beCore.h"
#include "beShared.h"
#include "beComponentInfo.h"
#include "beExchangeContainers.h"
#include <stdexcept>

namespace beCore
{

/// Resource collision exception.
template <class ResourceT>
class ResourceCollision : public std::runtime_error
{
public:
	ResourceT *const Resource;	///< Colliding resource.

	/// Constructor.
	LEAN_INLINE ResourceCollision(ResourceT *resource, const char *what)
		: std::runtime_error(what),
		Resource(resource) { }
};

/// Resource manager interface for named resources.
template <class ResourceT>
class LEAN_INTERFACE ResourceManager : public beCore::Resource
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(ResourceManager)

public:
	/// Value type.
	typedef ResourceT Resource;

	/// Gets information on the given resource.
	virtual ComponentInfo GetInfo(const Resource *resource) const = 0;
	/// Gets information on the resources currently available.
	virtual ComponentInfoVector GetInfo() const = 0;

	/// Sets the given name for the given resource.
	virtual void SetName(Resource *resource, const utf8_ntri &name, bool bKeepOldName = false) = 0;
	/// Gets the name of the given resource.
	virtual utf8_ntr GetName(const Resource *resource) const = 0;
	/// Gets the resource by name.
	virtual Resource* GetByName(const utf8_ntri &name, bool bThrow = false) const = 0;

	/// Gets a unique resource name from the given name.
	virtual Exchange::utf8_string GetUniqueName(const utf8_ntri &name) const = 0;

	/// Replaces the given old resource by the given new resource.
	virtual void Replace(Resource *oldResource, Resource *newResource) = 0;

	/// Commits changes / reacts to changes.
	virtual void Commit() { }
};

/// Resource manager interface for named and filed resources.
template <class ResourceT>
class LEAN_INTERFACE FiledResourceManager : public ResourceManager<ResourceT>
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(FiledResourceManager)

public:
	/// Value type.
	typedef ResourceT Resource;

	/// Changes the file of the given resource.
	virtual void SetFile(Resource *resource, const utf8_ntri &file) = 0;
	/// Unsets the file of the given resource.
	virtual void Unfile(Resource *resource) = 0;
	/// Gets the file of the given resource.
	virtual utf8_ntr GetFile(const Resource *resource) const = 0;
};

} // namespace

#endif