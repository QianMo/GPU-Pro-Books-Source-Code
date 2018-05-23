/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_RESOURCE_MANAGER_IMPL_PP
#define BE_CORE_RESOURCE_MANAGER_IMPL_PP

#include "beCore.h"
#include "beResourceManagerImpl.h"
#include <lean/pimpl/pimpl_ptr.h>

#include <lean/logging/errors.h>

namespace beCore
{

/// Default resource info construction implementation, replace using ADL.
template <class Derived>
LEAN_INLINE typename Derived::M::Info MakeResourceInfo(typename Derived::M &m, typename Derived::Resource *resource, Derived*)
{
	// ASSERT: resource != nullptr
	return typename Derived::M::Info(resource);
}

/// Default resource retrieval from resource index, replace using ADL.
template <class M, class Iterator>
LEAN_INLINE typename M::resources_t::Resource* GetResource(const M&, Iterator it)
{
	// ASSERT: Iterator always valid
	return it->resource;
}

/// Default resource replacement, replace using ADL.
template <class M, class Iterator>
LEAN_INLINE void SetResource(M&, Iterator it, typename M::resources_t::Resource *resource)
{
	// ASSERT: Iterator, resource always valid
	it->resource = resource;
}

/// Default resource index key retrieval from resource, replace using ADL.
template <class M>
LEAN_INLINE typename M::resources_t::Resource* GetResourceKey(const M&, const typename M::resources_t::Resource *pResource)
{
	// NOTE IN IMPLEMENTATIONS: pResource may be nullptr!
	return const_cast<typename M::resources_t::Resource*>(pResource);
}

/// Default resource cache retrieval, replace using ADL.
template <class Derived>
LEAN_INLINE Derived* GetResourceCache(typename Derived::M &m, Derived *cache)
{
	// ASSERT: cache always valid
	return cache;
}

/// Default resource cache setting, replace using ADL.
template <class Resource, class Derived>
LEAN_INLINE void SetResourceCache(typename Derived::M &m, Resource *resource, Derived *impl)
{
	// ASSERT: resource always valid
	resource->SetCache(GetResourceCache(m, impl));
}

/// Default successor setting, replace using ADL.
template <class M, class Resource>
LEAN_INLINE void SetResourceSuccessor(M&, Resource *oldResource, Resource *newResource)
{
	// ASSERT: resources always valid
	oldResource->SetSuccessor(newResource);
}

namespace
{

template <class Monitor, class Resource>
LEAN_INLINE void MonitorAddChanged(Monitor &monitor, const Resource *resource)
{
	monitor.AddChanged(Resource::GetComponentType());
}

} // namespace

/// Default resource change monitoring implementation. Replace using ADL.
template <class M, class Iterator>
LEAN_INLINE void ResourceChanged(M &m, Iterator it)
{
	if (m.pComponentMonitor)
		MonitorAddChanged(m.pComponentMonitor->Replacement, GetResource(m ,it));
}

/// Default resource management change monitoring implementation. Replace using ADL.
template <class M, class Iterator>
LEAN_INLINE void ResourceManagementChanged(M &m, Iterator it)
{
	if (m.pComponentMonitor)
		MonitorAddChanged(m.pComponentMonitor->Management, GetResource(m ,it));
}

// NOTE: Also implement ResourceFileChanged(M&, resources_t::iterator) using ADL

/// Default notes, replace using ADL.
template <class M, class Iterator>
LEAN_INLINE Exchange::utf8_string GetResourceNotes(const M&, Iterator it)
{
	return Exchange::utf8_string();
}

// Gets information on the given resource.
template <class R, class D, class B>
ComponentInfo ResourceManagerImpl<R, D, B>::GetInfo(const Resource *resource) const
{
	ComponentInfo result;
	LEAN_PIMPL_IN_CONST(typename Derived);
	typename M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, resource) );
	
	if (it != m.resourceIndex.End())
	{
		result.Name = m.resourceIndex.GetName(it).to<Exchange::utf8_string>();
		result.Notes = GetResourceNotes(m, it);
	}
	
	return result;
}

// Gets information on the resources currently available.
template <class R, class D, class B>
ComponentInfoVector ResourceManagerImpl<R, D, B>::GetInfo() const
{
	LEAN_PIMPL_IN_CONST(typename Derived);

	ComponentInfoVector info(m.resourceIndex.Count());
	uint4 infoIdx = 0;

	for (typename M::resources_t::const_name_iterator it = m.resourceIndex.BeginByName(), itEnd = m.resourceIndex.EndByName(); it != itEnd; ++it)
	{
		info[infoIdx].Name = m.resourceIndex.GetName(it).to<Exchange::utf8_string>();
		info[infoIdx].Notes = GetResourceNotes(m, it);
		++infoIdx;
	}

	LEAN_ASSERT(infoIdx == info.size());
	return info;
}

// Sets the given name for the given resource.
template <class R, class D, class B>
void ResourceManagerImpl<R, D, B>::SetName(Resource *resource, const utf8_ntri &name, bool bKeepOldName)
{
	LEAN_PIMPL_IN(typename Derived);
	typename M::resources_t::iterator it = m.resourceIndex.Find( GetResourceKey(m, LEAN_THROW_NULL(resource)) );

	if (it != m.resourceIndex.End())
		m.resourceIndex.SetName(it, name, bKeepOldName);
	else
		try
		{
			it = m.resourceIndex.Insert( GetResourceKey(m, resource), name, MakeResourceInfo(m, resource, static_cast<Derived*>(this)) );
		}
		catch (const std::runtime_error &e)
		{
			typename M::resources_t::const_name_iterator it = m.resourceIndex.FindByName(name.to<utf8_string>());

			if (it != m.resourceIndex.EndByName())
				throw ResourceCollision<Resource>( GetResource(m, it), e.what() );
			else
				throw;
		}

	SetResourceCache(m, resource, static_cast<Derived*>(this));
	ResourceManagementChanged(m, it);
}
// Gets the name of the given resource.
template <class R, class D, class B>
utf8_ntr ResourceManagerImpl<R, D, B>::GetName(const Resource *resource) const
{
	LEAN_PIMPL_IN_CONST(typename Derived);
	typename M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, resource) );
	return (it != m.resourceIndex.End())
		? m.resourceIndex.GetName(it)
		: utf8_ntr("");
}

// Gets the resource by name.
template <class Resource, class D, class B>
Resource* ResourceManagerImpl<Resource, D, B>::GetByName(const utf8_ntri &name, bool bThrow) const
{
	LEAN_PIMPL_IN_CONST(typename Derived);
	typename M::resources_t::const_name_iterator it = m.resourceIndex.FindByName(name.to<utf8_string>());
	if (it != m.resourceIndex.EndByName())
		return GetResource(m, it);
	else if (bThrow)
		LEAN_THROW_ERROR_CTX("Resource of the given name unknown to the resource manager", name);
	return nullptr;
}

// Gets a unique resource name from the given name.
template <class Resource, class D, class B>
Exchange::utf8_string ResourceManagerImpl<Resource, D, B>::GetUniqueName(const utf8_ntri &name) const
{
	LEAN_PIMPL_IN_CONST(typename Derived);
	return lean::from_range<Exchange::utf8_string>( m.resourceIndex.GetUniqueName(name) );
}

// Replaces the given old resource by the given new resource.
template <class R, class D, class B>
void ResourceManagerImpl<R, D, B>::Replace(Resource *oldResource, Resource *newResource)
{
	LEAN_PIMPL_IN(typename Derived);
	typename M::resources_t::iterator it = m.resourceIndex.Find( GetResourceKey(m, LEAN_THROW_NULL(oldResource)) );

	if (it != m.resourceIndex.End())
	{
		// Establish two-way mapping
		m.resourceIndex.Link(it, GetResourceKey(m, LEAN_THROW_NULL(newResource)));
		SetResourceCache(m, newResource, static_cast<Derived*>(this));
		
		// IMPORTANT: Keep resource until successor set
		lean::resource_ptr<Resource> oldResourceRef = oldResource;

		// Replace
		SetResource(m, it, newResource);
		m.resourceIndex.Unlink(it, GetResourceKey(m, oldResource));
		SetResourceSuccessor(m, oldResource, newResource);

		ResourceChanged(m, it);
	}
	else
		LEAN_THROW_ERROR_MSG("Resource to be replaced unknown to the resource manager");
}

// Gets information on the given resource.
template <class R, class D, class B>
ComponentInfo FiledResourceManagerImpl<R, D, B>::GetInfo(const Resource *resource) const
{
	ComponentInfo result;
	LEAN_PIMPL_IN_CONST(typename Derived);
	typename M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, resource) );
	
	if (it != m.resourceIndex.End())
	{
		result.Name = m.resourceIndex.GetName(it).to<Exchange::utf8_string>();
		result.File = m.resourceIndex.GetFile(it).to<Exchange::utf8_string>();
		result.Notes = GetResourceNotes(m, it);
	}
	
	return result;
}

// Gets information on the resources currently available.
template <class R, class D, class B>
ComponentInfoVector FiledResourceManagerImpl<R, D, B>::GetInfo() const
{
	LEAN_PIMPL_IN_CONST(typename Derived);

	ComponentInfoVector info(m.resourceIndex.Count());
	uint4 infoIdx = 0;

	for (typename M::resources_t::const_name_iterator it = m.resourceIndex.BeginByName(), itEnd = m.resourceIndex.EndByName(); it != itEnd; ++it)
	{
		info[infoIdx].Name = m.resourceIndex.GetName(it).to<Exchange::utf8_string>();
		info[infoIdx].File = m.resourceIndex.GetFile(it).to<Exchange::utf8_string>();
		info[infoIdx].Notes = GetResourceNotes(m, it);
		++infoIdx;
	}

	LEAN_ASSERT(infoIdx == info.size());
	return info;
}


// Changes the file of the given resource.
template <class R, class D, class B>
void FiledResourceManagerImpl<R, D, B>::SetFile(Resource *resource, const utf8_ntri &file)
{
	LEAN_PIMPL_IN(typename Derived);
	typename M::resources_t::iterator it = m.resourceIndex.Find( GetResourceKey(m, resource) );

	if (it != m.resourceIndex.End())
	{
		utf8_string previousFile = m.resourceIndex.GetFile(it).to<utf8_string>();

		typename M::resources_t::iterator itUnfiled;
		bool bChanged;
		m.resourceIndex.SetFile(it, file, &bChanged, &itUnfiled);

		if (bChanged)
		{
			try
			{
				if (itUnfiled != m.resourceIndex.End())
					ResourceFileChanged(m, itUnfiled, "", file);
				ResourceFileChanged(m, it, file, previousFile);
			}
			LEAN_ASSERT_NOEXCEPT

			ResourceManagementChanged(m, it);
		}
	}
	else
		LEAN_THROW_ERROR_CTX("Cannot file resource unknown to this manager", file);
}

// Unsets the file of the given resource.
template <class R, class D, class B>
void FiledResourceManagerImpl<R, D, B>::Unfile(Resource *resource)
{
	LEAN_PIMPL_IN(typename Derived);
	typename M::resources_t::iterator it = m.resourceIndex.Find( GetResourceKey(m, resource) );

	if (it != m.resourceIndex.End())
	{
		utf8_string previousFile = m.resourceIndex.GetFile(it).to<utf8_string>();

		if (m.resourceIndex.Unfile(it))
			ResourceFileChanged(m, it, "", previousFile);
	}
	else
		LEAN_THROW_ERROR_MSG("Cannot unfile resource unknown to this manager");
}

// Gets the file of the given resource.
template <class R, class D, class B>
utf8_ntr FiledResourceManagerImpl<R, D, B>::GetFile(const Resource *resource) const
{
	LEAN_PIMPL_IN_CONST(typename Derived);
	typename M::resources_t::const_iterator it = m.resourceIndex.Find( GetResourceKey(m, resource) );
	return (it != m.resourceIndex.End())
		? m.resourceIndex.GetFile(it)
		: utf8_ntr("");
}

} // namespace

#endif