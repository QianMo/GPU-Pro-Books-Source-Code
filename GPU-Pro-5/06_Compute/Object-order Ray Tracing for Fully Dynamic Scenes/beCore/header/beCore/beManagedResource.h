/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_MANAGED_RESOURCE
#define BE_CORE_MANAGED_RESOURCE

#include "beCore.h"
#include <lean/smart/resource_ptr.h>

namespace beCore
{

/// Managed resource interface.
template <class Cache, class Interface = lean::empty_base>
class LEAN_INTERFACE AbstractManagedResource : public Interface
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(AbstractManagedResource)

public:
	/// Sets the managing resource cache.
	virtual void SetCache(Cache *cache) = 0;
	/// Gets the managing resource cache.
	virtual Cache* GetCache() const = 0;
};

/// Managed resource implementation.
template < class Cache, class Base = lean::empty_base, class CacheStorage = Cache* >
class LEAN_INTERFACE ManagedResource : public Base
{
	LEAN_BASE_BEHAVIOR(ManagedResource)

protected:
	CacheStorage m_pCache;

	LEAN_INLINE ManagedResource(Cache *pCache = nullptr) : Base(), m_pCache(pCache) { }	
	LEAN_INLINE ManagedResource(const ManagedResource &right) : Base(right), m_pCache() { }
	LEAN_INLINE ManagedResource& operator =(const ManagedResource &right) { this->Base::operator =(right); return *this; }

public:
	/// Sets the managing resource cache.
	void SetCache(Cache *cache) { m_pCache = cache; }
	/// Gets the managing resource cache.
	Cache* GetCache() const { return m_pCache; }
};

/// Trys to get the name of the given resource.
template <class String, class Resource, class Cache>
inline String GetCachedName(const Resource *pResource, const Cache *pCache)
{
	if (pCache)
		return pCache->GetName(pResource);
	return String();
}

/// Trys to get the file of the given resource.
template <class String, class Resource, class Cache>
inline String GetCachedFile(const Resource *pResource, const Cache *pCache)
{
	if (pCache)
		return pCache->GetFile(pResource);
	return String();
}

/// Trys to get the name of the given resource.
template <class String, class Resource>
inline String GetCachedName(const Resource *pResource)
{
	if (pResource)
		return GetCachedName<String>(pResource, pResource->GetCache());
	return String();
}

/// Trys to get the name of the given resource.
template <class String, class Resource>
inline String GetCachedFile(const Resource *pResource)
{
	if (pResource)
		return GetCachedFile<String>(pResource, pResource->GetCache());
	return String();
}

/// Hot-swapped resource interface.
template <class Successor, class Interface = lean::empty_base>
class LEAN_INTERFACE AbstractHotResource : public Interface
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(AbstractHotResource)

public:
	/// Sets the resource successor.
	virtual void SetSuccessor(Successor *successor) = 0;
	/// Gets the resource successor.
	virtual Successor* GetSuccessor() const = 0;
};

/// Hot-swapped resource implementation.
template < class Successor, class Base = lean::empty_base, class SuccessorStorage = lean::resource_ptr<Successor> >
class LEAN_INTERFACE HotResource : public Base
{
	LEAN_BASE_BEHAVIOR(HotResource)

protected:
	SuccessorStorage m_pSuccessor;

	LEAN_INLINE HotResource() : Base(), m_pSuccessor() { }	
	LEAN_INLINE HotResource(const HotResource &right) : Base(right), m_pSuccessor() { }
	LEAN_INLINE HotResource& operator =(const HotResource &right) { this->Base::operator =(right); return *this; }

public:
	/// Sets the resource successor.
	void SetSuccessor(Successor *successor) { m_pSuccessor = successor; }
	/// Gets the resource successor.
	Successor* GetSuccessor() const { return m_pSuccessor; }
};

/// Trys to get the successor of the given resource.
template <class Resource>
inline Resource* GetSuccessor(Resource *pResource)
{
	Resource *pSuccessor = pResource;

	if (pSuccessor)
		while (Resource *nextSuccessor = pSuccessor->GetSuccessor())
			pSuccessor = nextSuccessor;

	return pSuccessor;
}

/// Managed resource interface.
template <class Cache, class Successor, class Interface = lean::empty_base>
class LEAN_INTERFACE AbstractManagedHotResource : public AbstractManagedResource< Cache, AbstractHotResource<Successor, Interface> >
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(AbstractManagedHotResource)
};

/// Managed resource implementation.
template < class Cache, class Successor, class Base = lean::empty_base,
	class CacheStorage = Cache*,
	class SuccessorStorage = lean::resource_ptr<Successor>
>
class LEAN_INTERFACE ManagedHotResource : public ManagedResource< Cache, HotResource<Successor, Base, SuccessorStorage>, CacheStorage >
{
	LEAN_BASE_BEHAVIOR(ManagedHotResource)

protected:
	LEAN_INLINE ManagedHotResource(Cache *pCache = nullptr) : typename ManagedHotResource::ManagedResource(pCache) { }	
	LEAN_INLINE ManagedHotResource(const ManagedHotResource &right) : typename ManagedHotResource::ManagedResource(right) { }
	LEAN_INLINE ManagedHotResource& operator =(const ManagedHotResource &right) { this->ManagedResource::operator =(right); return *this; }
};

} // namespace

#endif