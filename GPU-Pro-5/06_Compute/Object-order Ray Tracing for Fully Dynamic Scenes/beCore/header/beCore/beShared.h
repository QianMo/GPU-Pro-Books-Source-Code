/*****************************************************/
/* breeze Engine Core Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_CORE_SHARED
#define BE_CORE_SHARED

#include "beCore.h"
#include <lean/memory/win_heap.h>
#include <lean/memory/heap_bound.h>
#include <lean/memory/heap_allocator.h>
#include <lean/smart/resource.h>
#include <lean/smart/resource_ptr.h>

#include <lean/containers/any.h>

namespace beCore
{

/// Provides complex types that may be shared across module boundaries.
namespace Exchange
{
	/// Exchange heap.
	typedef lean::win_heap exchange_heap;
	
	/// Defines an allocator type that may be used in STL-conformant containers intended for cross-module data exchange.
	template <class Type, size_t Alignment = alignof(Type)>
	struct exchange_allocator_t
	{
		/// Exchange allocator type.
		typedef lean::heap_allocator<Type, exchange_heap, Alignment> t;
	};

} // namespace

using Exchange::exchange_heap;
using Exchange::exchange_allocator_t;

template <class Resource>
struct any_resource_t
{
	typedef lean::any_value< lean::resource_ptr<Resource>, lean::var_union<Resource*> > t;
};
/// Shared object base class.
typedef lean::heap_bound<exchange_heap> Shared;

/// Makes a shared class a resource class.
typedef lean::resource<long, exchange_allocator_t<long>::t> SharedMakeResource;
/// Makes a shared class an optional resource class.
typedef lean::resource<long, exchange_allocator_t<long>::t, true> SharedMakeOptionalResource;

/// Shared resource base class.
class Resource : public Shared, public SharedMakeResource
{
	LEAN_STATIC_INTERFACE_BEHAVIOR(Resource)
};

/// Shared optional resource base class.
class OptionalResource : public Shared, public SharedMakeOptionalResource
{
	LEAN_STATIC_INTERFACE_BEHAVIOR(OptionalResource)
};

/// Shared resource interface.
class LEAN_INTERFACE AbstractResource : public Shared,
	public lean::resource_interface<long, exchange_allocator_t<long>::t>
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(AbstractResource)
};

/// 'COM' resource interface.
class LEAN_INTERFACE RefCounted
{
	LEAN_INTERFACE_BEHAVIOR(RefCounted)

public:
	/// Increses the reference count of this resource.
	virtual void AddRef() const = 0;
	/// Decreases the reference count of this resource, destroying the resource when the count reaches zero.
	virtual void Release() const = 0;
};

/// Resource as ref counted base class.
template <class Base = RefCounted, class ResourceBase = Resource>
class LEAN_INTERFACE ResourceAsRefCounted : public Base, public ResourceBase
{
	LEAN_SHARED_BASE_BEHAVIOR(ResourceAsRefCounted)

protected:
	LEAN_BASE_DELEGATE(ResourceAsRefCounted, Base)

public:
	/// Increses the reference count of this resource.
	void AddRef() const LEAN_OVERRIDE
	{
		lean::resource_ptr<const ResourceAsRefCounted>(this).unbind();
	}
	/// Decreases the reference count of this resource, destroying the resource when the count reaches zero.
	void Release() const LEAN_OVERRIDE
	{
		lean::resource_ptr<const ResourceAsRefCounted>(this, lean::bind_reference);
	}
};

/// Resource to ref counted adapter.
template <class Derived, class Base = RefCounted>
class LEAN_INTERFACE ResourceToRefCounted : public Base
{
	LEAN_SHARED_BASE_BEHAVIOR(ResourceToRefCounted)

protected:
	LEAN_BASE_DELEGATE(ResourceToRefCounted, Base)

public:
	/// Increses the reference count of this resource.
	void AddRef() const LEAN_OVERRIDE
	{
		lean::resource_ptr<const ResourceToRefCounted>(this).unbind();
	}
	/// Decreases the reference count of this resource, destroying the resource when the count reaches zero.
	void Release() const LEAN_OVERRIDE
	{
		lean::resource_ptr<const ResourceToRefCounted>(this, lean::bind_reference);
	}
};

/// Passive ref counted adapter.
template <class Base = RefCounted>
class LEAN_INTERFACE UnRefCounted : public Base
{
	LEAN_BASE_BEHAVIOR(UnRefCounted)

protected:
	LEAN_BASE_DELEGATE(UnRefCounted, Base)

public:
	/// Does nothing.
	void AddRef() const LEAN_OVERRIDE { }
	/// Does nothing.
	void Release() const LEAN_OVERRIDE { }
};

} // namespace

#endif