/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_RESOURCE_PTR
#define LEAN_SMART_RESOURCE_PTR

#include "../cpp0x.h"
#include "../functional/variadic.h"
#include "common.h"

namespace lean
{
namespace smart
{

// Prototypes
template <class Counter, class Allocator>
class ref_counter;

template <class Type, reference_state_t RefState, class ReleasePolicy>
class scoped_ptr;

/// Destroys the given resource by calling @code delete resource@endcode  (default policy implementation).
template <class Resource>
LEAN_INLINE void destroy_resource(Resource *resource)
{
	delete resource;
}

/// Resource pointer class that performs strong reference counting on the given resource type.
template <class Resource, bool Critical = false>
class resource_ptr
{
	template <class Resource>
	friend class weak_resource_ptr;

public:
	/// Type of the resource stored by this resource pointer.
	typedef Resource resource_type;
	/// Type of the pointer stored by this resource pointer.
	typedef Resource* value_type;

private:
	resource_type *m_resource;

	/// Acquires the given resource.
	template <class Counter, class Allocator>
	static resource_type* acquire(resource_type *resource, const ref_counter<Counter, Allocator>& refCounter)
	{
		return (resource && refCounter.increment_checked())
			? resource
			: nullptr;
	}

	/// Acquires the given resource.
	static resource_type* acquire(resource_type *resource)
	{
		if (resource)
			resource->ref_counter().increment();

		return resource;
	}

	/// Releases the given resource.
	static void release(resource_type *resource)
	{
		// Clean up, if this is the last reference
		if (resource && !resource->ref_counter().decrement())
			destroy_resource(resource);
	}

protected:
	/// Constructs a resource pointer from the given resource and reference counter.
	template <class Counter, class Allocator>
	resource_ptr(resource_type *resource, const ref_counter<Counter, Allocator>& refCounter)
		: m_resource( acquire(resource, refCounter) ) { };

public:
	/// Constructs a resource pointer from the given resource.
	resource_ptr(resource_type *resource = nullptr)
		: m_resource( acquire(resource) ) { };
	/// Constructs a resource pointer from the given resource.
	template <class Resource2>
	resource_ptr(Resource2 *resource)
		: m_resource( acquire(resource) ) { };
	
	/// Constructs a resource pointer from the given resource pointer.
	resource_ptr(const resource_ptr &right)
		: m_resource( acquire(right.m_resource) ) { };
	/// Constructs a resource pointer from the given resource pointer.
	template <class Resource2, bool Critical2>
	resource_ptr(const resource_ptr<Resource2, Critical2> &right)
		: m_resource( acquire(right.get()) ) { };
	
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a resource pointer from the given r-value resource pointer.
	template <class Resource2, bool Critical2>
	resource_ptr(resource_ptr<Resource2, Critical2> &&right) noexcept
		: m_resource(right.unbind()) { }
#endif

	/// Constructs a resource pointer from the given resource without incrementing its reference count.
	LEAN_INLINE resource_ptr(resource_type *resource, bind_reference_t) noexcept
		: m_resource(resource) { };

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Constructs a resource pointer from the given resource without incrementing its reference count.
	template <class Type, reference_state_t RefState, class ReleasePolicy>
	LEAN_INLINE resource_ptr(scoped_ptr<Type, RefState, ReleasePolicy> &&resource) noexcept
		: m_resource(resource.detach()) { };
#endif
	/// Constructs a resource pointer from the given resource without incrementing its reference count.
	template <class Type, reference_state_t RefState, class ReleasePolicy>
	LEAN_INLINE resource_ptr(move_ref< scoped_ptr<Type, RefState, ReleasePolicy> > resource) noexcept
		: m_resource(resource.moved().detach()) { };

	/// Destroys the resource pointer.
	~resource_ptr()
	{
		release(m_resource);
	}

	/// Binds the given resource reference to this resource pointer.
	static LEAN_INLINE resource_ptr<resource_type, true> bind(resource_type *resource)
	{
		return resource_ptr<resource_type, true>(resource, bind_reference);
	}
	/// Transfers the resource reference held by this resource pointer to a new resource pointer.
	resource_ptr<resource_type, true> transfer()
	{
		// Visual C++ won't inline delegating function calls
		return resource_ptr<resource_type, true>(unbind(), bind_reference);
	}

	/// Replaces the resource reference held by this resource pointer by the given reference.
	void rebind(resource_type *resource)
	{
		resource_type *prevResource = m_resource;
		m_resource = resource;
		release(prevResource);
	}
	/// Unbinds the resource reference held by this resource pointer.
	LEAN_INLINE resource_type* unbind()
	{
		resource_type *prevResource = m_resource;
		m_resource = nullptr;
		return prevResource;
	}
	/// Replaces the resource reference held by this resource pointer by a new reference to given resource.
	LEAN_INLINE void reset(resource_type *resource)
	{
		// Do not check for redundant assignment
		// -> The code handles redundant assignment just fine
		// -> Checking generates up to twice the code due to unfortunate compiler optimization application order
		rebind(acquire(resource));
	}
	/// Releases the resource reference held by this resource pointer.
	LEAN_INLINE void release()
	{
		rebind(nullptr);
	}
	
	/// Replaces the stored resource with the given resource. <b>[ESA]</b>
	resource_ptr& operator =(resource_type *resource)
	{
		reset(resource);
		return *this;
	}

	/// Replaces the stored resource with one stored by the given resource pointer. <b>[ESA]</b>
	resource_ptr& operator =(const resource_ptr &right)
	{
		reset(right.m_resource);
		return *this;
	}
	/// Replaces the stored resource with one stored by the given resource pointer. <b>[ESA]</b>
	template <class Resource2, bool Critical2>
	resource_ptr& operator =(const resource_ptr<Resource2, Critical2> &right)
	{
		reset(right.get());
		return *this;
	}

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Replaces the stored resource with the one stored by the given r-value resource pointer. <b>[ESA]</b>
	template <class Resource2, bool Critical2>
	resource_ptr& operator =(resource_ptr<Resource2, Critical2> &&right) noexcept
	{
		// Self-assignment would be wrong
		if ((void*) this != (void*) &right)
			rebind(right.unbind());
		return *this;
	}
	/// Replaces the stored resource with the one stored by the given r-value scoped pointer. <b>[ESA]</b>
	template <class Type, reference_state_t RefState, class ReleasePolicy>
	resource_ptr& operator =(scoped_ptr<Type, RefState, ReleasePolicy> &&right) noexcept
	{
		rebind(right.detach());
		return *this;
	}
#endif
	/// Replaces the stored resource with the one stored by the given r-value scoped pointer. <b>[ESA]</b>
	template <class Type, reference_state_t RefState, class ReleasePolicy>
	resource_ptr& operator =(move_ref< scoped_ptr<Type, RefState, ReleasePolicy> > right) noexcept
	{
		rebind(right.moved().detach());
		return *this;
	}

	/// Gets the resource stored by this resource pointer.
	LEAN_INLINE resource_type *const & get(void) const { return m_resource; }

	/// Gets the resource stored by this resource pointer.
	LEAN_INLINE resource_type& operator *() const { return *m_resource; }
	/// Gets the resource stored by this resource pointer.
	LEAN_INLINE resource_type* operator ->() const { return m_resource; }
	/// Gets the resource stored by this resource pointer (getter compatibility).
	LEAN_INLINE resource_type* operator ()() const { return m_resource; }

	/// Gets the resource stored by this resource pointer.
	LEAN_INLINE operator resource_type*() const
	{
		LEAN_STATIC_ASSERT_MSG_ALT(!Critical,
			"Cannot implicitly cast critical reference, use unbind() for (insecure) storage.",
			Cannot_implicitly_cast_critical_reference__use_unbind_for_insecure_storage);
		return m_resource;
	}
};

/// Binds the given resource reference to a new resource pointer.
template <class Resource>
LEAN_INLINE resource_ptr<Resource, true> bind_resource(Resource *resource)
{
	// Visual C++ won't inline delegating function calls
	return resource_ptr<Resource, true>(resource, bind_reference);
}

/// Binds a new reference of the given resource to a resource pointer.
template <class Resource>
LEAN_INLINE resource_ptr<Resource, true> secure_resource(Resource *resource)
{
	// Visual C++ won't inline delegating function calls
	return resource_ptr<Resource, true>(resource);
}

#ifdef DOXYGEN_READ_THIS
	/// Creates a new resource using operator new.
	template <class Resource>
	resource_ptr<Resource, true> make_resource(...);
#else
	#define LEAN_MAKE_RESOURCE_FUNCTION_TPARAMS class Resource
	#define LEAN_MAKE_RESOURCE_FUNCTION_DECL inline resource_ptr<Resource, true> make_resource
	#define LEAN_MAKE_RESOURCE_FUNCTION_BODY(call) { return resource_ptr<Resource, true>( new Resource##call, bind_reference ); }
	LEAN_VARIADIC_TEMPLATE_T(LEAN_FORWARD, LEAN_MAKE_RESOURCE_FUNCTION_DECL, LEAN_MAKE_RESOURCE_FUNCTION_TPARAMS, LEAN_NOTHING, LEAN_MAKE_RESOURCE_FUNCTION_BODY)
#endif

struct new_resource_ptr_t { };
template <class T>
LEAN_INLINE resource_ptr<T, true> operator *(new_resource_ptr_t, T *p) { return resource_ptr<T, true>(p, bind_reference); }

} // namespace

using smart::resource_ptr;
using smart::bind_resource;
using smart::secure_resource;
using smart::make_resource;

} // namespace

#ifndef LEAN_NO_RESOURCE_PTR_NEW

/// @addtogroup ResourceMacros
/// @{

/// Modified operator new that returns a resource_ptr.
#define new_resource ::lean::smart::new_resource_ptr_t() * new

/// @}

#endif

#endif