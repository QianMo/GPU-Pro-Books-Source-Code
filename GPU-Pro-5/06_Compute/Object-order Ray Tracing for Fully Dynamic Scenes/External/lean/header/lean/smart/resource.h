/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_RESOURCE
#define LEAN_SMART_RESOURCE

#include "ref_counter.h"

/// @addtogroup ResourceMacros Resource macros
/// @see lean::smart::resource
/// @{

/// Allows resource pointers access to a class' reference counter.
#define LEAN_MAKE_RESOURCE \
	template <class Resource, bool Critical> \
	friend class lean::smart::resource_ptr; \
	template <class Resource> \
	friend class lean::smart::weak_resource_ptr;

/// Overrides the @code resource_interface@endcode methods required for a resource interface to treated like a resource.
/// WARNING: Changes class visibility.
#define LEAN_RENEW_RESOURCE \
	public: \
		const lean_resource_base::ref_counter_type& ref_counter() const { return lean_resource_base::ref_counter(); } \
		lean_resource_base::ref_counter_type::counter_type ref_count() const { return lean_resource_base::ref_count(); } \
	private:

/// @}

namespace lean
{
namespace smart
{

// Prototypes
template <class Resource, bool Critical>
class resource_ptr;
template <class Resource>
class weak_resource_ptr;

/// Base class that may be used to tag a specific class as a resource.
template < class Counter = long, class Allocator = std::allocator<Counter>, bool Lazy = false >
class resource
{
	LEAN_MAKE_RESOURCE

public:
	/// Resource base type.
	typedef resource lean_resource_base;
	/// Reference counter type.
	typedef ref_counter<Counter, Allocator> ref_counter_type;

protected:
	/// Reference counter type.
	mutable ref_counter_type m_refCounter;

private:
	/// Creates a new reference counter.
	LEAN_NOINLINE void create_ref_counter() const
	{
		LEAN_ASSERT(m_refCounter.is_null());
		m_refCounter = ref_counter_type();
	}

	/// Returns the reference counter of this resource.
	const ref_counter_type& ref_counter() const
	{
		if (Lazy)
		{
			if (m_refCounter.is_null())
				create_ref_counter();
		}
		return m_refCounter;
	}

protected:
	/// Default constructor.
	resource()
		: m_refCounter( Lazy ? ref_counter_type::null() : ref_counter_type() ) { }
	/// Allocator constructor.
	resource(const typename ref_counter_type::allocator_type& allocator)
		: m_refCounter(allocator) { }
	/// Copy constructor.
	LEAN_INLINE resource(const resource& right) noexcept { } // TODO: Copy allocator?
	/// Assignment operator.
	LEAN_INLINE resource& operator =(const resource& right) noexcept { return *this; }
#ifndef LEAN_OPTIMIZE_DEFAULT_DESTRUCTOR
	/// Destructor.
	LEAN_INLINE ~resource() noexcept { }
#endif

public:
	/// Gets the reference count.
	typename ref_counter_type::counter_type ref_count() const { return m_refCounter.count(); }
};

/// Interface that may be used to tag a specific class as a resource.
template < class Counter = long, class Allocator = std::allocator<Counter> >
class LEAN_INTERFACE resource_interface
{
	LEAN_SHARED_INTERFACE_BEHAVIOR(resource_interface)

public:
	/// Reference counter type.
	typedef ref_counter<Counter, Allocator> ref_counter_type;
	/// Returns the reference counter of this resource.
	virtual const ref_counter_type& ref_counter() const = 0;
	/// Gets the reference count.
	virtual typename ref_counter_type::counter_type ref_count() const = 0;
};

} // namespace

using smart::resource;
using smart::resource_interface;

} // namespace

#endif