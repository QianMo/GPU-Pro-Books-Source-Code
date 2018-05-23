/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_REF_COUNTER
#define LEAN_SMART_REF_COUNTER

#include <memory>
#include "../lean.h"
#include "../concurrent/atomic.h"
#include "../containers/allocator_aware.h"

namespace lean
{
namespace smart
{

/// Reference counter class that maintains strong and weak reference counts.
template < class Counter = long, class Allocator = std::allocator<Counter> >
class ref_counter
{
public:
	/// Type of the counter used by this class.
	typedef Counter counter_type;
	
private:
	/// Reference counter structure.
	class ref_counts : private containers::allocator_aware_base<typename Allocator::template rebind<ref_counts>::other>
	{
	private:
		typedef containers::allocator_aware_base<typename Allocator::template rebind<ref_counts>::other> base_type;

	public:
		/// Allocator type.
		typedef typename base_type::allocator_type allocator_type;

	protected:
		/// Constructor.
		ref_counts(const allocator_type& allocator, counter_type references, counter_type weakReferences)
			: base_type(allocator),
			references(references),
			weakReferences(weakReferences) { }
#ifndef LEAN_OPTIMIZE_DEFAULT_DESTRUCTOR
		// Destructor.
		~ref_counts() noexcept { }
#endif
	public:
		/// Reference counter.
		counter_type references;

		/// Weak reference count.
		counter_type weakReferences;

		/// Allocates and constructs a new internal reference counting object from the given parameters.
		static ref_counts* create(allocator_type allocator, counter_type references, counter_type weakReferences)
		{
			ref_counts *counts = allocator.allocate(1);

			try
			{
				new (counts) ref_counts(allocator, references, weakReferences);
			}
			catch (...)
			{
				allocator.deallocate(counts, 1);
				throw;
			}

			return counts;
		}

		/// Destructs and deallocates the given internal reference counting object.
		static void destroy(const ref_counts *counts)
		{
			LEAN_ASSERT(counts);

			ref_counts *countsPtr = const_cast<ref_counts*>(counts);

			allocator_type allocator( LEAN_MOVE(countsPtr->allocator()) );
			countsPtr->~ref_counts();

			allocator.deallocate(countsPtr, 1);
		}
	};

public:
	/// Type of the allocator used by this class.
	typedef typename ref_counts::allocator_type allocator_type;

private:
	// Reference counts
	ref_counts *m_counts;

	/// Acquires the given internal reference counting object.
	static ref_counts* acquire(ref_counts *counts)
	{
		if (counts)
			atomic_increment(counts->weakReferences);

		return counts;
	}

	/// Releases the given internal reference counting object.
	void release(ref_counts *counts)
	{
		// Clean up, if this is the last reference
		if (counts && !atomic_decrement(counts->weakReferences))
			ref_counts::destroy(counts);
	}

	/// Special constructor.
	explicit ref_counter(ref_counts *counts)
		: m_counts(counts) { }

public:
	/// Constructor.
	explicit ref_counter(counter_type references = 1)
		: m_counts( ref_counts::create(typename ref_counts::allocator_type(), references, 1) ) { }
	/// Constructor.
	explicit ref_counter(const allocator_type &allocator)
		: m_counts( ref_counts::create(allocator, 1, 1) ) { }
	/// Constructor.
	ref_counter(counter_type references, const allocator_type &allocator)
		: m_counts( ref_counts::create(allocator, references, 1) ) { }
	/// Copy constructor.
	ref_counter(const ref_counter& refCounter)
		: m_counts( acquire(refCounter.m_counts) ) { }
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Move constructor.
	ref_counter(ref_counter&& refCounter) noexcept
		: m_counts(std::move(refCounter.m_counts))
	{
		// Warning: this effectively "breaks" the other object
		refCounter.m_counts = nullptr;
	}
#endif
	/// Destructor.
	~ref_counter()
	{
		release(m_counts);
	}

	/// Gets a null reference counter that may only be copied from and assigned to.
	static LEAN_INLINE ref_counter null()
	{
		// Warning: this object is effectively "broken"
		return ref_counter(nullptr);
	}

	/// Assignment operator. <b>[ESA]</b>
	ref_counter& operator =(const ref_counter& refCounter)
	{
		// Don't re-assign the same
		if(m_counts != refCounter.m_counts)
		{
			ref_counts *prevReferences = m_counts;
			m_counts = acquire(refCounter.m_counts);
			release(prevReferences);
		}

		return *this;
	}
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	/// Move assignment operator. [ESA]
	ref_counter& operator =(ref_counter&& refCounter) noexcept
	{
		// Don't re-assign the same
		if(m_counts != refCounter.m_counts)
		{
			ref_counts *prevReferences = m_counts;

			m_counts = ::std::move(refCounter.m_counts);
			// Warning: this effectively "breaks" the other object
			refCounter.m_counts = nullptr;

			release(prevReferences);
		}

		return *this;
	}
#endif

	/// Increments the current reference count.
	bool increment_checked() const
	{
		LEAN_ASSERT(m_counts);

		for (;;)
		{
			counter_type references = (volatile counter_type&)m_counts->references;
			
			// Make sure reference count has not become 0 yet
			if (references < 1)
				return false;
			// Make sure value has not changed until it has successfully been updated
			else if (atomic_test_and_set(m_counts->references, references, references + 1))
				return true;
		}

		LEAN_ASSERT_UNREACHABLE();
	}
	/// Increments the current reference count.
	counter_type increment() const
	{
		LEAN_ASSERT(m_counts);

		return atomic_increment(m_counts->references);
	}
	/// Decrements the current reference count.
	counter_type decrement() const
	{
		LEAN_ASSERT(m_counts);

		return atomic_decrement(m_counts->references);
	}

	/// Checks whether this reference counter is in a valid state.
	LEAN_INLINE bool is_null() const { return (m_counts == nullptr); }

	/// Gets the current reference count.
	LEAN_INLINE counter_type count() const { LEAN_ASSERT(m_counts); return m_counts->references; };
	/// Gets whether the reference-counted object still exists (<==> reference count > 0).
	LEAN_INLINE bool valid() const { LEAN_ASSERT(m_counts); return (m_counts->references > 0); };

	/// Gets the current weak reference count.
	LEAN_INLINE counter_type weak_count() const { LEAN_ASSERT(m_counts); return m_counts->weakReferences; };

	/// Increments the current reference count.
	LEAN_INLINE ref_counter& operator ++() { ++const_cast<const ref_counter&>(*this); return *this; };
	/// Increments the current reference count.
	LEAN_INLINE const ref_counter& operator ++() const { increment(); return *this; };
	/// Decrements the current reference count.
	LEAN_INLINE ref_counter& operator --() { --const_cast<const ref_counter&>(*this); return *this; };
	/// Decrements the current reference count.
	LEAN_INLINE const ref_counter& operator --() const { decrement(); return *this; };
	/// Increments the current reference count.
	LEAN_INLINE counter_type operator ++(int) const { counter_type prevCount = count(); ++(*this); return prevCount; };
	/// Decrements the current reference count.
	LEAN_INLINE counter_type operator --(int) const { counter_type prevCount = count(); --(*this); return prevCount;};

	/// Gets the current reference count.
	LEAN_INLINE operator counter_type() const { return count(); };
	/// Gets whether the reference-counted object still exists (<==> reference count > 0).
	LEAN_INLINE operator bool() const { return valid(); };
};

} // namespace

using smart::ref_counter;

} // namespace

#endif