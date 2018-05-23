/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_ALLOCATOR_AWARE
#define LEAN_CONTAINERS_ALLOCATOR_AWARE

#include "../lean.h"
#include "../meta/type_traits.h"

namespace lean 
{
namespace containers
{

/// Stores an allocator object if the given allocator type is non-empty.
template <class Allocator, bool IsEmpty = is_empty<Allocator>::value>
class allocator_aware_base
{
public:
	typedef Allocator allocator_type;

private:
	Allocator m_allocator;

protected:
	/// Retrieves the allocator.
	LEAN_INLINE Allocator& allocator() { return m_allocator; };
	/// Retrieves the allocator.
	LEAN_INLINE const Allocator& allocator() const { return m_allocator; };

	/// Allows for unrestricted usage of the stored allocator object.
	struct allocator_ref
	{
		Allocator &allocator;

		LEAN_INLINE allocator_ref(allocator_aware_base &self)
			: allocator(self.allocator()) { }
	};

	/// Default allocator.
	LEAN_INLINE allocator_aware_base() { }
	/// Exeplicit allocator construction.
	LEAN_INLINE explicit allocator_aware_base(const Allocator &allocator)
		: m_allocator(allocator) { }

#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Move construction.
	LEAN_INLINE allocator_aware_base(allocator_aware_base &&right) noexcept
		: m_allocator(std::move(right.m_allocator)) { }
	/// Move assignment.
	LEAN_INLINE allocator_aware_base& operator =(allocator_aware_base &&right) noexcept
	{
		m_allocator = std::move(right.m_allocator);
		return *this;
	}
#endif

	/// Swaps the allocators of this allocator-aware object and the given allocator-aware object.
	LEAN_INLINE void swap(allocator_aware_base &right) noexcept
	{
		using std::swap;

		swap(m_allocator, right.m_allocator);
	}
};

/// Stores an allocator object if the given allocator type is non-empty.
template <class Allocator>
class allocator_aware_base<Allocator, true>
{
public:
	typedef Allocator allocator_type;

protected:
	/// Retrieves the allocator.
	LEAN_INLINE Allocator allocator() const { return Allocator(); };

	/// Allows for unrestricted usage of the stored allocator object.
	struct allocator_ref
	{
		Allocator allocator;

		LEAN_INLINE allocator_ref(allocator_aware_base &self) { }
	};

	/// Default allocator.
	LEAN_INLINE allocator_aware_base() noexcept { }
	/// Exeplicit allocator construction.
	LEAN_INLINE explicit allocator_aware_base(const Allocator &allocator) noexcept { }

	/// Swaps the allocators of this allocator-aware object and the given allocator-aware object.
	LEAN_INLINE void swap(allocator_aware_base &right) noexcept { }
};

} // namespace

} // namespace

#endif