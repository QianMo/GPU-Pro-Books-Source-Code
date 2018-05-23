/*****************************************************/
/* lean Containers              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONTAINERS_VECTOR_POLICIES
#define LEAN_CONTAINERS_VECTOR_POLICIES

#include "../lean.h"
#include "construction.h"

namespace lean 
{
namespace containers
{

/// Defines construction policies for vector classes.
namespace vector_policies
{
	/// Simple vector element construction policy.
	template <bool RawMove = false, bool RawCopy = false, bool NoDestruct = false, bool ZeroInit = false, bool NoInit = false>
	struct policy
	{
		/// Specifies whether memory containing constructed elements may be moved as a whole, without invoking the contained elements' copy or move constructors.
		static const bool raw_move = RawMove;
		/// Specifies whether memory containing constructed elements may be copied as a whole, without invoking the contained elements' copy constructors.
		static const bool raw_copy = RawCopy;
		/// Specifies whether memory containing constructed elements may be freed as a whole, without invoking the contained elements' destructors.
		static const bool no_destruct = NoDestruct;
		/// Specifies whether memory may be initialized with zeroes.
		static const bool zero_init = ZeroInit;
		/// Specifies whether memory needs to be initialized at all.
		static const bool no_init = NoInit;

		/// Move construction tag matching raw_move.
		typedef typename conditional_type<raw_move, trivial_construction_t, nontrivial_construction_t>::type move_tag;
		/// Copy construction tag matching raw_copy.
		typedef typename conditional_type<raw_copy, trivial_construction_t, nontrivial_construction_t>::type copy_tag;
		/// Destruction tag matching no_destruct.
		typedef typename conditional_type<no_destruct, trivial_construction_t, nontrivial_construction_t>::type destruct_tag;
		/// Construction tag matching no_construct.
		typedef typename conditional_type<zero_init, trivial_construction_t, nontrivial_construction_t>::type construct_tag;
	};

	/// Default element construction policy.
	typedef policy<> nonpod;
	/// Semi-POD element construction policy (raw move, yet properly copied and destructed).
	typedef policy<true> semipod;
	/// Initialize-POD element construction policy (raw move, no destruction, yet properly constructed).
	typedef policy<true, true, true> inipod;
	/// POD element zero initialization policy.
	typedef policy<true, true, true, true> pod;
	/// POD element no-construction policy.
	typedef policy<true, true, true, true, true> uninipod;
}

/// Default vector binder.
template <template <class E, class A> class Vector, template <class T> class Allocator>
struct vector_binder
{
	/// Constructs a vector type from the given element type.
	template <class Type>
	struct rebind
	{
		typedef Allocator<Type> allocator_type;
		typedef vector_policies::nonpod policy;
		typedef Vector<Type, allocator_type> type;
	};
};

/// Default vector binder.
template <template <class E, class P, class A> class Vector, class Policy, template <class T> class Allocator>
struct policy_vector_binder
{
	/// Constructs a vector type from the given element type.
	template <class Type>
	struct rebind
	{
		typedef Allocator<Type> allocator_type;
		typedef Policy policy;
		typedef Vector<Type, policy, allocator_type> type;
	};
};

} // namespace

namespace vector_policies = containers::vector_policies;

} // namespace

#endif