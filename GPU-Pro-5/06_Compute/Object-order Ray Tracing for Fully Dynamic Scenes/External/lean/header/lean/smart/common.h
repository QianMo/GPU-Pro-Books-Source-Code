/*****************************************************/
/* lean Smart                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_SMART_COMMON
#define LEAN_SMART_COMMON

#include "../lean.h"

namespace lean
{
namespace smart
{

#ifdef DOXYGEN_READ_THIS
/// Allows for the binding of existing COM object / resource references on smart pointer construction.
enum bind_reference_t
{
	bind_reference ///< Allows for the binding of existing COM object / resource references on smart pointer construction.
};
#else
/// Allows for the binding of existing COM object / resource references on smart pointer construction.
typedef consume_t bind_reference_t;
/// Allows for the binding of existing COM object / resource references on smart pointer construction.
const bind_reference_t bind_reference = consume;
#endif

/// Determines whether a reference is in a critical state.
enum reference_state_t
{
	stable_ref = false,		///< Reference is stable, referenced object cannot accidentally be destroyed.
	critical_ref = true		///< Reference is critical, referenced object might accidentally get destroyed.
};

/// Determines wrapper semantic.
enum ptr_semantic_t
{
	val_sem = false,	///< Value semantics (const-transitive, non-null).
	ptr_sem = true		///< Pointer semantics (const-intransitive, null).
};

#ifndef LEAN0X_NO_DECLTYPE
namespace impl
{
	template <class Type>
	struct move_ref_deref_helper
	{
		template <class T>
		static auto maybe_type(T *v) -> decltype(**v);
		static void maybe_type(...);

		typedef decltype(maybe_type(static_cast<Type*>(nullptr))) type;
	};
}
#endif

/// Move reference wrapper.
template <class Type>
struct move_ref
{
	/// Moved object type.
	typedef Type value_type;
	/// Pointer to the object to be moved.
	value_type *const moved_ptr;

	/// Constructs a move reference object for the given value.
	move_ref(value_type &v)
		: moved_ptr(addressof(v)) { }

	/// Gets a reference to the object to be moved.
	value_type& moved() const { return *moved_ptr; }
};

} // namespace

using smart::bind_reference_t;
using smart::bind_reference;

using smart::ptr_semantic_t;
using smart::val_sem;
using smart::ptr_sem;

using smart::reference_state_t;
using smart::stable_ref;
using smart::critical_ref;

} // namespace

#endif