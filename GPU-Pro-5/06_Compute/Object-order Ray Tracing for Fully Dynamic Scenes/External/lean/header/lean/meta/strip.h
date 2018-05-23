/*****************************************************/
/* lean Meta                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_META_STRIP
#define LEAN_META_STRIP

#include "../cpp0x.h"

namespace lean
{
namespace meta
{

/// Strips a reference from the given type.
template <class Type>
struct strip_reference
{
	/// Type without reference.
	typedef Type type;
	/// True, if any reference stripped.
	static const bool stripped = false;

	/// Adds any reference stripped.
	template <class Other>
	struct undo
	{
		/// Type with reference, if any reference stripped.
		typedef Other type;
	};
};

#ifndef DOXYGEN_SKIP_THIS

template <class Type>
struct strip_reference<Type&>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef Other& type; };
};

#ifndef LEAN0X_NO_RVALUE_REFERENCES
template <class Type>
struct strip_reference<Type&&>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef Other&& type; };
};
#endif

#endif

/// Inherits the stripped type.
template <class Type>
struct inh_strip_reference : public strip_reference<Type>::type { };

/// Strips a const modifier from the given type.
template <class Type>
struct strip_const
{
	/// Type without const modifier.
	typedef Type type;
	/// True, if any modifiers stripped.
	static const bool stripped = false;

	/// Adds any modifiers stripped.
	template <class Other>
	struct undo
	{
		/// Type with modifier, if any modifier stripped.
		typedef Other type;
	};
};

#ifndef DOXYGEN_SKIP_THIS

template <class Type>
struct strip_const<const Type>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef const Other type; };
};

#endif

/// Strips a volatile modifier from the given type.
template <class Type>
struct strip_volatile
{
	/// Type without volatile modifier.
	typedef Type type;
	/// True, if any modifiers stripped.
	static const bool stripped = false;

	/// Adds any modifiers stripped.
	template <class Other>
	struct undo
	{
		/// Type with modifier, if any modifier stripped.
		typedef Other type;
	};
};

#ifndef DOXYGEN_SKIP_THIS

template <class Type>
struct strip_volatile<volatile Type>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef volatile Other type; };
};

#endif

/// Strips cv-modifiers from the given type.
template <class Type>
struct strip_modifiers
{
	/// Type without cv-modifiers.
	typedef typename strip_volatile<typename strip_const<Type>::type>::type type;
	/// True, if any modifiers stripped.
	static const bool stripped = strip_volatile<Type>::stripped || strip_const<Type>::stripped;

	/// Adds any modifiers stripped.
	template <class Other>
	struct undo
	{
		/// Type with modifiers, if any modifiers stripped.
		typedef typename strip_const<Type>::template undo<typename strip_volatile<Type>::template undo<Other>::type>::type type;
	};
};

/// Strips cv-modifiers and references from the given type.
template <class Type>
struct strip_modref
{
	/// Value type.
	typedef typename strip_reference<Type>::type value_type;
	/// Type without cv-modifiers and references.
	typedef typename strip_modifiers<value_type>::type type;
	/// True, if any modifiers or references stripped.
	static const bool stripped = strip_reference<Type>::stripped || strip_modifiers<value_type>::stripped;

	/// Adds any modifiers and references stripped.
	template <class Other>
	struct undo
	{
		/// Type with modifiers and references, if any modifiers or references stripped.
		typedef typename strip_reference<Type>::template undo<typename strip_modifiers<value_type>::template undo<Other>::type>::type type;
	};
};

/// Inherits the stripped type.
template <class Type>
struct inh_strip_modref : public strip_modref<Type>::type { };

namespace impl
{

template <class Type>
struct do_strip_pointer
{
	typedef Type type;
	static const bool stripped = false;
	template <class Other>
	struct undo { typedef Other type; };
};

template <class Type>
struct do_strip_pointer<Type*>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef Other* type; };
};

} // namespace

/// Strips a pointer from the given type.
template <class Type>
struct strip_pointer
{
	/// Pointer type without modifiers.
	typedef typename strip_modifiers<Type>::type pointer;
	/// Type without pointer.
	typedef typename impl::do_strip_pointer<pointer>::type type;
	/// True, if any pointer stripped.
	static const bool stripped = impl::do_strip_pointer<pointer>::stripped;

	/// Adds any pointer stripped.
	template <class Other>
	struct undo
	{
		/// Pointer type without modifiers.
		typedef typename impl::do_strip_pointer<pointer>::template undo<Other>::type pointer;
		/// Type with pointer, if any pointer stripped.
		typedef typename strip_modifiers<Type>::template undo<pointer>::type type;
	};
};

/// Inherits the stripped type.
template <class Type>
struct inh_strip_pointer : public strip_pointer<Type>::type { };

namespace impl
{

template <class Type>
struct do_strip_array
{
	typedef Type type;
	static const bool stripped = false;
	template <class Other>
	struct undo { typedef Other type; };
};

template <class Type>
struct do_strip_array<Type[]>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef Other type[]; };
};

template <class Type, size_t Size>
struct do_strip_array<Type[Size]>
{
	typedef Type type;
	static const bool stripped = true;
	template <class Other>
	struct undo { typedef Other type[Size]; };
};

} // namespace

/// Strips a array from the given type.
template <class Type>
struct strip_array
{
	/// Pointer type without modifiers.
	typedef typename strip_modifiers<Type>::type array;
	/// Type without array.
	typedef typename impl::do_strip_array<array>::type type;
	/// True, if any array stripped.
	static const bool stripped = impl::do_strip_array<array>::stripped;

	/// Adds any array stripped.
	template <class Other>
	struct undo
	{
		/// Pointer type without modifiers.
		typedef typename impl::do_strip_array<array>::template undo<Other>::type array;
		/// Type with array, if any array stripped.
		typedef typename strip_modifiers<Type>::template undo<array>::type type;
	};
};

/// Inherits the stripped type.
template <class Type>
struct inh_strip_array : public strip_array<Type>::type { };

namespace impl
{

template <class Type, bool Continue = true>
struct do_rec_strip_modifiers
{
	typedef Type type;
	static const bool stripped = false;
};

template <class Type>
struct do_rec_strip_modifiers<Type, true>
{
	typedef strip_modref<Type> modref_stripper;
	typedef strip_array<typename modref_stripper::type> array_stripper;
	typedef strip_pointer<typename array_stripper::type> pointer_stripper;
	static const bool continue_stripping = array_stripper::stripped || pointer_stripper::stripped;

	static const bool stripped = modref_stripper::stripped
		|| do_rec_strip_modifiers<typename pointer_stripper::type, continue_stripping>::stripped;

	typedef typename array_stripper::template undo<
				typename pointer_stripper::template undo<
					typename do_rec_strip_modifiers<typename pointer_stripper::type, continue_stripping>::type
				>::type
			>::type type;
};

} // namespace

/// Recursively strips all modifiers from the given type.
template <class Type>
struct rec_strip_modifiers
{
	/// Type without modifieres.
	typedef typename impl::do_rec_strip_modifiers<Type>::type type;
	/// True, if any modifiers stripped.
	static const bool stripped = impl::do_rec_strip_modifiers<Type>::stripped;
};

} // namespace

using meta::strip_array;
using meta::strip_pointer;
using meta::strip_reference;
using meta::strip_const;
using meta::strip_volatile;
using meta::strip_modifiers;
using meta::strip_modref;
using meta::rec_strip_modifiers;

using meta::inh_strip_pointer;
using meta::inh_strip_array;
using meta::inh_strip_reference;
using meta::inh_strip_modref;

} // namespace

#endif