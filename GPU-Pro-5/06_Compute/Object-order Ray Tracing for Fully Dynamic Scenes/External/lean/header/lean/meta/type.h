/*****************************************************/
/* lean Meta                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_META_TYPE
#define LEAN_META_TYPE

namespace lean
{
namespace meta
{

/// True if types are equal, false otherwise.
template <class Type1, class Type2>
struct is_equal
{
	/// True if types are equal, false otherwise.
	static const bool value = false;
};

#ifndef DOXYGEN_SKIP_THIS

template <class Type>
struct is_equal<Type, Type>
{
	static const bool value = true;
};

#endif

/// Empty base class.
class empty_base
{
protected:
#ifdef LEAN0X_NO_DELETE_METHODS
	LEAN_INLINE empty_base() noexcept { }
	LEAN_INLINE empty_base(const empty_base&) noexcept { }
	LEAN_INLINE empty_base& operator =(const empty_base&) noexcept { return *this; }
#ifndef LEAN_OPTIMIZE_DEFAULT_DESTRUCTOR
	LEAN_INLINE ~empty_base() noexcept { }
#endif
#else
	empty_base() noexcept = default;
	empty_base(const empty_base&) noexcept = default;
	empty_base& operator =(const empty_base&) noexcept = default;
	~empty_base() noexcept = default;
#endif
};

/// Redefines the given type.
template <class Type>
struct identity
{
	/// Type.
	typedef Type type;
};

/// Redefines the given type if true, empty otherwise.
template <bool Condition, class Type>
struct enable_if : identity<Type> { };
#ifndef DOXYGEN_SKIP_THIS
template <class Type>
struct enable_if<false, Type> { };
#endif

/// Redefines the given type if the given forward type is either an r-value-ref or a const ref of the given value type, empty otherwise.
template <class Value, class Forward, class Type>
struct enable_move { };
#ifndef DOXYGEN_SKIP_THIS
	template <class Value, class Type>
	struct enable_move<Value, const Value&, Type> : identity<Type> { };
	#ifndef LEAN0X_NO_RVALUE_REFERENCES
		template <class Value, class Type>
		struct enable_move<Value, Value&&, Type> : identity<Type> { };
	#endif
#endif

/// Redefines TrueType if condition fulfilled, FalseType otherwise.
template <bool Condition, class TrueType, class FalseType>
struct conditional_type : identity<FalseType> { };
#ifndef DOXYGEN_SKIP_THIS
template <class TrueType, class FalseType>
struct conditional_type<true, TrueType, FalseType> : identity<TrueType> { };
#endif

/// Redefines Type1 if not void, else Type2 if not void, nothing otherwise.
template <class Type1, class Type2>
struct first_non_void : identity<Type1> { };
#ifndef DOXYGEN_SKIP_THIS
template <class Type2>
struct first_non_void<void, Type2> : identity<Type2> { };
template <>
struct first_non_void<void, void> { };
#endif

/// Redefines FullType if complete (and derived from BaseType), BaseType otherwise.
template <class FullType, class BaseType>
struct complete_type_or_base
{
private:
	typedef char complete[1];
	typedef char incomplete[2];

	static complete& check_type(const BaseType*);
	static incomplete& check_type(const void*);

public:
	/// Specifies whether FullType is complete and was derived from BaseType.
	static const bool is_complete = (
		sizeof( check_type( static_cast<FullType*>(nullptr) ) )
		==
		sizeof(complete) );
	
	/// Full type if complete (and derived from BaseType), BaseType otherwise.
	typedef typename conditional_type<is_complete, FullType, BaseType>::type type;
};

/// Absorbs the given values.
LEAN_MAYBE_EXPORT void absorb(...);
/// Cast functions to this type.
typedef void (*absorbfun)();

} // namespace

using meta::is_equal;
using meta::identity;
using meta::empty_base;
using meta::enable_if;
using meta::enable_move;

using meta::conditional_type;
using meta::first_non_void;
using meta::complete_type_or_base;

using meta::absorb;
using meta::absorbfun;

} // namespace

#ifdef LEAN_INCLUDE_LINKED
#include "source/support.cpp"
#endif

#endif