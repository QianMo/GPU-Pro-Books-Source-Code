/*****************************************************/
/* lean Meta                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_META_TYPE_TRAITS
#define LEAN_META_TYPE_TRAITS

#include "strip.h"

#ifndef LEAN0X_NO_STL
#include <type_traits>
#endif

namespace lean
{
namespace meta
{

/// Checks if the given integer type is unsigned.
template <class Integer>
struct is_unsigned
{
	static const Integer min = Integer(0);
	static const Integer max = Integer(-1);

	/// True, if @code Integer@endcode is unsigned, false otherwise.
	static const bool value = (max > min);
};

/// True if Type is derived from Base, false otherwise.
template <class Type, class Base>
struct is_derived
{
private:
	typedef char yes[1];
	typedef char no[2];

	static yes& sfinae_check(Base*);
	static no& sfinae_check(void*);

public:
	/// Specifies whether Type is derived from Base.
	static const bool value = (
		sizeof( is_derived::sfinae_check( static_cast<Type*>(nullptr) ) )
		==
		sizeof(yes) );
};

#ifdef LEAN0X_NO_STL

/// True if Type is an empty class.
template <class Type>
struct is_empty
{
private:
	// MONITOR: Make use of empty base class optimization
	template <class T>
	struct check : private T { int i; };

public:
	/// Specifies whether Type is empty.
	static const bool value = (sizeof(check<Type>) == sizeof(int));
};

#else

using std::is_empty;

#endif

/// Returns an rval expression of type T.
template <class T>
T make_rval();
/// Returns an lval expression of type T.
template <class T>
T& make_lval();

template <size_t Size, class Type = void>
struct nonzero_to_type { typedef Type type; };
template <class Type>
struct nonzero_to_type<0, Type> { };

} // namespace

using meta::is_unsigned;
using meta::is_derived;
using meta::is_empty;

using meta::make_rval;
using meta::make_lval;

/// True if Type defines the given type, false otherwise.
#define LEAN_DEFINE_HAS_TYPE(TypeName)													\
	template <class Type>																\
	class has_type_##TypeName															\
	{																					\
	private:																			\
		typedef char yes[1];															\
		typedef char no[2];																\
																						\
		template <class T>																\
		static yes& sfinae_check(T*, typename T::TypeName* = nullptr);					\
		static no& sfinae_check(...);													\
																						\
	public:																				\
		static const bool value =														\
			sizeof( has_type_##TypeName::sfinae_check( static_cast<Type*>(nullptr) ) )	\
			==																			\
			sizeof(yes);																\
	};

/// True if the given expression is valid, false otherwise. Use T for Type in your expression.
#define LEAN_DEFINE_IS_VALID(Name, Expr)																			\
	template <class Type>																							\
	class is_valid_##Name																							\
	{																												\
	public:																											\
		typedef char yes[1];																						\
		typedef char no[2];																							\
																													\
		template <class T>																							\
		static yes& sfinae_check(T*, char (*)[sizeof(Expr)] = nullptr);												\
		static no& sfinae_check(...);																				\
																													\
	public:																											\
		static const bool value =																					\
			sizeof( is_valid_##Name::sfinae_check( static_cast<Type*>(nullptr) ) )									\
			==																										\
			sizeof(yes);																							\
	};

/// True if Type defines the given type as X, false otherwise.
#define LEAN_DEFINE_HAS_TYPE_AND_X(TypeName, X)											\
	LEAN_DEFINE_HAS_TYPE(TypeName)														\
	template <class Type, bool HasType = has_type_##TypeName<Type>::value>				\
	struct has_type_and_##X##TypeName													\
	{																					\
		static const bool value = (Type::TypeName::value == X);							\
	};																					\
	template <class Type>																\
	struct has_type_and_##X##TypeName<Type, false>										\
	{																					\
		static const bool value = false;												\
	};
/// True if Type defines the given type as true, false otherwise.
#define LEAN_DEFINE_HAS_TYPE_AND_TRUE(TypeName) LEAN_DEFINE_HAS_TYPE_AND_X(TypeName, true)
/// True if Type defines the given type as false, false otherwise.
#define LEAN_DEFINE_HAS_TYPE_AND_FALSE(TypeName) LEAN_DEFINE_HAS_TYPE_AND_X(TypeName, false)
		
} // namespace

namespace lean
{
namespace meta
{

LEAN_DEFINE_HAS_TYPE(iterator_category);

/// Checks if the given type is an iterator.
template <class Type>
struct is_iterator
{
	/// True if the given type is an iterator.
	static const bool value = strip_pointer<typename strip_reference<Type>::type>::stripped
		|| has_type_iterator_category<typename strip_reference<Type>::type>::value;
};

#ifdef _MSC_VER

/// Checks if the given type is trivial.
template <class Type>
struct is_trivial
{
	/// True, if the given type is trivial.
	static const bool value = __has_trivial_constructor(Type) && __has_trivial_copy(Type) && __has_trivial_assign(Type) && __has_trivial_destructor(Type) || __is_pod(Type);
};

/// Checks if the given type is trivially copyable.
template <class Type>
struct is_trivially_copyable
{
	/// True, if the given type is trivial.
	static const bool value = __has_trivial_copy(Type) && __has_trivial_assign(Type) && __has_trivial_destructor(Type) || __is_pod(Type);
};

/// Checks if the given type is trivially destructible.
template <class Type>
struct is_trivially_destructible
{
	/// True, if the given type is trivial.
	static const bool value = __has_trivial_destructor(Type) || __is_pod(Type);
};

#else

using std::is_trivial;
using std::is_trivially_copyable;
using std::is_trivially_destructible;

#endif

} // namespace

using meta::is_iterator;

using meta::is_trivial;
using meta::is_trivially_copyable;
using meta::is_trivially_destructible;

} // namespace

#endif