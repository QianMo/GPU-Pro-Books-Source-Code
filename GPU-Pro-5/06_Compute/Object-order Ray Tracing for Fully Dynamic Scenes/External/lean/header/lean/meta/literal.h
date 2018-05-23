/*****************************************************/
/* lean Meta                    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_META_LITERAL
#define LEAN_META_LITERAL

namespace lean
{
namespace meta
{

/// Defines value as Value.
template <class Type, Type Value>
struct literal_constant
{
	static const Type value = Value;
};

/// Helper struct that may be used to pass constant booleans (compile-time literals).
template <bool Value>
struct literal_bool : literal_constant<bool, Value> { };
/// Helper struct that may be used to pass constant integers (compile-time literals).
template <int Value>
struct literal_int : literal_constant<int, Value> { };

/// Defines value as false.
struct false_type : literal_bool<false> { };
/// Defines value as true.
struct true_type : literal_bool<true> { };

/// Defines a false literal, ignoring any template arguments.
template <class T>
struct dependent_false : false_type { };
/// Defines a false literal, ignoring any template arguments.
template <int N>
struct int_dependent_false : false_type { };

} // namespace

using meta::literal_constant;
using meta::literal_bool;
using meta::literal_int;
using meta::false_type;
using meta::true_type;
using meta::dependent_false;
using meta::int_dependent_false;

} // namespace

#endif