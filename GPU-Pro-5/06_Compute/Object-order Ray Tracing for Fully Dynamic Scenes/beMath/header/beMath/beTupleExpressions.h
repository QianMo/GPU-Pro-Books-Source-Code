/*****************************************************/
/* breeze Engine Math Module    (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef BE_MATH_TUPLE_EXPRESSIONS
#define BE_MATH_TUPLE_EXPRESSIONS

#include "beMath.h"
#include <lean/meta/strip.h>

namespace beMath
{

namespace Impl
{

struct op_identity
{
	template <class Result, class Operand>
	static Result apply(const Operand &operand) { return operand; }
};
struct op_negate
{
	template <class Result, class Operand>
	static Result apply(const Operand &operand) { return -operand; }
};
struct op_add
{
	template <class Result, class Left, class Right>
	static Result apply(const Left &left, const Right &right) { return left + right; }
};
struct op_subtract
{
	template <class Result, class Left, class Right>
	static Result apply(const Left &left, const Right &right) { return left - right; }
};
struct op_multiply
{
	template <class Result, class Left, class Right>
	static Result apply(const Left &left, const Right &right) { return left * right; }
};
struct op_divide
{
	template <class Result, class Left, class Right>
	static Result apply(const Left &left, const Right &right) { return left / right; }
};
struct op_modulo
{
	template <class Result, class Left, class Right>
	static Result apply(const Left &left, const Right &right) { return left % right; }
};

struct acc_tuple
{
	template <class Result, class Operand, class Index>
	LEAN_INLINE static Result component(const Operand &operand, Index n) { return operand[n]; }
};

struct acc_scalar
{
	template <class Result, class Operand, class Index>
	LEAN_INLINE static Result component(const Operand &operand, Index) { return operand; }
};

/// Tuple expression mask class.
template <class Derived, class Element, size_t Count>
class tuple_expr_mask
{
protected:
	tuple_expr_mask() { }
	tuple_expr_mask(const tuple_expr_mask&) { }
	tuple_expr_mask& operator =(const tuple_expr_mask&) { return *this; }

public:
	/// Most derived type.
	typedef Derived expr_type;
	/// Value type.
	typedef Element value_type;
	/// Size type.
	typedef size_t size_type;
	/// Number of elements.
	static const size_type count = Count;

	/// Returns a pointer to the most-derived expression.
	LEAN_INLINE expr_type& expression() { return static_cast<expr_type&>(*this); }
	/// Returns a pointer to the most-derived expression.
	LEAN_INLINE const expr_type& expression() const { return static_cast<const expr_type&>(*this); }

	/// Accesses the n-th value.
	LEAN_INLINE value_type operator [](typename size_type n) const
	{
		return this->expression()[n];
	}

	/// Gets the number of elements in this tuple expression.
	LEAN_INLINE size_type size() const { return count; }

	/// Compares this tuple expression to the given tuple expression.
	template <class OtherExpr, class Element, size_t Count>
	bool operator ==(const tuple_expr_mask<OtherExpr, Element, Count> &right) const
	{
		for (size_t i = 0; i < Count; ++i)
			if ((*this)[i] != right[i])
				return false;
		return true;
	}
};

/// Tuple expression class.
template <class Derived, class Element, size_t Count, class SubExpression = Element>
class tuple_expr : public tuple_expr_mask<Derived, Element, Count>
{
protected:
	tuple_expr() { }
	tuple_expr(const tuple_expr&) { }
	tuple_expr& operator =(const tuple_expr&) { return *this; }

public:
	/// Mask type.
	typedef tuple_expr_mask<Derived, Element, Count> mask_type;
	/// Subexpression type.
	typedef SubExpression subexpr_type;

	/// Accesses the n-th subexpression.
	LEAN_INLINE subexpr_type operator [](typename mask_type::size_type n) const
	{
		return this->expression()[n];
	}
};

/// Checks whether the given type is a tuple expression.
template <class Expression>
class is_tuple_expr
{
private:
	typedef char yes[1];
	typedef char no[2];

	template <class Derived, class Element, size_t Count>
	static yes& check(const tuple_expr_mask<Derived, Element, Count>*);
	static no& check(...);

public:
	/// True, if Expression is a compatible tuple expression.
	static const bool value = (
		sizeof( check( static_cast<typename lean::strip_modref<Expression>::type*>(nullptr) ) )
		==
		sizeof(yes) );
};

/// Checks whether the given type has a compatible type.
template <class Expression>
class has_compatible_type
{
private:
	typedef char yes[1];
	typedef char no[2];

	template <class ActualExpression>
	static yes& check(const ActualExpression*, const typename lean::strip_modref<ActualExpression>::type::compatible_type* = nullptr);
	static no& check(...);

public:
	/// True, if Expression has a compatible type.
	static const bool value = (
		sizeof( check( static_cast<typename lean::strip_modref<Expression>::type*>(nullptr) ) )
		==
		sizeof(yes) );
};

/// Redefines the given type as compatible_type.
template <class CompatibleType>
struct inh_compatible_type
{
	/// Compatible type.
	typedef CompatibleType compatible_type; 
};

template <class Expression, bool HasCompatibleType>
struct inh_compatible_type_of_impl { };
template <class Expression>
struct inh_compatible_type_of_impl<Expression, true>
	: public inh_compatible_type<typename lean::inh_strip_modref<Expression>::compatible_type> { };

/// Redefines the given expression's compatible_type, if available.
template <class Expression>
class inh_compatible_type_of :
	public inh_compatible_type_of_impl<Expression, has_compatible_type<Expression>::value> { };

template <class Operator, class LeftNCVR, class RightNCVR>
struct result_of_op_default { };
template <class Operator, class OperandsNCVR>
struct result_of_op_default<Operator, OperandsNCVR, OperandsNCVR>
{
	typedef OperandsNCVR type;
};

template <bool IsTupleExpr, class Operator, class Left, class Right = Left>
struct result_of_op_tuple_expr_route
	: public result_of_op_default<
		Operator,
		typename lean::strip_modref<Left>::type,
		typename lean::strip_modref<Right>::type> { };

/// Defines the result type of the given operator applied to the given operand types.
template <class Operator, class Left, class Right = Left>
struct result_of_op
	: public result_of_op_tuple_expr_route<
		is_tuple_expr<Left>::value || is_tuple_expr<Right>::value,
		Operator, Left, Right > { };

template <class Operator, class Left, class Right = Left>
struct result_of_subexpr_op
{
	typedef typename result_of_op<
		Operator,
		typename lean::inh_strip_modref<Left>::subexpr_type,
		typename lean::inh_strip_modref<Right>::subexpr_type >::type type;
};

/// Unary per-component expression.
template <
	class Operator, class Operand, class Accessor = acc_tuple,
	class Element = typename lean::inh_strip_modref<Operand>::value_type,
	size_t Count = lean::inh_strip_modref<Operand>::count,
	class SubExpression = typename result_of_subexpr_op<Operator, Operand>::type,
	class InhCompatibleType = inh_compatible_type_of<Operand> >
class unary_tuple_expr
	: public tuple_expr< unary_tuple_expr<Operator, Operand, Accessor, Element, Count, SubExpression, InhCompatibleType>, Element, Count, SubExpression >,
	public InhCompatibleType
{
	template <class A, class B, class C, class D, size_t E, class F, class G>
	friend class unary_tuple_expr;

private:
	typedef tuple_expr< unary_tuple_expr, Element, Count, SubExpression > base_type;

	Operand m_operand;

public:
	typedef typename lean::conditional_type<
		lean::strip_reference<Operand>::stripped,
		Operand,
		const Operand&>::type operand_reference;
	
	unary_tuple_expr(operand_reference operand)
		: m_operand(operand) { }
	// Persistent copy constructor
	template <class OperandMod, class SubExpressionMod, class InhCompatibleTypeMod>
	unary_tuple_expr(const unary_tuple_expr<Operator, OperandMod, Accessor, Element, Count, SubExpressionMod, InhCompatibleTypeMod> &mod)
		: m_operand(mod.m_operand)
	{ }

	typedef typename base_type::subexpr_type subexpr_type;
	typedef typename base_type::size_type size_type;

	LEAN_INLINE subexpr_type operator [](size_type n) const
	{
		return Operator::apply<subexpr_type>( Accessor::component<subexpr_type>(m_operand, n) );
	}
};

/// Binary per-component expression.
template <
	class Operator, class Left, class Right,
	class Element = typename lean::inh_strip_modref<Left>::value_type,
	size_t Count = lean::inh_strip_modref<Left>::count,
	class SubExpression = typename result_of_subexpr_op<Operator, Left, Right>::type,
	class InhCompatibleType = inh_compatible_type_of<Left> >
class binary_tuple_expr
	: public tuple_expr< binary_tuple_expr<Operator, Left, Right, Element, Count, SubExpression, InhCompatibleType>, Element, Count, SubExpression >,
	public InhCompatibleType
{
	template <class A, class B, class C, class D, size_t E, class F, class G>
	friend class binary_tuple_expr;

private:
	typedef tuple_expr< binary_tuple_expr, Element, Count, SubExpression > base_type;

	Left m_left;
	Right m_right;

public:
	typedef typename lean::conditional_type<
		lean::strip_reference<Left>::stripped,
		Left,
		const Left&>::type left_reference;
	typedef typename lean::conditional_type<
		lean::strip_reference<Right>::stripped,
		Right,
		const Right&>::type right_reference;

	binary_tuple_expr(left_reference left, right_reference right)
		: m_left(left),
		m_right(right) { }
	// Persistent copy constructor
	template <class LeftMod, class RightMod, class SubExpressionMod, class InhCompatibleTypeMod>
	binary_tuple_expr(const binary_tuple_expr<Operator, LeftMod, RightMod, Element, Count, SubExpressionMod, InhCompatibleTypeMod> &mod)
		: m_left(mod.m_left),
		m_right(mod.m_right)
	{ }

	typedef typename base_type::subexpr_type subexpr_type;
	typedef typename base_type::size_type size_type;

	LEAN_INLINE subexpr_type operator [](size_type n) const
	{
		return Operator::apply<subexpr_type>(m_left[n], m_right[n]);
	}
};

template <class OperandNMR, class Operand, class CoOperandNMR, class CoOperand>
struct wrap_scalar_tuple_expr_impl
{
	typedef Operand type;
};
template <class CoOperandNMR, class Compatible, class CoOperand>
struct wrap_scalar_tuple_expr_impl<typename CoOperandNMR::compatible_type, Compatible, CoOperandNMR, CoOperand>
{
	typedef unary_tuple_expr<
		op_identity, Compatible, acc_scalar,
		typename CoOperandNMR::value_type, CoOperandNMR::count, Compatible,
		inh_compatible_type_of<CoOperandNMR> > type;
};
template <class Operand, class CoOperand>
struct wrap_scalar_tuple_expr : public wrap_scalar_tuple_expr_impl<
	typename lean::strip_modref<Operand>::type, Operand,
	typename lean::strip_modref<CoOperand>::type, CoOperand> { };

template <class Operator, class Left, class Right>
struct binary_tuple_expr_t
{
	typedef binary_tuple_expr<Operator,
		typename wrap_scalar_tuple_expr<Left, Right>::type,
		typename wrap_scalar_tuple_expr<Right, Left>::type> type;
};

/// Defines the result type of the given operator applied to the given tuple expression types.
template <class Operator, class Left, class Right = Left>
struct result_of_op_tuple_expr
{
	typedef typename binary_tuple_expr_t<Operator, Left, Right>::type type;
};
template <class Operand>
struct result_of_op_tuple_expr<op_identity, Operand, Operand>
{
	typedef unary_tuple_expr<op_identity, Operand> type;
};
template <class Operand>
struct result_of_op_tuple_expr<op_negate, Operand, Operand>
{
	typedef unary_tuple_expr<op_negate, Operand> type;
};

// Redirect if expression contains tuple expressions
template <class Operator, class Left, class Right>
struct result_of_op_tuple_expr_route<true, Operator, Left, Right>
	: public result_of_op_tuple_expr<Operator, Left, Right> { };

/// Makes a unary tuple expression from the given tuple expressions.
template <class Operator, class Operand, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<Operator, const Operand&>::type make_unary_tuple_expr(
	const tuple_expr_mask<Operand, Element, Count> &operand)
{
	return typename result_of_op_tuple_expr<Operator, const Operand&>::type( operand.expression() );
}
/// Makes a binary tuple expression from the given tuple expressions.
template <class Operator, class Left, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<Operator, const Left&, const Right&>::type make_binary_tuple_expr(
	const tuple_expr_mask<Left, Element, Count> &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return typename result_of_op_tuple_expr<Operator, const Left&, const Right&>::type(
		left.expression(),
		right.expression() );
}

/// Negates the given value.
template <class Operand, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_negate, const Operand&>::type operator -(
	const tuple_expr_mask<Operand, Element, Count> &operand)
{
	return make_unary_tuple_expr<op_negate>(operand);
}

/// Adds the given value to this values.
template <class Left, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_add, const Left&, const Right&>::type operator +(
	const tuple_expr_mask<Left, Element, Count> &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_tuple_expr<op_add>(left, right);
}

/// Subtracts the given right value from the given left value.
template <class Left, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_subtract, const Left&, const Right&>::type operator -(
	const tuple_expr_mask<Left, Element, Count> &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_tuple_expr<op_subtract>(left, right);
}

/// Multiplies the given left value by the given right value.
template <class Left, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_multiply, const Left&, const Right&>::type operator *(
	const tuple_expr_mask<Left, Element, Count> &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_tuple_expr<op_multiply>(left, right);
}

/// Multiplies the given left value by the given right value.
template <class Left, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_divide, const Left&, const Right&>::type operator /(
	const tuple_expr_mask<Left, Element, Count> &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_tuple_expr<op_divide>(left, right);
}

/// Multiplies the given left value by the given right value.
template <class Left, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_modulo, const Left&, const Right&>::type operator %(
	const tuple_expr_mask<Left, Element, Count> &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_tuple_expr<op_modulo>(left, right);
}

/// Makes a binary tuple expression from the given tuple expression and scalar.
template <class Operator, class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<Operator, const typename Right::compatible_type&, const Right&>::type make_binary_mixed_tuple_expr(
	const typename Right::compatible_type &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return typename result_of_op_tuple_expr<Operator, const typename Right::compatible_type&, const Right&>::type(
		typename wrap_scalar_tuple_expr<const typename Right::compatible_type&, Right>::type(left),
		right.expression() );
}
/// Makes a binary tuple expression from the given tuple expression and scalar.
template <class Operator, class Left, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<Operator, const Left&, const typename Left::compatible_type&>::type make_binary_mixed_tuple_expr(
	const tuple_expr_mask<Left, Element, Count> &left,
	const typename Left::compatible_type &right)
{
	return typename result_of_op_tuple_expr<Operator, const Left&, const typename Left::compatible_type&>::type(
		left.expression(),
		typename wrap_scalar_tuple_expr<const typename Left::compatible_type&, Left>::type(right) );
}

/// Adds the given value to this values.
template <class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_add, const typename Right::compatible_type&, const Right&>::type operator +(
	const typename Right::compatible_type &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_mixed_tuple_expr<op_add>(left, right);
}

/// Subtracts the given right value from the given left value.
template <class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_subtract, const typename Right::compatible_type&, const Right&>::type operator -(
	const typename Right::compatible_type &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_mixed_tuple_expr<op_subtract>(left, right);
}

/// Multiplies the given left value by the given right value.
template <class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_multiply, const typename Right::compatible_type&, const Right&>::type operator *(
	const typename Right::compatible_type &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_mixed_tuple_expr<op_multiply>(left, right);
}

/// Multiplies the given left value by the given right value.
template <class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_divide, const typename Right::compatible_type&, const Right&>::type operator /(
	const typename Right::compatible_type &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_mixed_tuple_expr<op_divide>(left, right);
}

/// Multiplies the given left value by the given right value.
template <class Right, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_modulo, const typename Right::compatible_type&, const Right&>::type operator %(
	const typename Right::compatible_type &left,
	const tuple_expr_mask<Right, Element, Count> &right)
{
	return make_binary_mixed_tuple_expr<op_modulo>(left, right);
}

/// Adds the given value to this values.
template <class Left, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_add, const Left&, const typename Left::compatible_type&>::type operator +(
	const tuple_expr_mask<Left, Element, Count> &left,
	const typename Left::compatible_type &right)
{
	return make_binary_mixed_tuple_expr<op_add>(left, right);
}

/// Subtracts the given left value from the given right value.
template <class Left, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_subtract, const Left&, const typename Left::compatible_type&>::type operator -(
	const tuple_expr_mask<Left, Element, Count> &left,
	const typename Left::compatible_type &right)
{
	return make_binary_mixed_tuple_expr<op_subtract>(left, right);
}

/// Multiplies the given right value by the given left value.
template <class Left, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_multiply, const Left&, const typename Left::compatible_type&>::type operator *(
	const tuple_expr_mask<Left, Element, Count> &left,
	const typename Left::compatible_type &right)
{
	return make_binary_mixed_tuple_expr<op_multiply>(left, right);
}

/// Multiplies the given right value by the given left value.
template <class Left, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_divide, const Left&, const typename Left::compatible_type&>::type operator /(
	const tuple_expr_mask<Left, Element, Count> &left,
	const typename Left::compatible_type &right)
{
	return make_binary_mixed_tuple_expr<op_divide>(left, right);
}

/// Multiplies the given right value by the given left value.
template <class Left, class Element, size_t Count>
LEAN_INLINE typename result_of_op_tuple_expr<op_modulo, const Left&, const typename Left::compatible_type&>::type operator %(
	const tuple_expr_mask<Left, Element, Count> &left,
	const typename Left::compatible_type &right)
{
	return make_binary_mixed_tuple_expr<op_modulo>(left, right);
}

} // namespace

using Impl::operator-;
using Impl::operator+;
using Impl::operator*;
using Impl::operator/;
using Impl::operator%;

} // namespace

#endif