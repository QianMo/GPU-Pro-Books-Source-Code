/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FUNCTIONAL_VARIADIC
#define LEAN_FUNCTIONAL_VARIADIC

#include "../macros.h"

/// @addtogroup GlobalMacros
/// @{

/// 'False' has-*-parameter-argument.
#define LEAN_VARIADIC_NO_PARAMS
/// 'True' has-*-parameter-argument.
#define LEAN_VARIADIC_HAS_PARAMS ,

#define LEAN_VARIADIC_TEMPLATE__LEAN_VARIADIC_NO_PARAMS(tparams)
#define LEAN_VARIADIC_TEMPLATE__LEAN_VARIADIC_HAS_PARAMS(tparams) template <tparams>

#define LEAN_ASSERT_NO_VA_ARGS()

/// Defines a variadic function template, optionally including additional template & function parameters.
#define LEAN_VARIADIC_TEMPLATE_XXL(fwdop, fun, tparams, has_tparams, params, has_params, modifiers, body, args, has_args, ...) \
	/* 0 */ \
	LEAN_VARIADIC_TEMPLATE__##has_tparams(tparams##__VA_ARGS__)	\
	fun(params) modifiers							\
	body((args))									\
	/* 1 */ \
	template <tparams has_tparams					\
			class __lean__t1						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1)	\
		))											\
	/* 2 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2)	\
		))											\
	/* 3 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3)	\
		))											\
	/* 4 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3,						\
			class __lean__t4						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3,		\
			__lean__t4 LEAN_FW_REF __lean__p4		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3),	\
			fwdop(__lean__t4, __lean__p4)	\
		))											\
	/* 5 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3,						\
			class __lean__t4,						\
			class __lean__t5						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3,		\
			__lean__t4 LEAN_FW_REF __lean__p4,		\
			__lean__t5 LEAN_FW_REF __lean__p5		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3),	\
			fwdop(__lean__t4, __lean__p4),	\
			fwdop(__lean__t5, __lean__p5)	\
		))											\
	/* 6 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3,						\
			class __lean__t4,						\
			class __lean__t5,						\
			class __lean__t6						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3,		\
			__lean__t4 LEAN_FW_REF __lean__p4,		\
			__lean__t5 LEAN_FW_REF __lean__p5,		\
			__lean__t6 LEAN_FW_REF __lean__p6		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3),	\
			fwdop(__lean__t4, __lean__p4),	\
			fwdop(__lean__t5, __lean__p5),	\
			fwdop(__lean__t6, __lean__p6)	\
		))											\
	/* 7 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3,						\
			class __lean__t4,						\
			class __lean__t5,						\
			class __lean__t6,						\
			class __lean__t7						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3,		\
			__lean__t4 LEAN_FW_REF __lean__p4,		\
			__lean__t5 LEAN_FW_REF __lean__p5,		\
			__lean__t6 LEAN_FW_REF __lean__p6,		\
			__lean__t7 LEAN_FW_REF __lean__p7		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3),	\
			fwdop(__lean__t4, __lean__p4),	\
			fwdop(__lean__t5, __lean__p5),	\
			fwdop(__lean__t6, __lean__p6),	\
			fwdop(__lean__t7, __lean__p7)	\
		))											\
	/* 8 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3,						\
			class __lean__t4,						\
			class __lean__t5,						\
			class __lean__t6,						\
			class __lean__t7,						\
			class __lean__t8						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3,		\
			__lean__t4 LEAN_FW_REF __lean__p4,		\
			__lean__t5 LEAN_FW_REF __lean__p5,		\
			__lean__t6 LEAN_FW_REF __lean__p6,		\
			__lean__t7 LEAN_FW_REF __lean__p7,		\
			__lean__t8 LEAN_FW_REF __lean__p8		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3),	\
			fwdop(__lean__t4, __lean__p4),	\
			fwdop(__lean__t5, __lean__p5),	\
			fwdop(__lean__t6, __lean__p6),	\
			fwdop(__lean__t7, __lean__p7),	\
			fwdop(__lean__t8, __lean__p8)	\
		))											\
	/* 9 */ \
	template <tparams has_tparams					\
			class __lean__t1,						\
			class __lean__t2,						\
			class __lean__t3,						\
			class __lean__t4,						\
			class __lean__t5,						\
			class __lean__t6,						\
			class __lean__t7,						\
			class __lean__t8,						\
			class __lean__t9						\
		>											\
	fun( params has_params							\
			__lean__t1 LEAN_FW_REF __lean__p1,		\
			__lean__t2 LEAN_FW_REF __lean__p2,		\
			__lean__t3 LEAN_FW_REF __lean__p3,		\
			__lean__t4 LEAN_FW_REF __lean__p4,		\
			__lean__t5 LEAN_FW_REF __lean__p5,		\
			__lean__t6 LEAN_FW_REF __lean__p6,		\
			__lean__t7 LEAN_FW_REF __lean__p7,		\
			__lean__t8 LEAN_FW_REF __lean__p8,		\
			__lean__t9 LEAN_FW_REF __lean__p9		\
		) modifiers									\
	body(( args has_args							\
			fwdop(__lean__t1, __lean__p1),	\
			fwdop(__lean__t2, __lean__p2),	\
			fwdop(__lean__t3, __lean__p3),	\
			fwdop(__lean__t4, __lean__p4),	\
			fwdop(__lean__t5, __lean__p5),	\
			fwdop(__lean__t6, __lean__p6),	\
			fwdop(__lean__t7, __lean__p7),	\
			fwdop(__lean__t8, __lean__p8),	\
			fwdop(__lean__t9, __lean__p9)	\
		))											\
	// End

/// Copies the given value.
#define LEAN_COPY(type, arg) arg

/// Defines variadic perfect-forwarding function including additional template & function parameters.
#define LEAN_VARIADIC_TEMPLATE_TPA(fwdop, fun, tparams, params, modifiers, body, args, ...) \
	LEAN_VARIADIC_TEMPLATE_XXL(fwdop##__VA_ARGS__, fun##__VA_ARGS__, tparams##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, params##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, modifiers##__VA_ARGS__, body##__VA_ARGS__, args##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS) LEAN_ASSERT_NO_VA_ARGS(__VA_ARGS__)

/// Defines variadic perfect-forwarding function including additional template & function parameters.
#define LEAN_VARIADIC_TEMPLATE_TP(fwdop, fun, tparams, params, modifiers, body, ...) \
	LEAN_VARIADIC_TEMPLATE_XXL(fwdop##__VA_ARGS__, fun##__VA_ARGS__, tparams##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, params##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, modifiers##__VA_ARGS__, body##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS) LEAN_ASSERT_NO_VA_ARGS(__VA_ARGS__)

/// Defines variadic perfect-forwarding function including additional template parameters.
#define LEAN_VARIADIC_TEMPLATE_T(fwdop, fun, tparams, modifiers, body, ...) \
	LEAN_VARIADIC_TEMPLATE_XXL(fwdop##__VA_ARGS__, fun##__VA_ARGS__, tparams##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS, modifiers##__VA_ARGS__, body##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS) LEAN_ASSERT_NO_VA_ARGS(__VA_ARGS__)

/// Defines variadic perfect-forwarding function including additional function parameters.
#define LEAN_VARIADIC_TEMPLATE_P(fwdop, fun, params, modifiers, body, ...) \
	LEAN_VARIADIC_TEMPLATE_XXL(fwdop##__VA_ARGS__, fun##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS, params##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, modifiers##__VA_ARGS__, body##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS) LEAN_ASSERT_NO_VA_ARGS(__VA_ARGS__)

/// Defines variadic perfect-forwarding function including additional function parameters.
#define LEAN_VARIADIC_TEMPLATE_PA(fwdop, fun, params, modifiers, body, args, ...) \
	LEAN_VARIADIC_TEMPLATE_XXL(fwdop##__VA_ARGS__, fun##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS, params##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS, modifiers##__VA_ARGS__, body##__VA_ARGS__, args##__VA_ARGS__, LEAN_VARIADIC_HAS_PARAMS) LEAN_ASSERT_NO_VA_ARGS(__VA_ARGS__)

/// Defines variadic perfect-forwarding function.
#define LEAN_VARIADIC_TEMPLATE(fwdop, fun, modifiers, body, ...) \
	LEAN_VARIADIC_TEMPLATE_XXL(fwdop##__VA_ARGS__, fun##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS, modifiers##__VA_ARGS__, body##__VA_ARGS__, LEAN_NOTHING, LEAN_VARIADIC_NO_PARAMS) LEAN_ASSERT_NO_VA_ARGS(__VA_ARGS__)

/// @}

#endif