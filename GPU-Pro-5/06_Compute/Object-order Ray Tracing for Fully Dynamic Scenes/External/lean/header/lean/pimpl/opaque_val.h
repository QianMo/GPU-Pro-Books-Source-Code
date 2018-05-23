/*****************************************************/
/* lean PImpl                   (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_PIMPL_OPAQUE_VAL
#define LEAN_PIMPL_OPAQUE_VAL

#include "../cpp0x.h"
#include "../meta/dereference.h"

namespace lean
{
namespace pimpl
{

/// Opaque value class that stores values of the type wrapped by
/// OpaqueTypeWrapperBase, providing access to these values via
/// the type wrapped by OpaqueTypeWrapper, provided OpaqueTypeWrapper
/// is both fully defined and derived from OpaqueTypeWrapperBase.
template <class OpaqueTypeWrapper, class OpaqueTypeWrapperBase>
class opaque_val
{
private:
	typename OpaqueTypeWrapperBase::type m_value;

public:
	/// Opaque type of the value concealed by this wrapper.
	typedef typename OpaqueTypeWrapperBase::type opaque_type;
	/// Actual type of the value concealed by this wrapper, if fully defined.
	typedef typename complete_type_or_base<OpaqueTypeWrapper, OpaqueTypeWrapperBase>::type::type actual_type;
	/// Dereferenced type of the value concealed by this wrapper.
	typedef typename maybe_dereference_once<actual_type>::value_type dereferenced_type;
	/// Dereferenced return type of the value concealed by this wrapper.
	typedef typename conditional_type<
		maybe_dereference_once<actual_type>::dereferenced,
		dereferenced_type&, dereferenced_type>::type dereferenced_return_type;
	/// True if this wrapper currently is in an opaque state, false otherwise.
	static const bool is_opaque = !complete_type_or_base<OpaqueTypeWrapper, OpaqueTypeWrapperBase>::is_complete;

	/// Constructs an opaque value object from the given (or default) value.
	opaque_val(const actual_type &value = actual_type())
		: m_value(value)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(!is_opaque, "cannot construct value in opaque state", cannot_construct_value_in_opaque_state);
	}

#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Constructs an opaque value object from the given value.
	opaque_val(opaque_val &&right) : m_value(std::move(right.m_value)) { }
#endif

	/// Replaces the stored value with the given new value.
	opaque_val& operator =(const actual_type &value)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(!is_opaque, "cannot assign value in opaque state", cannot_assign_value_in_opaque_state);

		m_value = value;
		return *this;
	}

#ifdef LEAN0X_NEED_EXPLICIT_MOVE
	/// Replaces the stored value with the given new value.
	opaque_val& operator =(opaque_val &&right)
	{
		m_value = std::move(right.m_value);
		return *this;
	}
#endif

	/// Gets the value concealed by this opaque wrapper.
	actual_type get(void) const { return static_cast<actual_type>(m_value); }
	
	/// Gets the value concealed by this opaque wrapper.
	dereferenced_return_type operator *() const { return maybe_dereference_once<actual_type>::dereference(get()); }
	/// Gets the value concealed by this opaque wrapper.
	actual_type operator ->() const { return get(); }

	/// Gets the value concealed by this opaque wrapper.
	operator actual_type() const { return get(); }
};

} // namespace

using pimpl::opaque_val;

} // namespace

/// @addtogroup PImplMacros PImpl macros
/// @see lean::pimpl
/// @{

/// Declares an opaque type of the given name, setting its internal opaque
/// value type to the given opaque type.
#define DECLARE_OPAQUE_TYPE(NAME, OPAQUE_TYPE)	\
	struct NAME##_opaque_type_wrapper			\
	{											\
		typedef OPAQUE_TYPE type;				\
	};											\
												\
	struct NAME##_actual_type_wrapper;			\
												\
	typedef ::lean::opaque_val<					\
		NAME##_actual_type_wrapper,				\
		NAME##_opaque_type_wrapper> NAME;

/// Defines the previously declared opaque type of the given name, setting
/// its actual value type to the given actual type and thus changing its
/// state from opaque to transparent.
#define DEFINE_OPAQUE_TYPE(NAME, ACTUAL_TYPE)								\
	struct NAME##_actual_type_wrapper : public NAME##_opaque_type_wrapper	\
	{																		\
		typedef ACTUAL_TYPE type;											\
	};

/// @}

#endif