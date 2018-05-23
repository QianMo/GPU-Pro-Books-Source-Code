/*****************************************************/
/* lean C++0x Config            (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CPP0X
#define LEAN_CPP0X

#include "macros.h"
#include "meta/literal.h"

/// @addtogroup CPP0X C++0x-related macros
/// @see GlobalMacros
/// @{

#ifdef DOXYGEN_READ_THIS
	/// @ingroup GlobalSwitches
	/// Define this to disable all C++0x features
	#define LEAN0X_DISABLE
#endif


// Disable all C++0x features by default when working with older C++ standards
#if (201100L > __cplusplus) || defined(LEAN0X_DISABLE)
	/// Indicates that built-in nullptr is not available.
	#define LEAN0X_NO_NULLPTR
	/// Indicates that built-in noexcept is not available.
	#define LEAN0X_NO_NOEXCEPT
	/// Indicates that r-value references are not available.
	#define LEAN0X_NO_RVALUE_REFERENCES
	/// Indicates that implicit move constructors are not available.
	#define LEAN0X_NO_IMPLICIT_MOVE
	/// Indicates that built-in static_assert is not available.
	#define LEAN0X_NO_STATIC_ASSERT
	/// Indicates that built-in alignment modifiers are not available.
	#define LEAN0X_NO_ALIGN
	/// Indicates that default / delete method specifiers not available.
	#define LEAN0X_NO_DELETE_METHODS
	/// Indicates that the standard library enhancements are not available.
	#define LEAN0X_NO_STL
	/// Indicates that attributes are not available.
	#define LEAN0X_NO_ATTRIBUTES
	/// Indicates that decltype is not available.
	#define LEAN0X_NO_DECLTYPE
	/// Indicates that override is not available.
	#define LEAN0X_NO_OVERRIDE
#endif

#ifndef LEAN0X_DISABLE

	// Enable Visual Studio 2010 C++11 features
	#if (_MSC_VER >= 1600) && defined(_MSC_EXTENSIONS)
		#undef LEAN0X_NO_NULLPTR
		#undef LEAN0X_NO_RVALUE_REFERENCES
		#undef LEAN0X_NO_STATIC_ASSERT
		#undef LEAN0X_NO_STL
		#undef LEAN0X_NO_DECLTYPE
		#undef LEAN0X_NO_OVERRIDE
		// WORKAROUND: Need to patch in missing C++11 features
		#define _ALLOW_KEYWORD_MACROS
	#endif

	#if !defined(LEAN0X_NO_RVALUE_REFERENCES) && defined(LEAN0X_NO_IMPLICIT_MOVE)
		/// Indicates that move constructors need to be defined explicitly.
		#define LEAN0X_NEED_EXPLICIT_MOVE
	#endif

#endif

// Fix missing nullptr
#if defined(LEAN0X_NO_NULLPTR) && !defined(nullptr)
	#define nullptr 0
#endif

// Fix missing noexcept
#if defined(LEAN0X_NO_NOEXCEPT) && !defined(noexcept)
	#define noexcept throw()
#endif

// Add override safety
#ifndef LEAN0X_NO_OVERRIDE
	/// Enforces virtual method overriding.
	#define LEAN_OVERRIDE override
#else
	/// Enforces virtual method overriding.
	#define LEAN_OVERRIDE
#endif

// Emulate static_assert
#if defined(LEAN0X_NO_STATIC_ASSERT) && !defined(static_assert)

#ifndef DOXYGEN_SKIP_THIS
	namespace lean
	{
		struct static_assertion_error;

		namespace impl
		{
			// Accepts literal to circumvent the requirement of typename (which dependens on context)
			template <bool Triggered>
			struct emit_static_assertion_error;

			template <bool Assertion, class Error>
			struct trigger_static_assertion_error
			{
				static const bool triggered = false;
			};

			template <class Error>
			struct trigger_static_assertion_error<false, Error>
			{
				// Defines literal instead of type to circumvent the requirement of typename (which dependens on context)
				static const bool triggered = sizeof(Error) || true;
			};
		}
	}
#endif
	
	/// Static assertion triggering a compiler error on failure.
	#define LEAN_STATIC_ASSERT(expr) typedef \
		::lean::impl::emit_static_assertion_error< \
			::lean::impl::trigger_static_assertion_error<(expr), ::lean::static_assertion_error>::triggered \
		> LEAN_JOIN_VALUES(static_assertion_error_, __LINE__)

	/// Static assertion incorporating the given message in a compiler error on failure.
	#define LEAN_STATIC_ASSERT_MSG(expr, msg) typedef \
		::lean::impl::emit_static_assertion_error< \
			::lean::impl::trigger_static_assertion_error<(expr), ::lean::static_assertion_error>::triggered \
		> LEAN_JOIN_VALUES(static_assertion_error_, __LINE__)

	/// Static assertion incorporating either the given message or the given type name in a compiler error on failure.
	#define LEAN_STATIC_ASSERT_MSG_ALT(expr, msg, msgtype) struct static_assertion_error__##msgtype; typedef \
		::lean::impl::emit_static_assertion_error< \
			::lean::impl::trigger_static_assertion_error<(expr), static_assertion_error__##msgtype>::triggered \
		> LEAN_JOIN_VALUES(static_assertion_error_, __LINE__)

	// Emulate static_assert
	#define static_assert(expr, msg) LEAN_STATIC_ASSERT_MSG(expr, msg)

#else
	/// Static assertion triggering a compiler error on failure.
	#define LEAN_STATIC_ASSERT(expr) static_assert(expr, #expr)
	
	/// Static assertion incorporating the given message in a compiler error on failure.
	#define LEAN_STATIC_ASSERT_MSG(expr, msg) static_assert(expr, msg)
	
	/// Static assertion incorporating either the given message or the given incomplete type in a compiler error on failure.
	#define LEAN_STATIC_ASSERT_MSG_ALT(expr, msg, msgtype) static_assert(expr, msg)
#endif

#ifndef LEAN0X_NO_ATTRIBUTES
	/// Indicates that the corresponding function will never return.
	#define LEAN_NORETURN [[noreturn]]
#else
	/// Indicates that the corresponding function will never return.
	#define LEAN_NORETURN __declspec(noreturn)
#endif

#ifndef LEAN0X_NO_RVALUE_REFERENCES
	// Automatically include utility for move semantics
	#include <utility>

	/// Forwarding reference.
	#define LEAN_FW_REF &&

	/// Forwards the given value.
	#define LEAN_FORWARD(type, arg) std::forward<type>(arg)

	/// Moves the given value.
	#define LEAN_MOVE(arg) std::move(arg)
#else
	/// Forwarding reference.
	#define LEAN_FW_REF const &

	/// Forwards the given value.
	#define LEAN_FORWARD(type, arg) arg

	/// Moves the given value.
	#define LEAN_MOVE(arg) arg
#endif

/// Turns the given enum wrapper struct into an enum struct.
#define LEAN_MAKE_ENUM_STRUCT(name) \
	T v; \
	name(T v) : v(v) { } \
	operator T() const { return v; } \
	template <class S> operator S() const \
	{ \
		LEAN_STATIC_ASSERT_MSG_ALT(lean::dependent_false<S>::value, "Unsafe enum cast attempted", Unsafe_enum_cast_attempted); \
		return v; \
	}

/// @}

#ifdef DOXYGEN_READ_THIS
	// Re-enable move semantics for documentation
	#undef LEAN0X_DISABLE
	#undef LEAN0X_NO_RVALUE_REFERENCES
	#undef LEAN0X_NO_DECLTYPE
	#undef LEAN0X_NO_ATTRIBUTES
#endif

#endif