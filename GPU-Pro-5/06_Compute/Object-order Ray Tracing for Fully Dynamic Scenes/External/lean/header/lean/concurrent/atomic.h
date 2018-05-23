/*****************************************************/
/* lean Concurrent              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_CONCURRENT_ATOMIC
#define LEAN_CONCURRENT_ATOMIC

#include "../lean.h"
#include "../meta/strip.h"

#ifdef _MSC_VER

#include <intrin.h>

namespace lean
{
namespace concurrent
{
	namespace impl
	{
		//// Long ////

		/// Atomically increments the given value, returning the results.
		__forceinline long atomic_increment(volatile long &value)
		{
			return _InterlockedIncrement(&value);
		}

		/// Atomically decrements the given value, returning the results.
		__forceinline long atomic_decrement(volatile long &value)
		{
			return _InterlockedDecrement(&value);
		}

		/// Atomically tests if the given value is equal to the given expected value, assigning the given new value on success.
		__forceinline bool atomic_test_and_set(volatile long &value, long expectedValue, long newValue)
		{
			return (_InterlockedCompareExchange(&value, newValue, expectedValue) == expectedValue);
		}

		/// Atomically sets the given value.
		__forceinline long atomic_set(volatile long &value, long newValue)
		{
			return _InterlockedExchange(&value, newValue);
		}

		//// Short ////

		/// Atomically increments the given value, returning the results.
		__forceinline short atomic_increment(volatile short &value)
		{
			return _InterlockedIncrement16(&value);
		}

		/// Atomically decrements the given value, returning the results.
		__forceinline short atomic_decrement(volatile short &value)
		{
			return _InterlockedDecrement16(&value);
		}

		/// Atomically tests if the given value is equal to the given expected value, assigning the given new value on success.
		__forceinline bool atomic_test_and_set(volatile short &value, short expectedValue, short newValue)
		{
			return (_InterlockedCompareExchange16(&value, newValue, expectedValue) == expectedValue);
		}

		/// Atomically sets the given value.
		__forceinline short atomic_set(volatile short &value, short newValue)
		{
			return _InterlockedExchange16(&value, newValue);
		}

		//// Integers ////

		template <size_t Size>
		struct atomic_type
		{
			// Always checked, therefore use static_assert with care
			LEAN_STATIC_ASSERT_MSG_ALT(Size & ~Size, // = false, dependent
				"Atomic operations on integers of the given type unsupported.",
				Atomic_operations_on_integers_of_the_given_type_unsupported);
		};

		template <> struct atomic_type<sizeof(short)> { typedef short type; };
		template <> struct atomic_type<sizeof(long)> { typedef long type; };

		//// Pointers ////

		/// Atomically tests if the given pointer is equal to the given expected pointer, assigning the given new pointer on success.
		__forceinline bool atomic_test_and_set(void *volatile &ptr, void *expectedPtr, void *newPtr)
		{
#ifdef _M_IX86
			return atomic_test_and_set(
				reinterpret_cast<volatile long&>(ptr),
				reinterpret_cast<long>(expectedPtr),
				reinterpret_cast<long>(newPtr) );
#else
			return (_InterlockedCompareExchangePointer(&ptr, newPtr, expectedPtr) == expectedPtr);
#endif
		}

		/// Atomically sets the given pointer.
		__forceinline void* atomic_set(void *volatile &ptr, void *newPtr)
		{
#ifdef _M_IX86
			return reinterpret_cast<void*>( atomic_set(
				reinterpret_cast<volatile long&>(ptr),
				reinterpret_cast<long>(newPtr) ) );
#else
			return _InterlockedExchangePointer(&ptr, newPtr);
#endif
		}

	} // namespace

} // namespace
} // namespace

#else

#error Unknown compiler, intrinsics unavailable.

#endif

namespace lean
{
namespace concurrent
{
	//// Integers ////

	/// Atomically increments the given value, returning the results.
	template <class Integer>
	LEAN_INLINE Integer atomic_increment(volatile Integer &value)
	{
		typedef typename impl::atomic_type<sizeof(Integer)>::type atomic_int;

		return static_cast<Integer>( impl::atomic_increment(
			reinterpret_cast<volatile atomic_int&>(value) ) );
	}

	/// Atomically decrements the given value, returning the results.
	template <class Integer>
	LEAN_INLINE Integer atomic_decrement(volatile Integer &value)
	{
		typedef typename impl::atomic_type<sizeof(Integer)>::type atomic_int;

		return static_cast<Integer>( impl::atomic_decrement(
			reinterpret_cast<volatile atomic_int&>(value) ) );
	}

	/// Atomically tests if the given value is equal to the given expected value, assigning the given new value on success.
	template <class Integer>
	LEAN_INLINE bool atomic_test_and_set(volatile Integer &value, typename identity<Integer>::type expectedValue, typename identity<Integer>::type newValue)
	{
		typedef typename impl::atomic_type<sizeof(Integer)>::type atomic_int;

		return impl::atomic_test_and_set(
			reinterpret_cast<volatile atomic_int&>(value),
			static_cast<atomic_int>(expectedValue),
			static_cast<atomic_int>(newValue) );
	}

	/// Atomically sets the given value.
	template <class Integer>
	LEAN_INLINE Integer atomic_set(volatile Integer &value, typename identity<Integer>::type newValue)
	{
		typedef typename impl::atomic_type<sizeof(Integer)>::type atomic_int;

		return static_cast<Integer>( impl::atomic_set(
			reinterpret_cast<volatile atomic_int&>(value),
			static_cast<atomic_int>(newValue) ) );
	}

	/// Atomically tests if the given value is equal to the given expected value, assigning the given new value on success.
	template <class Pointer>
	LEAN_INLINE bool atomic_test_and_set(Pointer *volatile &value, typename identity<Pointer>::type *expectedValue, typename identity<Pointer>::type *newValue)
	{
		return impl::atomic_test_and_set(
			const_cast<void *volatile &>(reinterpret_cast<const void *volatile &>(const_cast<const Pointer *volatile &>(value))),
			const_cast<void*>(static_cast<const void*>(expectedValue)),
			const_cast<void*>(static_cast<const void*>(newValue)) );
	}

	/// Atomically sets the given value.
	template <class Pointer>
	LEAN_INLINE Pointer* atomic_set(Pointer *volatile &value, typename identity<Pointer>::type *newValue)
	{
		return static_cast<Pointer*>( impl::atomic_set(
			const_cast<void *volatile &>(reinterpret_cast<const void *volatile &>(const_cast<const Pointer *volatile &>(value))),
			const_cast<void*>(static_cast<const void*>(newValue)) ) );
	}

} // namespace

using concurrent::atomic_increment;
using concurrent::atomic_decrement;
using concurrent::atomic_test_and_set;
using concurrent::atomic_set;

} // namespace

#endif