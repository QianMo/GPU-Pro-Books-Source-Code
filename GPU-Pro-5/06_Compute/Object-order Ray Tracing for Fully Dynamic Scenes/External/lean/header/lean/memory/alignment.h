/*****************************************************/
/* lean Memory                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_MEMORY_ALIGNMENT
#define LEAN_MEMORY_ALIGNMENT

#include "../lean.h"

#ifdef LEAN0X_NO_ALIGN
	
	#ifdef _MSC_VER
		#ifndef alignas
			/// Emulated alignas using MSVC storage class specification.
			#define alignas(alignment) __declspec( align(alignment) )
		#endif
		#ifndef alignof

			namespace lean
			{
				namespace memory
				{
					namespace impl
					{
						/// Workaround for Visual Studio bug: Alignment of some template classes not evaluated until size has been queried
						template <size_t Size, size_t Alignment>
						struct alignof_fix
						{
							LEAN_STATIC_ASSERT(Size != 0U && Alignment != 0U);
							static const size_t alignment = Alignment;
						};

						template <class Type>
						struct alignof_t
						{
							static const size_t alignment = alignof_fix<sizeof(Type), __alignof(Type)>::alignment;
						};
						template <>
						struct alignof_t<void> : alignof_t<void*> { };
					}
				}
			}

			/// Emulated alignof using MSVC-specific alignof operator extension.
			#define alignof(type) ::lean::memory::impl::alignof_t<type>::alignment
		#endif
	#else
		#error Unknown compiler, alignment specifiers unavailable.
	#endif

#endif

namespace lean
{
namespace memory
{
	/// Checks whether the given alignment is a valid power of two.
	template <size_t Alignment>
	struct is_valid_alignment
	{
		/// Specifies whether the given alignment is a valid power of two.
		static const bool value = Alignment && !(Alignment & (Alignment - 1));
	};

	/// Checks whether the given alignment is valid.
	LEAN_INLINE bool check_alignment(size_t aligment)
	{
		return aligment && !(aligment & (aligment - 1));
	}

	/// (Negatively) aligns the given unsigned integer on the given alignment boundaries.
	template <size_t Alignment, class Integer>
	LEAN_INLINE Integer lower_align_integer(Integer integer)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(is_valid_alignment<Alignment>::value,
			"Alignment is required to be power of two.", Alignment_is_required_to_be_power_of_two);

		// Widen BEFORE complement, otherwise higher-order bits might be lost
		return integer & ~static_cast<typename int_type<sign_class::no_sign, sizeof(Integer)>::type>(Alignment - 1);
	}

	/// (Negatively) aligns the given pointer on the given alignment boundaries.
	template <size_t Alignment, class Value>
	LEAN_INLINE Value* lower_align(Value *pointer)
	{
		return reinterpret_cast<Value*>(
			lower_align_integer<Alignment>( reinterpret_cast<uintptr_t>(pointer) ) );
	}

	/// Aligns the given unsigned integer on the given alignment boundaries.
	template <size_t Alignment, class Integer>
	LEAN_INLINE Integer align_integer(Integer integer)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(is_valid_alignment<Alignment>::value,
			"Alignment is required to be power of two.", Alignment_is_required_to_be_power_of_two);

		integer += (Alignment - 1);
		// Widen BEFORE complement, otherwise higher-order bits might be lost
		return integer & ~static_cast<typename int_type<sign_class::no_sign, sizeof(Integer)>::type>(Alignment - 1);
	}

	/// Aligns the given pointer on the given alignment boundaries.
	template <size_t Alignment, class Value>
	LEAN_INLINE Value* align(Value *pointer)
	{
		return reinterpret_cast<Value*>(
			align_integer<Alignment>( reinterpret_cast<uintptr_t>(pointer) ) );
	}

	/// Aligns the given unsigned integer on the given alignment boundaries, incrementing it at least by one.
	template <size_t Alignment, class Integer>
	LEAN_INLINE Integer upper_align_integer(Integer integer)
	{
		LEAN_STATIC_ASSERT_MSG_ALT(is_valid_alignment<Alignment>::value,
			"Alignment is required to be power of two.", Alignment_is_required_to_be_power_of_two);

		integer += Alignment;
		// Widen BEFORE complement, otherwise higher-order bits might be lost
		return integer & ~static_cast<typename int_type<sign_class::no_sign, sizeof(Integer)>::type>(Alignment - 1);
	}

	/// Aligns the given pointer on the given alignment boundaries, incrementing it at least by one.
	template <size_t Alignment, class Value>
	LEAN_INLINE Value* upper_align(Value *pointer)
	{
		return reinterpret_cast<Value*>(
			upper_align_integer<Alignment>( reinterpret_cast<uintptr_t>(pointer) ) );
	}

	/// Aligns derived classes according to the given alignment template argument when instances are created on the stack.
	/** @remarks MSC adds padding to make the size of aligned structures a multiple of their alignment, make sure to specify
	  * this base class first to allow for empty base class optimization. */
	template <size_t Alignment>
	struct stack_aligned
	{
		// Always checked, therefore use static_assert with care
		LEAN_STATIC_ASSERT_MSG_ALT(Alignment & ~Alignment, // = false, dependent
			"Alignment is required to be power of two.",
			Alignment_is_required_to_be_power_of_two);
	};

#ifndef DOXYGEN_SKIP_THIS

#ifdef _MSC_VER
	// MSC adds padding to make the size of aligned structures a multiple of their alignment...
	#pragma warning(push)
	#pragma warning(disable : 4324)
#endif

	template <> struct alignas(1) stack_aligned<1> { };
	template <> struct alignas(2) stack_aligned<2> { };
	template <> struct alignas(4) stack_aligned<4> { };
	template <> struct alignas(8) stack_aligned<8> { };
	template <> struct alignas(16) stack_aligned<16> { };
	template <> struct alignas(32) stack_aligned<32> { };
	template <> struct alignas(64) stack_aligned<64> { };
	template <> struct alignas(128) stack_aligned<128> { };

#ifdef _MSC_VER
	#pragma warning(pop)
#endif

#endif

} // namespace

using memory::is_valid_alignment;
using memory::check_alignment;
using memory::stack_aligned;

} // namespace

#endif