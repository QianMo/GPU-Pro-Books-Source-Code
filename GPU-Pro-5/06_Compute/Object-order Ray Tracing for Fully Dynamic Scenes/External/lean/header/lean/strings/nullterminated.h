/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_NULLTERMINATED
#define LEAN_STRINGS_NULLTERMINATED

#include "../lean.h"
#include "../meta/strip.h"
#include "../meta/type_traits.h"
#include "char_traits.h"

namespace lean
{
namespace strings
{

struct nullterminated_incompatible { };

/// Spezialize this class to make your string class compatible with the nullterminated character range class.
template <class Compatible, class Char, class Traits>
struct nullterminated_compatible : public nullterminated_incompatible
{
#ifdef DOXYGEN_READ_THIS
	/// Converts an object of your type into a nullterminated character range,
	/// returning the beginning of the range.
	static const Char* from(const Compatible &from);
	/// Converts an object of your type into a nullterminated character range,
	/// returning the end of the range (must be null character). You may use
	/// Traits::length(begin), if unknown.
	static const Char* from(const Compatible &from, const Char *begin);

	/// Converts a nullterminated character range into an object of your type.
	static Compatible to(const Char *begin);
	/// Converts a nullterminated character range into an object of your type.
	static Compatible to(const Char *begin, const Char *end);
#endif
};

/// Checks if the given type is nullterminated-character-range-compatible.
template <class Compatible, class Char, class Traits>
struct is_nullterminated_compatible
{
	/// True, if the given type is compatible.
	static const bool value = !is_derived<
		nullterminated_compatible<Compatible, Char, Traits>,
		nullterminated_incompatible>::value;
};

/// Checks if the given type is either nullterminated-character-range-compatible or a character range.
template <class Compatible, class Char, class Traits>
struct is_nullterminated_convertible
{
	/// True, if the given type is convertible.
	static const bool value = is_equal<
				typename strip_modifiers<typename strip_pointer<typename strip_reference<Compatible>::type>::type>::type,
				typename strip_modifiers<Char>::type >::value
			|| is_equal<
				typename strip_modifiers<typename strip_array<typename strip_reference<Compatible>::type>::type>::type,
				typename strip_modifiers<Char>::type >::value
			|| is_nullterminated_compatible<Compatible, Char, Traits>::value;
};

/// Asserts that the given type is nullterminated-character-range-compatible.
template <class Compatible, class Char, class Traits>
struct assert_nullterminated_compatible
{
	static const bool is_compatible = is_nullterminated_compatible<Compatible, Char, Traits>::value;

	LEAN_STATIC_ASSERT_MSG_ALT(is_compatible,
		"Type incompatible with nullterminated character range class.",
		Type_incompatible_with_nullterminated_character_range_class);

	/// Dummy that may be used in asserting typedef to enforce template instantiation.
	typedef void type;
};

template <class Char, class Traits>
struct null_char
{
	static const Char value = static_cast<Char>(0);
};

/// Nullterminated character half-range class that may be IMPLICITLY constructed from arbitrary string classes.
/// May be used in parameter lists, not recommended elsewhere.
template < class Char, class Traits = char_traits<typename strip_const<Char>::type> >
class nullterminated_implicit
{
public:
	/// Type of the characters referenced by this range.
	typedef typename strip_const<Char>::type value_type;
	
	/// Type of the size returned by this range.
	typedef size_t size_type;
	/// Type of the difference between the addresses of two elements in this range.
	typedef ptrdiff_t difference_type;

	/// Type of pointers to the elements contained by this range.
	typedef value_type* pointer;
	/// Type of constant pointers to the elements contained by this range.
	typedef const value_type* const_pointer;
	/// Type of references to the elements contained by this range.
	typedef value_type& reference;
	/// Type of constant references to the elements contained by this range.
	typedef const value_type& const_reference;

	/// Type of iterators to the elements contained by this range.
	typedef pointer iterator;
	/// Type of constant iterators to the elements contained by this range.
	typedef const_pointer const_iterator;

	/// Character traits used by this range.
	typedef Traits traits_type;

protected:
	const_pointer m_begin;

	// /// Gets a pointer to this null-terminated range.
	// LEAN_INLINE operator Char*() const { return m_begin; }
	// DESIGN: Only permit implicit conversion to compatible container types, otherwise pointers might accidentally dangle.

public:
	/// Constructs a (half) character range from the given C string.
	LEAN_INLINE nullterminated_implicit(const_pointer begin = &null_char<Char, Traits>::value)
		: m_begin(begin)
	{
		LEAN_ASSERT(m_begin);
	}
	/// Constructs a (half) character range from the given compatible object.
	template <class Compatible>
	LEAN_INLINE nullterminated_implicit(const Compatible &from,
			typename enable_if<is_nullterminated_compatible<Compatible, value_type, traits_type>::value, const void*>::type = nullptr)
		: m_begin(nullterminated_compatible<Compatible, value_type, traits_type>::from(from))
	{
		LEAN_ASSERT(m_begin);
	}

	/// Gets whether this character range is currently empty.
	LEAN_INLINE bool empty() const { return traits_type::empty(m_begin); }
	/// Gets the length of this character range, in characters (same as compute_size()). O(n).
	LEAN_INLINE size_type compute_length() const { return traits_type::length(m_begin); }
	/// Gets the length of this character range, in characters (same as compute_length()). O(n).
	LEAN_INLINE size_type compute_size() const { return compute_length(); }
	/// Gets the length of this character range, in code points (might differ from length() and size()). O(n).
	LEAN_INLINE size_type compute_count() const { return traits_type::count(m_begin); }
	
	/// Gets an element by position, access violation on failure.
	LEAN_INLINE const_reference operator [](size_type pos) const { return m_begin[pos]; }
	
	/// Gets a pointer to this nullterminated range.
	LEAN_INLINE const_pointer c_str() const { return m_begin; }
	/// Gets a pointer to this nullterminated range.
	LEAN_INLINE const_pointer data() const { return m_begin; }

	/// Returns a constant iterator to the first element contained by this character range. O(1).
	LEAN_INLINE const_iterator begin() const { return m_begin; }
	/// Returns a constant iterator  the last element contained by this character range. O(n).
	LEAN_INLINE const_iterator compute_end() const { return m_begin + compute_length(); }

	/// Swaps the contents of this range with the contents of the given range.
	LEAN_INLINE void swap(nullterminated_implicit& right)
	{
		const_pointer right_begin = right.m_begin;
		right.m_begin = m_begin;
		m_begin = right_begin;
	}

	/// Constructs a compatible object from this null-terminated character range.
	template <class Compatible>
	LEAN_INLINE Compatible to() const
	{
		typedef typename assert_nullterminated_compatible<
				Compatible,
				value_type, traits_type
			>::type assert_compatible;
		return nullterminated_compatible<Compatible, value_type, traits_type>::to(m_begin);
	}
};

/// Nullterminated character half-range class that may be constructed from arbitrary string classes.
template < class Char, class Traits = char_traits<typename strip_const<Char>::type> >
class nullterminated : public nullterminated_implicit<Char, Traits>
{
public:
	/// Corresponding implicit type.
	typedef nullterminated_implicit<Char, Traits> implicit_type;

	/// Constructs a (half) character range from the given C string.
	explicit LEAN_INLINE nullterminated(typename implicit_type::const_pointer begin = &null_char<Char, Traits>::value)
		: implicit_type(begin)  { }
	/// Constructs a (half) character range from the given compatible object.
	template <class Compatible>
	explicit LEAN_INLINE nullterminated(const Compatible &from,
			typename enable_if<is_nullterminated_compatible<Compatible, value_type, traits_type>::value, const void*>::type = nullptr)
		: implicit_type(from) { }
	/// Constructs a (half) character range from the given implicit half range.
	explicit LEAN_INLINE nullterminated(const implicit_type &right)
		: implicit_type(right)  { }
};

/// Makes an explicit nullterminated (half) range from the given implicit range.
template <class Char, class Traits>
LEAN_INLINE nullterminated<Char, Traits> make_nt(const nullterminated_implicit<Char, Traits> &range)
{
	return nullterminated<Char, Traits>(range);
}
/// Makes an explicit nullterminated (half) range from the given implicit range.
template <class Char>
LEAN_INLINE nullterminated<Char> make_nt(const Char *range)
{
	return nullterminated<Char>(range);
}
/// Makes an explicit nullterminated (half) range from the given implicit range.
template <class Char, class Compatible>
LEAN_INLINE typename enable_if<
		is_nullterminated_compatible<Compatible, Char, typename nullterminated<Char>::traits_type>::value,
		nullterminated<Char>
	>::type	make_nt(const Compatible &compatible)
{
	return nullterminated<Char>(compatible);
}

/// Comparison operator.
template <class Char, class Traits>
LEAN_INLINE bool operator ==(const nullterminated_implicit<Char, Traits>& left, const nullterminated_implicit<Char, Traits>& right)
{
	return nullterminated_implicit<Char, Traits>::traits_type::equal(left.c_str(), right.c_str());
}

template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator ==(const nullterminated_implicit<Char, Traits>& left, const Compatible& right) { return left == make_nt<Char, Traits>(right); }
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator ==(const Compatible& left, const nullterminated_implicit<Char, Traits>& right) { return make_nt<Char, Traits>(left) == right; }

/// Comparison operator.
template <class Char, class Traits>
LEAN_INLINE bool operator !=(const nullterminated_implicit<Char, Traits>& left, const nullterminated_implicit<Char, Traits>& right)
{
	return !(left == right);
}
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator !=(const nullterminated_implicit<Char, Traits>& left, const Compatible& right) { return left != make_nt<Char, Traits>(right); }
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator !=(const Compatible& left, const nullterminated_implicit<Char, Traits>& right) { return make_nt<Char, Traits>(left) != right; }

/// Comparison operator.
template <class Char, class Traits>
LEAN_INLINE bool operator <(const nullterminated_implicit<Char, Traits>& left, const nullterminated_implicit<Char, Traits>& right)
{
	return nullterminated_implicit<Char, Traits>::traits_type::less(left.c_str(), right.c_str());
}
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator <(const nullterminated_implicit<Char, Traits>& left, const Compatible& right) { return left < <make_nt<Char, Traits>(right); }
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator <(const Compatible& left, const nullterminated_implicit<Char, Traits>& right) { return make_nt<Char, Traits>(left) < right; }

/// Comparison operator.
template <class Char, class Traits>
LEAN_INLINE bool operator >(const nullterminated_implicit<Char, Traits>& left, const nullterminated_implicit<Char, Traits>& right)
{
	return (right < left);
}
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator >(const nullterminated_implicit<Char, Traits>& left, const Compatible& right) { return left > <make_nt<Char, Traits>(right); }
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator >(const Compatible& left, const nullterminated_implicit<Char, Traits>& right) { return make_nt<Char, Traits>(left) > right; }

/// Comparison operator.
template <class Char, class Traits>
LEAN_INLINE bool operator <=(const nullterminated_implicit<Char, Traits>& left, const nullterminated_implicit<Char, Traits>& right)
{
	return !(right < left);
}
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator <=(const nullterminated_implicit<Char, Traits>& left, const Compatible& right) { return left <= <make_nt<Char, Traits>(right); }
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator <=(const Compatible& left, const nullterminated_implicit<Char, Traits>& right) { return make_nt<Char, Traits>(left) <= right; }

/// Comparison operator.
template <class Char, class Traits>
LEAN_INLINE bool operator >=(const nullterminated_implicit<Char, Traits>& left, const nullterminated_implicit<Char, Traits>& right)
{
	return !(left < right);
}
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator >=(const nullterminated_implicit<Char, Traits>& left, const Compatible& right) { return left >= <make_nt<Char, Traits>(right); }
template <class Char, class Traits, class Compatible>
LEAN_INLINE typename enable_if<is_nullterminated_convertible<Compatible, Char, Traits>::value, bool>::type
	operator >=(const Compatible& left, const nullterminated_implicit<Char, Traits>& right) { return make_nt<Char, Traits>(left) >= right; }

/// Swaps the elements of two nullterminated character ranges.
template <class Char, class Traits>
LEAN_INLINE void swap(nullterminated_implicit<Char, Traits>& left, nullterminated_implicit<Char, Traits>& right)
{
	left.swap(right);
}

} // namespace

using strings::nullterminated_implicit;
using strings::nullterminated;

using strings::make_nt;

} // namespace

#endif