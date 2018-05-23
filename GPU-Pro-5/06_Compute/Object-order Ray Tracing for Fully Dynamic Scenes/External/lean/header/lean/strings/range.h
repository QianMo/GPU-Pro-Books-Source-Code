/*****************************************************/
/* lean Strings                 (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_STRINGS_RANGE
#define LEAN_STRINGS_RANGE

#include "../lean.h"
#include "../meta/strip.h"
#include "../meta/type_traits.h"

// NOTE: <iterator> includes loads of cruft
namespace std { template<class Iterator> struct iterator_traits; }

namespace lean
{
namespace strings
{

template <class Type>
struct iterator_types
{
	typedef typename std::iterator_traits<Type>::value_type value_type;
	typedef typename std::iterator_traits<Type>::difference_type difference_type;
	typedef typename std::iterator_traits<Type>::pointer pointer;
	typedef typename std::iterator_traits<Type>::reference reference;
};

template<class Type>
struct iterator_types<Type*>
{
	typedef Type value_type;
	typedef ptrdiff_t difference_type;
	typedef Type* pointer;
	typedef Type& reference;
};
template<class Type>
struct iterator_types<const Type*>
{
	typedef Type value_type;
	typedef ptrdiff_t difference_type;
	typedef const Type* pointer;
	typedef const Type& reference;
};

/*
// NOTE: WORKAROUND: MSVC dereferencing in SFINAE expressions buggy,
// defaults to T for non-dereferenceable types. '&' on r-value finally fails.
LEAN_DEFINE_IS_VALID(iterator_deref, &*make_rval<T>())
*/

template < class Type, bool HasDeref = is_iterator<Type>::value >
struct iterator_maybe : iterator_types<Type> { };
template <class Type>
struct iterator_maybe<Type, false>{ };

template < class Type, bool HasDeref = is_iterator<Type>::value >
struct iterator_reflexive : iterator_types<Type> { };
template <class Type>
struct iterator_reflexive<Type, false>
{
	typedef Type value_type;
	typedef ptrdiff_t difference_type;
	typedef const Type* pointer;
	typedef const Type& reference;
};

/// Iterator range.
template <class Iterator>
class range
{
private:
	Iterator m_begin;
	Iterator m_end;

public:
	/// Iterator type.
	typedef Iterator iterator;
	/// Iterator type.
	typedef Iterator const_iterator;

	/// Constructs an empty iterator range.
	LEAN_INLINE range()
		: m_begin(), m_end() { }
	/// Constructs an iterator range.
	LEAN_INLINE range(iterator begin, iterator end)
		: m_begin(begin), m_end(end) { }
	/// Constructs an iterator range.
	template <class Range>
	LEAN_INLINE explicit range(const Range &range)
		: m_begin(range.begin()), m_end(range.end()) { }

	/// Assigns the given iterators to this range.
	LEAN_INLINE void assign(iterator begin, iterator end)
	{
		m_begin = begin;
		m_end = end;
	}
	/// Assigns the given iterators to this range.
	template <class Range>
	LEAN_INLINE void assign(const Range &range)
	{
		m_begin = range.begin();
		m_end = range.end();
	}

	template <class Range>
	LEAN_INLINE range& operator =(const Range &range)
	{
		assign(range);
		return *this;
	}

	/// Gets whether this range is empty.
	LEAN_INLINE bool empty() const { return (m_begin == m_end); }
	/// Gets the size of this range (only valid for random access iterators).
	LEAN_INLINE size_t size() const { return m_end - m_begin; }

	/// Gets the beginning of this range.
	LEAN_INLINE iterator& begin() { return m_begin; }
	/// Gets the beginning of this range.
	LEAN_INLINE iterator begin() const { return m_begin; }
	/// Gets the end of this range.
	LEAN_INLINE iterator& end() { return m_end; }
	/// Gets the beginning of this range.
	LEAN_INLINE iterator end() const { return m_end; }

	/// Gets the n-th element.
	LEAN_INLINE typename iterator_reflexive<iterator>::reference operator [](ptrdiff_t n) const { return *(m_begin + n); }
};


/// Casts the iterators of the given range into iterators of the given type.
template <class DestRange, class Range>
LEAN_INLINE DestRange const_range_cast(const Range &right)
{
	return DestRange(
			const_cast<typename DestRange::iterator>(right.begin()),
			const_cast<typename DestRange::iterator>(right.end())
		);
}

/// Casts the iterators of the given range into iterators of the given type.
template <class DestRange, class Range>
LEAN_INLINE DestRange reinterpret_range_cast(const Range &right)
{
	return DestRange(
			reinterpret_cast<typename DestRange::iterator>(right.begin()),
			reinterpret_cast<typename DestRange::iterator>(right.end())
		);
}

/// Casts the iterators of the given range into iterators of the given type.
/// WARNING: In most cases, this is effectively the same as reinterpret_cast!
template <class DestRange, class Range>
LEAN_INLINE DestRange static_range_cast(const Range &right)
{
	return DestRange(
			static_cast<typename DestRange::iterator>(right.begin()),
			static_cast<typename DestRange::iterator>(right.end())
		);
}


/// Makes a range from the given pair of iterators.
template <class Iterator>
LEAN_INLINE range<Iterator> make_range(Iterator begin, Iterator end)
{
	return range<Iterator>(begin, end);
}
/// Makes a range from the given pair of iterators.
template <class Iterator>
LEAN_INLINE range<Iterator> make_range_n(Iterator begin, size_t len)
{
	return range<Iterator>(begin, begin + len);
}
/// Makes a range from the given range-compatible object.
template <class Range>
LEAN_INLINE range<typename Range::iterator> make_range(Range &r)
{
	return range<typename Range::iterator>(r.begin(), r.end());
}
/// Makes a range from the given range-compatible object.
template <class Range>
LEAN_INLINE range<typename Range::const_iterator> make_range(const Range &r)
{
	return range<typename Range::const_iterator>(r.begin(), r.end());
}

/// Makes a range from the given range-compatible vector.
template <class Range>
LEAN_INLINE range<typename Range::pointer> make_range_v(Range &r)
{
	return range<typename Range::pointer>(&r[0], &r[0] + r.size());
}
/// Makes a range from the given range-compatible vector.
template <class Range>
LEAN_INLINE range<typename Range::const_pointer> make_range_v(const Range &r)
{
	return range<typename Range::const_pointer>(&r[0], &r[0] + r.size());
}


/// Makes a range from the given null-terminated charcter string.
template <class Char>
LEAN_INLINE range<Char*> make_char_range(Char *nts)
{
	return range<Char*>(
		nts,
		nts + std::char_traits<typename strip_modifiers<Char>::type>::length(nts) );
}
/// Returns an unmodified reference to the given range. (generic convenience overload)
template <class Range>
LEAN_INLINE Range& make_char_range(Range &range) { return range; }


/// Constructs an object of the given type from the given range.
template <class Class, class Range>
LEAN_INLINE Class from_range(const Range &range)
{
	return Class(range.begin(), range.end());
}

template <class String>
struct string_traits;

/// Constructs an object of the given type from the given range.
template <class String, class Range>
LEAN_INLINE String string_from_range(const Range &range)
{
	return string_traits<String>::construct(range.begin(), range.end());
}


namespace impl
{
	LEAN_DEFINE_HAS_TYPE(iterator);
}

/// Checks if the given type is a range type.
template <class Type>
struct is_range
{
	/// True, if Type is a range type.
	static const bool value = impl::has_type_iterator<Type>::value;
};

/// Redefines the given type if Range is a range, empty otherwise.
template <class Range, class Type>
struct enable_if_range : public enable_if<is_range<Range>::value, Type> { };
/// Redefines the given type if Range is not a range, empty otherwise.
template <class Range, class Type>
struct enable_if_not_range : public enable_if<!is_range<Range>::value, Type> { };
/// Redefines the given type if both Range1 and Range2 are ranges, empty otherwise.
template <class Range1, class Range2, class Type>
struct enable_if_range2 : public enable_if<is_range<Range1>::value && is_range<Range2>::value, Type> { };
/// Redefines the given type if either Range1 or Range2 is not a range, empty otherwise.
template <class Range1, class Range2, class Type>
struct enable_if_not_range2 : public enable_if<!is_range<Range1>::value || !is_range<Range2>::value, Type> { };

/// Gets the character type for the given null-terminated array of character.
template <class Chars1>
struct range_char_type;
template <class Char>
struct range_char_type<Char*> { typedef typename strip_modifiers<Char>::type type; };

/// Gets the character type for the given null-terminated array of character.
template <class Chars1, class Chars2>
struct range_char_type2;
template <class Char, class Chars2>
struct range_char_type2<Char*, Chars2> { typedef typename strip_modifiers<Char>::type type; };
template <class Chars1, class Char>
struct range_char_type2<Chars1, Char*> { typedef typename strip_modifiers<Char>::type type; };
template <class Char1, class Char2>
struct range_char_type2<Char1*, Char2*> { typedef typename strip_modifiers<Char1>::type type; };

} // namespace

using strings::iterator_types;
using strings::iterator_maybe;
using strings::iterator_reflexive;

using strings::range;
using strings::make_range;
using strings::make_range_n;
using strings::make_range_v;
using strings::make_char_range;
using strings::from_range;
using strings::string_from_range;

using strings::is_range;
using strings::enable_if_range;
using strings::enable_if_not_range;
using strings::enable_if_range2;
using strings::enable_if_not_range2;
using strings::range_char_type;
using strings::range_char_type2;

using strings::const_range_cast;
using strings::reinterpret_range_cast;
using strings::static_range_cast;

} // namespace

#endif