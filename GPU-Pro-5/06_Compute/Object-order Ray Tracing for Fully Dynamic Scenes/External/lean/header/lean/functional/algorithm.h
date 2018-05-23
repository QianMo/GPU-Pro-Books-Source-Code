/*****************************************************/
/* lean Functional              (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FUNCTIONAL_ALGORITHM
#define LEAN_FUNCTIONAL_ALGORITHM

#include "../lean.h"
#include "../tags/noncopyable.h"
#include "../strings/range.h"
#include "../containers/construction.h"
#include "../meta/type_traits.h"
#include <functional>
#include <algorithm>

namespace lean
{
namespace functional
{

/// Compares the elements in the given ranges.
template <class Iterator1, class Iterator2>
inline bool equal(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2)
{
	while (true)
	{
		bool ended1 = (begin1 == end1);
		bool ended2 = (begin2 == end2);

		if (ended1 || ended2 || !(*begin1 == *begin2))
			return ended1 && ended2;

		++begin1;
		++begin2;
	}
}

/// Compares the elements in the given ranges using the given predicate.
template <class Iterator1, class Iterator2, class Pred>
inline bool equal(Iterator1 begin1, Iterator1 end1, Iterator2 begin2, Iterator2 end2, Pred pred)
{
	while (true)
	{
		bool ended1 = (begin1 == end1);
		bool ended2 = (begin2 == end2);

		if (ended1 || ended2 || !pred(*begin1, *begin2))
			return ended1 && ended2;

		++begin1;
		++begin2;
	}
}

/// Compares the elements in the given ranges.
template <class Range1, class Range2>
inline bool equal(const Range1 &range1, const Range2 &range2)
{
	return equal(range1.begin(), range1.end(), range2.begin(), range2.end());
}

/// Compares the elements in the given ranges using the given predicate.
template <class Range1, class Range2, class Pred>
inline bool equal(const Range1 &range1, const Range2 &range2, Pred pred)
{
	return equal(range1.begin(), range1.end(), range2.begin(), range2.end(), pred);
}

/// Compares the elements in the given ranges.
template <class Range1, class Range2>
inline bool lexicographical_compare(const Range1 &range1, const Range2 &range2)
{
	return std::lexicographical_compare(range1.begin(), range1.end(), range2.begin(), range2.end());
}

/// Compares the elements in the given ranges using the given predicate.
template <class Range1, class Range2, class Pred>
inline bool lexicographical_compare(const Range1 &range1, const Range2 &range2, Pred pred)
{
	return std::lexicographical_compare(range1.begin(), range1.end(), range2.begin(), range2.end(), pred);
}

/// Moves the element at last to first.
template <class Iterator>
inline void move_to_front(Iterator first, Iterator last, containers::nontrivial_construction_t = containers::nontrivial_construction_t())
{
	typedef typename iterator_types<Iterator>::value_type value_type;
#ifndef LEAN0X_NO_RVALUE_REFERENCES
	value_type temp( std::move(*last) );
	std::move_backward(first, last, lean::next(last));
	*first = std::move(temp);
#else
	std::rotate(first, last, lean::next(last));
#endif
}

/// Moves the element at last to first.
template <class Iterator>
inline void move_to_front(Iterator first, Iterator last, containers::trivial_construction_t)
{
	char temp[sizeof(*last)];
	memcpy(temp, lean::addressof(*last), sizeof(*last));
	memmove(lean::addressof(*first) + 1, lean::addressof(*first), sizeof(*first) * (last - first));
	memcpy(lean::addressof(*first), temp, sizeof(*first));
}

/// Inserts the element pointed at by <code>last</code> into the given sorted range <code>[first, last)</code>.
template <class Iterator, class MoveTag>
inline typename enable_if<is_derived<MoveTag, containers::construction_t>::value, Iterator>::type insert_last(Iterator first, Iterator last, MoveTag moveTag)
{
	Iterator pos = std::upper_bound(first, last, *last);
	move_to_front(pos, last, moveTag);
	return pos;
}
template <class Iterator>
LEAN_INLINE Iterator insert_last(Iterator first, Iterator last) { return insert_last(first, last, containers::nontrivial_construction_t()); }

/// Inserts the element pointed at by <code>last</code> into the given sorted range <code>[first, last)</code>.
template <class Iterator, class Predicate, class MoveTag>
inline Iterator insert_last(Iterator first, Iterator last, Predicate predicate, MoveTag moveTag)
{
	Iterator pos = std::upper_bound(first, last, *last, predicate);
	move_to_front(pos, last, moveTag);
	return pos;
}
template <class Iterator, class Predicate>
LEAN_INLINE typename enable_if<!is_derived<Predicate, containers::construction_t>::value, Iterator>::type insert_last(Iterator first, Iterator last, Predicate predicate)
{
	return insert_last(first, last, predicate, containers::nontrivial_construction_t());
}

/// Pushes the given element onto the given vector.
template <class Vector, class Value>
inline typename Vector::iterator push_unique(Vector &vector, Value LEAN_FW_REF value, bool *pNew = nullptr)
{
	typename Vector::iterator pos = std::find( vector.begin(), vector.end(), value );

	bool bNew = (pos == vector.end());
	if (bNew)
		pos = vector.insert( pos, LEAN_FORWARD(Value, value) );
	if (pNew)
		*pNew = bNew;

	return pos;
}

/// Pushes the given element into the given sorted vector.
template <class Vector, class Value>
inline typename Vector::iterator push_sorted(Vector &vector, Value LEAN_FW_REF value)
{
	typename Vector::iterator pos = std::upper_bound( vector.begin(), vector.end(), value );
	return vector.insert( pos, LEAN_FORWARD(Value, value) );
}

/// Pushes the given element into the given sorted vector.
template <class Vector, class Value, class Predicate>
inline typename Vector::iterator push_sorted(Vector &vector, Value LEAN_FW_REF value, Predicate LEAN_FW_REF predicate)
{
	typename Vector::iterator pos = std::upper_bound( vector.begin(), vector.end(), value, LEAN_FORWARD(Predicate, predicate) );
	return vector.insert( pos, LEAN_FORWARD(Value, value) );
}

/// Pushes the given element into the given sorted vector.
template <class Vector, class Value>
inline typename Vector::iterator push_sorted_unique(Vector &vector, Value LEAN_FW_REF value)
{
	typename Vector::iterator pos = std::lower_bound( vector.begin(), vector.end(), value );
	if (pos == vector.end() || value < *pos)
		pos = vector.insert( pos, LEAN_FORWARD(Value, value) );
	return pos;
}

/// Pushes the given element into the given sorted vector.
template <class Vector, class Value, class Predicate>
inline typename Vector::iterator push_sorted_unique(Vector &vector, Value LEAN_FW_REF value, Predicate predicate)
{
	typename Vector::iterator pos = std::lower_bound( vector.begin(), vector.end(), value, predicate);
	if (pos == vector.end() || predicate(value, *pos))
		pos = vector.insert( pos, LEAN_FORWARD(Value, value) );
	return pos;

	typename Vector::iterator pos = std::upper_bound( vector.begin(), vector.end(), value, LEAN_FORWARD(Predicate, predicate) );
	return vector.insert( pos, LEAN_FORWARD(Value, value) );
}

/// Locates the position of the first occurence of the given element in the given sorted range.
template <class Iterator, class Value>
inline Iterator find_sorted(Iterator begin, Iterator end, const Value &value)
{
	Iterator element = std::lower_bound(begin, end, value);

	if (element != end && value < *element)
		element = end;

	return element;
}

/// Locates the position of the first occurence of the given element in the given sorted range.
template <class Iterator, class Value, class Ord>
inline Iterator find_sorted(Iterator begin, Iterator end, const Value &value, Ord order)
{
	Iterator element = std::lower_bound(begin, end, value, order);

	if (element != end && order(value, *element))
		element = end;

	return element;
}

/// Removes the given element from the given vector.
template <class Vector, class Value>
inline bool remove(Vector &vector, const Value &value)
{
	typename Vector::iterator newEnd = std::remove(vector.begin(), vector.end(), value);
	bool bRemoved = (newEnd != vector.end());
	vector.erase(newEnd, vector.end());
	return bRemoved;
}

template <class CmpIt>
struct in_range_predicate
{
	CmpIt removeBegin, removeEnd;

	in_range_predicate(CmpIt begin, CmpIt end)
		: removeBegin(begin),
		removeEnd(end) { }

	template <class T>
	bool operator ()(const T &v)
	{
		for (CmpIt it = removeBegin; it != removeEnd; ++it)
			if (v == *it)
				return true;
		return false;
	}
};

/// Removes the given elements from the given vector.
template <class RangeIt, class CmpIt>
inline RangeIt remove_all(RangeIt begin, RangeIt end, CmpIt removeBegin, CmpIt removeEnd)
{
	return std::remove_if(begin, end, in_range_predicate<CmpIt>(removeBegin, removeEnd));
}

/// Removes the given elements from the given vector.
template <class Vector, class CmpIt>
inline bool remove_all(Vector &vector, CmpIt removeBegin, CmpIt removeEnd)
{
	typename Vector::iterator newEnd = remove_all(vector.begin(), vector.end(), removeBegin, removeEnd);
	bool bRemoved = (newEnd != vector.end());
	vector.erase(newEnd, vector.end());
	return bRemoved;
}

/// Removes the given element from the given vector.
template <class Vector, class Value>
inline bool remove_unordered(Vector &vector, const Value &value)
{
	typename Vector::const_iterator it = vector.begin();
	typename Vector::const_iterator newEnd = vector.end();

	while (it != newEnd)
		if (*it == value)
#ifndef LEAN0X_NO_RVALUE_REFERENCES
			*it = std::move(*--newEnd);
#else
			swap(*it, *--newEnd);
#endif
		else
			++it;

	bool bRemoved = (newEnd != vector.end());
	vector.erase(newEnd, vector.end());
	return bRemoved;
}

/// Generates a sequence of consecutive numbers.
template <class Counter>
struct increment_gen
{
	typedef Counter value_type;
	value_type counter;

	increment_gen(value_type init)
		: counter(init) { }

	LEAN_INLINE value_type operator ()() { return counter++; }
};

/// Generates a sequence of consecutive numbers.
template <class Counter>
struct decrement_gen
{
	typedef Counter value_type;
	value_type counter;

	decrement_gen(value_type onePlusInit)
		: counter(onePlusInit) { }

	LEAN_INLINE value_type operator ()() { return --counter; }
};

} // namespace

using functional::equal;
using functional::lexicographical_compare;
using functional::insert_last;
using functional::push_unique;
using functional::push_sorted_unique;
using functional::push_sorted;
using functional::find_sorted;
using functional::remove;
using functional::remove_all;
using functional::remove_unordered;

using functional::increment_gen;
using functional::decrement_gen;

} // namespace

#endif