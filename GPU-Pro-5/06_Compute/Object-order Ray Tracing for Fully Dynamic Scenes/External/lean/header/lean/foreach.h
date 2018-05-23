/*****************************************************/
/* lean Macros                  (c) Tobias Zirr 2011 */
/*****************************************************/

#pragma once
#ifndef LEAN_FOREACH_H
#define LEAN_FOREACH_H

#include "lean.h"
#include <iterator>
#include "meta/type_traits.h"

namespace lean
{
	namespace impl
	{

		template <class Range>
		struct foreach_range_iterator { typedef typename Range::iterator type; };
		template <class Range>
		struct foreach_range_iterator<const Range> { typedef typename Range::const_iterator type; };

		template <class Iterator, bool IsIterator>
		struct foreach_iterator_traits
		{
			typedef typename std::iterator_traits<Iterator>::pointer pointer;
			typedef typename std::iterator_traits<Iterator>::reference reference;
		};
		template <class Iterator>
		struct foreach_iterator_traits<Iterator, false>
		{
			typedef const int* pointer;
			typedef const int& reference;
		};

		struct foreach_iterator_base
		{
			LEAN_INLINE operator bool() const { return false; }
		};

		using std::begin;
		using std::end;
		
		template <class Range>
		struct foreach_iterator : public foreach_iterator_base
		{
			typedef typename foreach_range_iterator<Range>::type iterator;
			iterator m_end;
			mutable iterator it;

			LEAN_INLINE foreach_iterator() { }
			LEAN_INLINE foreach_iterator(Range &r)
				: m_end(end(r)),
				it(begin(r)) { }

			typedef foreach_iterator_traits<iterator, is_iterator<iterator>::value> traits;
			LEAN_INLINE typename traits::reference operator *() const { return *it; }
			LEAN_INLINE typename traits::pointer operator ->() const { return &*it; }
			LEAN_INLINE operator iterator() const { return it; }
		};
		
		template <class Range>
		LEAN_INLINE Range* foreach_range_nullptr(Range &r) { return 0; }
		template <class Range>
		LEAN_INLINE const Range* foreach_range_nullptr(const Range &r) { return 0; }

		template <class Range>
		LEAN_INLINE foreach_iterator<Range> make_foreach_iterator(Range*)
		{
			return foreach_iterator<Range>();
		}
		template <class Range>
		LEAN_INLINE foreach_iterator<Range> make_foreach_iterator(Range &r)
		{
			return foreach_iterator<Range>(r);
		}
		template <class Range>
		LEAN_INLINE foreach_iterator<const Range> make_foreach_iterator(const Range &r)
		{
			return foreach_iterator<const Range>(r);
		}

		template <class Range>
		LEAN_INLINE bool check_foreach_iterator(const foreach_iterator_base &i, Range*)
		{
			const foreach_iterator<Range> &iter = static_cast<const foreach_iterator<Range>&>(i);
			return (iter.it != iter.m_end);
		}
		template <class Range>
		LEAN_INLINE void increment_foreach_iterator(const foreach_iterator_base &i, Range*)
		{
			const foreach_iterator<Range> &iter = static_cast<const foreach_iterator<Range>&>(i);
			++iter.it;
		}

	} // namespace
} // namespace

/// @addtogroup GlobalMacros
/// @{
/*
/// Iterates over all elements in the given range.
#define LEAN_FOREACH(iterator, range_type, range_var, range) \
	if (const ::lean::impl::foreach_iterator_base &__lean__iterator__ = ::lean::impl::make_foreach_iterator(true ? 0 : ::lean::impl::foreach_range_nullptr(range))) { } else \
		for (range_type range_var = range; \
			::lean::impl::check_foreach_iterator(iterator, true ? 0 : ::lean::impl::foreach_range_nullptr(range)); \
			::lean::impl::increment_foreach_iterator(iterator, true ? 0 : ::lean::impl::foreach_range_nullptr(range)))


/// Iterates over all elements in the given range.
#define LEAN_FOREACH(iterator, range) \
	for (const ::lean::impl::foreach_iterator_base &iterator = ::lean::impl::make_foreach_iterator(range); \
		::lean::impl::check_foreach_iterator(iterator, true ? 0 : ::lean::impl::foreach_range_nullptr(range)); \
		::lean::impl::increment_foreach_iterator(iterator, true ? 0 : ::lean::impl::foreach_range_nullptr(range)))
*/

/// @}

#endif