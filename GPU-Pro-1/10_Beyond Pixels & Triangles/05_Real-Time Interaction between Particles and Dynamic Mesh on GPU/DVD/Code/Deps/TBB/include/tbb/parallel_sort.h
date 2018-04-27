/*
    Copyright 2005-2008 Intel Corporation.  All Rights Reserved.

    This file is part of Threading Building Blocks.

    Threading Building Blocks is free software; you can redistribute it
    and/or modify it under the terms of the GNU General Public License
    version 2 as published by the Free Software Foundation.

    Threading Building Blocks is distributed in the hope that it will be
    useful, but WITHOUT ANY WARRANTY; without even the implied warranty
    of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Threading Building Blocks; if not, write to the Free Software
    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

    As a special exception, you may use this file as part of a free software
    library without restriction.  Specifically, if other files instantiate
    templates or use macros or inline functions from this file, or you compile
    this file and link it with other files to produce an executable, this
    file does not by itself cause the resulting executable to be covered by
    the GNU General Public License.  This exception does not however
    invalidate any other reasons why the executable file might be covered by
    the GNU General Public License.
*/

#ifndef __TBB_parallel_sort_H
#define __TBB_parallel_sort_H

#include "parallel_for.h"
#include <algorithm>
#include <iterator>
#include <functional>

namespace tbb {

//! @cond INTERNAL
namespace internal {

//! Range used in quicksort to split elements into subranges based on a value.
/** The split operation selects a splitter and places all elements less than or equal to the value in the first range and the remaining elements in the second range.
    @ingroup algorithms */
template<typename RandomAccessIterator, typename Compare>
struct quick_sort_range {
    static const size_t grainsize = 500;
    const Compare &comp;
    RandomAccessIterator begin;
    size_t size;

    quick_sort_range( RandomAccessIterator begin_, size_t size_, const Compare &comp_ ) :
        comp(comp_), begin(begin_), size(size_) {}

    bool empty() const {return size==0;}
    bool is_divisible() const {return size>=grainsize;}

    quick_sort_range( quick_sort_range& range, split ) : comp(range.comp) {
        RandomAccessIterator array = range.begin;
        RandomAccessIterator key0 = range.begin; 
        size_t m = range.size/2u;
        std::swap ( array[0], array[m] );

        size_t i=0;
        size_t j=range.size;
        // Partition interval [i+1,j-1] with key *key0.
        for(;;) {
            __TBB_ASSERT( i<j, NULL );
            // Loop must terminate since array[l]==*key0.
            do {
                --j;
                __TBB_ASSERT( i<=j, "bad ordering relation?" );
            } while( comp( *key0, array[j] ));
            do {
                __TBB_ASSERT( i<=j, NULL );
                if( i==j ) goto partition;
                ++i;
            } while( comp( array[i],*key0 ));
            if( i==j ) goto partition;
            std::swap( array[i], array[j] );
        }
partition:
        // Put the partition key were it belongs
        std::swap( array[j], *key0 );
        // array[l..j) is less or equal to key.
        // array(j..r) is greater or equal to key.
        // array[j] is equal to key
        i=j+1;
        begin = array+i;
        size = range.size-i;
        range.size = j;
    }
};

//! Body class used to sort elements in a range that is smaller than the grainsize.
/** @ingroup algorithms */
template<typename RandomAccessIterator, typename Compare>
struct quick_sort_body {
    void operator()( const quick_sort_range<RandomAccessIterator,Compare>& range ) const {
        //SerialQuickSort( range.begin, range.size, range.comp );
        std::sort( range.begin, range.begin + range.size, range.comp );
    }
};

//! Wrapper method to initiate the sort by calling parallel_for.
/** @ingroup algorithms */
template<typename RandomAccessIterator, typename Compare>
void parallel_quick_sort( RandomAccessIterator begin, RandomAccessIterator end, const Compare& comp ) {
    parallel_for( quick_sort_range<RandomAccessIterator,Compare>(begin, end-begin, comp ), quick_sort_body<RandomAccessIterator,Compare>() );
}

} // namespace internal
//! @endcond

/** \page parallel_sort_iter_req Requirements on iterators for parallel_sort
    Requirements on value type \c T of \c RandomAccessIterator for \c parallel_sort:
    - \code void swap( T& x, T& y ) \endcode        Swaps \c x and \c y
    - \code bool Compare::operator()( const T& x, const T& y ) \endcode
                                                    True if x comes before y;
**/

/** \name parallel_sort
    See also requirements on \ref parallel_sort_iter_req "iterators for parallel_sort". **/
//@{

//! Sorts the data in [begin,end) using the given comparator 
/** The compare function object is used for all comparisons between elements during sorting.
    The compare object must define a bool operator() function.
    @ingroup algorithms **/
template<typename RandomAccessIterator, typename Compare>
void parallel_sort( RandomAccessIterator begin, RandomAccessIterator end, const Compare& comp) { 
    const int min_parallel_size = 500; 
    if( end > begin ) {
        if (end - begin < min_parallel_size) { 
            std::sort(begin, end, comp);
        } else {
            internal::parallel_quick_sort(begin, end, comp);
        }
    }
}

//! Sorts the data in [begin,end) with a default comparator \c std::less<RandomAccessIterator>
/** @ingroup algorithms **/
template<typename RandomAccessIterator>
inline void parallel_sort( RandomAccessIterator begin, RandomAccessIterator end ) { 
    parallel_sort( begin, end, std::less< typename std::iterator_traits<RandomAccessIterator>::value_type >() );
}

//! Sorts the data in the range \c [begin,end) with a default comparator \c std::less<T>
/** @ingroup algorithms **/
template<typename T>
inline void parallel_sort( T * begin, T * end ) {
    parallel_sort( begin, end, std::less< T >() );
}   
//@}


} // namespace tbb

#endif

