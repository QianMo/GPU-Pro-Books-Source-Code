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

#ifndef __TBB_concurrent_hash_map_H
#define __TBB_concurrent_hash_map_H

#include <stdexcept>
#include <iterator>
#include <utility>      // Need std::pair from here
#include "tbb_stddef.h"
#include "cache_aligned_allocator.h"
#include "tbb_allocator.h"
#include "spin_rw_mutex.h"
#include "atomic.h"
#include "aligned_space.h"
#if TBB_PERFORMANCE_WARNINGS
#include <typeinfo>
#endif

namespace tbb {

template<typename Key, typename T, typename HashCompare, typename A = tbb_allocator<std::pair<Key, T> > >
class concurrent_hash_map;

//! @cond INTERNAL
namespace internal {
    //! base class of concurrent_hash_map
    class hash_map_base {
    public:
        // Mutex types for each layer of the container
        typedef spin_rw_mutex node_mutex_t;
        typedef spin_rw_mutex chain_mutex_t;
        typedef spin_rw_mutex segment_mutex_t;

        //! Type of a hash code.
        typedef size_t hashcode_t;
        //! Log2 of n_segment
        static const size_t n_segment_bits = 6;
        //! Number of segments 
        static const size_t n_segment = size_t(1)<<n_segment_bits; 
        //! Maximum size of array of chains
        static const size_t max_physical_size = size_t(1)<<(8*sizeof(hashcode_t)-n_segment_bits);
    };

    template<typename Iterator>
    class hash_map_range;

    struct hash_map_segment_base {
        //! Mutex that protects this segment
        hash_map_base::segment_mutex_t my_mutex;

        // Number of nodes
        atomic<size_t> my_logical_size;

        // Size of chains
        /** Always zero or a power of two */
        size_t my_physical_size;

        //! True if my_logical_size>=my_physical_size.
        /** Used to support Intel(R) Thread Checker. */
        bool internal_grow_predicate() const;
    };

    //! Meets requirements of a forward iterator for STL */
    /** Value is either the T or const T type of the container.
        @ingroup containers */ 
    template<typename Container, typename Value>
    class hash_map_iterator
#if defined(_WIN64) && defined(_MSC_VER) 
        // Ensure that Microsoft's internal template function _Val_type works correctly.
        : public std::iterator<std::forward_iterator_tag,Value>
#endif /* defined(_WIN64) && defined(_MSC_VER) */
    {
        typedef typename Container::node node;
        typedef typename Container::chain chain;

        //! concurrent_hash_map over which we are iterating.
        Container* my_table;

        //! Pointer to node that has current item
        node* my_node;

        //! Index into hash table's array for current item
        size_t my_array_index;

        //! Index of segment that has array for current item
        size_t my_segment_index;

        template<typename C, typename T, typename U>
        friend bool operator==( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend bool operator!=( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );

        template<typename C, typename T, typename U>
        friend ptrdiff_t operator-( const hash_map_iterator<C,T>& i, const hash_map_iterator<C,U>& j );
    
        template<typename C, typename U>
        friend class internal::hash_map_iterator;

        template<typename I>
        friend class internal::hash_map_range;

        void advance_to_next_node() {
            size_t i = my_array_index+1;
            do {
                while( i<my_table->my_segment[my_segment_index].my_physical_size ) {
                    my_node = my_table->my_segment[my_segment_index].my_array[i].node_list;
                    if( my_node ) goto done;
                    ++i;
                }
                i = 0;
            } while( ++my_segment_index<my_table->n_segment );
        done:
            my_array_index = i;
        }
#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
        template<typename Key, typename T, typename HashCompare, typename A>
        friend class tbb::concurrent_hash_map;
#else
    public: // workaround
#endif
        hash_map_iterator( const Container& table, size_t segment_index, size_t array_index=0, node* b=NULL );
    public:
        //! Construct undefined iterator
        hash_map_iterator() {}
        hash_map_iterator( const hash_map_iterator<Container,typename Container::value_type>& other ) :
            my_table(other.my_table),
            my_node(other.my_node),
            my_array_index(other.my_array_index),
            my_segment_index(other.my_segment_index)
        {}
        Value& operator*() const {
            __TBB_ASSERT( my_node, "iterator uninitialized or at end of container?" );
            return my_node->item;
        }
        Value* operator->() const {return &operator*();}
        hash_map_iterator& operator++();
        
        //! Post increment
        Value* operator++(int) {
            Value* result = &operator*();
            operator++();
            return result;
        }

        // STL support

        typedef ptrdiff_t difference_type;
        typedef Value value_type;
        typedef Value* pointer;
        typedef Value& reference;
        typedef const Value& const_reference;
        typedef std::forward_iterator_tag iterator_category;
    };

    template<typename Container, typename Value>
    hash_map_iterator<Container,Value>::hash_map_iterator( const Container& table, size_t segment_index, size_t array_index, node* b ) : 
        my_table(const_cast<Container*>(&table)),
        my_node(b),
        my_array_index(array_index),
        my_segment_index(segment_index)
    {
        if( segment_index<my_table->n_segment ) {
            if( !my_node ) {
                chain* first_chain = my_table->my_segment[segment_index].my_array;
                if( first_chain ) my_node = first_chain[my_array_index].node_list;
            }
            if( !my_node ) advance_to_next_node();
        }
    }

    template<typename Container, typename Value>
    hash_map_iterator<Container,Value>& hash_map_iterator<Container,Value>::operator++() {
        my_node=my_node->next;
        if( !my_node ) advance_to_next_node();
        return *this;
    }

    template<typename Container, typename T, typename U>
    bool operator==( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
        return i.my_node==j.my_node;
    }

    template<typename Container, typename T, typename U>
    bool operator!=( const hash_map_iterator<Container,T>& i, const hash_map_iterator<Container,U>& j ) {
        return i.my_node!=j.my_node;
    }

    //! Range class used with concurrent_hash_map
    /** @ingroup containers */ 
    template<typename Iterator>
    class hash_map_range {
    private:
        Iterator my_begin;
        Iterator my_end;
        mutable Iterator my_midpoint;
        size_t my_grainsize;
        //! Set my_midpoint to point approximately half way between my_begin and my_end.
        void set_midpoint() const;
        template<typename U> friend class hash_map_range;
    public:
        //! Type for size of a range
        typedef std::size_t size_type;
        typedef typename Iterator::value_type value_type;
        typedef typename Iterator::reference reference;
        typedef typename Iterator::const_reference const_reference;
        typedef typename Iterator::difference_type difference_type;
        typedef Iterator iterator;

        //! True if range is empty.
        bool empty() const {return my_begin==my_end;}

        //! True if range can be partitioned into two subranges.
        bool is_divisible() const {
            return my_midpoint!=my_end;
        }
        //! Split range.
        hash_map_range( hash_map_range& r, split ) : 
            my_end(r.my_end),
            my_grainsize(r.my_grainsize)
        {
            r.my_end = my_begin = r.my_midpoint;
            set_midpoint();
            r.set_midpoint();
        }
        //! type conversion
        template<typename U>
        hash_map_range( hash_map_range<U>& r) : 
            my_begin(r.my_begin),
            my_end(r.my_end),
            my_midpoint(r.my_midpoint),
            my_grainsize(r.my_grainsize)
        {}
        //! Init range with iterators and grainsize specified
        hash_map_range( const Iterator& begin_, const Iterator& end_, size_type grainsize = 1 ) : 
            my_begin(begin_), 
            my_end(end_), 
            my_grainsize(grainsize) 
        {
            set_midpoint();
            __TBB_ASSERT( grainsize>0, "grainsize must be positive" );
        }
        const Iterator& begin() const {return my_begin;}
        const Iterator& end() const {return my_end;}
        //! The grain size for this range.
        size_type grainsize() const {return my_grainsize;}
    };

    template<typename Iterator>
    void hash_map_range<Iterator>::set_midpoint() const {
        size_t n = my_end.my_segment_index-my_begin.my_segment_index;
        if( n>1 || (n==1 && my_end.my_array_index>0) ) {
            // Split by groups of segments
            my_midpoint = Iterator(*my_begin.my_table,(my_end.my_segment_index+my_begin.my_segment_index)/2u);
        } else {
            // Split by groups of nodes
            size_t m = my_end.my_array_index-my_begin.my_array_index;
            if( m>my_grainsize ) {
                my_midpoint = Iterator(*my_begin.my_table,my_begin.my_segment_index,m/2u);
            } else {
                my_midpoint = my_end;
            }
        }
        __TBB_ASSERT( my_midpoint.my_segment_index<=my_begin.my_table->n_segment, NULL );
    }  
} // namespace internal
//! @endcond

//! Unordered map from Key to T.
/** concurrent_hash_map is associative container with concurrent access.

@par Compatibility
    The class meets all Container Requirements from C++ Standard (See ISO/IEC 14882:2003(E), clause 23.1).

@par Exception Safety
    - Hash function is not permitted to throw an exception. User-defined types Key and T are forbidden from throwing an exception in destructors.
    - If exception happens during insert() operations, it has no effect (unless exception raised by HashCompare::hash() function during grow_segment).
    - If exception happens during operator=() operation, the container can have a part of source items, and methods size() and empty() can return wrong results.

@par Changes since TBB 2.0
    - Fixed exception-safety
    - Added template argument for allocator
    - Added allocator argument in constructors
    - Added constructor from a range of iterators
    - Added several new overloaded insert() methods
    - Added get_allocator()
    - Added swap()
    - Added count()
    - Added overloaded erase(accessor &) and erase(const_accessor&)
    - Added equal_range() [const]
    - Added [const_]pointer, [const_]reference, and allocator_type types
    - Added global functions: operator==(), operator!=(), and swap() 

    @ingroup containers */
template<typename Key, typename T, typename HashCompare, typename A>
class concurrent_hash_map : protected internal::hash_map_base {
    template<typename Container, typename Value>
    friend class internal::hash_map_iterator;

    template<typename I>
    friend class internal::hash_map_range;

    struct node;
    friend struct node;
    typedef typename A::template rebind<node>::other node_allocator_type;

public:
    class const_accessor;
    friend class const_accessor;
    class accessor;

    typedef Key key_type;
    typedef T mapped_type;
    typedef std::pair<const Key,T> value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef internal::hash_map_iterator<concurrent_hash_map,value_type> iterator;
    typedef internal::hash_map_iterator<concurrent_hash_map,const value_type> const_iterator;
    typedef internal::hash_map_range<iterator> range_type;
    typedef internal::hash_map_range<const_iterator> const_range_type;
    typedef A allocator_type;

    //! Combines data access, locking, and garbage collection.
    class const_accessor {
        friend class concurrent_hash_map;
        friend class accessor;
        void operator=( const accessor& ) const; // Deny access
        const_accessor( const accessor& );       // Deny access
    public:
        //! Type of value
        typedef const std::pair<const Key,T> value_type;

        //! True if result is empty.
        bool empty() const {return !my_node;}

        //! Set to null
        void release() {
            if( my_node ) {
                my_lock.release();
                my_node = NULL;
            }
        }

        //! Return reference to associated value in hash table.
        const_reference operator*() const {
            __TBB_ASSERT( my_node, "attempt to dereference empty accessor" );
            return my_node->item;
        }

        //! Return pointer to associated value in hash table.
        const_pointer operator->() const {
            return &operator*();
        }

        //! Create empty result
        const_accessor() : my_node(NULL) {}

        //! Destroy result after releasing the underlying reference.
        ~const_accessor() {
            my_node = NULL; // my_lock.release() is called in scoped_lock destructor
        }
    private:
        node* my_node;
        node_mutex_t::scoped_lock my_lock;
        hashcode_t my_hash;
    };

    //! Allows write access to elements and combines data access, locking, and garbage collection.
    class accessor: public const_accessor {
    public:
        //! Type of value
        typedef std::pair<const Key,T> value_type;

        //! Return reference to associated value in hash table.
        reference operator*() const {
            __TBB_ASSERT( this->my_node, "attempt to dereference empty accessor" );
            return this->my_node->item;
        }

        //! Return pointer to associated value in hash table.
        pointer operator->() const {
            return &operator*();
        }       
    };

    //! Construct empty table.
    concurrent_hash_map(const allocator_type &a = allocator_type())
        : my_allocator(a)

    {
        initialize();
    }

    //! Copy constructor
    concurrent_hash_map( const concurrent_hash_map& table, const allocator_type &a = allocator_type())
        : my_allocator(a)
    {
        initialize();
        internal_copy(table);
    }

    //! Construction with copying iteration range and given allocator instance
    template<typename I>
    concurrent_hash_map(I first, I last, const allocator_type &a = allocator_type())
        : my_allocator(a)
    {
        initialize();
        internal_copy(first, last);
    }

    //! Assignment
    concurrent_hash_map& operator=( const concurrent_hash_map& table ) {
        if( this!=&table ) {
            clear();
            internal_copy(table);
        } 
        return *this;
    }


    //! Clear table
    void clear();

    //! Clear table and destroy it.  
    ~concurrent_hash_map();

    //------------------------------------------------------------------------
    // Parallel algorithm support
    //------------------------------------------------------------------------
    range_type range( size_type grainsize=1 ) {
        return range_type( begin(), end(), grainsize );
    }
    const_range_type range( size_type grainsize=1 ) const {
        return const_range_type( begin(), end(), grainsize );
    }

    //------------------------------------------------------------------------
    // STL support - not thread-safe methods
    //------------------------------------------------------------------------
    iterator begin() {return iterator(*this,0);}
    iterator end() {return iterator(*this,n_segment);}
    const_iterator begin() const {return const_iterator(*this,0);}
    const_iterator end() const {return const_iterator(*this,n_segment);}
    std::pair<iterator, iterator> equal_range( const Key& key ) { return internal_equal_range(key, end()); }
    std::pair<const_iterator, const_iterator> equal_range( const Key& key ) const { return internal_equal_range(key, end()); }
    
    //! Number of items in table.
    /** Be aware that this method is relatively slow compared to the 
        typical size() method for an STL container. */
    size_type size() const;

    //! True if size()==0.
    bool empty() const;

    //! Upper bound on size.
    size_type max_size() const {return (~size_type(0))/sizeof(node);}

    //! return allocator object
    allocator_type get_allocator() const { return this->my_allocator; }

    //! swap two instances
    void swap(concurrent_hash_map &table);

    //------------------------------------------------------------------------
    // concurrent map operations
    //------------------------------------------------------------------------

    //! Return count of items (0 or 1)
    size_type count( const Key& key ) const {
        return const_cast<concurrent_hash_map*>(this)->lookup</*insert*/false>(NULL, key, /*write=*/false, NULL );
    }

    //! Find item and acquire a read lock on the item.
    /** Return true if item is found, false otherwise. */
    bool find( const_accessor& result, const Key& key ) const {
        return const_cast<concurrent_hash_map*>(this)->lookup</*insert*/false>(&result, key, /*write=*/false, NULL );
    }

    //! Find item and acquire a write lock on the item.
    /** Return true if item is found, false otherwise. */
    bool find( accessor& result, const Key& key ) {
        return lookup</*insert*/false>(&result, key, /*write=*/true, NULL );
    }
        
    //! Insert item (if not already present) and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor& result, const Key& key ) {
        return lookup</*insert*/true>(&result, key, /*write=*/false, NULL );
    }

    //! Insert item (if not already present) and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor& result, const Key& key ) {
        return lookup</*insert*/true>(&result, key, /*write=*/true, NULL );
    }

    //! Insert item by copying if there is no such key present already and acquire a read lock on the item.
    /** Returns true if item is new. */
    bool insert( const_accessor& result, const value_type& value ) {
        return lookup</*insert*/true>(&result, value.first, /*write=*/false, &value.second );
    }

    //! Insert item by copying if there is no such key present already and acquire a write lock on the item.
    /** Returns true if item is new. */
    bool insert( accessor& result, const value_type& value ) {
        return lookup</*insert*/true>(&result, value.first, /*write=*/true, &value.second );
    }

    //! Insert item by copying if there is no such key present already
    /** Returns true if item is inserted. */
    bool insert( const value_type& value ) {
        return lookup</*insert*/true>(NULL, value.first, /*write=*/false, &value.second );
    }

    //! Insert range [first, last)
    template<typename I>
    void insert(I first, I last) {
        for(; first != last; ++first)
            insert( *first );
    }

    //! Erase item.
    /** Return true if item was erased by particularly this call. */
    bool erase( const Key& key );

    //! Erase item by const_accessor.
    /** Return true if item was erased by particularly this call. */
    bool erase( const_accessor& item_accessor ) {
        return exclude( item_accessor, /*readonly=*/ true );
    }

    //! Erase item by accessor.
    /** Return true if item was erased by particularly this call. */
    bool erase( accessor& item_accessor ) {
        return exclude( item_accessor, /*readonly=*/ false );
    }

private:
    //! Basic unit of storage used in chain.
    struct node {
        //! Next node in chain
        node* next;
        node_mutex_t mutex;
        value_type item;
        node( const Key& key ) : item(key, T()) {}
        node( const Key& key, const T& t ) : item(key, t) {}
        // exception-safe allocation, see C++ Standard 2003, clause 5.3.4p17
        void* operator new( size_t size, node_allocator_type& a ) {
            void *ptr = a.allocate(1);
            if(!ptr) throw std::bad_alloc();
            return ptr;
        }
        // match placement-new form above to be called if exception thrown in constructor
        void operator delete( void* ptr, node_allocator_type& a ) {return a.deallocate(static_cast<node*>(ptr),1); }
    };

    struct chain;
    friend struct chain;

    //! A linked-list of nodes.
    /** Should be zero-initialized before use. */
    struct chain {
        void push_front( node& b ) {
            b.next = node_list;
            node_list = &b;
        }
        chain_mutex_t mutex;
        node* node_list;
    };

    struct segment;
    friend struct segment;

    //! Segment of the table.
    /** The table is partioned into disjoint segments to reduce conflicts.
        A segment should be zero-initialized before use. */
    struct segment: internal::hash_map_segment_base {
#if TBB_DO_ASSERT
        ~segment() {
            __TBB_ASSERT( !my_array, "should have been cleared earlier" );
        }
#endif /* TBB_DO_ASSERT */

        // Pointer to array of chains
        chain* my_array;

        // Get chain in this segment that corresponds to given hash code.
        chain& get_chain( hashcode_t hashcode, size_t n_segment_bits ) {
            return my_array[(hashcode>>n_segment_bits)&(my_physical_size-1)];
        }
     
        //! Allocate an array with at least new_size chains. 
        /** "new_size" is rounded up to a power of two that occupies at least one cache line.
            Does not deallocate the old array.  Overwrites my_array. */
        void allocate_array( size_t new_size ) {
            size_t n=(internal::NFS_GetLineSize()+sizeof(chain)-1)/sizeof(chain);
            __TBB_ASSERT((n&(n-1))==0, NULL);
            while( n<new_size ) n<<=1;
            chain* array = cache_aligned_allocator<chain>().allocate( n );
            // storing earlier might help overcome false positives of in deducing "bool grow" in concurrent threads
            __TBB_store_with_release(my_physical_size, n);
            memset( array, 0, n*sizeof(chain) );
            my_array = array;
        }
    };

    segment& get_segment( hashcode_t hashcode ) {
        return my_segment[hashcode&(n_segment-1)];
    }

    node_allocator_type my_allocator;

    HashCompare my_hash_compare;

    segment* my_segment;

    node* create_node(const Key& key, const T* t) {
        // exception-safe allocation and construction
        if(t) return new( my_allocator ) node(key, *t);
        else  return new( my_allocator ) node(key);
    }

    void delete_node(node* b) {
        my_allocator.destroy(b);
        my_allocator.deallocate(b, 1);
    }

    node* search_list( const Key& key, chain& c ) const {
        node* b = c.node_list;
        while( b && !my_hash_compare.equal(key, b->item.first) )
            b = b->next;
        return b;
    }
    //! Returns an iterator for an item defined by the key, or for the next item after it (if upper==true)
    template<typename I>
    std::pair<I, I> internal_equal_range( const Key& key, I end ) const;

    //! delete item by accessor
    bool exclude( const_accessor& item_accessor, bool readonly );

    //! Grow segment for which caller has acquired a write lock.
    void grow_segment( segment_mutex_t::scoped_lock& segment_lock, segment& s );

    //! Does heavy lifting for "find" and "insert".
    template<bool op_insert>
    bool lookup( const_accessor* result, const Key& key, bool write, const T* t );

    //! Perform initialization on behalf of a constructor
    void initialize() {
        my_segment = cache_aligned_allocator<segment>().allocate(n_segment);
        memset( my_segment, 0, sizeof(segment)*n_segment );
     }

    //! Copy "source" to *this, where *this must start out empty.
    void internal_copy( const concurrent_hash_map& source );

    template<typename I>
    void internal_copy(I first, I last);
};

template<typename Key, typename T, typename HashCompare, typename A>
concurrent_hash_map<Key,T,HashCompare,A>::~concurrent_hash_map() {
    clear();
    cache_aligned_allocator<segment>().deallocate( my_segment, n_segment );
}

template<typename Key, typename T, typename HashCompare, typename A>
typename concurrent_hash_map<Key,T,HashCompare,A>::size_type concurrent_hash_map<Key,T,HashCompare,A>::size() const {
    size_type result = 0;
    for( size_t k=0; k<n_segment; ++k )
        result += my_segment[k].my_logical_size;
    return result;
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::empty() const {
    for( size_t k=0; k<n_segment; ++k )
        if( my_segment[k].my_logical_size )
            return false;
    return true;
}

template<typename Key, typename T, typename HashCompare, typename A>
template<bool op_insert>
bool concurrent_hash_map<Key,T,HashCompare,A>::lookup( const_accessor* result, const Key& key, bool write, const T* t ) {
    if( result /*&& result->my_node -- checked in release() */ )
        result->release();
    const hashcode_t h = my_hash_compare.hash( key );
    segment& s = get_segment(h);
restart:
    bool return_value = false;
    // first check in double-check sequence
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
    const bool grow = op_insert && s.internal_grow_predicate();
#else
    const bool grow = op_insert && s.my_logical_size >= s.my_physical_size
        && s.my_physical_size < max_physical_size; // check whether there are free bits
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT */
    segment_mutex_t::scoped_lock segment_lock( s.my_mutex, /*write=*/grow );
    if( grow ) { // Load factor is too high  
        grow_segment( segment_lock, s );
    }
    if( !s.my_array ) {
        __TBB_ASSERT( !op_insert, NULL );
        return false;
    }
    __TBB_ASSERT( (s.my_physical_size&(s.my_physical_size-1))==0, NULL );
    chain& c = s.get_chain( h, n_segment_bits );
    chain_mutex_t::scoped_lock chain_lock( c.mutex, /*write=*/false );

    node* b = search_list( key, c );
    if( op_insert ) {
        if( !b ) {
            b = create_node(key, t);
            // Search failed
            if( !chain_lock.upgrade_to_writer() ) {
                // Rerun search_list, in case another thread inserted the item during the upgrade.
                node* b_temp = search_list( key, c );
                if( b_temp ) { // unfortunately, it did
                    chain_lock.downgrade_to_reader();
                    delete_node( b );
                    b = b_temp;
                    goto done;
                }
            }
            ++s.my_logical_size; // we can't change it earlier due to correctness of size() and exception safety of equal()
            return_value = true;
            c.push_front( *b );
        }
    } else { // find or count
        if( !b )      return false;
        return_value = true;
    }
done:
    if( !result ) return return_value;
    if( !result->my_lock.try_acquire( b->mutex, write ) ) {
        // we are unlucky, prepare for longer wait
        internal::AtomicBackoff trials;
        do {
            if( !trials.bounded_pause() ) {
                // the wait takes really long, restart the operation
                chain_lock.release(); segment_lock.release();
                __TBB_Yield();
                goto restart;
            }
        } while( !result->my_lock.try_acquire( b->mutex, write ) );
    }
    result->my_node = b;
    result->my_hash = h;
    return return_value;
}

template<typename Key, typename T, typename HashCompare, typename A>
template<typename I>
std::pair<I, I> concurrent_hash_map<Key,T,HashCompare,A>::internal_equal_range( const Key& key, I end ) const {
    hashcode_t h = my_hash_compare.hash( key );
    size_t segment_index = h&(n_segment-1);
    segment& s = my_segment[segment_index ];
    size_t chain_index = (h>>n_segment_bits)&(s.my_physical_size-1);
    if( !s.my_array )
        return std::make_pair(end, end);
    chain& c = s.my_array[chain_index];
    node* b = search_list( key, c );
    if( !b )
        return std::make_pair(end, end);
    iterator lower(*this, segment_index, chain_index, b), upper(lower);
    return std::make_pair(lower, ++upper);
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::erase( const Key &key ) {
    hashcode_t h = my_hash_compare.hash( key );
    segment& s = get_segment( h );
    node* b;
    {
        bool chain_locked_for_write = false;
        segment_mutex_t::scoped_lock segment_lock( s.my_mutex, /*write=*/false );
        if( !s.my_array ) return false;
        __TBB_ASSERT( (s.my_physical_size&(s.my_physical_size-1))==0, NULL );
        chain& c = s.get_chain( h, n_segment_bits );
        chain_mutex_t::scoped_lock chain_lock( c.mutex, /*write=*/false );
    search:
        node** p = &c.node_list;
        b = *p;
        while( b && !my_hash_compare.equal(key, b->item.first ) ) {
            p = &b->next;
            b = *p;
        }
        if( !b ) return false;
        if( !chain_locked_for_write && !chain_lock.upgrade_to_writer() ) {
            chain_locked_for_write = true;
            goto search;
        }
        *p = b->next;
        --s.my_logical_size;
    }
    {
        node_mutex_t::scoped_lock item_locker( b->mutex, /*write=*/true );
    }
    // note: there should be no threads pretending to acquire this mutex again, do not try to upgrade const_accessor!
    delete_node( b ); // Only one thread can delete it due to write lock on the chain_mutex
    return true;        
}

template<typename Key, typename T, typename HashCompare, typename A>
bool concurrent_hash_map<Key,T,HashCompare,A>::exclude( const_accessor &item_accessor, bool readonly ) {
    __TBB_ASSERT( item_accessor.my_node, NULL );
    const hashcode_t h = item_accessor.my_hash;
    node *const b = item_accessor.my_node;
    item_accessor.my_node = NULL; // we ought release accessor anyway
    segment& s = get_segment( h );
    {
        segment_mutex_t::scoped_lock segment_lock( s.my_mutex, /*write=*/false );
        __TBB_ASSERT( s.my_array, NULL );
        __TBB_ASSERT( (s.my_physical_size&(s.my_physical_size-1))==0, NULL );
        chain& c = s.get_chain( h, n_segment_bits );
        chain_mutex_t::scoped_lock chain_lock( c.mutex, /*write=*/true );
        node** p = &c.node_list;
        while( *p && *p != b )
            p = &(*p)->next;
        if( !*p ) { // someone else was the first
            item_accessor.my_lock.release();
            return false;
        }
        __TBB_ASSERT( *p == b, NULL );
        *p = b->next;
        --s.my_logical_size;
    }
    if( readonly ) // need to get exclusive lock
        item_accessor.my_lock.upgrade_to_writer(); // return value means nothing here
    item_accessor.my_lock.release();
    delete_node( b ); // Only one thread can delete it due to write lock on the chain_mutex
    return true;
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::swap(concurrent_hash_map<Key,T,HashCompare,A> &table) {
    std::swap(this->my_allocator, table.my_allocator);
    std::swap(this->my_hash_compare, table.my_hash_compare);
    std::swap(this->my_segment, table.my_segment);
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::clear() {
#if TBB_PERFORMANCE_WARNINGS
    size_t total_physical_size = 0, min_physical_size = size_t(-1L), max_physical_size = 0; //< usage statistics
    static bool reported = false;
#endif
    for( size_t i=0; i<n_segment; ++i ) {
        segment& s = my_segment[i];
        size_t n = s.my_physical_size;
        if( chain* array = s.my_array ) {
            s.my_array = NULL;
            s.my_physical_size = 0;
            s.my_logical_size = 0;
            for( size_t j=0; j<n; ++j ) {
                while( node* b = array[j].node_list ) {
                    array[j].node_list = b->next;
                    delete_node(b);
                }
            }
            cache_aligned_allocator<chain>().deallocate( array, n );
        }
#if TBB_PERFORMANCE_WARNINGS
        total_physical_size += n;
        if(min_physical_size > n) min_physical_size = n;
        if(max_physical_size < n) max_physical_size = n;
    }
    if( !reported
        && ( (total_physical_size >= n_segment*48 && min_physical_size < total_physical_size/n_segment/2)
         || (total_physical_size >= n_segment*128 && max_physical_size > total_physical_size/n_segment*2) ) )
    {
        reported = true;
        internal::runtime_warning(
            "Performance is not optimal because the hash function produces bad randomness in lower bits in %s",
            typeid(*this).name() );
#endif
    }
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::grow_segment( segment_mutex_t::scoped_lock& segment_lock, segment& s ) {
    // Following is second check in a double-check.
    if( s.my_logical_size >= s.my_physical_size ) {
        chain* old_array = s.my_array;
        size_t old_size = s.my_physical_size;
        s.allocate_array( s.my_logical_size+1 );
        for( size_t k=0; k<old_size; ++k )
            while( node* b = old_array[k].node_list ) {
                old_array[k].node_list = b->next;
                hashcode_t h = my_hash_compare.hash( b->item.first );
                __TBB_ASSERT( &get_segment(h)==&s, "hash function changed?" );
                s.get_chain(h,n_segment_bits).push_front(*b);
            }
        cache_aligned_allocator<chain>().deallocate( old_array, old_size );
    }
    segment_lock.downgrade_to_reader();
}

template<typename Key, typename T, typename HashCompare, typename A>
void concurrent_hash_map<Key,T,HashCompare,A>::internal_copy( const concurrent_hash_map& source ) {
    for( size_t i=0; i<n_segment; ++i ) {
        segment& s = source.my_segment[i];
        __TBB_ASSERT( !my_segment[i].my_array, "caller should have cleared" );
        if( s.my_logical_size ) {
            segment& d = my_segment[i];
            d.allocate_array( s.my_logical_size );
            d.my_logical_size = s.my_logical_size;
            size_t s_size = s.my_physical_size;
            chain* s_array = s.my_array;
            chain* d_array = d.my_array;
            for( size_t k=0; k<s_size; ++k )
                for( node* b = s_array[k].node_list; b; b=b->next ) {
                    __TBB_ASSERT( &get_segment(my_hash_compare.hash( b->item.first ))==&d, "hash function changed?" );
                    node* b_new = create_node(b->item.first, &b->item.second);
                    d_array[k].push_front(*b_new); // hashcode is the same and segment and my_physical sizes are the same
                }
        }
    }
}

template<typename Key, typename T, typename HashCompare, typename A>
template<typename I>
void concurrent_hash_map<Key,T,HashCompare,A>::internal_copy(I first, I last) {
    for(; first != last; ++first)
        insert( *first );
}

template<typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator==(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b) {
    if(a.size() != b.size()) return false;
    typename concurrent_hash_map<Key, T, HashCompare, A1>::const_iterator i(a.begin()), i_end(a.end());
    typename concurrent_hash_map<Key, T, HashCompare, A2>::const_iterator j, j_end(b.end());
    for(; i != i_end; ++i) {
        j = b.equal_range(i->first).first;
        if( j == j_end || !(i->second == j->second) ) return false;
    }
    return true;
}

template<typename Key, typename T, typename HashCompare, typename A1, typename A2>
inline bool operator!=(const concurrent_hash_map<Key, T, HashCompare, A1> &a, const concurrent_hash_map<Key, T, HashCompare, A2> &b)
{    return !(a == b); }

template<typename Key, typename T, typename HashCompare, typename A>
inline void swap(concurrent_hash_map<Key, T, HashCompare, A> &a, concurrent_hash_map<Key, T, HashCompare, A> &b)
{    a.swap( b ); }

} // namespace tbb

#endif /* __TBB_concurrent_hash_map_H */
