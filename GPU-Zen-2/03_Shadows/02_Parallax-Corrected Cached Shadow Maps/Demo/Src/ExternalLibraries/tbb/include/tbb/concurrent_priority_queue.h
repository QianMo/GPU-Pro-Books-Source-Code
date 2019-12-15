/*
    Copyright 2005-2010 Intel Corporation.  All Rights Reserved.

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

#ifndef __TBB_concurrent_priority_queue_H
#define __TBB_concurrent_priority_queue_H

#if !TBB_PREVIEW_CONCURRENT_PRIORITY_QUEUE
#error Set TBB_PREVIEW_CONCURRENT_PRIORITY_QUEUE to include concurrent_priority_queue.h
#endif

#include "atomic.h"
#include "cache_aligned_allocator.h"
#include "tbb_exception.h"
#include "tbb_stddef.h"
#include "tbb_profiling.h"
#include <iterator>
#include <functional>

namespace tbb {
namespace interface5 {

//! Concurrent priority queue
template <typename T, typename Compare=std::less<T>, typename A=cache_aligned_allocator<T> >
class concurrent_priority_queue {
 public:
    //! Element type in the queue.
    typedef T value_type;

    //! Reference type
    typedef T& reference;

    //! Const reference type
    typedef const T& const_reference;

    //! Integral type for representing size of the queue.
    typedef size_t size_type;

    //! Difference type for iterator
    typedef ptrdiff_t difference_type;

    //! Allocator type
    typedef A allocator_type;

    //! Constructs a new concurrent_priority_queue with default capacity
    explicit concurrent_priority_queue(const allocator_type& a = allocator_type()) : my_helper(a) { 
        internal_construct(0); 
    }
    
    //! Constructs a new concurrent_priority_queue with init_sz capacity
    explicit concurrent_priority_queue(size_type init_capacity, const allocator_type& a = allocator_type()) : my_helper(a) { 
        internal_construct(init_capacity); 
    }
    
    //! [begin,end) constructor
    template<typename InputIterator>
    concurrent_priority_queue(InputIterator begin, InputIterator end, const allocator_type& a = allocator_type()) : my_helper(a)
    {
        internal_iterator_construct(begin, end, typename std::iterator_traits<InputIterator>::iterator_category());
    }
    
    //! Copy constructor
    /** State of this queue may not reflect results of pending operations on the copied queue. */
    concurrent_priority_queue(const concurrent_priority_queue& src, const allocator_type& a = allocator_type()) : my_helper(a)
    {
        internal_construct(src.my_size);
        my_size = src.my_size;
        mark = src.mark;
        __TBB_TRY {
            internal_copy(src.data, data, my_size);
        } __TBB_CATCH(...) {
            my_helper.deallocate(data, my_capacity);
            __TBB_RETHROW();
        }
        heapify();
    }

    //! Assignment operator
    /** State of this queue may not reflect results of pending operations on the copied queue. */
    concurrent_priority_queue& operator=(const concurrent_priority_queue& src) {
        if (this !=&src) {
            concurrent_priority_queue copy(src);
            copy.swap(*this);
        }
        return *this;
    }

    //! Destroys a concurrent_priority_queue 
    ~concurrent_priority_queue() { internal_destroy(); }

    //! Returns true if empty, false otherwise
    /** Returned value may not reflect results of pending operations. */
    bool empty() const { return my_size==0; }

    //! Returns the current number of elements contained in the queue
    /** Returned value may not reflect results of pending operations. */
    size_type size() const { return my_size; } 

    //! Returns the current capacity (i.e. allocated storage) of the queue
    /** Returned value may not reflect results of pending operations. */
    size_type capacity() const { 
        return my_capacity; 
    }

    //! Pushes elem onto the queue, increasing capacity of queue if necessary
    void push(const_reference elem) {
        cpq_operation op_data(elem, PUSH_OP);
        insert_handle_wait(&op_data);
        if (op_data.result == FAILED) { // Copy constructor with elem threw exception
            tbb::internal::throw_exception(tbb::internal::eid_bad_alloc);
        }
    }
    
    //! Gets a reference to and removes highest priority element
    /** If a highest priority element was found, sets elem and returns true, 
        otherwise returns false. */
    bool try_pop(reference elem) {
        cpq_operation op_data(POP_OP);
        op_data.elem = &elem;
        insert_handle_wait(&op_data);
        return op_data.result==SUCCESS;
    }

    //! If current capacity is less than new_cap, increases capacity to new_cap
    void reserve(size_type new_cap) {
        cpq_operation op_data(RESERVE_OP);
        op_data.sz = new_cap;
        insert_handle_wait(&op_data);
        if (op_data.result == FAILED) { // Copy constructors threw exception during array resize
            tbb::internal::throw_exception(tbb::internal::eid_bad_alloc);
        }
    }

    //! Clear the queue; not thread-safe
    /** Resets size, effectively emptying queue; does not free space.
        May not clear elements added in pending operations. */
    void clear() {
        for (size_type i=my_size; i>0; --i) {
            data[i-1].~value_type();
        }
        my_size = 0;
        mark = 0; 
    }

    //! Shrink queue capacity to current contents; not thread-safe
    void shrink_to_fit() {
        internal_reserve(my_size);
    }

    //! Swap this queue with another; not thread-safe
    void swap(concurrent_priority_queue& q) {
        std::swap(data, q.data);
        std::swap(my_size, q.my_size);        
        std::swap(my_capacity, q.my_capacity);
        std::swap(mark, q.mark);
        std::swap(my_helper, q.my_helper);
    }

    //! Return allocator object
    allocator_type get_allocator() const { return my_helper; }

private:
    enum operation_type {INVALID_OP, PUSH_OP, POP_OP, RESERVE_OP};
    enum operation_status { WAITING = 0, SUCCESS, FAILED };
    class cpq_operation {
     public:
        operation_type type;
        uintptr_t result;
        union {
            value_type *elem;
            size_type sz;
        };
        cpq_operation *next;
        cpq_operation(const_reference e, operation_type t) : 
            type(t), result(WAITING), elem(const_cast<value_type*>(&e)), next(NULL) {}
        cpq_operation(operation_type t) : type(t), result(WAITING), next(NULL) {}
    };

    //! An atomically updated list of pending operations
    tbb::atomic<cpq_operation *> operation_list;
    //! Padding added to avoid false sharing
    char padding1[ tbb::internal::NFS_MaxLineSize - sizeof(tbb::atomic<cpq_operation *>)];

    //! Flag to indicate that a thread is active in handling pending operations
    uintptr_t handler_busy;
    //! The space taken up by the queue
    size_type my_size;
    //! The point at which unsorted elements begin
    size_type mark;
    //! Padding added to avoid false sharing
    char padding2[ tbb::internal::NFS_MaxLineSize - sizeof(uintptr_t) - 2*sizeof(size_type)];

    //! The current capacity (allocated space) for the queue
    size_type my_capacity;
    //! Storage for the heap of elements in the queue, plus unheapified elements
    /** data has the following structure:
        
         binary unheapified
          heap   elements
        ____|_______|____
        |       |       |
        v       v       v
        [_|...|_|_|...|_| |...| ]
         0       ^       ^       ^
                 |       |       |__my_capacity
                 |       |__my_size
                 |__mark
                 

        Thus, data stores the binary heap starting at position 0 through
        mark-1 (it may be empty).  Then there are 0 or more elements 
        that have not yet been inserted into the heap, in positions 
        mark through my_size-1. */
    value_type *data;
    //! A helper object to save space
    struct helper_type : public allocator_type {
        //! A user-specified allocator; or cache_aligned_allocator by default
        helper_type(allocator_type const& a) : allocator_type(a) {}
        //! The comparison operator: if compare(A,B) then B has higher priority
        Compare compare;
    };
    helper_type my_helper;

    //! Internal constructor with an initial capacity
    void internal_construct(size_type init_sz) {
        my_size = my_capacity = mark = 0;
        data = NULL;
        operation_list = NULL;
        handler_busy = 0;
        internal_reserve(init_sz);
    }

    //! Internal constructor with forward, bidirectional and random access iterators
    template <typename ForwardIterator>
    void internal_iterator_construct(ForwardIterator begin, ForwardIterator end, std::forward_iterator_tag) {
        internal_construct(std::distance(begin, end));
        my_size = my_capacity;
        size_type i=0;
        __TBB_TRY {
            for(; begin != end; ++begin, ++i)
                new (&data[i]) value_type(*begin);
        } __TBB_CATCH(...) {
            my_size = i;
            clear();
            my_helper.deallocate(data, my_capacity);
            __TBB_RETHROW();
        }
        heapify();
    }

    //! Internal constructor with input iterators
    template <typename InputIterator>
    void internal_iterator_construct(InputIterator begin, InputIterator end, std::input_iterator_tag) {
        internal_construct(32); 
        size_type i=0;
        __TBB_TRY {
            for(; begin != end; ++begin, ++i) {
                if (i>=my_capacity)
                    internal_reserve(my_capacity<<1);
                new (&data[i]) value_type(*begin);
                ++my_size;
            }
        } __TBB_CATCH(...) {
            clear();
            my_helper.deallocate(data, my_capacity);
            __TBB_RETHROW();
        }
        heapify();
        shrink_to_fit();
    }

    //! Internal destructor
    void internal_destroy() { 
#if TBB_USE_ASSERT        
        cpq_operation *op_list = operation_list.fetch_and_store((cpq_operation *)(internal::poisoned_ptr));
        __TBB_ASSERT(op_list==NULL,"concurrent_priority_queue destroyed with pending operations.\n");
        __TBB_ASSERT(!handler_busy,"concurrent_priority_queue destroyed with pending operations.\n");
#endif
        clear();
        if (data) my_helper.deallocate(data, my_capacity);
    }

    //! Set available space to desired_capacity elements
    /** Checks for fit with my_size and will not reduce space below that. */
    void internal_reserve(size_type desired_capacity);

    //! Placement-new copy-construct contents of src to dst
    /** Assumes data has been preallocated to src.my_size. */
    void internal_copy(const value_type *src, value_type *dst, size_type sz) {
        size_type i=0;
        __TBB_TRY {
            for (; i<sz; ++i) new (&dst[i]) value_type(src[i]);
        } __TBB_CATCH(...) {
            // clean up dst
            for (; i>0; --i) {
                dst[i-1].~value_type();
            }
            __TBB_RETHROW();
        }
    }

    //! Merge unsorted elements into heap
    void heapify();

    //! Re-heapify after an extraction
    /** Re-heapify by pushing last element down the heap from the root. */
    void reheap();

    //! Grab and handle all the operations in the current operation_list
    void handle_operations();

    //! Put op in list and either handle list or wait for op to complete
    void insert_handle_wait(cpq_operation *op) {
        cpq_operation *res = operation_list, *tmp;

        using namespace tbb::internal;

        __TBB_ASSERT(operation_list!=(cpq_operation *)poisoned_ptr, "Attempt to use destroyed concurrent_priority_queue.\n");
        // insert the operation in the queue
        do {
            op->next = tmp = res;
            // ITT note: &operation_list+1 tag is used to cover accesses to all ops in operation_list.
            // This thread has created the operation, and now releases it so that the handler thread 
            // may handle the operations w/o triggering a race condition; thus this tag will be acquired
            // just before the operations are handled in handle_operations.
            call_itt_notify(releasing, &operation_list+1);
        } while ((res = operation_list.compare_and_swap(op, tmp)) != tmp);
        if (!tmp) { // first in the list; handle the operations
            // ITT note: &operation_list tag covers access to the handler_busy flag, which this
            // waiting handler thread will try to set upon entering handle_operations.
            call_itt_notify(acquired, &operation_list);
            handle_operations();
            __TBB_ASSERT(op->result, NULL);
        }
        else { // not first; wait for op to be ready
            call_itt_notify(prepare, &(op->result));
            spin_wait_while_eq(op->result, uintptr_t(WAITING));
            itt_load_word_with_acquire(op->result);
        }
    }
};
    
//! Set available space to desired_capacity elements
template <typename T, typename Compare, typename A>
void concurrent_priority_queue<T, Compare, A>::internal_reserve(size_type desired_capacity) {
    value_type *tmp_data = NULL;
    
    // don't reduce queue capacity below content size
    if (desired_capacity<my_size) desired_capacity = my_size;
    // handle special case of complete queue removal
    if (desired_capacity==0) {
        // assert that data was properly cleared before this
        __TBB_ASSERT(my_size==0, NULL);
        __TBB_ASSERT(mark==0, NULL);
        if (data) my_helper.deallocate(data, my_capacity);
        data = NULL;
        my_capacity = 0;
        return;
    }
    // allocate the new array
    tmp_data = static_cast<value_type*>(my_helper.allocate(desired_capacity));
    if( !tmp_data )
        tbb::internal::throw_exception(tbb::internal::eid_bad_alloc);
    if (data) {
        // fill new array with old contents, if any
        __TBB_TRY {
            internal_copy(data, tmp_data, my_size);
        } __TBB_CATCH(...) {
            my_helper.deallocate(tmp_data, desired_capacity);
            __TBB_RETHROW();
        }
        // clear and delete old array
        for (size_type i=my_size; i>0; --i) {
            data[i-1].~value_type();
        }
        my_helper.deallocate(data, my_capacity);
    }
    // else simply put the new array in data
    // update data and my_capacity
    data = tmp_data;
    my_capacity = desired_capacity;
}

//! Merge unsorted elements into heap
template <typename T, typename Compare, typename A>
void concurrent_priority_queue<T, Compare, A>::heapify() {
    value_type *loc = data;

    if (!mark) mark = 1;
    for (; mark<my_size; ++mark) { // for each unheapified element under my_size
        size_type cur_pos = mark; 
        value_type to_place = loc[mark];
        do { // push to_place up the heap
            size_type parent = (cur_pos-1)>>1;
            if (!my_helper.compare(loc[parent], to_place)) break;
            loc[cur_pos] = loc[parent];
            cur_pos = parent;
        } while( cur_pos );
        loc[cur_pos] = to_place;
    }
}
    
//! Push the last element in the array down the heap
/** Assumes data[0] was already extracted and my_size was already decremented; 
    thus the element to push down the heap is data[my_size]. */
template <typename T, typename Compare, typename A>
void concurrent_priority_queue<T, Compare, A>::reheap() {
    size_type cur_pos=0, child=1;
    value_type *loc = data;
    
    while (child < mark) {
        size_type target = child;
        if (child+1 < mark && my_helper.compare(loc[child], loc[child+1])) ++target;
        // target now has the higher priority child
        if (my_helper.compare(loc[target], loc[my_size])) break;
        loc[cur_pos] = loc[target];
        cur_pos = target;
        child = (cur_pos<<1)+1;
    }
    loc[cur_pos] = loc[my_size];
}

template <typename T, typename Compare, typename A>
void concurrent_priority_queue<T, Compare, A>::handle_operations() {
    cpq_operation *op_list, *tmp, *pop_list=NULL;

    using namespace tbb::internal;

    // get the handler_busy: only one thread can possibly spin here at a time
    // ITT note: &handler_busy tag covers access to the actual queue as it is passed between active 
    // and waiting handlers.  Below, the waiting handler waits until the active handler releases, 
    // and the waiting handler acquires &handler_busy as it becomes the active_handler. The 
    // release point is at the end of this function, when all operations in the list have been applied
    // to the queue.
    call_itt_notify(prepare, &handler_busy);
    spin_wait_until_eq(handler_busy, uintptr_t(0));
    call_itt_notify(acquired, &handler_busy);
    // acquire not necessary here due to causality rule and surrounding atomics
    __TBB_store_with_release(handler_busy, 1);

    // grab the operation list
    // ITT note: &operation_list tag covers access to the handler_busy flag itself.
    // Capturing the state of the operation_list signifies that handler_busy has been set and a new
    // active handler will now process that list's operations.
    call_itt_notify(releasing, &operation_list);
    op_list = operation_list.fetch_and_store(NULL);
    // ITT note: &operation_list+1 tag is used to cover accesses to all ops in operation_list.
    // The threads that created each operation released this tag so that this handler thread 
    // could handle the operations w/o triggering a race condition; thus this handler thread
    // now acquires this tag just before handling the operations.
    call_itt_notify(acquired, &operation_list+1);

    // first pass processes all constant time operations: pushes, tops, some pops. Also reserve.
    while (op_list) {
        __TBB_ASSERT(op_list->type != INVALID_OP, NULL);
        tmp = op_list;
        op_list = op_list->next;
        if (tmp->type == PUSH_OP) {
            __TBB_TRY {
                if (my_size >= my_capacity) internal_reserve(my_capacity?my_capacity<<1:1);
                new (&data[my_size]) value_type(*(tmp->elem)); // copy the data
                ++my_size;
                itt_store_word_with_release(tmp->result, uintptr_t(SUCCESS));
            } __TBB_CATCH(...) {
                itt_store_word_with_release(tmp->result, uintptr_t(FAILED));
            }
        }
        else if (tmp->type == POP_OP) {
            if (!my_size) {
                itt_store_word_with_release(tmp->result, uintptr_t(FAILED));
            }
            else {
                if (mark < my_size && my_helper.compare(data[0], data[my_size-1])) {
                    // there are newly pushed elems and the last one is higher than top
                    *(tmp->elem) = data[my_size-1]; // copy the data
                    data[my_size-1].~value_type();
                    --my_size;
                    itt_store_word_with_release(tmp->result, uintptr_t(SUCCESS));
                    __TBB_ASSERT(mark<=my_size, NULL);
                }
                else {
                    tmp->next = pop_list;
                    pop_list = tmp;
                }
            }
        }
        else {
            __TBB_ASSERT(tmp->type == RESERVE_OP, NULL);
            __TBB_TRY {
                internal_reserve(tmp->sz);
                itt_store_word_with_release(tmp->result, uintptr_t(SUCCESS));
            } __TBB_CATCH(...) {
                itt_store_word_with_release(tmp->result, uintptr_t(FAILED));
            };
        }
    }

    // second pass processes pop operations
    while (pop_list) {
        tmp = pop_list;
        pop_list = pop_list->next;
        __TBB_ASSERT(tmp->type == POP_OP, NULL);
        if (!my_size) {
            itt_store_word_with_release(tmp->result, uintptr_t(FAILED));
        }
        else {
            __TBB_ASSERT(mark<=my_size, NULL);
            if (mark < my_size && my_helper.compare(data[0], data[my_size-1])) {
                // there are newly pushed elems and the last one is higher than top
                *(tmp->elem) = data[my_size-1]; // copy the data
                --my_size;
                itt_store_word_with_release(tmp->result, uintptr_t(SUCCESS));
                data[my_size].~value_type();
            }
            else { // extract and push the last element down heap
                *(tmp->elem) = data[0]; // copy the data
                if (mark == my_size) --mark;
                --my_size;
                itt_store_word_with_release(tmp->result, uintptr_t(SUCCESS));
                data[0] = data[my_size];
                if (my_size > 1) // don't reheap for heap of size 1
                    reheap();
                data[my_size].~value_type();
            }
            __TBB_ASSERT(mark<=my_size, NULL);
        }
    }

    // heapify any leftover pushed elements before doing the next batch of operations
    if (mark<my_size) heapify();
    __TBB_ASSERT(mark<=my_size, NULL);
    
    // release the handler_busy
    itt_store_word_with_release(handler_busy, uintptr_t(0));
}

} // namespace interface5

using interface5::concurrent_priority_queue;

} // namespace tbb


#endif /* __TBB_concurrent_priority_queue_H */
