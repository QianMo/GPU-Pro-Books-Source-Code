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

#ifndef __TBB_tbb_allocator_H
#define __TBB_tbb_allocator_H

#include <new>
#include "tbb_stddef.h"

namespace tbb {

//! @cond INTERNAL
namespace internal {

    //! Deallocates memory using FreeHandler
    /** The function uses scalable_free if scalable allocator is available and free if not*/
    void deallocate_via_handler_v3( void *p );

    //! Allocates memory using MallocHandler
    /** The function uses scalable_malloc if scalable allocator is available and malloc if not*/
    void* allocate_via_handler_v3( size_t n );

    //! Returns true if standard malloc/free are used to work with memory.
    bool is_malloc_used_v3();
}
//! @endcond

//! Meets "allocator" requirements of ISO C++ Standard, Section 20.1.5
/** The class selects the best memory allocation mechanism available 
    from scalable_malloc and standard malloc.
    The members are ordered the same way they are in section 20.4.1
    of the ISO C++ standard.
    @ingroup memory_allocation */
template<typename T>
class tbb_allocator {
public:
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> struct rebind {
        typedef tbb_allocator<U> other;
    };

    //! Specifies current allocator
    enum malloc_type {
        scalable, 
        standard
    };

#if _WIN64
    //! Non-ISO method required by Microsoft's STL containers 
    /** Some versions of Microsoft's container classes seem to require that 
        allocators supply this method. */
    char* _Charalloc( size_type size ) {        
        return (char*)(internal::allocate_via_handler_v3( size * sizeof(T)));
    }
#endif /* _WIN64 */

    tbb_allocator() throw() {}
    tbb_allocator( const tbb_allocator& ) throw() {}
    template<typename U> tbb_allocator(const tbb_allocator<U>&) throw() {}

    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}
    
    //! Allocate space for n objects, starting on a cache/sector line.
    pointer allocate( size_type n, const void* /*hint*/ = 0) {
        return pointer(internal::allocate_via_handler_v3( n * sizeof(T) ));
    }

    //! Free block of memory that starts on a cache line
    void deallocate( pointer p, size_type ) {
        internal::deallocate_via_handler_v3(p);        
    }

    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        size_type max = static_cast<size_type>(-1) / sizeof (T);
        return (max > 0 ? max : 1);
    }
    
    //! Copy-construct value at location pointed to by p.
    void construct( pointer p, const T& value ) {new(static_cast<void*>(p)) T(value);}

    //! Destroy value at location pointed to by p.
    void destroy( pointer p ) {p->~T();}

    //! Returns current allocator
    static malloc_type allocator_type() {
        return internal::is_malloc_used_v3() ? standard : scalable;
    }
};

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<> 
class tbb_allocator<void> {
public:
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template<typename U> struct rebind {
        typedef tbb_allocator<U> other;
    };
};

template<typename T, typename U>
inline bool operator==( const tbb_allocator<T>&, const tbb_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const tbb_allocator<T>&, const tbb_allocator<U>& ) {return false;}

} // namespace ThreadBuildingBlocks 

#endif /* __TBB_tbb_allocator_H */
