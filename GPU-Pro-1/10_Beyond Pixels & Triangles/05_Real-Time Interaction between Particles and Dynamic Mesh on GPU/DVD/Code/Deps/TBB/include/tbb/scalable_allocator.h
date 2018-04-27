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

#ifndef __TBB_scalable_allocator_H
#define __TBB_scalable_allocator_H

#include <stddef.h> // Need ptrdiff_t and size_t from here.

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

//! The "malloc" analogue to allocate block of memory of size bytes.
/** @ingroup memory_allocation */
void * scalable_malloc (size_t size);

//! The "free" analogue to discard a previously allocated piece of memory
/** @ingroup memory_allocation */
void   scalable_free (void* ptr);

//! The "realloc" analogue complementing scalable_malloc
/** @ingroup memory_allocation */
void * scalable_realloc (void* ptr, size_t size);

//! The "calloc" analogue complementing scalable_malloc
/** @ingroup memory_allocation */
void * scalable_calloc (size_t nobj, size_t size);

#ifdef __cplusplus
} // extern "C"
#endif /* __cplusplus */

#ifdef __cplusplus

#include <new>      // To use new with the placement argument

namespace tbb {

//! Meets "allocator" requirements of ISO C++ Standard, Section 20.1.5
/** The members are ordered the same way they are in section 20.4.1
    of the ISO C++ standard.
    @ingroup memory_allocation */
template<typename T>
class scalable_allocator {
public:
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<class U> struct rebind {
        typedef scalable_allocator<U> other;
    };

    scalable_allocator() throw() {}
    scalable_allocator( const scalable_allocator& ) throw() {}
    template<typename U> scalable_allocator(const scalable_allocator<U>&) throw() {}

    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}

    //! Allocate space for n objects, starting on a cache/sector line.
    pointer allocate( size_type n, const void* /*hint*/ =0 ) {
        return static_cast<pointer>( scalable_malloc( n * sizeof(value_type) ) );
    }

    //! Free block of memory that starts on a cache line
    void deallocate( pointer p, size_type ) {
        scalable_free( p );
    }

    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        size_type absolutemax = static_cast<size_type>(-1) / sizeof (T);
        return (absolutemax > 0 ? absolutemax : 1);
    }
    void construct( pointer p, const T& val ) { new(static_cast<void*>(p)) T(val); }
    void destroy( pointer p ) {(static_cast<T*>(p))->~T();}
};

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<>
class scalable_allocator<void> {
public:
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template<class U> struct rebind {
        typedef scalable_allocator<U> other;
    };
};

template<typename T, typename U>
inline bool operator==( const scalable_allocator<T>&, const scalable_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const scalable_allocator<T>&, const scalable_allocator<U>& ) {return false;}

} // namespace tbb

#if _MSC_VER
    #if __TBB_BUILD && !defined(__TBBMALLOC_NO_IMPLICIT_LINKAGE)
        #define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
    #endif

    #if !__TBBMALLOC_NO_IMPLICIT_LINKAGE
        #ifdef _DEBUG
            #pragma comment(lib, "tbbmalloc_debug.lib")
        #else
            #pragma comment(lib, "tbbmalloc.lib")
        #endif
    #endif
#endif

#endif /* __cplusplus */

#endif /* __TBB_scalable_allocator_H */
