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

#ifndef __TBB_parallel_reduce_H
#define __TBB_parallel_reduce_H

#include "task.h"
#include "aligned_space.h"
#include "partitioner.h"
#include <new>

namespace tbb {

//! @cond INTERNAL
namespace internal {

    //! ITT instrumented routine that stores src into location pointed to by dst.
    void itt_store_pointer_with_release_v3( void* dst, void* src );

    //! ITT instrumented routine that loads pointer from location pointed to by src.
    void* itt_load_pointer_with_acquire_v3( const void* src );

    template<typename T> inline void parallel_reduce_store_body( T*& dst, T* src ) {
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
        itt_store_pointer_with_release_v3(&dst,src);
#else
        __TBB_store_with_release(dst,src);
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT */
    }

    template<typename T> inline T* parallel_reduce_load_body( T*& src ) {
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
        return static_cast<T*>(itt_load_pointer_with_acquire_v3(&src));
#else
        return __TBB_load_with_acquire(src);
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT */
    }

    //! Task type use to combine the partial results of parallel_reduce.
    /** @ingroup algorithms */
    template<typename Body>
    class finish_reduce: public task {
        Body* const my_body;
        bool has_right_zombie;
        aligned_space<Body,1> zombie_space;
        finish_reduce( Body* body ) : 
            my_body(body),
            has_right_zombie(false)
        {
        }
        task* execute() {
            if( has_right_zombie ) {
                // Right child was stolen.
                Body* s = zombie_space.begin();
                my_body->join( *s );
                s->~Body();
            }
            return NULL;
        }       
        template<typename Range,typename Body_, typename Partitioner>
        friend class start_reduce;
    };

    //! Task type used to split the work of parallel_reduce.
    /** @ingroup algorithms */
    template<typename Range, typename Body, typename Partitioner>
    class start_reduce: public task {
        typedef finish_reduce<Body> finish_type;
        Body* my_body;
        Range my_range;
        typename Partitioner::partition_type my_partition;
        /*override*/ task* execute();
        template<typename Body_>
        friend class finish_reduce;
    
        //! Constructor used for root task
        start_reduce( const Range& range, Body* body, const Partitioner &partitioner ) :
            my_body(body),
            my_range(range),
            my_partition(partitioner)
        {
        }
        //! Splitting constructor used to generate children.
        /** this becomes left child.  Newly constructed object is right child. */
        start_reduce( start_reduce& parent, split ) :
            my_body(parent.my_body),
            my_range(parent.my_range,split()),
            my_partition(parent.my_partition,split())
        {
        }
        //! Process a steal.
        /** Affinity is never set, so if this method is called, it must arise because this is stolen. */
        /*override*/ void note_affinity( affinity_id id ) {
            finish_type* p = static_cast<finish_type*>(parent() );
            Body* body = new( p->zombie_space.begin() ) Body(*my_body,split());
            p->has_right_zombie = true;
            my_body = body;
        }
    public:
#if !__TBB_EXCEPTIONS
        static void run( const Range& range, Body& body, const Partitioner &partitioner ) {
            if( !range.empty() ) 
                task::spawn_root_and_wait( *new(task::allocate_root()) start_reduce(range,&body,partitioner) );
        }
#else /* __TBB_EXCEPTIONS */
        static void run( const Range& range, Body& body, const Partitioner &partitioner ) {
            // Bound context prevents exceptions from body to affect nesting or sibling algorithms,
            // and allows users to handle exceptions safely by wrapping parallel_for in the try-block.
            if( !range.empty() )  {
                task_group_context context;
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_reduce(range,&body,partitioner) );
            }
        }
        static void run( const Range& range, Body& body, const Partitioner &partitioner, task_group_context& context ) {
            if( !range.empty() ) 
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_reduce(range,&body,partitioner) );
        }
#endif /* __TBB_EXCEPTIONS */
    };

    template<typename Range, typename Body, typename Partitioner>
    task* start_reduce<Range,Body,Partitioner>::execute() {
        if( !my_range.is_divisible() || my_partition.should_execute_range(*this) ) {
            (*my_body)( my_range );
            return NULL;
        } else {
            finish_type& c = *new( allocate_continuation()) finish_type(my_body);
            recycle_as_child_of(c);
            c.set_ref_count(2);    
            start_reduce& b = *new( c.allocate_child() ) start_reduce(*this,split());
            spawn(b);
            return this;
        }
    }
 
    //! 0 if root, 1 if a left child, 2 if a right child.
    /** Represented as a char, not enum, for compactness. */
    typedef char reduction_context;

    //! Task type use to combine the partial results of parallel_reduce with affinity_partitioner.
    /** @ingroup algorithms */
    template<typename Body>
    class finish_reduce_with_affinity: public task {
        //! Pointer to body, or NULL if the left child has not yet finished. 
        Body* my_body;
        bool has_right_zombie;
        const reduction_context my_context;
        aligned_space<Body,1> zombie_space;
        finish_reduce_with_affinity( char context ) : 
            my_body(NULL),
            has_right_zombie(false),
            my_context(context)
        {
        }
        task* execute() {
            if( has_right_zombie ) {
                // Right child was stolen.
                Body* s = zombie_space.begin();
                my_body->join( *s );
                s->~Body();
            }
            if( my_context==1 ) 
                parallel_reduce_store_body( static_cast<finish_reduce_with_affinity*>(parent())->my_body, my_body );
            return NULL;
        }       
        template<typename Range,typename Body_>
        friend class start_reduce_with_affinity;
    };

    //! Task type used to split the work of parallel_reduce with affinity_partitioner.
    /** @ingroup algorithms */
    template<typename Range, typename Body>
    class start_reduce_with_affinity: public task {
        typedef finish_reduce_with_affinity<Body> finish_type;
        Body* my_body;
        Range my_range;
        typename affinity_partitioner::partition_type my_partition;
        reduction_context my_context;
        /*override*/ task* execute();
        template<typename Body_>
        friend class finish_reduce_with_affinity;
    
        //! Constructor used for root task
        start_reduce_with_affinity( const Range& range, Body* body, affinity_partitioner& partitioner ) :
            my_body(body),
            my_range(range),
            my_partition(partitioner),
            my_context(0)
        {
        }
        //! Splitting constructor used to generate children.
        /** this becomes left child.  Newly constructed object is right child. */
        start_reduce_with_affinity( start_reduce_with_affinity& parent, split ) :
            my_body(parent.my_body),
            my_range(parent.my_range,split()),
            my_partition(parent.my_partition,split()),
            my_context(2)
        {
            my_partition.set_affinity(*this);
            parent.my_context = 1;
        }
        //! Update affinity info, if any
        /*override*/ void note_affinity( affinity_id id ) {
            my_partition.note_affinity( id );
        }

public:
        static void run( const Range& range, Body& body, affinity_partitioner& partitioner ) {
            if( !range.empty() ) {
#if !__TBB_EXCEPTIONS || TBB_JOIN_OUTER_TASK_GROUP
                task::spawn_root_and_wait( *new(task::allocate_root()) start_reduce_with_affinity(range,&body,partitioner) );
#else
                // Bound context prevents exceptions from body to affect nesting or sibling algorithms,
                // and allows users to handle exceptions safely by wrapping parallel_for in the try-block.
                task_group_context context;
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_reduce_with_affinity(range,&body,partitioner) );
#endif /* __TBB_EXCEPTIONS && !TBB_JOIN_OUTER_TASK_GROUP */
            }
        }
#if __TBB_EXCEPTIONS
        static void run( const Range& range, Body& body, affinity_partitioner& partitioner, task_group_context& context ) {
            if( !range.empty() ) 
                task::spawn_root_and_wait( *new(task::allocate_root(context)) start_reduce_with_affinity(range,&body,partitioner) );
        }
#endif /* __TBB_EXCEPTIONS */
    };

    template<typename Range, typename Body>
    task* start_reduce_with_affinity<Range,Body>::execute() {
        if( my_context==2 ) {
            finish_type* p = static_cast<finish_type*>(parent() );
            if( !parallel_reduce_load_body(p->my_body) ) {
                my_body = new( p->zombie_space.begin() ) Body(*my_body,split());
                p->has_right_zombie = true;
            } 
        }
        if( !my_range.is_divisible() || my_partition.should_execute_range(*this) ) {
            (*my_body)( my_range );
            if( my_context==1 ) 
                parallel_reduce_store_body(static_cast<finish_type*>(parent())->my_body, my_body );
            return my_partition.continue_after_execute_range(*this);
        } else {
            finish_type& c = *new( allocate_continuation()) finish_type(my_context);
            recycle_as_child_of(c);
            c.set_ref_count(2);    
            bool delay = my_partition.decide_whether_to_delay();
            start_reduce_with_affinity& b = *new( c.allocate_child() ) start_reduce_with_affinity(*this,split());
            my_partition.spawn_or_delay(delay,*this,b);
            return this;
        }
    } 
} // namespace internal
//! @endcond

// Requirements on Range concept are documented in blocked_range.h

/** \page parallel_reduce_body_req Requirements on parallel_reduce body
    Class \c Body implementing the concept of parallel_reduce body must define:
    - \code Body::Body( Body&, split ); \endcode        Splitting constructor.
                                                        Must be able to run concurrently with operator() and method \c join
    - \code Body::~Body(); \endcode                     Destructor
    - \code void Body::operator()( Range& r ); \endcode Function call operator applying body to range \c r
                                                        and accumulating the result
    - \code void Body::join( Body& b ); \endcode        Join results. 
                                                        The result in \c b should be merged into the result of \c this
**/

/** \name parallel_reduce
    See also requirements on \ref range_req "Range" and \ref parallel_reduce_body_req "parallel_reduce Body". **/
//@{

//! Parallel iteration with reduction and simple_partitioner, or default partitioner if no partitioner is specified.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const simple_partitioner& partitioner = simple_partitioner() ) {
    internal::start_reduce<Range,Body,simple_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const auto_partitioner& partitioner ) {
    internal::start_reduce<Range,Body,auto_partitioner>::run( range, body, partitioner );
}

//! Parallel iteration with reduction and auto_partitioner
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, affinity_partitioner& partitioner ) {
    internal::start_reduce_with_affinity<Range,Body>::run( range, body, partitioner );
}

#if __TBB_EXCEPTIONS
//! Parallel iteration with reduction, simple partitioner and user-supplied context.
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const simple_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce<Range,Body,simple_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, auto_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, const auto_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce<Range,Body,auto_partitioner>::run( range, body, partitioner, context );
}

//! Parallel iteration with reduction, affinity_partitioner and user-supplied context
/** @ingroup algorithms **/
template<typename Range, typename Body>
void parallel_reduce( const Range& range, Body& body, affinity_partitioner& partitioner, task_group_context& context ) {
    internal::start_reduce_with_affinity<Range,Body>::run( range, body, partitioner, context );
}
#endif /* __TBB_EXCEPTIONS */
//@}

} // namespace tbb

#endif /* __TBB_parallel_reduce_H */

