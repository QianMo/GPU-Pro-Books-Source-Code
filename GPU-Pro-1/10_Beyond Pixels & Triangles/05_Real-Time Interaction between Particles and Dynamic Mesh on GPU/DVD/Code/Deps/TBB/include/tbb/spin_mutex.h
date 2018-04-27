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

#ifndef __TBB_spin_mutex_H
#define __TBB_spin_mutex_H

#include <cstddef>
#include "tbb_stddef.h"
#include "tbb/tbb_machine.h"

namespace tbb {

//! A lock that occupies a single byte.
/** A spin_mutex is a spin mutex that fits in a single byte.  
    It should be used only for locking short critical sections 
    (typically &lt;20 instructions) when fairness is not an issue.  
    If zero-initialized, the mutex is considered unheld.
    @ingroup synchronization */
class spin_mutex {
    //! 0 if lock is released, 1 if lock is acquired.
    unsigned char flag;

public:
    //! Construct unacquired lock.
    /** Equivalent to zero-initialization of *this. */
    spin_mutex() : flag(0) {}

    //! Represents acquisition of a mutex.
    class scoped_lock : private internal::no_copy {
    private:
        //! Points to currently held mutex, or NULL if no lock is held.
        spin_mutex* my_mutex; 

        //! Value to store into spin_mutex::flag to unlock the mutex.
        internal::uintptr my_unlock_value;

        //! Like acquire, but with ITT instrumentation.
        void internal_acquire( spin_mutex& m );

        //! Like try_acquire, but with ITT instrumentation.
        bool internal_try_acquire( spin_mutex& m );

        //! Like release, but with ITT instrumentation.
        void internal_release();

    public:
        //! Construct without without acquiring a mutex.
        scoped_lock() : my_mutex(NULL), my_unlock_value(0) {}

        //! Construct and acquire lock on a mutex.
        scoped_lock( spin_mutex& m ) { 
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
            my_mutex=NULL;
            internal_acquire(m);
#else
            my_unlock_value = __TBB_LockByte(m.flag);
            my_mutex=&m;
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT*/
        }

        //! Acquire lock.
        void acquire( spin_mutex& m ) {
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
            internal_acquire(m);
#else
            my_unlock_value = __TBB_LockByte(m.flag);
            my_mutex = &m;
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT*/
        }

        //! Try acquiring lock (non-blocking)
        bool try_acquire( spin_mutex& m ) {
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
            return internal_try_acquire(m);
#else
            bool result = __TBB_TryLockByte(m.flag);
            if( result ) {
                my_unlock_value = 0;
                my_mutex = &m;
            }
            return result;
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT*/
        }

        //! Release lock
        void release() {
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
            internal_release();
#else
            __TBB_store_with_release(my_mutex->flag, static_cast<unsigned char>(my_unlock_value));
            my_mutex = NULL;
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT */
        }

        //! Destroy lock.  If holding a lock, releases the lock first.
        ~scoped_lock() {
            if( my_mutex ) {
#if TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT
                internal_release();
#else
                __TBB_store_with_release(my_mutex->flag, static_cast<unsigned char>(my_unlock_value));
#endif /* TBB_DO_THREADING_TOOLS||TBB_DO_ASSERT */
            }
        }
    };

    // Mutex traits
    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = false;

    friend class scoped_lock;
};

} // namespace tbb

#endif /* __TBB_spin_mutex_H */
