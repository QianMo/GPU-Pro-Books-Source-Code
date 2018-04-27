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

#ifndef __TBB_mutex_H
#define __TBB_mutex_H

#if _WIN32||_WIN64
#include <windows.h>
#if !defined(_WIN32_WINNT)
// The following Windows API function is declared explicitly;
// otherwise any user would have to specify /D_WIN32_WINNT=0x0400
extern "C" BOOL WINAPI TryEnterCriticalSection( LPCRITICAL_SECTION );
#endif

#else /* if not _WIN32||_WIN64 */
#include <pthread.h>
namespace tbb { namespace internal {
// Use this internal TBB function to throw an exception
extern void handle_perror( int error_code, const char* what );
} } //namespaces
#endif /* _WIN32||_WIN64 */

#include <stdio.h>
#include "tbb_stddef.h"

namespace tbb {

//! Wrapper around the platform's native reader-writer lock.
/** For testing purposes only.
    @ingroup synchronization */
class mutex {
public:
    //! Construct unacquired mutex.
    mutex() {
#if TBB_DO_ASSERT
    internal_construct();
#else
  #if _WIN32||_WIN64
        InitializeCriticalSection(&impl);
  #else
        int error_code = pthread_mutex_init(&impl,NULL);
        if( error_code )
            tbb::internal::handle_perror(error_code,"mutex: pthread_mutex_init failed");
  #endif /* _WIN32||_WIN64*/
#endif /* TBB_DO_ASSERT */
    };

    ~mutex() {
#if TBB_DO_ASSERT
        internal_destroy();
#else
  #if _WIN32||_WIN64
        DeleteCriticalSection(&impl);
  #else
        pthread_mutex_destroy(&impl); 

  #endif /* _WIN32||_WIN64 */
#endif /* TBB_DO_ASSERT */
    };

    class scoped_lock;
    friend class scoped_lock;

    //! The scoped locking pattern
    /** It helps to avoid the common problem of forgetting to release lock.
        It also nicely provides the "node" for queuing locks. */
    class scoped_lock : private internal::no_copy {
    public:
        //! Construct lock that has not acquired a mutex. 
        scoped_lock() : my_mutex(NULL) {};

        //! Acquire lock on given mutex.
        /** Upon entry, *this should not be in the "have acquired a mutex" state. */
        scoped_lock( mutex& mutex ) {
            acquire( mutex );
        }

        //! Release lock (if lock is held).
        ~scoped_lock() {
            if( my_mutex ) 
                release();
        }

        //! Acquire lock on given mutex.
        void acquire( mutex& mutex ) {
#if TBB_DO_ASSERT
            internal_acquire(mutex);
#else
            my_mutex = &mutex;
  #if _WIN32||_WIN64
            EnterCriticalSection(&mutex.impl);
  #else
            pthread_mutex_lock(&mutex.impl);
  #endif /* _WIN32||_WIN64 */
#endif /* TBB_DO_ASSERT */
        }

        //! Try acquire lock on given mutex.
        bool try_acquire( mutex& mutex ) {
#if TBB_DO_ASSERT
            return internal_try_acquire (mutex);
#else
            bool result;
  #if _WIN32||_WIN64
            result = TryEnterCriticalSection(&mutex.impl)!=0;
  #else
            result = pthread_mutex_trylock(&mutex.impl)==0;
  #endif /* _WIN32||_WIN64 */
            if( result )
                my_mutex = &mutex;
            return result;
#endif /* TBB_DO_ASSERT */
        }

        //! Release lock
        void release() {
#if TBB_DO_ASSERT
            internal_release ();
#else
  #if _WIN32||_WIN64
            LeaveCriticalSection(&my_mutex->impl);
  #else
            pthread_mutex_unlock(&my_mutex->impl);
  #endif /* _WIN32||_WIN64 */
            my_mutex = NULL;
#endif /* TBB_DO_ASSERT */
        }

    private:
        //! The pointer to the current mutex to work
        mutex* my_mutex;

        //! All checks from acquire using mutex.state were moved here
        void internal_acquire( mutex& m );

        //! All checks from try_acquire using mutex.state were moved here
        bool internal_try_acquire( mutex& m );

        //! All checks from release using mutex.state were moved here
        void internal_release();
    };

    // Mutex traits
    static const bool is_rw_mutex = false;
    static const bool is_recursive_mutex = false;
    static const bool is_fair_mutex = false;

private:
#if _WIN32||_WIN64
    CRITICAL_SECTION impl;    
    enum state_t {
        INITIALIZED=0x1234,
        DESTROYED=0x789A,
        HELD=0x56CD
    } state;
#else
    pthread_mutex_t impl;
#endif /* _WIN32||_WIN64 */

    //! All checks from mutex constructor using mutex.state were moved here
    void internal_construct();

    //! All checks from mutex destructor using mutex.state were moved here
    void internal_destroy();
};

} // namespace tbb 

#endif /* __TBB_mutex_H */
