#ifndef _FBXSDK_SCOPED_PTR_HPP_
#define _FBXSDK_SCOPED_PTR_HPP_
// FBX includes
#include "kdebug.h"

// Begin FBX namespace
#include <kbaselib_nsbegin.h> // namespace

//
//  KScopedPtr mimics a built-in pointer except that it guarantees deletion
//  of the object pointed to, either on destruction of the KScopedPtr or via
//  an explicit Reset()
//

//
// Deletion policy dictates the way the pointer is destroyed
// By default, KScopedPtr uses the DefaultDeletionPolicy
//
template<class T>
class DefaultDeletionPolicy
{
public:
    static inline void DeleteIt(T** ptr)
    {
        if ( *ptr != NULL )
        {
            delete *ptr;
            *ptr = NULL;
        }
    }
};

template<class T>
class FreeDeletionPolicy
{
public:
    static inline void DeleteIt(T** ptr)
    {
        if ( *ptr != NULL )
        {
            free( *ptr );
            *ptr = NULL;
        }
    }
};

//---------------------------------------------------------------------
template<class T, class DeletionPolicyT = DefaultDeletionPolicy<T> >
class KScopedPtr
{
private:
    T* ptr;

    // Non copyable object
    KScopedPtr(KScopedPtr const &);
    KScopedPtr& operator=(KScopedPtr const &);

    typedef KScopedPtr<T, DeletionPolicyT> ThisType;
    typedef DeletionPolicyT DeletionPolicy;

public:
    //////////////////////////////
    explicit KScopedPtr(T* p = 0): ptr(p){}

    //////////////////////////////
    ~KScopedPtr()
    {
        DeletionPolicy::DeleteIt(&ptr);
    }

    //////////////////////////////
    inline void Reset(T* p = 0)
    {
        K_ASSERT(p == 0 || p != ptr); // catch self-reset errors
        ThisType(p).Swap(*this);
    }

    //////////////////////////////
    inline T & operator*() const
    {
        K_ASSERT(ptr != 0);
        return *ptr;
    }

    //////////////////////////////
    inline T* operator->() const
    {
        K_ASSERT(ptr != 0);
        return ptr;
    }

    //////////////////////////////
    inline T* Get() const
    {
        return ptr;
    }

    inline operator T* () const
    {
        return ptr;
    }

    //////////////////////////////
    // Implicit conversion to "bool"
    operator bool () const
    {
        return ptr != 0;
    }

    //////////////////////////////
    bool operator! () const
    {
        return ptr == 0;
    }

    //////////////////////////////
    void Swap(KScopedPtr & b)
    {
        T * tmp = b.ptr;
        b.ptr = ptr;
        ptr = tmp;
    }

    //////////////////////////////
    T* Release()
    {
        T* tmp = ptr;
        ptr = NULL;

        return tmp;
    }
};

//----------------------------------------
//
// Deletion policy dictates the way the pointer is destroyed
// The FBXObjectDeletionPolicy, dictate the way we destroy
// KFbxObject. This policy is used by KFBXObjectScopedPtr
//
template <class FBXObjectT>
class FBXObjectDeletionPolicy
{
public:
    static inline void DeleteIt(FBXObjectT** ptr)
    {
        if (*ptr != NULL)
        {
            (*ptr)->Destroy();
            *ptr = NULL;
        }
    }
};

//---------------------------------
template <class FBXObjectT>
class KFBXObjectScopedPtr: public KScopedPtr<FBXObjectT, FBXObjectDeletionPolicy<FBXObjectT> >
{
public:
    explicit KFBXObjectScopedPtr(FBXObjectT* p = 0):KScopedPtr<FBXObjectT, FBXObjectDeletionPolicy<FBXObjectT> >(p){}
};

// End FBX namespace
#include <kbaselib_nsend.h>

#endif
