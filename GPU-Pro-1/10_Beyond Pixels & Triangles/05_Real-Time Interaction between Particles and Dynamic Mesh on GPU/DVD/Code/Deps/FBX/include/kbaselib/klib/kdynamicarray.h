#ifndef _FBXSDK_KDYNAMICARRAY_H_
#define _FBXSDK_KDYNAMICARRAY_H_

/**************************************************************************************

 Copyright © 2001 - 2008 Autodesk, Inc. and/or its licensors.
 All Rights Reserved.

 The coded instructions, statements, computer programs, and/or related material 
 (collectively the "Data") in these files contain unpublished information 
 proprietary to Autodesk, Inc. and/or its licensors, which is protected by 
 Canada and United States of America federal copyright law and by international 
 treaties. 
 
 The Data may not be disclosed or distributed to third parties, in whole or in
 part, without the prior written consent of Autodesk, Inc. ("Autodesk").

 THE DATA IS PROVIDED "AS IS" AND WITHOUT WARRANTY.
 ALL WARRANTIES ARE EXPRESSLY EXCLUDED AND DISCLAIMED. AUTODESK MAKES NO
 WARRANTY OF ANY KIND WITH RESPECT TO THE DATA, EXPRESS, IMPLIED OR ARISING
 BY CUSTOM OR TRADE USAGE, AND DISCLAIMS ANY IMPLIED WARRANTIES OF TITLE, 
 NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE OR USE. 
 WITHOUT LIMITING THE FOREGOING, AUTODESK DOES NOT WARRANT THAT THE OPERATION
 OF THE DATA WILL BE UNINTERRUPTED OR ERROR FREE. 
 
 IN NO EVENT SHALL AUTODESK, ITS AFFILIATES, PARENT COMPANIES, LICENSORS
 OR SUPPLIERS ("AUTODESK GROUP") BE LIABLE FOR ANY LOSSES, DAMAGES OR EXPENSES
 OF ANY KIND (INCLUDING WITHOUT LIMITATION PUNITIVE OR MULTIPLE DAMAGES OR OTHER
 SPECIAL, DIRECT, INDIRECT, EXEMPLARY, INCIDENTAL, LOSS OF PROFITS, REVENUE
 OR DATA, COST OF COVER OR CONSEQUENTIAL LOSSES OR DAMAGES OF ANY KIND),
 HOWEVER CAUSED, AND REGARDLESS OF THE THEORY OF LIABILITY, WHETHER DERIVED
 FROM CONTRACT, TORT (INCLUDING, BUT NOT LIMITED TO, NEGLIGENCE), OR OTHERWISE,
 ARISING OUT OF OR RELATING TO THE DATA OR ITS USE OR ANY OTHER PERFORMANCE,
 WHETHER OR NOT AUTODESK HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS
 OR DAMAGE. 

**************************************************************************************/

#include <kbaselib_h.h>

#include <klib/kcontainerallocators.h>
#include <klib/kdebug.h>

#include <new>

#include <kbaselib_nsbegin.h>

template <typename VALUE_TYPE, typename ALLOCATOR = KBaseAllocator>
class KDynamicArray
{
public:
    typedef VALUE_TYPE ValueType;
    typedef ALLOCATOR AllocatorType;

    KDynamicArray()
        : mArray(NULL)
        , mArrayCapacity(0)
        , mValueCount(0)
        , mAllocator(sizeof(ValueType))
    {
    }

    KDynamicArray(size_t const pInitialSize)
        : mArray(NULL)
        , mArrayCapacity(0)
        , mValueCount(0)
        , mAllocator(sizeof(ValueType))
    {
        Reserve(pInitialSize);
    }

    KDynamicArray(KDynamicArray const& pArray)
        : mArray(NULL)
        , mArrayCapacity(0)
        , mValueCount(0)
        , mAllocator(sizeof(ValueType))
    {
        Reserve(pArray.mArrayCapacity);
        CopyArray(mArray, pArray.mArray, pArray.mValueCount);
        mValueCount = pArray.mValueCount;
    }

    ~KDynamicArray()
    {
        size_t i;
        for (i = 0; i < mValueCount; i++)
        {
            mArray[i].~VALUE_TYPE();
        }

        mAllocator.FreeMemory(mArray);
    }

    size_t GetCapacity() const
    {
        return mArrayCapacity;
    }

    size_t GetSize() const
    {
        return mValueCount;
    }

    void Reserve(size_t const pCount)
    {
        if (pCount > mArrayCapacity)
        {
            // We don't use mAllocator.PreAllocate, because we want our array
            // to be continuous in memory.
            void* lBuffer = mAllocator.AllocateRecords(pCount);
            ValueType* lNewArray = reinterpret_cast<ValueType*>(lBuffer);

            MoveArray(lNewArray, mArray, mValueCount);

            mAllocator.FreeMemory(mArray);
            mArray = lNewArray;
            mArrayCapacity = pCount;
        }
    }

    void PushBack(ValueType const& pValue, size_t const pNCopies = 1)
    {
        if (mValueCount + pNCopies > mArrayCapacity)
        {
            // grow by 50%
            size_t lNewSize = mArrayCapacity + mArrayCapacity / 2;

            if (mValueCount + pNCopies > lNewSize)
            {
                lNewSize = mValueCount + pNCopies;
            }

            Reserve(lNewSize);
        }

        K_ASSERT(mValueCount + pNCopies <= mArrayCapacity);

        Fill(mArray + mValueCount, pValue, pNCopies);

        mValueCount += pNCopies;
    }

    void Insert(size_t const pIndex, ValueType const& pValue, size_t const pNCopies = 1)
    {
        K_ASSERT(pIndex >= 0);
        K_ASSERT(pIndex <= mValueCount);

        ValueType lValue = pValue; // in case pValue is in array

        if (pNCopies == 0)
        {
        }
        else if (pIndex >= mValueCount)
        {
            PushBack(pValue, pNCopies);
        }
        else if (mValueCount + pNCopies > mArrayCapacity)
        {
            // not enough room
            // grow by 50%
            size_t lNewSize = mArrayCapacity + mArrayCapacity / 2;

            if (mValueCount + pNCopies > lNewSize)
            {
                lNewSize = mValueCount + pNCopies;
            }

            void* lBuffer = mAllocator.AllocateRecords(lNewSize);
            ValueType* lNewArray = reinterpret_cast<ValueType*>(lBuffer);

            MoveArray(lNewArray, mArray, pIndex); // copy prefix
            Fill(lNewArray + pIndex, pValue, pNCopies); // copy values
            MoveArray(lNewArray + pIndex + pNCopies, mArray + pIndex, mValueCount - pIndex); // copy suffix

            mAllocator.FreeMemory(mArray);
            mArray = lNewArray;
            mValueCount += pNCopies;
            mArrayCapacity = lNewSize;
        }
        else
        {
            // copy suffix backwards
            MoveArrayBackwards(mArray + pIndex + pNCopies, mArray + pIndex, mValueCount - pIndex);
            Fill(mArray + pIndex, pValue, pNCopies); // copy values
            mValueCount += pNCopies;
        }
    }

    void PopBack(size_t pNElements = 1)
    {
        K_ASSERT(pNElements <= mValueCount);

        size_t i;
        for (i = mValueCount - pNElements; i < mValueCount; i++)
        {
            mArray[i].~VALUE_TYPE();
        }

        mValueCount -= pNElements;
    }

    void Remove(size_t const pIndex, size_t pNElements = 1)
    {
        K_ASSERT(pIndex >= 0);
        K_ASSERT(pIndex <= mValueCount);
        K_ASSERT(pIndex + pNElements <= mValueCount);

        if (pIndex + pNElements >= mValueCount)
        {
            PopBack(pNElements);
        }
        else
        {
            size_t i;
            for (i = pIndex; i < pIndex + pNElements; i++)
            {
                mArray[i].~VALUE_TYPE();
            }

            MoveOverlappingArray(mArray + pIndex, mArray + pIndex + pNElements, mValueCount - pNElements);

            mValueCount -= pNElements;
        }
    }

    ValueType& operator[](size_t const pIndex)
    {
        return *(mArray + pIndex);
    }

    ValueType const& operator[](size_t const pIndex) const
    {
        return *(mArray + pIndex);
    }

    KDynamicArray& operator=(KDynamicArray const& pArray)
    {
        Reserve(pArray.mArrayCapacity);
        CopyArray(mArray, pArray.mArray, pArray.mValueCount);
        mValueCount = pArray.mValueCount;

        return *this;
    }

private:
    static void CopyArray(ValueType* pDest, ValueType const* pSrc, size_t pCount)
    {
        int i;
        for (i = 0; i < pCount; i++)
        {
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma push_macro("new")
#undef new
#endif
            new(&(pDest[i])) ValueType(pSrc[i]);
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma pop_macro("new")
#endif
        }
    }

    static void MoveArray(ValueType* pDest, ValueType const* pSrc, size_t pCount)
    {
        int i;
        for (i = 0; i < pCount; i++)
        {
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma push_macro("new")
#undef new
#endif
            new(&(pDest[i])) ValueType(pSrc[i]);
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma pop_macro("new")
#endif
        }

        for (i = 0; i < pCount; i++)
        {
            pSrc[i].~VALUE_TYPE();
        }
    }

    static void MoveOverlappingArray(ValueType* pDest, ValueType const* pSrc, size_t pCount)
    {
        int i;
        for (i = 0; i < pCount; i++)
        {
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma push_macro("new")
#undef new
#endif
            new(&(pDest[i])) ValueType(pSrc[i]);
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma pop_macro("new")
#endif

            pSrc[i].~VALUE_TYPE();
        }
    }

    static void MoveArrayBackwards(ValueType* pDest, ValueType const* pSrc, size_t pCount)
    {
        size_t i;
        for (i = pCount - 1; i >= 0; i--)
        {
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma push_macro("new")
#undef new
#endif
            new(&(pDest[i])) ValueType(pSrc[i]);
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma pop_macro("new")
#endif

            pSrc[i].~VALUE_TYPE();
        }
    }

    static void Fill(ValueType* pDest, ValueType const& pValue, size_t pCount)
    {
        size_t i;
        for (i = 0; i < pCount; i++)
        {
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma push_macro("new")
#undef new
#endif
            new(&(pDest[i])) ValueType(pValue);
#if ( defined(MEMORY_DEBUG) && defined(_DEBUG) && defined(KARCH_ENV_WIN32))
#pragma pop_macro("new")
#endif
        }
    }


    ValueType* mArray;
    size_t mArrayCapacity;
    size_t mValueCount;

    AllocatorType mAllocator;
};

#include <kbaselib_nsend.h>

#endif
