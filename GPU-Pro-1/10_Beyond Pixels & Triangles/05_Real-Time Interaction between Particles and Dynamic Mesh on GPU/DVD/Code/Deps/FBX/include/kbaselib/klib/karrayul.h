/*!  \file karrayul.h
 */

#ifndef _FBXSDK_KARRAYUL_H_
#define _FBXSDK_KARRAYUL_H_

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

#ifndef K_PLUGIN
    #include <klib/kdebug.h>
#endif

#include <string.h>
#include <stdlib.h>

#define KARRAYUL_BLOCKSIZE 4

#include <object/i/iobject.h>

#include <kbaselib_forward.h>
#include <kbaselib_nsbegin.h> // namespace

    /***********************************************************************
        CLASS KStaticArray
    ************************************************************************/
    template< class Type > class KBaseStaticArray
    {
        protected:
        int  mCount;
        Type *mArrayBuf;

        public:
        inline int GetCount() { return mCount; }

        //! Access pointer at given index.
        inline Type &operator[](int pIndex)
        {
        #ifndef K_PLUGIN
            K_ASSERT_MSG( pIndex >= 0   , "Buffer underflow, appelle ton plombier.");
            K_ASSERT_MSG( pIndex < mCount,"Buffer overflow, appelle ton plombier.");
        #endif
            return mArrayBuf[pIndex];
        }
    };

    // Static Array
    template< class Type, int Count > class KStaticArray : public KBaseStaticArray<Type>
    {
        public:
        Type mArray[Count];
            inline KStaticArray(){ this->mArrayBuf = mArray; this->mCount = Count;}
    };

    template< class Type, int Count1, int Count2 > class KStaticArray2d
    {
        public:
#if defined(KARCH_DEV_MSC) && (_MSC_VER <= 1200)// VC6
        KStaticArray<Type,Count2> *mArray;

        KStaticArray2d() { mArray = new KStaticArray<Type,Count2>(Count1); }
        ~KStaticArray2d() { delete[] mArray; }
#else
        KStaticArray<Type,Count2> mArray[Count1];
#endif

        // Access pointer at given index.
        inline KStaticArray< Type, Count2 > &operator[](int pIndex)
        {
        #ifndef K_PLUGIN
            K_ASSERT_MSG( pIndex >= 0   , "Buffer underflow, appelle ton plombier.");
            K_ASSERT_MSG( pIndex < Count1,"Buffer overflow, appelle ton plombier.");
        #endif
            return mArray[pIndex];
        }
    };

    /***********************************************************************
        CLASS KArrayTemplate
    ************************************************************************/
    class  KBaseArraySize {
        public:
            KBaseArraySize( int pItemSize)
                : mItemSize(pItemSize)
            {
            }
            inline int GetTypeSize() const { return mItemSize; }
        private:
            int mItemSize;
    };

    template <class T> class  KBaseArraySizeType
    {
        public:
            inline int GetTypeSize() const { return sizeof(T); }
    };

    // Helpers
    KBASELIB_DLL void KBaseArrayFree(char*);
    KBASELIB_DLL char* KBaseArrayRealloc(char*, size_t);
    
    ///////////////////////////////////////////////////////////////////////////////
    //
    //  WARNING!
    //
    //  Anything beyond these lines may not be documented accurately and is
    //  subject to change without notice.
    //
    ///////////////////////////////////////////////////////////////////////////////
    #ifndef DOXYGEN_SHOULD_SKIP_THIS
        KBASELIB_DLL void* KBaseArrayGetAlloc();
    #endif

    template <class TypeSize> class  KBaseArray
    {
        protected:
            struct KHeader {
                int mArrayCount;
                int mBlockCount;
            };


        protected:
            /** Constructor.
            * \param pItemPerBlock Number of pointers per allocated block.
            * \param pItemSize Size of one item of the array.
            */
            inline KBaseArray(TypeSize pTypeSize)
                : mTypeSize(pTypeSize)
            {
                mBaseArray  = NULL;
                #ifndef K_PLUGIN
                    K_ASSERT( pItemPerBlock > 0 );
                #endif
            }


            //! Destructor.
            inline ~KBaseArray(){
                Clear ();
            }

            /** Insert an item at the given position.
            * \param pIndex Position where to insert the item.
            * \param pItem  Pointer to the item to be inserted.
            * \remark if pIndex is greater than the number of items already in the
            * array, the item will be appended at the end.
            * \return The actual position where the item as been inserted.
            */
            inline int      InsertAt(int pIndex, void *pItem)
            {
              int lArrayCount = GetArrayCount();
              int lBlockCount = GetBlockCount();

                #ifndef K_PLUGIN
                    K_ASSERT( pIndex >= 0 );
                #endif

                if (pIndex>lArrayCount) {
                    pIndex = GetArrayCount();
                }

                if (lArrayCount>= lBlockCount*KARRAYUL_BLOCKSIZE)
                {
                    // must Alloc.Realloc some new space

                    // double the number of blocks.
                    lBlockCount = ( 0 == lBlockCount ) ? 1 : lBlockCount * 2;
                    mBaseArray = KBaseArrayRealloc(mBaseArray, (size_t) (lBlockCount*KARRAYUL_BLOCKSIZE*GetTypeSize()) + GetHeaderOffset() );
                }

                if (pIndex<lArrayCount)
                {
                    // This is an insert
                    memmove (&(mBaseArray[(pIndex+1)*GetTypeSize()+ GetHeaderOffset() ]), &(mBaseArray[(pIndex)*GetTypeSize()+ GetHeaderOffset()] ), GetTypeSize()*(lArrayCount-pIndex));
                }

                memmove (&(mBaseArray[(pIndex)*GetTypeSize()+ GetHeaderOffset() ]), pItem, GetTypeSize());

                SetArrayCount(lArrayCount+1);
                SetBlockCount(lBlockCount);

                return pIndex;
            }


            /** Get the item at the given position.
            * \param pIndex The position of the item to access.
            * \return Pointer to the item.
            * \remarks This method assumes that the passed inded is
            * in the valid range of the array. No checks are made.
            */
            inline void*    GetAt(int pIndex)                           { return &(mBaseArray[(pIndex)*GetTypeSize()+ GetHeaderOffset() ]); }

            /** Removes the item at the given position.
            * \param pIndex The position of the item to remove.
            * \remarks If the index is not valid, nothing is performed. Otherwise,
            * the item is removed from the array and the items are shifted to fill the
            * empty slot.
            */
            inline void RemoveAt(int pIndex)
            {

            #if defined(_DEBUG) && !defined(KARCH_ENV_MACOSX)
                if (!ValidateIndex( pIndex ))
                {
                    return;
                }
            #endif
                int lArrayCount = GetArrayCount();
                if (pIndex+1<lArrayCount)
                {
                    memmove (&(mBaseArray[(pIndex)*GetTypeSize()+ GetHeaderOffset() ]), &(mBaseArray[(pIndex+1)*GetTypeSize()+ GetHeaderOffset() ]), GetTypeSize()*(lArrayCount-pIndex-1));
                }

                SetArrayCount( lArrayCount-1 );

            #ifdef _DEBUG
                memset( &(mBaseArray[(GetArrayCount())*GetTypeSize()+ GetHeaderOffset() ]),0,GetTypeSize());
            #endif
            }


            /** Check that the given position is inside the array boundaries.
            * \param pIndex Index value to validate.
            * \return \c true if the index value is within the array boundaries. \c false
            * otherwise.
            */
            inline bool ValidateIndex( int pIndex ) const
            {
                int lArrayCount = GetArrayCount();
                if (pIndex>=0 && pIndex<lArrayCount)
                {
                    return true;
                } else
                {
                    #ifndef K_PLUGIN
                        K_ASSERT_MSG_NOW(_T("ArrayTemplate : Index out of range"));
                    #endif
                    return false;
                }
            }


        public:
            /** Get number of pointers in the array.
            * \return The number of items in the array.
            */
            inline int GetCount() const { return GetArrayCount(); }

            //! Remove all pointers without deleting the associated objects.
            inline void Clear()
            {
                if (mBaseArray!=NULL)
                {
                    KBaseArrayFree(mBaseArray);
                    mBaseArray = NULL;
                }
            }


            //! Fast empty, set object count to zero but don't free any memory.
            inline void Empty()
            {
            #ifndef K_PLUGIN
                #ifdef _DEBUG
                    memset( mBaseArray+ GetHeaderOffset() ,0,GetArrayCount()*GetTypeSize());
                #endif
            #endif
                SetArrayCount(0);
            }


            /** Set array capacity to contain at least the specified number of elements without reallocating.
            * \param pCapacity Number of items that can be stored in the array before reallocating the memory.
            * \return The number of available slots in the array.
            * \remarks If capacity is lower than arrayCount, arrayCount is lowered to capacity.
            */
            inline int      Reserve(int pCapacity)
            {

                #ifndef K_PLUGIN
                    K_ASSERT( pCapacity > 0 );
                #endif

                if( pCapacity )
                {
                    const kUInt lTempNewBlockCount = ( (kUInt) (pCapacity + KARRAYUL_BLOCKSIZE - 1 ) / KARRAYUL_BLOCKSIZE );
                    const kUInt lNewBlockCount = (lTempNewBlockCount > 1 ? lTempNewBlockCount : 1);

                    int         lArrayCount   = GetArrayCount();
                    int         lBlockCount   = GetBlockCount();

                    const kUInt lOldArraySize = lArrayCount*GetTypeSize();
                    const kUInt lNewArraySize = lNewBlockCount*KARRAYUL_BLOCKSIZE*GetTypeSize();

                    if (lNewBlockCount != (kUInt) lBlockCount)
                        mBaseArray = KBaseArrayRealloc(mBaseArray, (size_t) lNewArraySize+ GetHeaderOffset()  );

                    if( lNewBlockCount > (kUInt) lBlockCount ) {
                        memset( ((char*)mBaseArray+ GetHeaderOffset() ) + lOldArraySize, 0, (size_t) (lNewArraySize-lOldArraySize) );
                        SetArrayCount(lArrayCount);
                    } else if (pCapacity < lArrayCount)
                    {
                        memset( ((char*)mBaseArray+ GetHeaderOffset() ) + pCapacity*GetTypeSize(), 0, (size_t) (lNewArraySize-pCapacity*GetTypeSize()) );
                        SetArrayCount(pCapacity);
                    }

                    SetBlockCount(lNewBlockCount);
                }

                return GetBlockCount()*KARRAYUL_BLOCKSIZE;
            }


            //! Set arrayCount to specified number of elements. The array capacity is adjusted accordingly.
            //  Differ from SetCount because array capacity can be lowewed.

            /** Force the array of elements to a given size.
            * \remarks If the array is upsized, the memory allocated is set to 0 and
            * no constructor is called. Thus, this function is not appropriate for
            * types of elements requiring initialization.
            */
            inline void     SetCount (int pCount)
            {
            #ifndef K_PLUGIN
            #ifdef _DEBUG
                if (pCount<0)
                {
                    K_ASSERT_MSG_NOW (_T("ArrayUL : Item count can't be negative"));
                    return ;
                }
            #endif
            #endif
                int lArrayCount = GetArrayCount();
                if (pCount > lArrayCount)
                {
                    AddMultiple( pCount-lArrayCount);
                } else
                {
                    SetArrayCount(pCount);
                }
            }

            inline void     Resize(int pItemCount)
            {
                #ifndef K_PLUGIN
                    K_ASSERT( pItemCount >= 0 );
                #endif

                const kUInt lTempNewBlockCount = ( (kUInt) (pItemCount + KARRAYUL_BLOCKSIZE - 1 ) / KARRAYUL_BLOCKSIZE );
                const kUInt lNewBlockCount = (lTempNewBlockCount > 1 ? lTempNewBlockCount : 1);

                int         lArrayCount     = GetArrayCount();
                int         lBlockCount   = GetBlockCount();

                const kUInt lOldArraySize   = lArrayCount*GetTypeSize();
                const kUInt lNewArraySize   = lNewBlockCount*KARRAYUL_BLOCKSIZE*GetTypeSize();

                if (lNewBlockCount != (kUInt) lBlockCount)
                    mBaseArray = KBaseArrayRealloc(mBaseArray, (size_t) lNewArraySize+ GetHeaderOffset()  );

                if( lNewBlockCount > (kUInt) lBlockCount )
                    memset( ((char*)mBaseArray+ GetHeaderOffset() ) + lOldArraySize, 0, (size_t) (lNewArraySize-lOldArraySize) );
                else if (pItemCount < lArrayCount)
                    memset( ((char*)mBaseArray+ GetHeaderOffset() ) + pItemCount*GetTypeSize(), 0, (size_t) (lNewArraySize-pItemCount*GetTypeSize()) );

                SetBlockCount(lNewBlockCount);
                SetArrayCount(pItemCount);
            }

            inline void     AddMultiple(int pItemCount)
            {
                #ifndef K_PLUGIN
                    K_ASSERT( pItemCount > 0 );
                #endif

                if( pItemCount )
                {
                    int         lArrayCount = GetArrayCount();
                    int         lBlockCount = GetBlockCount();
                    const kUInt lTempNewBlockCount = ( (kUInt) (lArrayCount+pItemCount + KARRAYUL_BLOCKSIZE - 1 ) / KARRAYUL_BLOCKSIZE );
                    const kUInt lNewBlockCount = (lTempNewBlockCount > 1 ? lTempNewBlockCount : 1);

                    const kUInt lOldArraySize = lArrayCount*GetTypeSize();
                    const kUInt lNewArraySize = lNewBlockCount*KARRAYUL_BLOCKSIZE*GetTypeSize();

                    #ifndef K_PLUGIN
                        K_ASSERT( lOldArraySize < lNewArraySize );
                    #endif

                    if( lNewBlockCount > (kUInt) lBlockCount )
                    {
                        mBaseArray = KBaseArrayRealloc(mBaseArray, (size_t) lNewArraySize+ GetHeaderOffset()  );
                        lBlockCount = lNewBlockCount;
                    }

                    memset( ((char*)mBaseArray+ GetHeaderOffset() ) + lOldArraySize, 0, (size_t) (lNewArraySize-lOldArraySize) );
                    SetArrayCount ( lArrayCount + pItemCount );
                    SetBlockCount (lBlockCount);
                }
            }


            inline int  GetTypeSize() const { return mTypeSize.GetTypeSize(); }

        ///////////////////////////////////////////////////////////////////////////////
        //
        //  WARNING!
        //
        //  Anything beyond these lines may not be documented accurately and is
        //  subject to change without notice.
        //
        ///////////////////////////////////////////////////////////////////////////////
        #ifndef DOXYGEN_SHOULD_SKIP_THIS

        protected:
            inline KHeader* const   GetHeader() const
            {
                return (KHeader*    const   )mBaseArray;
            }
            inline KHeader*     GetHeader()
            {
                return (KHeader*)mBaseArray;
            }
            inline int GetHeaderOffset() const
            {
                return sizeof(KHeader);
            }
            inline int GetArrayCount() const
            {
                return GetHeader() ? GetHeader()->mArrayCount : 0;
            }
            inline void SetArrayCount(int pArrayCount)
            {
                if (GetHeader()) GetHeader()->mArrayCount=pArrayCount;
            }
            inline int GetBlockCount() const
            {
                return GetHeader() ? GetHeader()->mBlockCount : 0;
            }
            inline void SetBlockCount(int pArrayCount)
            {
                if (GetHeader()) GetHeader()->mBlockCount=pArrayCount;
            }

        protected:
            char*           mBaseArray;
            TypeSize        mTypeSize;

        #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };


    //! Array of pointers.
    #define VC6

    template < class Type > class  KArrayTemplate : public KBaseArray< KBaseArraySizeType<Type> >
    {
        typedef KBaseArray< KBaseArraySizeType<Type> > ParentClass;


    public:
        /** Constructor.
        * \param pItemPerBlock Number of pointers per allocated block.
        */
        inline KArrayTemplate()
            : ParentClass (KBaseArraySizeType<Type>())
        {
        }

        //! Copy constructor.
        inline KArrayTemplate(const KArrayTemplate& pArrayTemplate)
            : ParentClass (KBaseArraySizeType<Type>())
        {
            *this = pArrayTemplate;
        }

        inline ~KArrayTemplate() {}

        /** Insert a pointer.
        * \param pIndex Position where to insert the pointer.
        * \param pItem  Item to insert.
        * \return Position of the inserted pointer in the array.
        * \remarks If the given index is out of range, the pointer is appended at the end.
        */
        inline int InsertAt(int pIndex, Type pItem)
        {
            return ParentClass::InsertAt( pIndex,&pItem );
        }

        /** Remove a pointer in the array without deleting the associated object.
        * \param pIndex Position of the item to remove.
        * \return Removed item.
        */
        inline Type RemoveAt(int pIndex)
        {
            Type tmpItem = GetAt(pIndex);
            ParentClass::RemoveAt( pIndex );
            return tmpItem;
        }

        /** Remove last pointer in the array without deleting the associated object.
        * \return Remove item.
        */
        inline Type RemoveLast()
        {
            return RemoveAt(ParentClass::GetArrayCount()-1);
        }

        /** Remove first matching pointer in the array without deleting the associated object.
        * \param pItem Item to remove.
        * \return \c true if a matching pointer is found and removed, \c false otherwise.
        */
        inline bool RemoveIt(Type pItem)
        {
            int Index = Find (pItem);
            if (Index>=0)
            {
                RemoveAt (Index);
                return true;
            }
            return false;
        }

        //! Access pointer at given index.
        inline Type &operator[](int pIndex) const
        {
        #if defined(_DEBUG) && !defined(KARCH_ENV_MACOSX)
            if (!ParentClass::ValidateIndex( pIndex ))
            {
                return (Type &)(ParentClass::mBaseArray[(0)*sizeof(Type)+ ParentClass::GetHeaderOffset() ]);
            }
        #endif
            return (Type &)(ParentClass::mBaseArray[(pIndex)*sizeof(Type)+ ParentClass::GetHeaderOffset() ]);
        }

        //! Set pointer at given index, must within valid range.
        inline void SetAt(int pIndex, Type pItem)
        {
        #if defined(_DEBUG) && !defined(KARCH_ENV_MACOSX)
            if (!ParentClass::ValidateIndex( pIndex ))
            {
                return;
            }
        #endif
            GetArray()[pIndex] = pItem;
        }

        //! Set last pointer, the array must contain at least one pointer.
        inline void SetLast(Type pItem)
        {
            SetAt (ParentClass::GetArrayCount()-1, pItem);
        }

        //! Get pointer at given index, must be within valid range.
        inline Type GetAt(int pIndex) const
        {
        #if defined(_DEBUG) && !defined(KARCH_ENV_MACOSX)
            if (!ParentClass::ValidateIndex( pIndex ))
            {
                return (Type &)(ParentClass::mBaseArray[(0)*sizeof(Type)+ ParentClass::GetHeaderOffset() ]);
            }
        #endif
            return (Type &)(ParentClass::mBaseArray[(pIndex)*sizeof(Type)+ ParentClass::GetHeaderOffset() ]);
        }

        /** Get first pointer, the array must contain at least one pointer.
        * \return First pointer.
        */
        inline Type GetFirst() const
        {
        #ifndef K_PLUGIN
            K_ASSERT( ParentClass::GetArrayCount() >= 1 );
        #endif
            return GetAt(0);
        }

        /** Get last pointer, the array must contain at least one pointer.
        * \return Last pointer.
        */
        inline Type GetLast() const
        {
        #ifndef K_PLUGIN
            K_ASSERT( ParentClass::GetArrayCount() >= 1 );
        #endif
            return GetAt(ParentClass::GetArrayCount()-1);
        }

        /** Find first matching pointer.
        * \return Index of first matching pointer found or -1 if  there is no
        * matching element.
        */
        inline int Find(Type pItem) const
        {
            return FindAfter( -1, pItem );
        }

        /** Find first matching pointer after given index.
        * \return Index of first matching pointer found after given index or
        * -1 if there is no matching pointer.
        * \remarks The index must be within valid range.
        */
        inline int FindAfter(int pAfterIndex, Type pItem) const
        {
        #ifndef K_PLUGIN
        #ifdef _DEBUG
            if ( pAfterIndex > ParentClass::GetArrayCount() || pAfterIndex < -1 )
            {
                K_ASSERT_MSG_NOW (_T("ArrayUL : Search Begin Index out of range"));
                return -1;
            }
        #endif
        #endif
            int Count;
            for ( Count=pAfterIndex+1; Count<ParentClass::GetArrayCount(); Count++)
            {
                if (GetAt(Count)==pItem)
                {
                    return Count;
                }
            }
            return -1;
        }

        /** Find first matching pointer before given index.
        * \return Index of first matching pointer found after given index or
        * -1 if there is no matching pointer.
        * \remarks The index must be within valid range.
        */
        inline int FindBefore(int pBeforeIndex, Type pItem) const
        {
        #ifndef K_PLUGIN
        #ifdef _DEBUG
            if ( pBeforeIndex > ParentClass::GetArrayCount() || pBeforeIndex <= 0 )
            {
                K_ASSERT_MSG_NOW (_T("ArrayUL : Search Begin Index out of range"));
                return -1;
            }
        #endif
        #endif
            int Count;
            for ( Count=pBeforeIndex-1; Count>=0; Count--)
            {
                if (GetAt(Count)==pItem)
                {
                    return Count;
                }
            }
            return -1;
        }

        /** Append a pointer at the end of the array.
        * \return Index of appended pointer.
        */
        inline int Add(Type pItem)
        {
            return InsertAt(ParentClass::GetArrayCount(), pItem);
        }

        /** Add Element at the end of array if not present.
        * \return Index of Element.
        */
        inline int AddUnique(Type pItem)
        {
            int lReturnIndex = Find(pItem);
            if (lReturnIndex == -1)
            {
                lReturnIndex = Add(pItem);
            }
            return lReturnIndex;
        }

        /**  Add multiple (init to zero) elements in the array, use SetAt or GetArray to set the value of the new elements
        *   \param pItemCount. How many new Array elements you want.
        */
        inline void AddMultiple( kUInt pItemCount )
        {
            ParentClass::AddMultiple( pItemCount );
        }

        inline void AddArray(KArrayTemplate<Type> &pArray)
        {
            int lSourceIndex, lCount = pArray.GetCount();
            if( lCount == 0 ) return;
            int lDestinationIndex = ParentClass::GetCount();
            AddMultiple(lCount);
            for( lSourceIndex = 0; lSourceIndex < lCount; lSourceIndex++)
            {
                SetAt(lDestinationIndex++, pArray[lSourceIndex]);
            }
        }

        inline void AddArrayNoDuplicate(KArrayTemplate<Type> &pArray)
        {
            int i, lCount = pArray.GetCount();
            for( i = 0; i < lCount; i++)
            {
                Type lItem = pArray[i];
                if (Find(lItem) == -1)
                {
                    Add(lItem);
                }
            }
        }
        inline void RemoveArray(KArrayTemplate<Type> &pArray)
        {
            int lRemoveIndex, lRemoveCount = pArray.GetCount();
            for( lRemoveIndex = 0; lRemoveIndex < lRemoveCount; lRemoveIndex++)
            {
                RemoveIt(pArray[lRemoveIndex]);
            }
        }

        //! Get pointer to internal array of pointers.
        inline Type* GetArray() const
        {
            if (ParentClass::mBaseArray == NULL)
                return NULL;

            return (Type*)(ParentClass::mBaseArray+ ParentClass::GetHeaderOffset()) ;
        }

        //! Copy array of pointers without copying the associated objects.
        inline KArrayTemplate<Type>& operator=(const KArrayTemplate<Type>& pArrayTemplate)
        {
            ParentClass::Clear();

            int i, lCount = pArrayTemplate.GetCount();

            for (i = 0; i < lCount; i++)
            {
                Add(pArrayTemplate[i]);
            }

            return (*this);
        }

        #ifdef K_PLUGIN
            //! Cast operator.
            inline operator Type* ()
            {
                return GetArray();
            }
        #endif
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    //  WARNING!
    //
    //  Anything beyond these lines may not be documented accurately and is
    //  subject to change without notice.
    //
    ///////////////////////////////////////////////////////////////////////////////

    #ifndef DOXYGEN_SHOULD_SKIP_THIS

    template <class Type> inline void DeleteAndClear(KArrayTemplate<Type>& Array)
    {
        kUInt lItemCount = Array.GetCount();
        while( lItemCount )
        {
            lItemCount--;
            Type& Item = (Array.operator[](lItemCount));
            delete Item;
            Item = NULL;
        }
        Array.Clear();
    }
/*    template <class Type> inline void DeleteAndClear(KArrayTemplate<Type>& Array)
    {
        kUInt lItemCount = Array.GetCount();
        while( lItemCount )
        {
            lItemCount--;
            Type& Item = (Array.operator[](lItemCount));
            delete Item;
            Item = NULL;
        }
        Array.Clear();
    }
*/
    typedef class KBASELIB_DLL KArrayTemplate<int *>    KArrayHkInt;
    typedef class KBASELIB_DLL KArrayTemplate<kUInt *>  KArrayHkUInt;
    typedef class KBASELIB_DLL KArrayTemplate<double *> KArrayHkDouble;
    typedef class KBASELIB_DLL KArrayTemplate<float *>  KArrayHkFloat;
    typedef class KBASELIB_DLL KArrayTemplate<void *>   KArrayVoid;
    typedef class KBASELIB_DLL KArrayTemplate<char *>   KArrayChar;
    typedef class KBASELIB_DLL KArrayTemplate<int>      KArraykInt;
    typedef class KBASELIB_DLL KArrayTemplate<kUInt>    KArraykUInt;
    typedef class KBASELIB_DLL KArrayTemplate<float>    KArraykFloat;
    typedef class KBASELIB_DLL KArrayTemplate<double>   KArraykDouble;

    typedef class KBASELIB_DLL KArrayTemplate<kReference>   KArrayUL;

    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

#include <kbaselib_nsend.h>

#endif // #define _FBXSDK_KARRAYUL_H_


