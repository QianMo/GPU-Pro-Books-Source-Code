/*!  \file kfbxmemoryallocator.h
 */
 
#ifndef _FBXSDK_MEMORY_ALLOCATOR_H_
#define _FBXSDK_MEMORY_ALLOCATOR_H_

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

#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <stdlib.h>


#include <fbxfilesdk_nsbegin.h>

/** \brief This class provides services for memory management.
  * \nosubgrouping
  * The FBX SDK Manager uses an object of type KFbxMemoryAllocator
  * to allocate and free memory. Implement your own class if your
  * application requires custom memory management.
  */
class KFBX_DLL KFbxMemoryAllocator
{
public:
	/** Constructor
	  * \param pMallocHandler      Pointer to a function implementing malloc. This function allocates memory blocks.
	  * \param pCallocHandler      Pointer to a function implementing calloc. This function allocates an array in memory with elements initialized to 0.
	  * \param pReallocHandler     Pointer to a function implementing realloc. This function reallocate memory blocks.
	  * \param pFreeHandler        Pointer to a function implementing free. This function deallocates memory blocks.
	  */
	KFbxMemoryAllocator(void* (*pMallocHandler)(size_t),
						void* (*pCallocHandler)(size_t,size_t),
						void* (*pReallocHandler)(void*,size_t),
						void  (*pFreeHandler)(void*))
		: mMallocHandler(pMallocHandler)
		, mCallocHandler(pCallocHandler)
		, mReallocHandler(pReallocHandler)
		, mFreeHandler(pFreeHandler)
		, mMallocHandler_debug(0)
		, mCallocHandler_debug(0)
		, mReallocHandler_debug(0)
		, mFreeHandler_debug(0)
	{
	}
	KFbxMemoryAllocator(void* (*pMallocHandler)(size_t),
						void* (*pCallocHandler)(size_t,size_t),
						void* (*pReallocHandler)(void*,size_t),
						void  (*pFreeHandler)(void*),
						void* (*pMallocHandler_debug)(size_t,int,const char *,int),
						void* (*pCallocHandler_debug)(size_t, size_t,int,const char *,int),
						void* (*pReallocHandler_debug)(void*, size_t,int,const char *,int),
						void  (*pFreeHandler_debug)(void*,int)
						)
		: mMallocHandler(pMallocHandler)
		, mCallocHandler(pCallocHandler)
		, mReallocHandler(pReallocHandler)
		, mFreeHandler(pFreeHandler)
		, mMallocHandler_debug(pMallocHandler_debug)
		, mCallocHandler_debug(pCallocHandler_debug)
		, mReallocHandler_debug(pReallocHandler_debug)
		, mFreeHandler_debug(pFreeHandler_debug)
	{
	}
	virtual ~KFbxMemoryAllocator() = 0;	

	void* (*mMallocHandler)(size_t);
	void* (*mCallocHandler)(size_t,size_t);
	void* (*mReallocHandler)(void*,size_t);
	void  (*mFreeHandler)(void*);
	void* (*mMallocHandler_debug)(size_t,int,const char *,int);
	void* (*mCallocHandler_debug)(size_t, size_t,int,const char *,int);
	void* (*mReallocHandler_debug)(void*, size_t,int,const char *,int);
	void  (*mFreeHandler_debug)(void*,int);
};

/** Default implementation of memory allocator.
  * \nosubgrouping
  * This default implementation uses malloc, calloc, realloc, and free from the C runtime library.
  */
class KFBX_DLL KFbxDefaultMemoryAllocator : public KFbxMemoryAllocator
{
public:
	KFbxDefaultMemoryAllocator();
	~KFbxDefaultMemoryAllocator();
};

#include <fbxfilesdk_nsend.h>

#endif // _FBXSDK_MEMORY_ALLOCATOR_H_



