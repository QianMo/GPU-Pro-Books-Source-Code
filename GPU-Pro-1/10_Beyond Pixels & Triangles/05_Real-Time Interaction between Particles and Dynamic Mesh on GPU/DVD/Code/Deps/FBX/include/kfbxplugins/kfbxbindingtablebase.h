#ifndef _FBXSDK_BINDING_TABLE_BASE_H_ 
#define _FBXSDK_BINDING_TABLE_BASE_H_

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

// FBX SDK includes
#include <kfbxplugins/kfbxobject.h>
#include <kfbxplugins/kfbxbindingtableentry.h>
#include <klib/kdynamicarray.h>

// FBX namespace
#include <fbxfilesdk_nsbegin.h>

/** \brief A binding table represents a collection of bindings
* from source types such as KFbxObjects, or KFbxLayerElements
* to destinations which can be of similar types. See KFbxBindingTableEntry.
* \nosubgrouping
*/
class KFBX_DLL KFbxBindingTableBase : public KFbxObject
{
	KFBXOBJECT_DECLARE_ABSTRACT(KFbxBindingTableBase,KFbxObject);

public:

	/** Adds a new entry to the binding table.
	* \return The new entry
	*/
	KFbxBindingTableEntry& AddNewEntry();

	/** Query the number of table entries.
	* \return The number of entries
	*/
	size_t GetEntryCount() const;

	/** Access a table entry. 
	* \param pIndex Valid range is [0, GetEntryCount()-1]
	* \return A valid table entry if pIndex is valid. Otherwise the value is undefined.
	*/
	KFbxBindingTableEntry const& GetEntry( size_t pIndex ) const;

	/** Access a table entry. 
	* \param pIndex Valid range is [0, GetEntryCount()-1]
	* \return A valid table entry if pIndex is valid. Otherwise the value is undefined.
	*/	
	KFbxBindingTableEntry& GetEntry( size_t pIndex );

	KFbxBindingTableBase& operator=(KFbxBindingTableBase const& pTable);

	/** Retrieve the table entry  for the given source value.
	* \param pSrcName The source value to query
	* \return The corresponding entry, or NULL if no entry in 
	* the table has a source equal in value to pSrcName.
	*/
	KFbxBindingTableEntry const* GetEntryForSource(char const* pSrcName) const;

	/** Retrieve the table entry for the given destination value.
	* \param pDestName The destination value to query
	* \return The corresponding entry, or NULL if no entry in 
	* the table has a destination equal in value to pDestName.
	*/
	KFbxBindingTableEntry const* GetEntryForDestination(char const* pDestName) const;

	///////////////////////////////////////////////////////////////////////////////
	//
	//  WARNING!
	//
	//	Anything beyond these lines may not be documented accurately and is 
	// 	subject to change without notice.
	//
	///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

protected:
	KFbxBindingTableBase(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxBindingTableBase() = 0; // make this class abstract

private:
	KDynamicArray<KFbxBindingTableEntry> mEntries;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // _FBXSDK_VERTEX_CACHE_DEFORMER_H_ 
