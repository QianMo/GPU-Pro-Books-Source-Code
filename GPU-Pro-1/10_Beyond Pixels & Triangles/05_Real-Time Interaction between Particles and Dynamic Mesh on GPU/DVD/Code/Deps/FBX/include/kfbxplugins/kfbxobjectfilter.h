/*!  \file kfbxobjectfilter.h
 */

#ifndef _FBXSDK_OBJECT_FILTER_H_
#define _FBXSDK_OBJECT_FILTER_H_

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

#include <kaydara.h>
#include <kbaselib/klib/kstring.h>
#include <kfbxplugins/kfbxobject.h>


#include <fbxfilesdk_nsbegin.h>

/** \brief This object represents a filter criteria on an object.
  * \nosubgrouping
  */
class KFBX_DLL KFbxObjectFilter
{

public:

	//! Tells if this filter match the given object
	virtual bool Match(const KFbxObject * pObjectPtr) const = 0;

	//! Tells if this filter does NOT match the given object
	virtual bool NotMatch(const KFbxObject * pObjectPtr) const { return !Match(pObjectPtr); };

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////
#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

/**\brief This class represents a name filter on an object.
  *\nosubgrouping
  */
class KFBX_DLL KFbxNameFilter : public KFbxObjectFilter
{
public:
	/**
	  * \name Constructor and Destructor
	  */
	//@{
	//!Constructor
    inline KFbxNameFilter( const char* pTargetName ) : mTargetName( pTargetName ) {};
	//@}

	//! Tells if this filter match the given object
    virtual bool Match(const KFbxObject * pObjectPtr) const { return pObjectPtr ? mTargetName == pObjectPtr->GetName() : false; }

private:
    KString mTargetName;
};

#include <fbxfilesdk_nsend.h>

#endif //_FBXSDK_OBJECT_FILTER_H_
