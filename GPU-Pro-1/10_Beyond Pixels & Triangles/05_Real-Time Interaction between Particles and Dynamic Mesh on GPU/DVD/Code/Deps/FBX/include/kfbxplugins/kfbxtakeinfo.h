/*!  \file kfbxtakeinfo.h
 */

#ifndef _FBXSDK_TAKE_INFO_H_
#define _FBXSDK_TAKE_INFO_H_

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

#include <klib/kstring.h>
#include <klib/ktime.h>
#include <klib/karrayul.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxThumbnail;

struct KLayerInfo
{
	KString	mName;
	int		mId;
};


/** Contains take information prefetched from an imported file
  * and exported to an output file. 
  */
class KFBX_DLL KFbxTakeInfo
{
public:
	//! Constructor.
	KFbxTakeInfo(); 
	virtual ~KFbxTakeInfo(); 
	KFbxTakeInfo(const KFbxTakeInfo& pTakeInfo);
	KFbxTakeInfo& operator=(const KFbxTakeInfo& pTakeInfo);

	//! Take name.
	KString mName;

	/** Take name once imported in a scene.
	  * Modify it if it has to be different than the take name in the imported file.
	  * \remarks This field is only used when importing a scene.
	  */
	KString mImportName;

	//! Take description.
	KString mDescription;

	/** Import/export flag.
	  * Set to \c true by default. Set to \c false if the take must not be imported or exported.
	  */
	bool mSelect;

	//! Local time span, set to animation interval if left to default value.
	KTimeSpan mLocalTimeSpan;

	//! Reference time span, set to animation interval if left to default value.
	KTimeSpan mReferenceTimeSpan;

	/** Time value to offset the animation keys once imported in a scene.
	  * Modify it if the animation of a take must be offset.
	  * Its effect depends on the state of \c mImportOffsetType.
	  * \remarks This field is only used when importing a scene.
	  */
	KTime mImportOffset;

	/** EImportOffsetType Import offset types.
	  * - \e eABSOLUTE
	  * - \e eRELATIVE
	  */
	typedef enum  
	{
		eABSOLUTE,
		eRELATIVE
	} EImportOffsetType;

	/** Import offset type.
	  * If set to \c eABSOLUTE, \c mImportOffset gives the absolute time of 
	  * the first animation key and the appropriate time shift is applied 
	  * to all of the other animation keys.
	  * If set to \c eRELATIVE, \c mImportOffset gives the relative time 
	  * shift applied to all the animation keys.
	  */
	EImportOffsetType mImportOffsetType;

	/**	Get the take thumbnail.
	  * \return Pointer to the thumbnail.
	  */
	KFbxThumbnail* GetTakeThumbnail();

	/** Set the take thumbnail.
	  * \param pTakeThumbnail The referenced thumbnail object.
	  */
	void SetTakeThumbnail(KFbxThumbnail* pTakeThumbnail);

	void CopyLayers(const KFbxTakeInfo& pTakeInfo);
	KArrayTemplate<KLayerInfo*>	mLayerInfoList;
	int							mCurrentLayer;

protected:
	KFbxThumbnail* mTakeThumbnail;
};

typedef KFbxTakeInfo* HKFbxTakeInfo;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_TAKE_INFO_H_


