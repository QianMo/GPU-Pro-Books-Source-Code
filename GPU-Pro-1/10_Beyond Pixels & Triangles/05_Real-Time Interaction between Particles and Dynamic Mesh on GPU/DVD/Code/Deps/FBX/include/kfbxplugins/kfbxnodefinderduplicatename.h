/*!  \file kfbxnodefinderduplicatename.h
 */

#ifndef _FBXSDK_NODE_FINDER_DUPLICATE_NAME_H_
#define _FBXSDK_NODE_FINDER_DUPLICATE_NAME_H_

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

#include <kfbxplugins/kfbxnodefinder.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxScene;
class KFbxGeometry;
class KFbxSurfaceMaterial;
class KFbxTexture;
class KFbxVideo;
class KFbxGenericNode;
class KFbxLayerElementTexture;

//
//	CLASS KFbxNodeFinderDuplicateName
//

//! KFbxNodeFinderDuplicateName 
class KFBX_DLL KFbxNodeFinderDuplicateName : public KFbxNodeFinder
{
public:

	/** \enum EState
	  * - \e eCHECK_NODE_NAME
	  * - \e eCHECK_MATERIAL_NAME
	  * - \e eCHECK_TEXTURE_NAME
	  * - \e eCHECK_VIDEO_NAME
	  * - \e eCHECK_GENERIC_NODE_NAME
	  * - \e eSTATE_COUNT
	  */
	typedef enum 
	{
		eCHECK_NODE_NAME,
		eCHECK_MATERIAL_NAME,
		eCHECK_TEXTURE_NAME,
		eCHECK_VIDEO_NAME,
		eCHECK_GENERIC_NODE_NAME,
		eSTATE_COUNT
	} EState;

	/** Constructor. 
	 *  When the destination scene is specified, duplicates are searched in both the destination scene and in the processed node tree.
	 *  \param pDestinationScene     Destination scene to search. \c NULL by default.
	 */
	KFbxNodeFinderDuplicateName(KFbxScene* pDestinationScene = NULL);

	//! Destructor.
	virtual ~KFbxNodeFinderDuplicateName();

	//! Reset the finder object
	virtual void Reset();

	/** GetState.
	*	\param pStateIndex     State index.
	*	\return                State of pStateIndex.
	*/
	bool GetState(int pStateIndex);

	/** SetState.
	*	\param pStateIndex     State index.
	*	\param pValue          
	*/
	void SetState(int pStateIndex, bool pValue);

	/** GetNodeArray.
	*	\return
	*/
	KArrayTemplate<KFbxNode*>& GetNodeArray();

	/** GetNodeArray.
	*	\return
	*/
	KArrayTemplate<KFbxNode*>& GetDuplicateNodeArray();

	/** GetMaterialArray.
	*	\return
	*/
	KArrayTemplate<KFbxSurfaceMaterial*>& GetMaterialArray();

	/** GetMaterialArray.
	*	\return
	*/
	KArrayTemplate<KFbxSurfaceMaterial*>& GetDuplicateMaterialArray();

	/** GetTextureArray.
	*	\return
	*/
	KArrayTemplate<KFbxTexture*>& GetTextureArray();

	/** GetTextureArray.
	*	\return
	*/
	KArrayTemplate<KFbxTexture*>& GetDuplicateTextureArray();

	/** GetVideoArray.
	*	\return
	*/
	KArrayTemplate<KFbxVideo*>& GetVideoArray();

	/** GetVideoArray.
	*	\return
	*/
	KArrayTemplate<KFbxVideo*>& GetDuplicateVideoArray();


	KArrayTemplate<KFbxGenericNode*>& GetGenericNodeArray();

	KArrayTemplate<KFbxGenericNode*>& GetDuplicateGenericNodeArray();


protected:

	/** Find all the node corresponding to the research criterium.
	*	\param iNode
	*/
	virtual void ApplyRecursive(KFbxNode& iNode);
	
	/** Check if a node answers to research criterium.
	*	Criteriums must be defined by child class.
	*	\param iNode
	*	\return
	*/
	virtual bool CheckNode(KFbxNode& iNode);

	/** Check for duplicate node name.
	*	\param pNode
	*	\return True if there is a duplicate node name, false otherwise.
	*/
	bool CheckNodeName(KFbxNode& pNode);

	/** Check for duplicate material name.
	*	\param pGeometry
	*   \param pNode
	*	\return True if there is a duplicate material name, false otherwise.
	*/
	bool CheckMaterialName(KFbxGeometry* pGeometry, KFbxNode* pNode);

	/** Check for duplicate texture name.
	*	\param pGeometry
	*	\return True if there is a duplicate texture name, false otherwise.
	*/
	bool CheckTextureName(KFbxGeometry* pGeometry);
	bool CheckLayerElementTextureName(KFbxLayerElementTexture* pLayerElementTexture);

	/** Check for duplicate video name.
	*	\param pGeometry
	*	\return True if there is a duplicate video name, false otherwise.
	*/
	bool CheckVideoName(KFbxGeometry* pGeometry);
	bool CheckLayerElementVideoName(KFbxLayerElementTexture* pLayerElementTexture);

	bool CheckGenericNodeName(char* pNodeName);

	bool mStates [eSTATE_COUNT];

	KArrayTemplate <KFbxNode*> mENodeArray;
	KArrayTemplate <KFbxNode*> mDuplicateNodeArray;
	KArrayTemplate <KFbxSurfaceMaterial*> mMaterialArray;
	KArrayTemplate <KFbxSurfaceMaterial*> mDuplicateMaterialArray;
    KArrayTemplate <KFbxTexture*> mTextureArray;
	KArrayTemplate <KFbxTexture*> mDuplicateTextureArray;
    KArrayTemplate <KFbxVideo*> mVideoArray;
	KArrayTemplate <KFbxVideo*> mDuplicateVideoArray;
	KArrayTemplate <KFbxGenericNode*> mGenericNodeArray;
	KArrayTemplate <KFbxGenericNode*> mDuplicateGenericNodeArray;

	#ifdef _MULTIENTITY
		KFbxScene* mSourceScene;
	#endif // #ifdef _MULTIENTITY
	KFbxScene* mDestinationScene;
};

#include <fbxfilesdk_nsend.h>

#endif


