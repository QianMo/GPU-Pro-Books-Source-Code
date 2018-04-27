/*!  \file kfbxpluginbase.h
 */

#ifndef _FBXSDK_PLUGIN_BASE_H_
#define _FBXSDK_PLUGIN_BASE_H_

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

#include <klib/kstring.h>
#include <klib/karrayul.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>


// Not part of the public FBX SDK
#ifndef K_FBXSDK

#include <fbxfilesdk_nsbegin.h>

class KFbxNode;


/**FBX SDK plugin base class
  *\nosubgrouping
  */
class KFBX_DLL KFbxPluginBase
{
public:
	/**
	  * \name Constructor and Destructor
	  */
	//@{

	//!Constructor.
	KFbxPluginBase();

	//!Destructor.
	virtual ~KFbxPluginBase();
    //@}


	/** Retrieve the prefix of the FbxRoot node.
	  * \return                  the prefix of the FbxRoot node.
	  */
	inline KString FbxRootPrefix() { return KString(mRootPrefix); }

    /** Mark alternative FbxRoots.
	  * \param pScene            Scene whose nodes are marked
	  */
	void MarkAlternativeFbxRoots( KFbxScene* pScene );

protected:
	/** Allow the concrete class to indicate that the received
	  * node need to be handled as an FbxRoot even thought it 
	  * does not have the FbxRootPrefix name.
	  * \param pNode            
	  * \return                \c false.
	  * \remarks                This method is implicitly called by the IsFbxRootNode.
	  */
	virtual bool IsAlternativeFbxRootNode(KFbxNode* pNode);

private:
	char* mRootPrefix;
};


/**FBX SDK plugin import base class
  *\nosubgrouping
  */
class KFBX_DLL KFbxPluginImportBase : public KFbxPluginBase
{
public:
	/**
	  * \name Constructor and Destructor.
	  */
	//@{

	//!Constructor.
	KFbxPluginImportBase();
    
	//!Destructor.
	virtual ~KFbxPluginImportBase();
    //@}


	/** Returns true if the passed argument is an FbxRoot node that 
	  * is excluded from the creation process.
	  * \param pNode
	  */
	bool IsExcludedFbxRoot(KFbxNode* pNode);

	/** Pre-process the first level of the Fbx scene to transform the FbxRoot nodes.
	  * Modifies the hierarchy received (pRoot).
	  * \param pRoot
	  */
	void Preprocess(KFbxNode* pRoot);

	/** Returns true if all of the scene roots are FBX_ROOTS.
	  * \param pRoot
	  */
	bool NeedPreProcess(KFbxNode* pRoot);

protected:

	/** Returns always true unless pNode was a child of a FbxRoot that has been re-parented to the Scene.
	  * \param pNode
	  */
	virtual bool NeedFbxRoot(KFbxNode* pNode);

	/** Allow the concrete class to compute whatever need to be done
	  * to define the World Transformation. This method is called
	  * in the Preprocess.
	  * \param pResMat                 Affine Matrix of World Transformation.
	  */
	virtual void ComputeWorldTransform(KgeAMatrix& pResMat);

	/** Allow the concrete class to customize the modification of the hierarchy
	  * (pRoot). By default this method set the worldTransform as the default TRS
	  * of pRoot.
	  * \param pRoot
	  * \param pResMat                 Affine Matrix of World Transformation.               
	  * \param isIdentity
	  */
	virtual void CustomApplyWorldTransform(KFbxNode* pRoot, KgeAMatrix& pResMat, bool isIdentity);

private:

	// keep the list of nodes that would be children
	// of the scene in the destination package (this 
	// includes Fbx_Root nodes even if their GM is
	// an identity and therefore are not processed
	// (these nodes will also be found in the mExcludedFbxRoot)
	KArrayTemplate<KFbxNode*> mFbxRootChildren;

	// keep a list of FbxRoot nodes that have an identity GM
	// and thus that should not be created in the destination
	// package. However they need to exist with their Fbx
	// transform to allow the correct processing of the rest
	// of the hierarchy objects
	KArrayTemplate<KFbxNode*> mExcludedFbxRoot;
};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef K_FBXSDK
#endif // _FBXSDK_PLUGIN_BASE_H_

