/*!  \file kfbxpath.h
 */

#ifndef _FBXSDK_PATH_H_
#define _FBXSDK_PATH_H_

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


#include <klib/karrayul.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

// Not part of the public FBX SDK
#ifndef K_FBXSDK

#include <fbxfilesdk_nsbegin.h>

class KFbxNode;

/** Class KFbxPath.
  *	Object describing the single path from the first parent, the root 
  *	node, to a node.  Methods to access parent nodes, compare paths
  *	together are provided.  
  *
  *	With some 3D system (maya), node can be named with the same name,
  *	as long as they do not share the same path.  In such a system, a 
  *	path is a unique identifier for a node.  Paths are useful to match
  *	nodes during a merge back process.
  */
class KFBX_DLL KFbxPath
{

public:
	
	//! Base constructor.
	KFbxPath();
	
	/** Copy constructor.
	  *	\param pPath Path.
	  */
	KFbxPath(KFbxPath& pPath);

	/** Constructor.
	  *	\param pNode Node.
	  */
	KFbxPath(KFbxNode* pNode);

	//! Destructor.
	~KFbxPath();

	/** Get the node on which the path leads.
	  *	\return the node on which the path leads
	  */
	KFbxNode* GetNode();
	
	/** Get the path's root node.
	  *	\return KFbxNode object.
	  */
	KFbxNode* GetRootNode();
	
	/** GetParentCount.
	  *	\return count of parent nodes.
	  */
	int GetParentCount() const;
	
	/** Retrieve parent node name.
	  *	\param pIndex    Parent index. 0 is the root node, the last parent is the immediate node parent.
	  *	\return name string.
	  */
	char const* GetParentNodeName(int pIndex) const;

	/** Get a parent node.
	  *	\param pIndex		Parent index. 0 is the root node, the last parent is the immediate node parent.
	  *	\return node found.
	  */
	KFbxNode* GetParentNode(int pIndex) const;

	/** Equivalence operator.
	  *	\param pPath Path.
	  *	\return \c true if pPath is equivalent to this path.
	  */
	bool operator== (KFbxPath& pPath);

	/** Assignment operator.
	  *	\param pPath Path
	  *	\return \c this path (which will now be equivalent to pPath).
	  */
	KFbxPath& operator= (KFbxPath& pPath);

	/** Array access operator.
	  *	\param pIndex Index.
	  *	\return node found.
	  */
	KFbxNode* operator[] (int pIndex);

	/** Construct path of a KFbxNode object.
	  *	\param pNode	KFbxNodeObject used to build the path.
	  */
	void Set(KFbxNode* pNode);

	/** Copy a path.
	  *	\param pPath	Path to be copied
	  */
	void Set(KFbxPath& pPath);

private:

	// The root is without parent
	KFbxNode* mNode;
	KFbxNode* mRoot;
	KArrayTemplate<kReference> mPath;

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef K_FBXSDK
#endif // #ifndef _FBXSDK_PATH_H_


