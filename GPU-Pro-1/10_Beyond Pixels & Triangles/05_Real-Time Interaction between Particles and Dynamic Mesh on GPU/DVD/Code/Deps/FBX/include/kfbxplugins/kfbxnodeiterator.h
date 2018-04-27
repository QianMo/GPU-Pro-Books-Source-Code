/*!  \file kfbxnodeiterator.h
 */

#ifndef _FBXSDK_NODE_ITERATOR_H_
#define _FBXSDK_NODE_ITERATOR_H_

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

#include <fbxfilesdk_nsbegin.h>

class KFbxNode; 


/**	This class and iterator type accesses the FbxNode hierarchy.
  * \nosubgrouping
  *	The iterator takes a root node that can be any parent node in the KFbxScene. 
  * The iterator will then only travel within the children of a given root.
  *
  * Since the iterator becomes invalid when the scene hierarchy changes, the 
  * iterator should only used in a fixed scene hierarchy. 
  */

class KFBX_DLL KFbxNodeIterator
{	
	
public:

	/** \enum TraversalType  Method by which the node hierarchy is traversed.
	  * - \e eDepthFirst           The leaf of the tree are first traversed
	  * - \e eBreadthFirst         Each child is traversed before going down to the leafs
	  * - \e eDepthFirstParent     Like depth first but the parent of the leafs are returned prior to the leafs themselves
	  */
	enum TraversalType
	{
		eDepthFirst,
		eBreadthFirst,
		eDepthFirstParent
	};

	/** Contructor
	  * \param pRootNode     The root of the iterator hierarchy.
	  * \param pType         The traversal type.
	  */
	KFbxNodeIterator( KFbxNode *pRootNode, TraversalType pType)	;
	/** Copy Constructor
	  * \param pCopy     Iterator to copy
	  */
	KFbxNodeIterator(const KFbxNodeIterator &pCopy);

	/** Destructor
	  */
	virtual ~KFbxNodeIterator();
	
	/** Get a pointer to the current KFbxNode.
	  * \return     The current KFbxNode pointer.
	  */
	virtual KFbxNode* Get();

	/** Get a pointer to the next KFbxNode.
	  * \return     The next KFbxNode pointer, or \c NULL if the end is reached.
	  */
	virtual KFbxNode* Next();

	/** Get a pointer to the previous KFbxNode pointer
	  * \return     The previous KFbxNode pointer, or \c NULL if the root of the iterator is reached.
	  */
	virtual KFbxNode* Prev();

	virtual void Reset() {mCurrentIndex = 0;};


///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

private:

	KFbxNode		**mArray;
	int				mCurrentIndex;
	int				mNodeCount;
	TraversalType	mTraversalType;

	void FillArray(int &lCounter, KFbxNode *pNode);
	
};

#include <fbxfilesdk_nsend.h>

#endif


