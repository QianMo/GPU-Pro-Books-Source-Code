/*!  \file kfbxshape.h
 */

#ifndef _FBXSDK_SHAPE_H_
#define _FBXSDK_SHAPE_H_

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

#include <kfbxplugins/kfbxgeometrybase.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;


/** A shape describes the deformation on a set of control points.
  * \nosubgrouping
  * Shapes are associated with instances of class KFbxGeometry.
  */
class KFBX_DLL KFbxShape : public KFbxGeometryBase
{
	KFBXOBJECT_DECLARE(KFbxShape,KFbxGeometryBase);

public:

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

public:
	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType = eDEEP_CLONE) const;

protected:

	/** Return the type of node attribute which is EAttributeType::eUNIDENTIFIED.
	  * EAttributeType::eUNIDENTIFIED is returned because this class derives
	  * accidentally from KFbxNodeAttribute but its instances are not meant to 
	  * be assigned to nodes.
	  */
	virtual EAttributeType GetAttributeType() const;

	
	KFbxShape(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxShape();

	virtual void Destruct(bool pRecursive, bool pDependents);

	//! Assignment operator.
	KFbxShape& operator=(KFbxShape const& pShape);

	friend class KFbxGeometry;
	friend class KFbxWriterFbx;
	friend class KFbxReaderFbx;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxShape* HKFbxShape;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_SHAPE_H_


