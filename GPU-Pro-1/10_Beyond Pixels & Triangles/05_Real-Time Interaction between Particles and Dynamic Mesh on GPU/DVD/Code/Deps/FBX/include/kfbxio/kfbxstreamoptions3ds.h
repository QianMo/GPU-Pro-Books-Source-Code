/*!  \file kfbxstreamoptions3ds.h
 */
 
#ifndef _FBXSDK_KFbxStreamOptions3dsWriter_H_
#define _FBXSDK_KFbxStreamOptions3dsWriter_H_

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


#define KFBXSTREAMOPT_3DS_AMBIENT_LIGHT "AMBIENT LIGHT"
#define KFBXSTREAMOPT_3DS_REFERENCENODE "REFERENCENODE"
#define KFBXSTREAMOPT_3DS_TEXTURE "TEXTURE"
#define KFBXSTREAMOPT_3DS_MATERIAL "MATERIAL"
#define KFBXSTREAMOPT_3DS_ANIMATION "ANIMATION"
#define KFBXSTREAMOPT_3DS_MESH "MESH"
#define KFBXSTREAMOPT_3DS_LIGHT "LIGHT"
#define KFBXSTREAMOPT_3DS_CAMERA "CAMERA"
#define KFBXSTREAMOPT_3DS_AMBIENT_LIGHT "AMBIENT LIGHT"
#define KFBXSTREAMOPT_3DS_RESCALING "RESCALING"
#define KFBXSTREAMOPT_3DS_FILTER "FILTER"
#define KFBXSTREAMOPT_3DS_SMOOTHGROUP "SMOOTHGROUP"
#define KFBXSTREAMOPT_3DS_TEXUVBYPOLY "TEXUVBYPOLY"
#define KFBXSTREAMOPT_3DS_TAKE_NAME "TAKE NAME"
#define KFBXSTREAMOPT_3DS_MESH_COUNT "MESH COUNT"
#define KFBXSTREAMOPT_3DS_LIGHT_COUNT "LIGHT COUNT"
#define KFBXSTREAMOPT_3DS_CAMERA_COUNT "CAMERA COUNT"


#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <kfbxplugins/kfbxsdkmanager.h>
#include <kfbxio/kfbxstreamoptions.h>
#include <fbxfilesdk_nsbegin.h>


/** \brief This class is used for accessing the Import options of 3ds files.
  * The content of KfbxStreamOptions3ds is stored in the inherited Property of its parent (KFbxStreamOptions).
  */

class KFBX_DLL KFbxStreamOptions3dsReader : public KFbxStreamOptions
{
	KFBXOBJECT_DECLARE(KFbxStreamOptions3dsReader,KFbxStreamOptions);
public:
	/** Reset all options to default values
	*/
	void Reset();
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
	

protected:

	virtual void Construct( const KFbxStreamOptions3dsReader* pFrom );
	virtual bool ConstructProperties( bool pForceSet );
	KFbxStreamOptions3dsReader(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxStreamOptions3dsReader();
	virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};


/**	\brief This class is used for accessing the Export options of 3ds files.
  * The content of KfbxStreamOptions3ds is stored in the inherited Property of its parent (KFbxStreamOptions).
  */

class KFBX_DLL KFbxStreamOptions3dsWriter : public KFbxStreamOptions
{
	KFBXOBJECT_DECLARE(KFbxStreamOptions3dsWriter,KFbxStreamOptions);

public:
	/** Reset all options to default values
	*/
	void Reset();
#ifndef DOXYGEN_SHOULD_SKIP_THIS
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
	

protected:

	void Construct( const KFbxStreamOptions3dsWriter* pFrom );
	bool ConstructProperties( bool pForceSet );
	KFbxStreamOptions3dsWriter(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxStreamOptions3dsWriter();
	virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};

#include <fbxfilesdk_nsend.h>

#endif

