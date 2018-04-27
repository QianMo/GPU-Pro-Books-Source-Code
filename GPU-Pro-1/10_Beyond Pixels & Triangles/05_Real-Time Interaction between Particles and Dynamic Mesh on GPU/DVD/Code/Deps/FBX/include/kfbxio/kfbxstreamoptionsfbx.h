/*!  \file kfbxstreamoptionsfbx.h
 */

#ifndef _FBXSDK_KFbxStreamOptionsFbxWriter_H_
#define _FBXSDK_KFbxStreamOptionsFbxWriter_H_
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


#define KFBXSTREAMOPT_FBX_CURRENT_TAKE_NAME "CURRENT TAKE NAME"
#define KFBXSTREAMOPT_FBX_PASSWORD "PASSWORD"
#define KFBXSTREAMOPT_FBX_PASSWORD_ENABLE "PASSWORD ENABLE"
#define KFBXSTREAMOPT_FBX_MODEL "MODEL"
#define KFBXSTREAMOPT_FBX_TEXTURE "TEXTURE"
#define KFBXSTREAMOPT_FBX_MATERIAL "MATERIAL"
#define KFBXSTREAMOPT_FBX_MEDIA "MEDIA"
#define KFBXSTREAMOPT_FBX_LINK "LINK"
#define KFBXSTREAMOPT_FBX_SHAPE "SHAPE"
#define KFBXSTREAMOPT_FBX_GOBO "GOBO"
#define KFBXSTREAMOPT_FBX_ANIMATION "ANIMATION"
#define KFBXSTREAMOPT_FBX_CHARACTER "CHARACTER"
#define KFBXSTREAMOPT_FBX_GLOBAL_SETTINGS "GLOBAL SETTINGS"
#define KFBXSTREAMOPT_FBX_PIVOT "PIVOT"
#define KFBXSTREAMOPT_FBX_MERGE_LAYER_AND_TIMEWARP "MERGE LAYER AND TIMEWARP"
#define KFBXSTREAMOPT_FBX_CONSTRAINT "CONSTRAINT"
#define KFBXSTREAMOPT_FBX_EMBEDDED "EMBEDDED"
#define KFBXSTREAMOPT_FBX_MODEL_COUNT "MODEL COUNT"
#define KFBXSTREAMOPT_FBX_DEVICE_COUNT "DEVICE COUNT"
#define KFBXSTREAMOPT_FBX_CHARACTER_COUNT "CHARACTER COUNT"
#define KFBXSTREAMOPT_FBX_ACTOR_COUNT "ACTOR COUNT"
#define KFBXSTREAMOPT_FBX_CONSTRAINT_COUNT "CONSTRAINT_COUNT"
#define KFBXSTREAMOPT_FBX_MEDIA_COUNT "MEDIA COUNT"
#define KFBXSTREAMOPT_FBX_TEMPLATE "TEMPLATE"
// Clone every external objects into the document when exporting?    (default: ON)
#define KFBXSTREAMOPT_FBX_COLLAPSE_EXTERNALS    "COLLAPSE EXTERNALS"
// Can we compress arrays of sufficient size in files?               (default: ON)
#define KFBXSTREAMOPT_FBX_COMPRESS_ARRAYS       "COMPRESS ARRAYS"

// ADVANCED OPTIONS -- SHOULD PROBABLY NOT BE IN ANY UI

// Property to skip when looking for things to embed.
// If you have more than one property to ignore (as is often the case) then you must
// create sub-properties.
// Property names must be the full hiearchical property name (ie: parent|child|child)
#define KFBXSTREAMOPT_FBX_EMBEDDED_PROPERTIES_SKIP  "EMBEDDED SKIP"

// Compression level, from 0 (no compression) to 9 (eat your CPU)    (default: speed)
#define KFBXSTREAMOPT_FBX_COMPRESS_LEVEL        "COMPRESS LEVEL"    

// Minimum size before compression is even attempted, in bytes.     
#define KFBXSTREAMOPT_FBX_COMPRESS_MINSIZE      "COMPRESS MINSIZE"

#include <kaydaradef.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <kfbxplugins/kfbxsdkmanager.h>
#include <kfbxio/kfbxstreamoptions.h>
#include <kfbxplugins/kfbxobject.h>
#include <klib/karrayul.h>
#include <kfbxplugins/kfbxtakeinfo.h>
#include <fbxcore/fbxcollection/kfbxdocumentinfo.h>



#include <fbxfilesdk_nsbegin.h>

/**	\brief This class is used for accessing the Import options of Fbx files.
  * The content of KfbxStreamOptionsFbx is stored in the inherited Property of its parent (KFbxStreamOptions).
  * The import options include that:
  * KFBXSTREAMOPT_FBX_CURRENT_TAKE_NAME    :Current take name 
  * KFBXSTREAMOPT_FBX_PASSWORD             :The password
  * KFBXSTREAMOPT_FBX_PASSWORD_ENABLE      :If password enable
  * KFBXSTREAMOPT_FBX_MODEL                :If model import
  * KFBXSTREAMOPT_FBX_TEXTURE              :If texture import
  * KFBXSTREAMOPT_FBX_MATERIAL             :If material import
  * KFBXSTREAMOPT_FBX_MEDIA                :If media import
  * KFBXSTREAMOPT_FBX_LINK                 :If link import
  * KFBXSTREAMOPT_FBX_SHAPE                :If shape import
  * KFBXSTREAMOPT_FBX_GOBO                 :If gobo import
  * KFBXSTREAMOPT_FBX_ANIMATION            :If animation import
  * KFBXSTREAMOPT_FBX_CHARACTER            :If character import
  * KFBXSTREAMOPT_FBX_GLOBAL_SETTINGS      :If global settings import
  * KFBXSTREAMOPT_FBX_PIVOT                :If pivot import
  * KFBXSTREAMOPT_FBX_MERGE_LAYER_AND_TIMEWARP  :If merge layer and timewarp
  * KFBXSTREAMOPT_FBX_CONSTRAINT           :If constrain import
  * KFBXSTREAMOPT_FBX_MODEL_COUNT          :The count of model
  * KFBXSTREAMOPT_FBX_DEVICE_COUNT         :The count of device
  * KFBXSTREAMOPT_FBX_CHARACTER_COUNT      :The count of character
  * KFBXSTREAMOPT_FBX_ACTOR_COUNT          :The count of actor
  * KFBXSTREAMOPT_FBX_CONSTRAINT_COUNT     :The count of constrain
  * KFBXSTREAMOPT_FBX_MEDIA_COUNT          :The count of media
  * KFBXSTREAMOPT_FBX_TEMPLATE             :If template import
  * 
  */
class KFBX_DLL KFbxStreamOptionsFbxReader : public KFbxStreamOptions
{

	KFBXOBJECT_DECLARE(KFbxStreamOptionsFbxReader,KFbxStreamOptions);
public:

/** Reset all options to default values
  *The default values is :
  * KFBXSTREAMOPT_FBX_CURRENT_TAKE_NAME    :Null
  * KFBXSTREAMOPT_FBX_PASSWORD             :Null
  * KFBXSTREAMOPT_FBX_PASSWORD_ENABLE      :false
  * KFBXSTREAMOPT_FBX_MODEL                :true
  * KFBXSTREAMOPT_FBX_TEXTURE              :true
  * KFBXSTREAMOPT_FBX_MATERIAL             :true
  * KFBXSTREAMOPT_FBX_MEDIA                :true
  * KFBXSTREAMOPT_FBX_LINK                 :true
  * KFBXSTREAMOPT_FBX_SHAPE                :true
  * KFBXSTREAMOPT_FBX_GOBO                 :true
  * KFBXSTREAMOPT_FBX_ANIMATION            :true
  * KFBXSTREAMOPT_FBX_CHARACTER            :true
  * KFBXSTREAMOPT_FBX_GLOBAL_SETTINGS      :true
  * KFBXSTREAMOPT_FBX_PIVOT                :true
  * KFBXSTREAMOPT_FBX_MERGE_LAYER_AND_TIMEWARP  :false
  * KFBXSTREAMOPT_FBX_CONSTRAINT           :true
  * KFBXSTREAMOPT_FBX_MODEL_COUNT          :0
  * KFBXSTREAMOPT_FBX_DEVICE_COUNT         :0
  * KFBXSTREAMOPT_FBX_CHARACTER_COUNT      :0
  * KFBXSTREAMOPT_FBX_ACTOR_COUNT          :0
  * KFBXSTREAMOPT_FBX_CONSTRAINT_COUNT     :0
  * KFBXSTREAMOPT_FBX_MEDIA_COUNT          :0
  * KFBXSTREAMOPT_FBX_TEMPLATE             :false
  */
	
	void Reset();

#ifndef DOXYGEN_SHOULD_SKIP_THIS
public:
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
	
	KArrayTemplate<HKFbxTakeInfo> mTakeInfo;
	HKFbxDocumentInfo mDocumentInfo;
	
protected:
	KFbxStreamOptionsFbxReader(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxStreamOptionsFbxReader();
	virtual void Construct(const KFbxStreamOptionsFbxReader* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};




/**	\brief This class is used for accessing the Export options of Fbx files.
  * The content of KfbxStreamOptionsFbx is stored in the inherited Property of its parent (KFbxStreamOptions).
  * The export options include that:
  * KFBXSTREAMOPT_FBX_CURRENT_TAKE_NAME    :Current take name 
  * KFBXSTREAMOPT_FBX_PASSWORD             :The password
  * KFBXSTREAMOPT_FBX_PASSWORD_ENABLE      :If password enable
  * KFBXSTREAMOPT_FBX_MODEL                :If model export
  * KFBXSTREAMOPT_FBX_TEXTURE              :If texture export
  * KFBXSTREAMOPT_FBX_MATERIAL             :If material export
  * KFBXSTREAMOPT_FBX_MEDIA                :If media export
  * KFBXSTREAMOPT_FBX_LINK                 :If link export
  * KFBXSTREAMOPT_FBX_SHAPE                :If shape export
  * KFBXSTREAMOPT_FBX_GOBO                 :If gobo export
  * KFBXSTREAMOPT_FBX_ANIMATION            :If animation export
  * KFBXSTREAMOPT_FBX_CHARACTER            :If character export
  * KFBXSTREAMOPT_FBX_GLOBAL_SETTINGS      :If global settings export
  * KFBXSTREAMOPT_FBX_PIVOT                :If pivot export
  * KFBXSTREAMOPT_FBX_EMBEDDED             :If embedded
  * KFBXSTREAMOPT_FBX_CONSTRAINT           :If constrain export
  * KFBXSTREAMOPT_FBX_TEMPLATE             :If template export
  * KFBXSTREAMOPT_FBX_MODEL_COUNT          :The count of model
  * KFBXSTREAMOPT_FBX_DEVICE_COUNT         :The count of device
  * KFBXSTREAMOPT_FBX_CHARACTER_COUNT      :The count of character
  * KFBXSTREAMOPT_FBX_ACTOR_COUNT          :The count of actor
  * KFBXSTREAMOPT_FBX_CONSTRAINT_COUNT     :The count of constrain
  * KFBXSTREAMOPT_FBX_MEDIA_COUNT          :The count of media
  * KFBXSTREAMOPT_FBX_COLLAPSE_EXTERNALS   :Clone every external objects into the document when exporting
  * KFBXSTREAMOPT_FBX_COMPRESS_ARRAYS      :If compress arrays of sufficient size in files
  * KFBXSTREAMOPT_FBX_EMBEDDED_PROPERTIES_SKIP   :Property to skip when looking for things to embed.
  * KFBXSTREAMOPT_FBX_COMPRESS_LEVEL       :Compression level, from 0 (no compression) to 9
  * KFBXSTREAMOPT_FBX_COMPRESS_MINSIZE     :Minimum size before compression
  * 
  */
class KFBX_DLL KFbxStreamOptionsFbxWriter : public KFbxStreamOptions
{
	KFBXOBJECT_DECLARE(KFbxStreamOptionsFbxWriter,KFbxStreamOptions);
public:
	
/** Reset all options to default values
  *The default values is :
  * KFBXSTREAMOPT_FBX_CURRENT_TAKE_NAME    :Null
  * KFBXSTREAMOPT_FBX_PASSWORD             :Null
  * KFBXSTREAMOPT_FBX_PASSWORD_ENABLE      :false
  * KFBXSTREAMOPT_FBX_MODEL                :true
  * KFBXSTREAMOPT_FBX_TEXTURE              :true
  * KFBXSTREAMOPT_FBX_MATERIAL             :true
  * KFBXSTREAMOPT_FBX_MEDIA                :true
  * KFBXSTREAMOPT_FBX_LINK                 :true
  * KFBXSTREAMOPT_FBX_SHAPE                :true
  * KFBXSTREAMOPT_FBX_GOBO                 :true
  * KFBXSTREAMOPT_FBX_ANIMATION            :true
  * KFBXSTREAMOPT_FBX_CHARACTER            :true
  * KFBXSTREAMOPT_FBX_GLOBAL_SETTINGS      :true
  * KFBXSTREAMOPT_FBX_PIVOT                :true
  * KFBXSTREAMOPT_FBX_EMBEDDED             :false
  * KFBXSTREAMOPT_FBX_CONSTRAINT           :true
  * KFBXSTREAMOPT_FBX_MODEL_COUNT          :0
  * KFBXSTREAMOPT_FBX_DEVICE_COUNT         :0
  * KFBXSTREAMOPT_FBX_CHARACTER_COUNT      :0
  * KFBXSTREAMOPT_FBX_ACTOR_COUNT          :0
  * KFBXSTREAMOPT_FBX_CONSTRAINT_COUNT     :0
  * KFBXSTREAMOPT_FBX_MEDIA_COUNT          :0
  * KFBXSTREAMOPT_FBX_TEMPLATE             :false
  * KFBXSTREAMOPT_FBX_COLLAPSE_EXTERNALS   :true
  * KFBXSTREAMOPT_FBX_COMPRESS_ARRAYS      :true
  * KFBXSTREAMOPT_FBX_EMBEDDED_PROPERTIES_SKIP  :Null
  * KFBXSTREAMOPT_FBX_COMPRESS_LEVEL       :1
  * KFBXSTREAMOPT_FBX_COMPRESS_MINSIZE     :1024
  */
	virtual void Reset();
#ifndef DOXYGEN_SHOULD_SKIP_THIS
public:
	
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
	KArrayTemplate<HKFbxTakeInfo> mTakeInfo;
	HKFbxDocumentInfo mDocumentInfo;
protected:
	KFbxStreamOptionsFbxWriter(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxStreamOptionsFbxWriter();
	virtual void Construct(const KFbxStreamOptionsFbxWriter* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};




#include <fbxfilesdk_nsend.h>
#endif
