/*!  \file kfbxskeleton.h
 */

#ifndef _FBXSDK_SKELETON_H_
#define _FBXSDK_SKELETON_H_

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

#include <klib/kerror.h>

#include <kfbxplugins/kfbxnodeattribute.h>
#include <kfbxplugins/kfbxcolor.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;

/**	This node attribute contains the properties of a skeleton segment.
  * \nosubgrouping
  */
class KFBX_DLL KFbxSkeleton : public KFbxNodeAttribute
{
	KFBXOBJECT_DECLARE(KFbxSkeleton,KFbxNodeAttribute);

public:
 	//! Return the type of node attribute which is EAttributeType::eSKELETON.
	virtual EAttributeType GetAttributeType() const;

    //! Reset the skeleton to default values and type to \c eROOT.
	void Reset();

	/**
	  * \name Skeleton Properties
	  */
	//@{

	/** \enum ESkeletonType Skeleton types.
	  * - \e eROOT
	  * - \e eLIMB
	  * - \e eLIMB_NODE
	  * - \e eEFFECTOR
	  *
	  * \remarks \e eEFFECTOR is synonymous to \e eROOT.
	  * \remarks The \e eLIMB_NODE type is a bone defined uniquely by a transform and a size value while
	  * \remarks the \e eLIMB type is a bone defined by a transform and a length.
	  * 
	  */
    typedef enum   
    {
	    eROOT, 
	    eLIMB, 
	    eLIMB_NODE, 
	    eEFFECTOR
    } ESkeletonType;    

    /** Set the skeleton type.
	  * \param pSkeletonType Skeleton type identifier.
	  */
    void SetSkeletonType(ESkeletonType pSkeletonType);

	/** Get the skeleton type.
	  * \return Skeleton type identifier.
	  */
    ESkeletonType GetSkeletonType() const;

	/** Get a flag to know if the skeleton type was set.
	  * \return \c true if a call to SetSkeletonType() has been made.
	  * \remarks When the attribute is not set, the application can choose to ignore the attribute or use the default value.
	  * \remarks The flag is set back to \c false when Reset() is called.
      */
	bool GetSkeletonTypeIsSet() const;

	/** Get the default value for the skeleton type.
	  * \return \c eROOT
	  */
	ESkeletonType GetSkeletonTypeDefaultValue() const;
		
	/** Set limb length.
	  * \param pLength Length of the limb.
	  * \return \c true if skeleton type is \c eLIMB, \c false otherwise.
	  * \remarks Limb length is only set if skeleton type is \c eLIMB.
	  * \remarks This function is deprecated. Use property LimbLength.Set(pLength) instead.
      */
	K_DEPRECATED bool SetLimbLength(double pLength);
	
	/** Get limb length.
	  * \return limb length.
	  * \remarks Limb length is only valid if skeleton type is \c eLIMB.
	  * \remarks This function is deprecated. Use property LimbLength.Get() instead.
      */
	K_DEPRECATED double GetLimbLength() const;

	/** Get a flag to know if the limb length was set.
	  * \return \c true if a call to SetLimbLength() has been made.
	  * \remarks When the attribute is not set, the application can choose to ignore the attribute or use the default value.
	  * \remarks The flag is set back to \c false when Reset() is called.
      */
	K_DEPRECATED bool GetLimbLengthIsSet() const;

	/** Get the default value for the limb length.
	  * \return 1.0
	  */
	double GetLimbLengthDefaultValue() const;
	
	/** Set skeleton limb node size.
	  * \param pSize Size of the limb node.
	  * \remarks This function is deprecated. Use property Size.Set(pSize) instead.
	  */
	K_DEPRECATED bool SetLimbNodeSize(double pSize);
	
	/** Get skeleton limb node size.
	  * \return Limb node size value.
      * \remarks This function is deprecated. Use property Size.Get() instead.
	  */
	K_DEPRECATED double GetLimbNodeSize() const;

	/** Get a flag to know if the limb node size was set.
	  * \return \c true if a call to SetLimbNodeSize() has been made.
	  * \remarks When the attribute is not set, the application can choose to ignore the attribute or use the default value.
	  * \remarks The flag is set back to \c false when Reset() is called.
      * \remarks     This function is OBSOLETE, DO NOT USE.  It will always return false.  It will be removed on in the next release.
      * \remarks     This function is deprecated. Use property Size instead.
      */
	K_DEPRECATED bool GetLimbNodeSizeIsSet() const;

	/** Get the default value for the limb node size.
	  * \return 100.0
	  */
	double GetLimbNodeSizeDefaultValue() const;

	/** Set limb or limb node color.
	  * \param pColor RGB values for the limb color.
	  * \return \c true if skeleton type is \c eLIMB or \c eLIMB_NODE, \c false otherwise.
	  * \remarks Limb or limb node color is only set if skeleton type is \c eLIMB or \c eLIMB_NODE.
      */
	bool SetLimbNodeColor(const KFbxColor& pColor);
	
	/** Get limb or limb node color.
	  * \return Currently set limb color.
	  * \remarks Limb or limb node color is only valid if skeleton type is \c eLIMB or \c eLIMB_NODE.
      */
	KFbxColor GetLimbNodeColor() const;

	/** Get a flag to know if the limb node color was set.
	  * \return \c true if a call to SetLimbNodeColor() has been made.
	  * \remarks When the attribute is not set, the application can choose to ignore the attribute or use the default value.
	  * \remarks The flag is set back to \c false when Reset() is called.
      */
	bool GetLimbNodeColorIsSet() const;

	/** Get the default value for the limb node color.
	  * \return R=0.8, G=0.8, B=0.8
	  */
	KFbxColor GetLimbNodeColorDefaultValue() const;

	//@}


	/**
	  * \name Property Names
	  */
	static const char*			sSize;
	static const char*			sLimbLength;

	/**
	  * \name Property Default Values
	  */
	//@{	
	static const fbxDouble1		sDefaultSize;
	static const fbxDouble1		sDefaultLimbLength;


	//////////////////////////////////////////////////////////////////////////
	//
	// Properties
	//
	//////////////////////////////////////////////////////////////////////////
	
	/** This property handles the limb node size.
	  *
      * To access this property do: Size.Get().
      * To set this property do: Size.Set(fbxDouble1).
      *
	  * Default value is 100.0
	  */
	KFbxTypedProperty<fbxDouble1>		Size;

	/** This property handles the skeleton limb length.
	  *
      * To access this property do: LimbLength.Get().
      * To set this property do: LimbLength.Set(fbxDouble1).
      *
	  * Default value is 1.0
	  */
	KFbxTypedProperty<fbxDouble1>			LimbLength;

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
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

    KFbxSkeleton(KFbxSdkManager& pManager, char const* pName);
	~KFbxSkeleton();

	virtual void Construct(const KFbxSkeleton* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);

	void Reset( bool pResetProperties );

	//! Assignment operator.
    KFbxSkeleton& operator=(KFbxSkeleton const& pSkeleton);

	virtual KString		GetTypeName() const;
	virtual KStringList	GetTypeFlags() const;

    ESkeletonType mSkeletonType;

	bool mLimbLengthIsSet;
	bool mLimbNodeSizeIsSet;
	bool mLimbNodeColorIsSet;
	bool mSkeletonTypeIsSet;

	friend class KFbxReaderFbx;
	friend class KFbxWriterFbx;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxSkeleton* HKFbxSkeleton;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_SKELETON_H_


