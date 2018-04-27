/*!  \file kfbxmarker.h
 */

#ifndef _FBXSDK_MARKER_H_
#define _FBXSDK_MARKER_H_

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

#include <kfbxplugins/kfbxnodeattribute.h>
#include <kfbxplugins/kfbxcolor.h>
#include <kfbxmath/kfbxvector4.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxTakeNode;
class KFbxSdkManager;

/**	This node attribute contains the properties of a marker.
  * \nosubgrouping
  */
class KFBX_DLL KFbxMarker : public KFbxNodeAttribute
{
	KFBXOBJECT_DECLARE(KFbxMarker,KFbxNodeAttribute);

public:
	//! Return the type of node attribute which is EAttributeType::eMARKER.
	virtual EAttributeType GetAttributeType() const;

	//! Reset the marker to default values.
	void Reset();

	/** \enum EType Marker types.
	  * - \e eSTANDARD
	  * - \e eOPTICAL
	  * - \e eFK_EFFECTOR
	  * - \e eIK_EFFECTOR
	  */
	typedef enum { 
		eSTANDARD, 
		eOPTICAL, 
		eFK_EFFECTOR,
		eIK_EFFECTOR
	} EType;

	/** Set marker type.
	  * \param pType The type of marker.
	  */
	void SetType(EType pType);

	/** Get marker type.
	  * \return The type of the marker.
	  */
	EType GetType() const;

	/** \enum ELook Marker look.
	  * - \e eCUBE
	  * - \e eHARD_CROSS
	  * - \e eLIGHT_CROSS
	  * - \e eSPHERE
	  */
	typedef enum { 
		eCUBE, 
		eHARD_CROSS, 
		eLIGHT_CROSS, 
		eSPHERE
	} ELook;
	
	/** Set marker look.
	  * \param pLook The look of the marker.
	  * \remarks This function is deprecated. Use property SetLook.Set(pLook) instead.
	  */
	K_DEPRECATED void SetLook(ELook pLook);

	/** Get marker look.
	  * \return The look of the marker.
	  * \remarks This function is deprecated. Use property SetLook.Get() instead.
	  */
	K_DEPRECATED ELook GetLook() const;

	/** Set marker size.
	  * \param pSize The size of the marker.
	  * \remarks This function is deprecated. Use property Size.Set(pSize) instead.
	  */
	K_DEPRECATED void SetSize(double pSize);

	/** Get marker size.
	  * \return The currently set marker size.
	  * \remarks This function is deprecated. Use property Size.Get() instead.
	  */
	K_DEPRECATED double GetSize() const;

	/** Set whether a marker label is shown.
	  * \param pShowLabel If set to \c true the marker label is visible.
	  * \remarks This function is deprecated. Use property ShowLabel.Set(pShowLabel) instead.
	  */
	K_DEPRECATED void SetShowLabel(bool pShowLabel);

	/** Get whether a marker label is shown.
	  * \return \c true if the marker label is visible.
	  * \remarks This function is deprecated. Use property ShowLabel.Get() instead.
	  */
	K_DEPRECATED bool GetShowLabel() const;

	/** Set the IK pivot position.
	  * \param pIKPivot The translation in local coordinates.
	  * \remarks This function is deprecated. Use property IKPivot.Set(pIKPivot) instead.
	  */
	K_DEPRECATED void SetIKPivot(KFbxVector4& pIKPivot);

	/**  Get the IK pivot position.
	  * \return The pivot position vector.
	  * \remarks This function is deprecated. Use property IKPivot.Get() instead.
	  */
	K_DEPRECATED KFbxVector4 GetIKPivot() const;

	/**
	  * \name Default Animation Values
	  * This set of functions provides direct access to default
	  * animation values specific to a marker. The default animation 
	  * values are found in the default take node of the associated node.
	  * Hence, these functions only work if the marker has been associated
	  * with a node.
	  */
	//@{

	/** Get default occlusion.
	  * \return 0.0 if optical marker animation is valid by default, 1.0 if it is occluded by default.
	  * \remarks This function only works if marker type is set to KFbxMarker::eOPTICAL.
	  */
	double GetDefaultOcclusion() const;

	/** Set default occlusion.
	  * \param pOcclusion 0.0 if optical marker animation is valid by default, 1.0 if it is occluded by default.
	  * \remarks This function only works if marker type is set to KFbxMarker::eOPTICAL.
	  */
	void SetDefaultOcclusion(double pOcclusion);

	/** Get default IK reach translation.
	  * \return A value between 0.0 and 100.0, 100.0 means complete IK reach.
	  * \remarks This function only works if marker type is set to KFbxMarker::eIK_EFFECTOR.
	  */
	double GetDefaultIKReachTranslation() const;

	/** Set default IK reach translation.
	  * \param pIKReachTranslation A value between 0.0 and 100.0, 100.0 means complete IK reach.
	  * \remarks This function only works if marker type is set to KFbxMarker::eIK_EFFECTOR.
	  */
	void SetDefaultIKReachTranslation(double pIKReachTranslation);

	/** Get default IK reach rotation.
	  * \return A value between 0.0 and 100.0, 100.0 means complete IK reach.
	  * \remarks This function only works if marker type is set to KFbxMarker::eIK_EFFECTOR.
	  */
	double GetDefaultIKReachRotation() const;

	/** Set default IK reach rotation.
	  * \param pIKReachRotation A value between 0.0 and 100.0, 100.0 means complete IK reach.
	  * \remarks This function only works if marker type is set to KFbxMarker::eIK_EFFECTOR.
	  */
	void SetDefaultIKReachRotation(double pIKReachRotation);

	//@}

	/**
	  * \name Obsolete functions
	  */
	//@{

	/** Get default color.
	  * \return Input parameter filled with appropriate data.
	  * \remarks Marker color can not be animated anymore.
	  */
	KFbxColor& GetDefaultColor(KFbxColor& pColor) const;

	/** Set default color.
	  * \remarks Marker color can not be animated anymore.
	  */
	void SetDefaultColor(KFbxColor& pColor);

	//@}

	/**
	  * \name Property Names
	  */
	static const char*			sLook;
	static const char*			sSize;
	static const char*			sShowLabel;
	static const char*			sIKPivot;

	/**
	  * \name Property Default Values
	  */
	static const ELook			sDefaultLook;
	static const fbxDouble1		sDefaultSize;
	static const fbxBool1		sDefaultShowLabel;
	static const fbxDouble3		sDefaultIKPivot;

	//////////////////////////////////////////////////////////////////////////
	//
	// Properties
	//
	//////////////////////////////////////////////////////////////////////////
	
	/** This property handles the marker's look.
	  *
      * To access this property do: Look.Get().
      * To set this property do: Look.Set(ELook).
      *
	  * Default value is eCUBE
	  */
	KFbxTypedProperty<ELook> Look;
	
	/** This property handles the marker's size.
	  *
      * To access this property do: Size.Get().
      * To set this property do: Size.Set(fbxDouble1).
      *
	  * Default value is 100
	  */
	KFbxTypedProperty<fbxDouble1> Size;
	
	/** This property handles the marker's label visibility.
	  *
      * To access this property do: ShowLabel.Get().
      * To set this property do: ShowLabel.Set(fbxBool1).
      *
	  * Default value is false
	  */
	KFbxTypedProperty<fbxBool1> ShowLabel;
	
	/** This property handles the marker's pivot position.
	  *
      * To access this property do: IKPivot.Get().
      * To set this property do: IKPivot.Set(fbxDouble3).
      *
	  * Default value is (0., 0., 0.)
	  */
	KFbxTypedProperty<fbxDouble3> IKPivot;

	// Dynamic properties

	/** This method grants access to the occlusion property.
	  * \remark If the marker is not of type Optical or the property
	  * is invalid, return NULL
	  */
	KFbxProperty* GetOcclusion();

	/** This method grants access to the IKReachTranslation property.
	  * \remark If the marker is not of type IK Effector or the property
	  * is invalid, return NULL
	  */
	KFbxProperty* GetIKReachTranslation();
	/** This method grants access to the IKReachRotation property.
	  * \remark If the marker is not of type IK Effector or the property
	  * is invalid, return NULL
	  */
	KFbxProperty* GetIKReachRotation();

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

	KFbxMarker(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxMarker();

	virtual void Construct(const KFbxMarker* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	virtual void Destruct(bool pRecursive, bool pDependents);
	void Reset( bool pResetProperties );

	//! Assignment operator.
    KFbxMarker& operator=(KFbxMarker const& pMarker);

	/**
	  *	Used to retrieve the KProperty list from an attribute.
	  */

	virtual KString		GetTypeName() const;
	virtual KStringList	GetTypeFlags() const;

	EType mType;

	KFbxProperty dynProp; // temporary placeholder for either
	// the Occlusion, IKReachTranslation or IKReachRotation 
	// properties. Its address is returned in the GetOcclusion(),
	// GetIKReachTranslation() and GetIKReachRotation() if the property
	// is valid

	friend class KFbxReaderFbx;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxMarker* HKFbxMarker;

inline EFbxType FbxTypeOf( KFbxMarker::ELook const &pItem )			{ return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_MARKER_H_


