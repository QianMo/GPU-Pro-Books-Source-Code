/*!  \file kfbxnodelimits.h
 */

#ifndef _FBXSDK_NODE_LIMITS_H_
#define _FBXSDK_NODE_LIMITS_H_

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

#include <kfbxmath/kfbxvector4.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxNode;

 /** \enum ELimitedProperty
  * - \e eTRANSLATION
  * - \e eROTATION
  * - \e eSCALE
  */
typedef enum 
	{
		eTRANSLATION,          
		eROTATION,            
		eSCALE,		
	} ELimitedProperty;

/** \brief KFbxLimits defines a 3 component min, max limit. 
  * KFbxLimits uses KFbxVector4 objects to store the values. Although the members are identified as
  * X, Y and Z (the W component is ignored) at this level, they are unitless values and will only 
  * have meaning within the context they are queried.
  * \nosubgrouping
  */
class KFBX_DLL KFbxLimits
{
	public:
		//! Constructor.
		KFbxLimits(KMBTransform *pMBTransform = NULL);		

		//! Destructor
		virtual ~KFbxLimits();
		

		/** Set the active state of min limit.
		  * \param pXActive     Set to \c true, to activate the X component min limit.
		  * \param pYActive     Set to \c true, to activate the Y component min limit.
		  * \param pZActive     Set to \c true, to activate the Z component min limit.
		  */
		void SetLimitMinActive(const bool pXActive, const bool pYActive, const bool pZActive);

		/** Get the active states of the three components of the min limit.
		  * \param pXActive     \c true if the X component of the min limit is active.
		  * \param pYActive     \c true if the Y component of the min limit is active.
		  * \param pZActive     \c true if the Z component of the min limit is active.
		  */
		void GetLimitMinActive(bool& pXActive, bool& pYActive, bool& pZActive) const;

		/** Set the active state of max limit.
		  * \param pXActive     Set to \c true, to activate the X component max limit.
		  * \param pYActive     Set to \c true, to activate the Y component max limit.
		  * \param pZActive     Set to \c true, to activate the Z component max limit.
		  */
		void SetLimitMaxActive(const bool pXActive, const bool pYActive, const bool pZActive);

		/** Get the active states of the three components of the max limit.
		  * \param pXActive     \c true if the X component of the max limit is active.
		  * \param pYActive     \c true if the Y component of the max limit is active.
		  * \param pZActive     \c true if the Z component of the max limit is active.
		  */
		void GetLimitMaxActive(bool& pXActive, bool& pYActive, bool& pZActive) const;

		/** Check if at least one of the active flags is set.
		  * \return     \c true if one of the six active flags is set.
		  */
		bool GetLimitSomethingActive() const;
		
		/** Set the min limit.
		  * \param pMin     The X, Y and Z values to be set for the min limit.
		  */
		void        SetLimitMin(const KFbxVector4& pMin);

		/** Get the min limit.
		  * \return     The current X, Y and Z values for the min limit.
		  */
		KFbxVector4 GetLimitMin() const;

		/** Set the max limit.
		  * \param pMax    The X, Y and Z values to be set for the max limit.
		  */
		void        SetLimitMax(const KFbxVector4& pMax);

		/** Get the max limit.
		  * \return     The current X, Y and Z values for the max limit.
		  */
		KFbxVector4 GetLimitMax() const;
		
		/** Set the property that is limited
		  * \param pProperty     The limited property
		  */
		inline void SetLimitedProperty(ELimitedProperty pProperty)
		{
			mProperty = pProperty;
		}

		/** Get the property that is limited
		  * \return     The current limited property
		  */
		inline ELimitedProperty  GetLimitedProperty()
		{
			return mProperty;
		}

		KFbxLimits& operator = (const KFbxLimits& pFbxLimit);


	private:	
		bool mNeedsMBTransformDelete;
		KMBTransform*		mMBTransform;
		ELimitedProperty mProperty;
};

/** The KFbxNodeLimits defines limits for transforms.
  * \nosubgrouping
  */
class KFBX_DLL KFbxNodeLimits
{
	
	public:
		/** Constructor.
		  * \param pLimitedNode     Pointer to the node to which these limits apply.
		  * \param pMBTransform
		  */
		KFbxNodeLimits(KFbxNode *pLimitedNode,KMBTransform *pMBTransform);

		/** Get the limited node.
		  * \return     Pointer to the node to which these limits apply. This node is the same pointer as 
		  *             the one passed to the constructor.
		  */
		KFbxNode* GetLimitedNode();

		/**
		  * \name Node Translation Limits
		  */
		//@{

		/** Change the translation limit active flag.
		  * \param pActive     State of the translation limits active flag.
		  * \remarks           If this flag is set to \c false, the values in the mTranslationLimits are ignored.
		  */
		void SetTranslationLimitActive(bool pActive);

		/** Get the translation limit active flag.
		  * \return     Translation limit active flag state.
		  * \remarks    If this flag is \c false, the values in the mTranslationLimits are ignored.
		  */
		bool GetTranslationLimitActive();

		//! The translation limits.
		KFbxLimits mTranslationLimits;

		//@}

		/**
		  * \name Node Rotation Limits
		  */
		//@{

		/** Change the rotation limit active flag.
		  * \param pActive     State of the rotation limits active flag.
		  * \remarks           If this flag is set to \c false, the values in the mRotationLimits are ignored.
		  */
		void SetRotationLimitActive(bool pActive);

		/** Get the rotation limit active flag.
		  * \return     Rotation limit active flag state. 
		  * \remarks    If this flag is \c false, the values in the mRotationLimits are ignored.
		  */
		bool GetRotationLimitActive();

		//! The rotation limits.
		KFbxLimits mRotationLimits;

		//@}

		/**
		  * \name Node Scale Limits
		  */
		//@{

		/** Change the scaling limit active flag.
		  * \param pActive     State of the scaling limits active flag.
		  * \remarks           If this flag is set to \c false, the values in the mScalingLimits are ignored.
		  */
		void SetScalingLimitActive(bool pActive);

		/** Get the scaling limit active flag.
		  * \return      Scaling limit active flag state.
		  * \remarks     If this flag is \c false, the values in the mScalingLimits are ignored.
		  */
		bool GetScalingLimitActive();

		//! The scaling limits.
		KFbxLimits mScalingLimits;

		//@}

#ifndef DOXYGEN_SHOULD_SKIP_THIS	

		KFbxNodeLimits& operator=(KFbxNodeLimits const& pNodeLimits); 

	private:
		//! Node on which the limits are applied
		KFbxNode*			  mLimitedNode;
		KMBTransform *		  mMBTransform;
	friend class KFbxNode;
#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

typedef KFbxLimits* HKFbxLimits;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_NODE_LIMITS_H_


