/*!  \file kfbxconstraintaim.h
 */

#ifndef _FBXSDK_CONSTRAINT_AIM_H_
#define _FBXSDK_CONSTRAINT_AIM_H_

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

#include <kfbxplugins/kfbxconstraint.h>
#include <kfbxplugins/kfbxgroupname.h>
#include <kfbxmath/kfbxvector4.h>

#include <klib/kerror.h>

#include <fbxfilesdk_nsbegin.h>

	class KFbxSdkManager;

    /** \brief This constraint class contains methods for accessing the properties of an aim constraint.
      * A aim constraint lets you constrain an object based on the properties of one or more aim objects.
	  * \nosubgrouping
	  */
	class KFBX_DLL KFbxConstraintAim : public KFbxConstraint
	{
		KFBXOBJECT_DECLARE(KFbxConstraintAim,KFbxConstraint);

		/**
		  * \name Properties
		  */
		//@{
			KFbxTypedProperty<fbxBool1>		Lock;
			KFbxTypedProperty<fbxBool1>		Active;

			KFbxTypedProperty<fbxDouble1>	Weight;
			KFbxTypedProperty<fbxDouble3>	RotationOffset;

			KFbxTypedProperty<fbxReference> AimAtObjects;
			KFbxTypedProperty<fbxReference> ConstrainedObject;

			KFbxTypedProperty<fbxEnum>		WorldUpType;
			KFbxTypedProperty<fbxReference> WorldUpObject;
			KFbxTypedProperty<fbxDouble3>	WorldUpVector;
			KFbxTypedProperty<fbxDouble3>	UpVector;
			KFbxTypedProperty<fbxDouble3>	AimVector;
			
			KFbxTypedProperty<fbxBool1>		AffectX;
			KFbxTypedProperty<fbxBool1>		AffectY;
			KFbxTypedProperty<fbxBool1>		AffectZ;
		//@}
		

	public:
		typedef enum 
		{
			eAimAtSceneUp,
			eAimAtObjectUp,
			eAimAtObjectRotationUp,
			eAimAtVector,
			eAimAtNone,
			eAimAtCount
		} EAimConstraintWoldUpType;

		/** Set the constraint lock.
		  * \param pLock State of the lock flag.
		  */
		inline void SetLock(bool pLock) { Lock.Set(pLock); }

		/** Retrieve the constraint lock state.
		  * \return Current lock flag.
		  */
		inline bool GetLock()			{ return Lock.Get(); }

		/** Set the constraint active.
		  * \param pActive State of the active flag.
		  */
		inline void SetActive(bool pActive) { Active.Set(pActive); }

		/** Retrieve the constraint active state.
		  * \return Current active flag.
		  */
		inline bool GetActive() { return Active.Get(); }

		/** Set the weight of the constraint.
		  * \param pWeight New weight value.
		  */
		void SetWeight(double pWeight);

		/** Set the rotation offset.
		  * \param pRotation New offset vector.
		  */
		virtual void SetOffset(KFbxVector4 pRotation);

		/** Retrieve the constraint rotation offset.
		  * \return Current rotation offset.
		  */
		KFbxVector4 GetOffset();

		/** Get the weight of a source.
		  * \param pObject Source object.
		  */
		double GetSourceWeight(KFbxObject* pObject);

		/** Add a source to the constraint.
		  * \param pObject New source object.
		  * \param pWeight Weight of the source object.
		  */
		void AddConstraintSource(KFbxObject* pObject, double pWeight = 100);

		/** Retrieve the constraint source count.
		  * \return Current constraint source count.
		  */
		int GetConstraintSourceCount();

		/** Retrieve a constraint source object.
		  * \return Current source at the specified index.
		  */
		KFbxObject* GetConstraintSource(int pIndex);

		/** Set the constrainted object.
		  * \param pObject The constrained object.
		  */
		void SetConstrainedObject(KFbxObject* pObject);

		/** Retrieve the constrainted object.
		  * \return Current constrained object.
		  */
		KFbxObject* GetConstrainedObject();

		/** Set the world up type.
		  * \param pType The world up type.
		  */
		void SetWorldUpType(EAimConstraintWoldUpType pType);

		/** Retrieve the world up type.
		  * \return Current world up type.
		  */
		EAimConstraintWoldUpType GetWorldUpType();

		/** Set the world up object.
		  * \param pObject The world up object.
		  */
		void SetWorldUpObject(KFbxObject* pObject);

		/** Retrieve the world up object.
		  * \return Current world up object.
		  */
		KFbxObject* GetWorldUpObject();

		/** Set the world up vector.
		  * \param pVector The world up vector.
		  */
		void SetWorldUpVector(KFbxVector4 pVector);

		/** Retrieve the world up vector.
		  * \return Current world up vector.
		  */
		KFbxVector4 GetWorldUpVector();

		/** Set the up vector.
		  * \param pVector The up vector.
		  */
		void SetUpVector(KFbxVector4 pVector);

		/** Retrieve the up vector.
		  * \return Current up vector.
		  */
		KFbxVector4 GetUpVector();

		/** Set the aim vector.
		  * \param pVector The up vector.
		  */
		void SetAimVector(KFbxVector4 pVector);

		/** Retrieve the aim vector.
		  * \return Current up vector.
		  */
		KFbxVector4 GetAimVector();

		/** Set the constraint X-axe effectiveness.
		  * \param pAffect State of the effectivness on the X axe.
		  */
		inline void SetAffectX(bool pAffect) { AffectX.Set(pAffect); }

		/** Retrieve the constraint X-axe effectiveness.
		  * \return Current state flag.
		  */
		inline bool GetAffectX() { return AffectX.Get(); }

		/** Set the constraint Y-axe effectiveness.
		  * \param pAffect State of the effectivness on the X axe.
		  */
		inline void SetAffectY(bool pAffect){ AffectY.Set(pAffect); }

		/** Retrieve the constraint Y-axe effectiveness.
		  * \return Current state flag.
		  */
		inline bool GetAffectY() { return AffectY.Get(); }

		/** Set the constraint Z-axe effectiveness.
		  * \param pAffect State of the effectivness on the X axe.
		  */
		inline void SetAffectZ(bool pAffect) { AffectZ.Set(pAffect); }

		/** Retrieve the constraint Z-axe effectiveness.
		  * \return Current state flag.
		  */
		inline bool GetAffectZ() { return AffectZ.Get(); }

	#ifndef DOXYGEN_SHOULD_SKIP_THIS

		// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

	protected:

		KFbxConstraintAim(KFbxSdkManager& pManager, char const* pName);
		~KFbxConstraintAim();

		virtual bool ConstructProperties(bool pForceSet);
		virtual void Destruct(bool pRecursive, bool pDependents);

		virtual EConstraintType GetConstraintType();
		virtual	KString	GetTypeName() const;

		friend class KFbxWriterFbx6;
		friend class KFbxReaderFbx;

	#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
	};

inline EFbxType FbxTypeOf( KFbxConstraintAim::EAimConstraintWoldUpType const &pItem )				{ return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif // _FBXSDK_CONSTRAINT_AIM_H_


