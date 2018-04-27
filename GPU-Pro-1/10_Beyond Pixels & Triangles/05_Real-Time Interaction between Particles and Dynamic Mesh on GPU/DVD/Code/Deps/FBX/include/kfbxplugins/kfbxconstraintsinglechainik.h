/*!  \file kfbxconstraintsinglechainik.h
 */

#ifndef _FBXSDK_CONSTRAINT_SINGLE_CHAIN_IK_H_
#define _FBXSDK_CONSTRAINT_SINGLE_CHAIN_IK_H_

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

#include <klib/kerror.h>

#include <kfbxmath/kfbxvector4.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;

/** \brief This constraint class contains methods for accessing the properties of a single chain IK constraint.
  * \nosubgrouping
  */
class KFBX_DLL KFbxConstraintSingleChainIK : public KFbxConstraint
{
	KFBXOBJECT_DECLARE(KFbxConstraintSingleChainIK,KFbxConstraint);

		/**
		  * \name Properties
		  */
		//@{
			KFbxTypedProperty<fbxBool1>		Lock;
			KFbxTypedProperty<fbxBool1>		Active;
			
			KFbxTypedProperty<fbxEnum>		PoleVectorType;
			KFbxTypedProperty<fbxEnum>		SolverType;
			KFbxTypedProperty<fbxEnum>		EvaluateTSAnim;

			KFbxTypedProperty<fbxDouble1>	Weight;
			//KFbxTypedProperty<fbxReference> PoleVectorObjectWeights;
			KFbxTypedProperty<fbxReference>	PoleVectorObjects;
			KFbxTypedProperty<fbxDouble3>	PoleVector;
			KFbxTypedProperty<fbxDouble1>	Twist;

			KFbxTypedProperty<fbxReference> FirstJointObject;
			KFbxTypedProperty<fbxReference> EndJointObject;
			KFbxTypedProperty<fbxReference> EffectorObject;
		//@}

public:
	/** \enum ESolverType Pole vector type.
	  * - \e eRP_SOLVER
	  * - \e eSC_SOLVER
	  */
	typedef enum 
    {
		eRP_SOLVER,
		eSC_SOLVER
	} ESolverType;

	/** \enum EPoleVectorType Pole vector type.
	  * - \e ePOLE_VECTOR_TYPE_VECTOR
	  * - \e ePOLE_VECTOR_TYPE_OBJECT
	  */
    typedef enum 
    {
		ePOLE_VECTOR_TYPE_VECTOR,
		ePOLE_VECTOR_TYPE_OBJECT
	} EPoleVectorType;

	/** \enum EEvalTS If the constaints read its animation on Translation and Scale for the nodes it constraints.
	  * - \e eEVALTS_NEVER
	  * - \e eEVALTS_AUTODETECT
	  * = \e eEVALTS_ALWAYS
	  */
	typedef enum
	{
		eEVAL_TS_NEVER,
		eEVAL_TS_AUTO_DETECT,
		eEVAL_TS_ALWAYS
	} EEvalTS;

	/** Set the constraint lock.
	  * \param pLock     State of the lock flag.
	  */
	void SetLock(bool pLock);

	/** Retrieve the constraint lock state.
	  * \return     Current lock flag.
	  */
	bool GetLock();

	/** Set the constraint active.
	  * \param pActive     State of the active flag.
	  */
	void SetActive(bool pActive);

	/** Retrieve the constraint active state.
	  * \return     Current active flag.
	  */
	bool GetActive();

	/** Set the Pole Vector type.
	  * \param pType     New type for the pole vector.
	  */
	void SetPoleVectorType(EPoleVectorType pType);

	/** Retrieve the pole vector type.
	  * \return     Current pole vector type.
	  */
	EPoleVectorType GetPoleVectorType();

	/** Set the Solver type.
	  * \param pType     New type for the solver.
	  */
	void SetSolverType(ESolverType pType);

	/** Retrieve the solver type.
	  * \return     Current solver type.
	  */
	ESolverType GetSolverType();

	/** Sets the EvalTS
	  * \param pEval     New type of EvalTS 
	  */
	void SetEvalTS(EEvalTS pEval);
	
	/** Retrieve the EvalTS
	  * \return     The current EvalTS type
	  */
	EEvalTS GetEvalTS();

	/** Set the weight of the constraint.
	  * \param pWeight     New weight value.
	  */
	void SetWeight(double pWeight);

	/** Get the weight of the constraint.
	  * \return     The current weight value.
	  */
	double GetWeight();

	
//	void AddPoleVectorObjectWeight(double pWeight);

	/** Get the weight of a source.
	  * \param pObject     Source object that we want the weight.
	  */
	double GetPoleVectorObjectWeight(KFbxObject* pObject);

	/** Add a source to the constraint.
	  * \param pObject     New source object.
	  * \param pWeight     Weight value of the source object expressed as a percentage.
      * \remarks           pWeight value is 100 percent by default.
	  */
	void AddPoleVectorObject(KFbxObject* pObject, double pWeight = 100);

	/** Retrieve the constraint source count.
	  * \return     Current constraint source count.
	  */
	int GetConstraintPoleVectorCount();

	/** Retrieve a constraint source object.
	  * \param pIndex     Index of constraint source object.
	  * \return           Current source at the specified index.
	  */
	KFbxObject* GetPoleVectorObject(int pIndex);

	/** Set the pole vector.
	  * \param pVector     New pole vector.
	  */
	void SetPoleVector(KFbxVector4 pVector);

	/** Retrieve the pole vector.
	  * \return     Current pole vector.
	  */
	KFbxVector4 GetPoleVector();

	/** Set the twist value.
	* \param pTwist    New twist value.
	*/
	void SetTwist(double pTwist);

	/** Retrieve the twist value.
	  * \return     Current twist value.
	  */
	double GetTwist();

	/** Set the first joint object.
	  * \param pObject     The first joint object.
	  */
	void SetFirstJointObject(KFbxObject* pObject);

	/** Retrieve the first joint object.
	  * \return Current first joint object.
	  */
	KFbxObject* GetFirstJointObject();

	/** Set the end joint object.
	  * \param pObject     The end joint object.
	  */
	void SetEndJointObject(KFbxObject* pObject);

	/** Retrieve the end joint object.
	  * \return     Current end joint object.
	  */
	KFbxObject* GetEndJointObject();

	/** Set the effector object.
	  * \param pObject     The effector object.
	  */
	void SetEffectorObject(KFbxObject* pObject);

	/** Retrieve the effector object.
	  * \return     Current effector object.
	  */
	KFbxObject* GetEffectorObject();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:
	
	KFbxConstraintSingleChainIK(KFbxSdkManager& pManager, char const* pName);
	~KFbxConstraintSingleChainIK();

	virtual bool ConstructProperties( bool pForceSet );
	virtual void Destruct(bool pRecursive, bool pDependents);

	virtual EConstraintType GetConstraintType();
	virtual	KString	GetTypeName() const;

	friend class KFbxWriterFbx6;
	friend class KFbxReaderFbx;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

inline EFbxType FbxTypeOf( KFbxConstraintSingleChainIK::EPoleVectorType const &pItem )				{ return eENUM; }
inline EFbxType FbxTypeOf( KFbxConstraintSingleChainIK::ESolverType const &pItem )				{ return eENUM; }
inline EFbxType FbxTypeOf( KFbxConstraintSingleChainIK::EEvalTS const &pItem )				{ return eENUM; }


#include <fbxfilesdk_nsend.h>

#endif // _FBXSDK_CONSTRAINT_SINGLE_CHAIN_IK_H_


