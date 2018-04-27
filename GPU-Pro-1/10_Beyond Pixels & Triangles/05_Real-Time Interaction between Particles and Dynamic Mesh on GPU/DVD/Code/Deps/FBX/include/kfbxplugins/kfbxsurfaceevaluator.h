/*!  \file kfbxsurfaceevaluator.h
 */

#ifndef _FBXSDK_SURFACE_EVALUATOR_H_
#define _FBXSDK_SURFACE_EVALUATOR_H_

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

#include <fbxfilesdk_nsbegin.h>

class KFbxWeightedMapping;

//#define KSURFACE_EVALUATOR_NORMALIZE
#define KSURFACE_EVALUATOR_4D

#ifdef KSURFACE_EVALUATOR_4D
#define	N_Dimension		4
#else
#define	N_Dimension		3
#endif

enum
 {
	KSURFACE_NORMAL, 
	KSURFACE_CLOSE 
};


/***************************************************************************
  Class KFBXSurfaceEvaluator
 ***************************************************************************/

class KFBXSurfaceEvaluator  
{
public :

	 //! Constructor.
    KFBXSurfaceEvaluator ();

	//! Destructor.
    virtual ~KFBXSurfaceEvaluator ();

	//!
	virtual void EvaluateSurface (KFbxWeightedMapping* lWeightedMapping = NULL);
	
	// Evaluation settings function
	
	//!
	virtual void SetEvaluationModeU (int pModeU);
	
	//!
	virtual void SetEvaluationModeV	(int pModeV);
	
	//!
	virtual void SetEvaluationStepU (kUInt pStepU);
	
	//!
	virtual void SetEvaluationStepV (kUInt pStepV);
	
	// Output Data
	
	//!
	virtual bool GetBottomCapU (void);
	
	//!
	virtual bool GetTopCapU (void);
	
	//!
	virtual bool GetBottomCapV (void);
	
	//!
	virtual bool GetTopCapV (void);
	
	//!
	virtual void SetDestinationArray (double* pArray);
	
	//!
	virtual kUInt GetCurvePointCountX (void);
	
	//!
	virtual kUInt GetCurvePointCountY (void);
	
	//!
	virtual void SetDestinationNormalArray (double* pArray);
  	
	// Input Data

	//!
	virtual void SetSurfaceTensionU (double pTensionU);

	//!
	virtual void SetSurfaceTensionV (double pTensionV);

	//!
	virtual void SetSourceArray (double* pArray, kUInt NPointX, kUInt NPointY);

	//!
	virtual void SetAuxSourceArray (int pIdentification, double* pArray);

	//!
	void Destroy (int IsLocal = false);

	//!
	inline int GetEvaluationModeU () {return mEvaluation_Mode_U;}

	//!
	inline int GetEvaluationModeV () {return mEvaluation_Mode_V;}

	//!
	void Set_U_Blending_Parameters (const double pMatrice4x4[16]);
	
	//!
	void Set_V_Blending_Parameters (const double pMatrice4x4[16]);

	//!
	void SetOrderU (kUInt pOrderU);
	
	//!
	void SetOrderV (kUInt pOrderV);
	
	//!
	void SetAfterStepU (kUInt uf);
	
	//!
	void SetAfterStepV (kUInt vf);


protected :
	
	inline kUInt ClipX (kUInt PosiX) 
	{	
		if (PosiX >= mSource_Point_Count_X)
			return PosiX -= mSource_Point_Count_X;
		return PosiX;
	}

	inline kUInt ClipY (kUInt PosiY) 
	{	
		if (PosiY >= mSource_Point_Count_Y)
			return PosiY -= mSource_Point_Count_Y;
		return PosiY;
	}

	inline double GetData (kUInt PosiX, kUInt PosiY, kUInt Dimension)
	{
		return mSource_Array [(PosiX + PosiY * mSource_Point_Count_X)* N_Dimension	+ Dimension];
	}
	   
	// Evaluation settings

	int mEvaluation_Mode_U;
	int mEvaluation_Mode_V;

	kUInt mOrder_U;
	kUInt mOrder_V;

	kUInt mN_Step_U;
	kUInt mN_Step_V;

	kUInt mAfter_Step_U;
	kUInt mAfter_Step_V;

	// Output Data

	kUInt mDestination_Point_Count_X;
	kUInt mDestination_Point_Count_Y;

	double* mDestination_Array;
	double* mNormal_Array;

	// Input Data

	double mTension_U;
	double mTension_V;

	kUInt mSource_Point_Count_X;
	kUInt mSource_Point_Count_Y;
	double* mSource_Array;

	// Internal Data

	double* mTangent_U;
	double* mTangent_V;

	kUInt mNeed_ReCompute_Table;

	kUInt mNeed_Check_Cap;

	bool mCap_Bottom_U;
	bool mCap_Top_U;
	bool mCap_Bottom_V;
	bool mCap_Top_V;


	double	ABi1, BBi1, CBi1, DBi1,
			ABi2, BBi2, CBi2, DBi2,
			ABi3, BBi3, CBi3, DBi3,
			ABi4, BBi4, CBi4, DBi4;

	double	ABj1, BBj1, CBj1, DBj1,
			ABj2, BBj2, CBj2, DBj2,
			ABj3, BBj3, CBj3, DBj3,
			ABj4, BBj4, CBj4, DBj4;

	double	BdBi1, CdBi1, DdBi1,
			BdBi2, CdBi2, DdBi2,
			BdBi3, CdBi3, DdBi3,
			BdBi4, CdBi4, DdBi4;

	double	BdBj1, CdBj1, DdBj1,
			BdBj2, CdBj2, DdBj2,
			BdBj3, CdBj3, DdBj3,
			BdBj4, CdBj4, DdBj4;

	kUInt mBi_Table_Count;
	kUInt mBj_Table_Count;
	kUInt mBij_Table_Count;

	double* mBi_Table;
	double* mBj_Table;

	double* mBdi_Table;
	double* mBdj_Table;


	double* mBij_Table;
	double* mBdij_Table;
	double* mBidj_Table;

	// Internal routine

	virtual	void SetBiTable (void);
	virtual	void SetBjTable (void);
	virtual	void SetBijTable (void);
	virtual	void SetBdijTable (void);
	virtual	void SetBidjTable (void);
	virtual void SetOtherTable (void) {};

	virtual	void EvaluateSurfaceExactNormal();
	virtual void ComputeBlendingFactor(KFbxWeightedMapping* lWeightedMapping);
	virtual void ComputeBlendingCV( double *pCVArray, 
								    double **pCVs, 
									kUInt pVertexSize, 
									double *pBij, 
									double *pWeight /* may be set to NULL */, 
						            double *pSourceBlendFactor, 
									kUInt *pSourceIndex  );
	virtual void AddBlendingCV( KFbxWeightedMapping* lWeightedMapping, 
							    double* pSourceBlendFactor, 
								kUInt* pSourceIndex, 
								kUInt pDestinationIndex );

	void Correct_Cap_Normal ();
	void Set_Cap_Flag ();
};

//! Create a KFBXSurfaceEvaluator object
KFBX_DLL KFBXSurfaceEvaluator* KFBXSurfaceEvaluatorCreate ();

#include <fbxfilesdk_nsend.h>

#endif


