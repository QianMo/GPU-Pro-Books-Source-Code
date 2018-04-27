/*!  \file kfbxsurfevalmacrosutil.h
 */

#ifndef _FBXSDK_SURFACE_EVALUATOR_MACROS_UTILS_H_
#define _FBXSDK_SURFACE_EVALUATOR_MACROS_UTILS_H_

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
#include <klib/kdebug.h>

#include <fbxfilesdk_nsbegin.h>

#define	Epsilon		0.00000001


/** Surface evaluator macros util **/

/** Compute Point macros **/
//_____________________________________________________________________________
#define Normalize_Normal(  )													\
							Norme_Normal = pNormal[0]*pNormal[0] + pNormal[1]*pNormal[1] + pNormal[2]*pNormal[2];\
							Norme_Normal = sqrt(Norme_Normal);													 \
																												 \
							pNormal[0]/= Norme_Normal;															 \
							pNormal[1]/= Norme_Normal;															 \
							pNormal[2]/= Norme_Normal;															 \
//_____________________________________________________________________________

#ifdef KSURFACE_EVALUATOR_4D
	#define Normal_Dimension_Code()	{/*pNormal[3]	= 1.0;*/}
#else
	#define Normal_Dimension_Code()	{}
#endif

#ifdef	KSURFACE_EVALUATOR_NORMALIZE
	#define Normal_Code()	{Normalize_Normal(); Normal_Dimension_Code();}
#else
	#define Normal_Code()	{Normal_Dimension_Code();}
#endif

#ifdef KSURFACE_EVALUATOR_4D
	#define Dimension_Code()	{ /* *pDestination	= 1.0;*/ pDestination	++;	}
#else
	#define Dimension_Code()	{}
#endif
		
/*
//_____________________________________________________________________________
#define Compute_Point_And_Normal( Eval_Type, Normal_Eval_Type )			\
																		\
						pPointNormalU = NormalU; pPointNormalV = NormalV;																	\
																																			\
						Compute_UNR_Point_Dimension##Eval_Type();			Compute_UNR_Point_NormalUV_Dimension##Normal_Eval_Type();		\
						pDestination++;	Increment_Control_Point_Pointers();	pPointNormalU++; pPointNormalV++;								\
																																			\
						Compute_UNR_Point_Dimension##Eval_Type();			Compute_UNR_Point_NormalUV_Dimension##Normal_Eval_Type();		\
						pDestination++;	Increment_Control_Point_Pointers();	pPointNormalU++; pPointNormalV++;								\
																																			\
						Compute_UNR_Point_Dimension##Eval_Type();			Compute_UNR_Point_NormalUV_Dimension##Normal_Eval_Type();		\
						pDestination++;										pPointNormalU++; pPointNormalV++;								\
																																			\
						pNormal[0] = (NormalU[1]*NormalV[2] - NormalU[2]*NormalV[1]);														\
						pNormal[1] = (NormalU[2]*NormalV[0] - NormalU[0]*NormalV[2]);														\
						pNormal[2] = (NormalU[0]*NormalV[1] - NormalU[1]*NormalV[0]);														\
																																			\
						Normal_Code();																										\
						Dimension_Code();																									\
																																			\
						pNormal += N_Dimension;																								\
//_____________________________________________________________________________	
  */
//_____________________________________________________________________________
#define Compute_Point_And_Normal_Table( Eval_Type, Normal_Eval_Type, Rational )	\
																				\
						Compute_##Rational##_Point_Dimension##Eval_Type();			Compute_##Rational##_Point_NormalUV_Dimension##Normal_Eval_Type();		\
						pDestination++;	Increment_Control_Point_Pointers();	pPointNormalU++; pPointNormalV++;								\
																																			\
						Compute_##Rational##_Point_Dimension##Eval_Type();			Compute_##Rational##_Point_NormalUV_Dimension##Normal_Eval_Type();		\
						pDestination++;	Increment_Control_Point_Pointers();	pPointNormalU++; pPointNormalV++;								\
																																			\
						Compute_##Rational##_Point_Dimension##Eval_Type();			Compute_##Rational##_Point_NormalUV_Dimension##Normal_Eval_Type();		\
						pDestination++;																										\
																																			\
						pPointNormalU --; pPointNormalU--;  pPointNormalV --; pPointNormalV --;												\
																																			\
						pNormal[0] = (pPointNormalV[1]*pPointNormalU[2] - pPointNormalV[2]*pPointNormalU[1]);								\
						pNormal[1] = (pPointNormalV[2]*pPointNormalU[0] - pPointNormalV[0]*pPointNormalU[2]);								\
						pNormal[2] = (pPointNormalV[0]*pPointNormalU[1] - pPointNormalV[1]*pPointNormalU[0]);								\
																																			\
						Normal_Code();																										\
						Dimension_Code();																									\
																																			\
						pNormal += N_Dimension;	pPointNormalU += N_Dimension;  pPointNormalV += N_Dimension;								\
//_____________________________________________________________________________	

//_____________________________________________________________________________
#define Compute_Normal_Table( Normal_Eval_Type, Rational )														\
																												\
						Compute_##Rational##_Point_NormalUV_Dimension##Normal_Eval_Type();					\
						Increment_Control_Point_Pointers();	pPointNormalU++; pPointNormalV++;					\
																												\
						Compute_##Rational##_Point_NormalUV_Dimension##Normal_Eval_Type();					\
						Increment_Control_Point_Pointers();	pPointNormalU++; pPointNormalV++;					\
																												\
						Compute_##Rational##_Point_NormalUV_Dimension##Normal_Eval_Type();					\
																												\
						pPointNormalU --; pPointNormalU --;  pPointNormalV --; pPointNormalV --;				\
																												\
						pNormal[0] =  (pPointNormalV[1]*pPointNormalU[2] - pPointNormalV[2]*pPointNormalU[1]);	\
						pNormal[1] =  (pPointNormalV[2]*pPointNormalU[0] - pPointNormalV[0]*pPointNormalU[2]);	\
						pNormal[2] =  (pPointNormalV[0]*pPointNormalU[1] - pPointNormalV[1]*pPointNormalU[0]);	\
																												\
						Normal_Code();																			\
																												\
						pNormal += N_Dimension;	pPointNormalU += N_Dimension;  pPointNormalV += N_Dimension;	\
//_____________________________________________________________________________	


//_____________________________________________________________________________
#define Set_Cap_Normal()																		\
		Normal[0] = (pTangentV[1]*pTangentU[2] - pTangentV[2]*pTangentU[1]);					\
		Normal[1] = (pTangentV[2]*pTangentU[0] - pTangentV[0]*pTangentU[2]);					\
		Normal[2] = (pTangentV[0]*pTangentU[1] - pTangentV[1]*pTangentU[0]);					\
																								\
		do{																						\
			pNormal[0] = Normal[0];   pNormal[1] = Normal[1];   pNormal[2] = Normal[2];			\
			pNormal += Increment;																\
		}while(--Count);																		\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_Point( Eval_Type , Rational )								\
																			\
						Compute_##Rational##_Point_Dimension##Eval_Type();\
						pDestination++;	Increment_Control_Point_Pointers();	\
																			\
						Compute_##Rational##_Point_Dimension##Eval_Type();\
						pDestination++;	Increment_Control_Point_Pointers();	\
																			\
						Compute_##Rational##_Point_Dimension##Eval_Type();\
						pDestination++;										\
																			\
						Dimension_Code();									\
//_____________________________________________________________________________	

//_____________________________________________________________________________
#define	Compute_Cap_U_Normal()															\
																						\
	Go_Down = mDestination_Point_Count_X * N_Dimension;									\
	NormalV[0]=NormalU[0] = NormalV[1]=NormalU[1] = NormalV[2]=NormalU[2] = 0.0;		\
																						\
	pNormal  = mNormal_Array+ Go_Down - N_Dimension*2;									\
	pNormal2 = mNormal_Array+ N_Dimension;												\
	Count_Y = mDestination_Point_Count_Y;												\
																						\
	do{																					\
		NormalU[0]+=pNormal[0];   NormalU[1]+=pNormal[1];   NormalU[2]+=pNormal[2];		\
		NormalV[0]+=pNormal2[0];  NormalV[1]+=pNormal2[1];  NormalV[2]+=pNormal2[2];	\
																						\
		pNormal += Go_Down;	pNormal2+= Go_Down;											\
	}while(--Count_Y);																	\
																						\
	pNormal  = mNormal_Array+ Go_Down - N_Dimension;									\
	pNormal2 = mNormal_Array;															\
	Count_Y = mDestination_Point_Count_Y;												\
	do{																					\
		pNormal[0] = NormalU[0];   pNormal[1] = NormalU[1];   pNormal[2] = NormalU[2];	\
		pNormal2[0] = NormalV[0];  pNormal2[1] = NormalV[1];  pNormal2[2] = NormalV[2];	\
																						\
		pNormal += Go_Down;  pNormal2+= Go_Down;										\
	}while(--Count_Y);																	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define	Compute_Cap_V_Normal()															\
																						\
	Go_Down = mDestination_Point_Count_X * N_Dimension;									\
	NormalV[0]=NormalU[0] = NormalV[1]=NormalU[1] = NormalV[2]=NormalU[2] = 0.0;		\
																						\
	pNormal  = mNormal_Array+ Go_Down * (mDestination_Point_Count_Y-2);					\
	pNormal2 = mNormal_Array+ Go_Down;													\
	Count_X = mDestination_Point_Count_X;												\
	do{																					\
		NormalU[0]+=pNormal[0];   NormalU[1]+=pNormal[1];   NormalU[2]+=pNormal[2];		\
		NormalV[0]+=pNormal2[0];  NormalV[1]+=pNormal2[1];  NormalV[2]+=pNormal2[2];	\
																						\
		pNormal += N_Dimension;	pNormal2+= N_Dimension;									\
	}while(--Count_X);																	\
																						\
	pNormal2 = mNormal_Array;															\
	Count_X  = mDestination_Point_Count_X;												\
	do{																					\
		pNormal[0] = NormalU[0];   pNormal[1] = NormalU[1];   pNormal[2] = NormalU[2];	\
		pNormal2[0] = NormalV[0];  pNormal2[1] = NormalV[1];  pNormal2[2] = NormalV[2];	\
																						\
		pNormal += N_Dimension;  pNormal2+= N_Dimension;								\
	}while(--Count_X);																	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define	Compute_Cap_U_Normal_Fast()														\
																						\
	Go_Down = mDestination_Point_Count_X * N_Dimension;									\
																						\
	kUInt Go_NStep_V_Down = Go_Down * N_Step_V;										\
																						\
	NormalV[0]=NormalU[0] = NormalV[1]=NormalU[1] = NormalV[2]=NormalU[2] = 0.0;		\
																						\
	pNormal  = mNormal_Array+ Go_Down - (N_Dimension*2);								\
	pNormal2 = mNormal_Array+ N_Dimension;												\
	Count_Y  = (mDestination_Point_Count_Y/N_Step_V);									\
	do{																					\
		NormalU[0]+=pNormal[0];   NormalU[1]+=pNormal[1];   NormalU[2]+=pNormal[2];		\
		NormalV[0]+=pNormal2[0];  NormalV[1]+=pNormal2[1];  NormalV[2]+=pNormal2[2];	\
																						\
		pNormal += Go_NStep_V_Down;	pNormal2+= Go_NStep_V_Down;							\
	}while(--Count_Y);																	\
																						\
	pNormal  = mNormal_Array+ Go_Down - N_Dimension;									\
	pNormal2 = mNormal_Array;															\
	Count_Y = mDestination_Point_Count_Y;												\
	do{																					\
		pNormal[0] = NormalU[0];   pNormal[1] = NormalU[1];   pNormal[2] = NormalU[2];	\
		pNormal2[0] = NormalV[0];  pNormal2[1] = NormalV[1];  pNormal2[2] = NormalV[2];	\
																						\
		pNormal += Go_Down;  pNormal2+= Go_Down;										\
	}while(--Count_Y);																	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define	Compute_Cap_V_Normal_Fast()														\
																						\
	kUInt Go_Right = N_Dimension*N_Step_U;												\
	Go_Down = mDestination_Point_Count_X * N_Dimension;									\
	NormalV[0]=NormalU[0] = NormalV[1]=NormalU[1] = NormalV[2]=NormalU[2] = 0.0;		\
																						\
	pNormal  = mNormal_Array+ Go_Down * (mDestination_Point_Count_Y-2);					\
	pNormal2 = mNormal_Array+ Go_Down;													\
	Count_X  = (mDestination_Point_Count_X/N_Step_U);									\
	do{																					\
		NormalU[0]+=pNormal[0];   NormalU[1]+=pNormal[1];   NormalU[2]+=pNormal[2];		\
		NormalV[0]+=pNormal2[0];  NormalV[1]+=pNormal2[1];  NormalV[2]+=pNormal2[2];	\
																						\
		pNormal += Go_Right;	pNormal2+= Go_Right;									\
	}while(--Count_X);																	\
																						\
	pNormal  = mNormal_Array+ Go_Down * (mDestination_Point_Count_Y-1);					\
	pNormal2 = mNormal_Array;															\
	Count_X  = mDestination_Point_Count_X;												\
	do{																					\
		pNormal[0] = NormalU[0];   pNormal[1] = NormalU[1];   pNormal[2] = NormalU[2];	\
		pNormal2[0] = NormalV[0];  pNormal2[1] = NormalV[1];  pNormal2[2] = NormalV[2];	\
																						\
		pNormal += N_Dimension;  pNormal2+= N_Dimension;								\
	}while(--Count_X);																	\
//_____________________________________________________________________________





/** Compute macros **/
//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_V0_U0()							\
					*pDestination =												\
						pBij[0]  * (*pPoint00) + pBij[1]  * (*pPoint01) + pBij[2]  * (*pPoint02)+	\
						pBij[4]  * (*pPoint10) + pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12)+	\
						pBij[8]  * (*pPoint20) + pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22);	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_V0()								\
					*pDestination =												\
						pBij[0]  * (*pPoint00) + pBij[1]  * (*pPoint01) + pBij[2]  * (*pPoint02) +  pBij[3]  * (*pPoint03) +	\
						pBij[4]  * (*pPoint10) + pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +  pBij[7]  * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +  pBij[11] * (*pPoint23) ;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_U0()								\
					*pDestination =												\
						pBij[0]  * (*pPoint00) + pBij[1]  * (*pPoint01) + pBij[2]  * (*pPoint02) +	\
						pBij[4]  * (*pPoint10) + pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +	\
						pBij[8]  * (*pPoint20) + pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +	\
						pBij[12] * (*pPoint30) + pBij[13] * (*pPoint31) + pBij[14] * (*pPoint32) ;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_U1()									\
					*pDestination =												\
						pBij[1]  * (*pPoint01) + pBij[2]  * (*pPoint02) +  pBij[3]  * (*pPoint03) +	\
						pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +  pBij[7]  * (*pPoint13) +	\
						pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +  pBij[11] * (*pPoint23) +	\
						pBij[13] * (*pPoint31) + pBij[14] * (*pPoint32) +  pBij[15] * (*pPoint33) ;	\
//_____________________________________________________________________________
  

//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_V0_U1()							\
					*pDestination =												\
						pBij[1]  * (*pPoint01) + pBij[2]  * (*pPoint02) +  pBij[3]  * (*pPoint03) +	\
						pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +  pBij[7]  * (*pPoint13) +	\
						pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +  pBij[11] * (*pPoint23);	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_V1()								\
					*pDestination =												\
						pBij[4]  * (*pPoint10) + pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +  pBij[7]  * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +  pBij[11] * (*pPoint23) +	\
						pBij[12] * (*pPoint30) + pBij[13] * (*pPoint31) + pBij[14] * (*pPoint32) +  pBij[15] * (*pPoint33) ;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_V1_U1()								\
					*pDestination =													\
						pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +  pBij[7]  * (*pPoint13) +	\
						pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +  pBij[11] * (*pPoint23) +	\
						pBij[13] * (*pPoint31) + pBij[14] * (*pPoint32) +  pBij[15] * (*pPoint33) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij_V1_U0()							\
					*pDestination =												\
						pBij[4]  * (*pPoint10) + pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) + 	\
						pBij[8]  * (*pPoint20) + pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) + 	\
						pBij[12] * (*pPoint30) + pBij[13] * (*pPoint31) + pBij[14] * (*pPoint32) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension_With_pBij()									\
					*pDestination =												\
						pBij[0]  * (*pPoint00) + pBij[1]  * (*pPoint01) + pBij[2]  * (*pPoint02) +  pBij[3]  * (*pPoint03) +	\
						pBij[4]  * (*pPoint10) + pBij[5]  * (*pPoint11) + pBij[6]  * (*pPoint12) +  pBij[7]  * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9]  * (*pPoint21) + pBij[10] * (*pPoint22) +  pBij[11] * (*pPoint23) +	\
						pBij[12] * (*pPoint30) + pBij[13] * (*pPoint31) + pBij[14] * (*pPoint32) +  pBij[15] * (*pPoint33) ;	\
//_____________________________________________________________________________




//_____________________________________________________________________________
#define Compute_UNR_Point_Dimension()																				\
					*pDestination =																					\
						Bij11 * (*pPoint00) + Bij21 * (*pPoint01) + Bij31 * (*pPoint02) +  Bij41 * (*pPoint03) +	\
						Bij12 * (*pPoint10) + Bij22 * (*pPoint11) + Bij32 * (*pPoint12) +  Bij42 * (*pPoint13) +	\
						Bij13 * (*pPoint20) + Bij23 * (*pPoint21) + Bij33 * (*pPoint22) +  Bij43 * (*pPoint23) +	\
						Bij14 * (*pPoint30) + Bij24 * (*pPoint31) + Bij34 * (*pPoint32) +  Bij44 * (*pPoint33) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_V0_U0()									\
					*pPointNormalU =																	\
						pBdij[0]  * (*pPoint00) + pBdij[1]  * (*pPoint01) + pBdij[2]  * (*pPoint02) +	\
						pBdij[4]  * (*pPoint10) + pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) +	\
						pBdij[8]  * (*pPoint20) + pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) ; 	\
																										\
					*pPointNormalV =																	\
						pBidj[0]  * (*pPoint00) + pBidj[1]  * (*pPoint01) + pBidj[2]  * (*pPoint02) +	\
						pBidj[4]  * (*pPoint10) + pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) +	\
						pBidj[8]  * (*pPoint20) + pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) ;	\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_U0()										\
					*pPointNormalU =																	\
						pBdij[0]  * (*pPoint00) + pBdij[1]  * (*pPoint01) + pBdij[2]  * (*pPoint02) +	\
						pBdij[4]  * (*pPoint10) + pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) +	\
						pBdij[8]  * (*pPoint20) + pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) +	\
						pBdij[12] * (*pPoint30) + pBdij[13] * (*pPoint31) + pBdij[14] * (*pPoint32) ;	\
																										\
					*pPointNormalV =																	\
						pBidj[0]  * (*pPoint00) + pBidj[1]  * (*pPoint01) + pBidj[2]  * (*pPoint02) +	\
						pBidj[4]  * (*pPoint10) + pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) +	\
						pBidj[8]  * (*pPoint20) + pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) +	\
						pBidj[12] * (*pPoint30) + pBidj[13] * (*pPoint31) + pBidj[14] * (*pPoint32) ;	\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_U1()										\
					*pPointNormalU =																	\
						pBdij[1]  * (*pPoint01) + pBdij[2]  * (*pPoint02) + pBdij[3]  * (*pPoint03) +	\
						pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) + pBdij[7]  * (*pPoint13) +	\
						pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) + pBdij[11] * (*pPoint23) +	\
						pBdij[13] * (*pPoint31) + pBdij[14] * (*pPoint32) + pBdij[15] * (*pPoint33) ;	\
																										\
					*pPointNormalV =																	\
						pBidj[1]  * (*pPoint01) + pBidj[2]  * (*pPoint02) + pBidj[3]  * (*pPoint03) +	\
						pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) + pBidj[7]  * (*pPoint13) +	\
						pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) + pBidj[11] * (*pPoint23) +	\
						pBidj[13] * (*pPoint31) + pBidj[14] * (*pPoint32) + pBidj[15] * (*pPoint33) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_V1_U0()									\
					*pPointNormalU =																	\
						pBdij[4]  * (*pPoint10) + pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) +	\
						pBdij[8]  * (*pPoint20) + pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) +	\
						pBdij[12] * (*pPoint30) + pBdij[13] * (*pPoint31) + pBdij[14] * (*pPoint32) ;	\
																										\
					*pPointNormalV =																	\
						pBidj[4]  * (*pPoint10) + pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) +	\
						pBidj[8]  * (*pPoint20) + pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) +	\
						pBidj[12] * (*pPoint30) + pBidj[13] * (*pPoint31) + pBidj[14] * (*pPoint32) ;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_V1()																		\
					*pPointNormalU =																				\
						pBdij[4]  * (*pPoint10) + pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) + pBdij[7]  * (*pPoint13) +	\
						pBdij[8]  * (*pPoint20) + pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) + pBdij[11] * (*pPoint23) +	\
						pBdij[12] * (*pPoint30) + pBdij[13] * (*pPoint31) + pBdij[14] * (*pPoint32) + pBdij[15] * (*pPoint33) ;	\
																													\
					*pPointNormalV =																				\
						pBidj[4]  * (*pPoint10) + pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) + pBidj[7]  * (*pPoint13) +	\
						pBidj[8]  * (*pPoint20) + pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) + pBidj[11] * (*pPoint23) +	\
						pBidj[12] * (*pPoint30) + pBidj[13] * (*pPoint31) + pBidj[14] * (*pPoint32) + pBidj[15] * (*pPoint33) ;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_V1_U1()									\
					*pPointNormalU =																	\
						pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) + pBdij[7]  * (*pPoint13) +	\
						pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) + pBdij[11] * (*pPoint23) +	\
						pBdij[13] * (*pPoint31) + pBdij[14] * (*pPoint32) + pBdij[15] * (*pPoint33) ;	\
																										\
					*pPointNormalV =																	\
						pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) + pBidj[7]  * (*pPoint13) +	\
						pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) + pBidj[11] * (*pPoint23) +	\
						pBidj[13] * (*pPoint31) + pBidj[14] * (*pPoint32) + pBidj[15] * (*pPoint33) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_V0_U1()									\
					*pPointNormalU =																	\
						pBdij[1]  * (*pPoint01) + pBdij[2]  * (*pPoint02) + pBdij[3]  * (*pPoint03) +	\
						pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) + pBdij[7]  * (*pPoint13) +	\
						pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) + pBdij[11] * (*pPoint23) ;	\
																										\
					*pPointNormalV =																	\
						pBidj[1]  * (*pPoint01) + pBidj[2]  * (*pPoint02) + pBidj[3]  * (*pPoint03) +	\
						pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) + pBidj[7]  * (*pPoint13) +	\
						pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) + pBidj[11] * (*pPoint23) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj_V0()										\
					*pPointNormalU =																	\
						pBdij[0]  * (*pPoint00) + pBdij[1]  * (*pPoint01) + pBdij[2]  * (*pPoint02) + pBdij[3]  * (*pPoint03) +	\
						pBdij[4]  * (*pPoint10) + pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) + pBdij[7]  * (*pPoint13) +	\
						pBdij[8]  * (*pPoint20) + pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) + pBdij[11] * (*pPoint23); 	\
																													\
					*pPointNormalV =																				\
						pBidj[0]  * (*pPoint00) + pBidj[1]  * (*pPoint01) + pBidj[2]  * (*pPoint02) + pBidj[3]  * (*pPoint03) +	\
						pBidj[4]  * (*pPoint10) + pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) + pBidj[7]  * (*pPoint13) +	\
						pBidj[8]  * (*pPoint20) + pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) + pBidj[11] * (*pPoint23);	\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension_With_pBdij_pBidj()																		\
					*pPointNormalU =																				\
						pBdij[0]  * (*pPoint00) + pBdij[1]  * (*pPoint01) + pBdij[2]  * (*pPoint02) + pBdij[3]  * (*pPoint03) +	\
						pBdij[4]  * (*pPoint10) + pBdij[5]  * (*pPoint11) + pBdij[6]  * (*pPoint12) + pBdij[7]  * (*pPoint13) +	\
						pBdij[8]  * (*pPoint20) + pBdij[9]  * (*pPoint21) + pBdij[10] * (*pPoint22) + pBdij[11] * (*pPoint23) +	\
						pBdij[12] * (*pPoint30) + pBdij[13] * (*pPoint31) + pBdij[14] * (*pPoint32) + pBdij[15] * (*pPoint33) ;	\
																													\
					*pPointNormalV =																				\
						pBidj[0]  * (*pPoint00) + pBidj[1]  * (*pPoint01) + pBidj[2]  * (*pPoint02) + pBidj[3]  * (*pPoint03) +	\
						pBidj[4]  * (*pPoint10) + pBidj[5]  * (*pPoint11) + pBidj[6]  * (*pPoint12) + pBidj[7]  * (*pPoint13) +	\
						pBidj[8]  * (*pPoint20) + pBidj[9]  * (*pPoint21) + pBidj[10] * (*pPoint22) + pBidj[11] * (*pPoint23) +	\
						pBidj[12] * (*pPoint30) + pBidj[13] * (*pPoint31) + pBidj[14] * (*pPoint32) + pBidj[15] * (*pPoint33) ;	\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define Compute_UNR_Point_NormalUV_Dimension()																		\
					*pPointNormalU =																				\
						Bdij11 * (*pPoint00) + Bdij21 * (*pPoint01) + Bdij31 * (*pPoint02) + Bdij41 * (*pPoint03) +	\
						Bdij12 * (*pPoint10) + Bdij22 * (*pPoint11) + Bdij32 * (*pPoint12) + Bdij42 * (*pPoint13) +	\
						Bdij13 * (*pPoint20) + Bdij23 * (*pPoint21) + Bdij33 * (*pPoint22) + Bdij43 * (*pPoint23) +	\
						Bdij14 * (*pPoint30) + Bdij24 * (*pPoint31) + Bdij34 * (*pPoint32) + Bdij44 * (*pPoint33) ;	\
																													\
					*pPointNormalV =																				\
						Bidj11 * (*pPoint00) + Bidj21 * (*pPoint01) + Bidj31 * (*pPoint02) +  Bidj41 * (*pPoint03) +\
						Bidj12 * (*pPoint10) + Bidj22 * (*pPoint11) + Bidj32 * (*pPoint12) +  Bidj42 * (*pPoint13) +\
						Bidj13 * (*pPoint20) + Bidj23 * (*pPoint21) + Bidj33 * (*pPoint22) +  Bidj43 * (*pPoint23) +\
						Bidj14 * (*pPoint30) + Bidj24 * (*pPoint31) + Bidj34 * (*pPoint32) +  Bidj44 * (*pPoint33) ;\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_Normal_Point_Dimension( PosiX, PosiY, Dimension )				\
				*pPointNormalU =												\
						Bdij11 * GetData( ClipX(PosiX+0), ClipY( PosiY+0 ), Dimension ) + Bdij21 * GetData( ClipX(PosiX+1), ClipY( PosiY+0 ), Dimension ) + Bdij31 * GetData( ClipX(PosiX+2), ClipY( PosiY+0 ), Dimension )  +  Bdij41 * GetData( ClipX(PosiX+3), ClipY( PosiY+0 ), Dimension ) +	\
						Bdij12 * GetData( ClipX(PosiX+0), ClipY( PosiY+1 ), Dimension ) + Bdij22 * GetData( ClipX(PosiX+1), ClipY( PosiY+1 ), Dimension ) + Bdij32 * GetData( ClipX(PosiX+2), ClipY( PosiY+1 ), Dimension )  +  Bdij42 * GetData( ClipX(PosiX+3), ClipY( PosiY+1 ), Dimension ) +	\
						Bdij13 * GetData( ClipX(PosiX+0), ClipY( PosiY+2 ), Dimension ) + Bdij23 * GetData( ClipX(PosiX+1), ClipY( PosiY+2 ), Dimension ) + Bdij33 * GetData( ClipX(PosiX+2), ClipY( PosiY+2 ), Dimension )  +  Bdij43 * GetData( ClipX(PosiX+3), ClipY( PosiY+2 ), Dimension ) +	\
						Bdij14 * GetData( ClipX(PosiX+0), ClipY( PosiY+3 ), Dimension ) + Bdij24 * GetData( ClipX(PosiX+1), ClipY( PosiY+3 ), Dimension ) + Bdij34 * GetData( ClipX(PosiX+2), ClipY( PosiY+3 ), Dimension )  +  Bdij44 * GetData( ClipX(PosiX+3), ClipY( PosiY+3 ), Dimension );	\
																																																																									\
				*pPointNormalV =																																																																	\
						Bidj11 * GetData( ClipX(PosiX+0), ClipY( PosiY+0 ), Dimension ) + Bidj21 * GetData( ClipX(PosiX+1), ClipY( PosiY+0 ), Dimension ) + Bidj31 * GetData( ClipX(PosiX+2), ClipY( PosiY+0 ), Dimension )  +  Bidj41 * GetData( ClipX(PosiX+3), ClipY( PosiY+0 ), Dimension ) +	\
						Bidj12 * GetData( ClipX(PosiX+0), ClipY( PosiY+1 ), Dimension ) + Bidj22 * GetData( ClipX(PosiX+1), ClipY( PosiY+1 ), Dimension ) + Bidj32 * GetData( ClipX(PosiX+2), ClipY( PosiY+1 ), Dimension )  +  Bidj42 * GetData( ClipX(PosiX+3), ClipY( PosiY+1 ), Dimension ) +	\
						Bidj13 * GetData( ClipX(PosiX+0), ClipY( PosiY+2 ), Dimension ) + Bidj23 * GetData( ClipX(PosiX+1), ClipY( PosiY+2 ), Dimension ) + Bidj33 * GetData( ClipX(PosiX+2), ClipY( PosiY+2 ), Dimension )  +  Bidj43 * GetData( ClipX(PosiX+3), ClipY( PosiY+2 ), Dimension ) +	\
						Bidj14 * GetData( ClipX(PosiX+0), ClipY( PosiY+3 ), Dimension ) + Bidj24 * GetData( ClipX(PosiX+1), ClipY( PosiY+3 ), Dimension ) + Bidj34 * GetData( ClipX(PosiX+2), ClipY( PosiY+3 ), Dimension )  +  Bidj44 * GetData( ClipX(PosiX+3), ClipY( PosiY+3 ), Dimension );	
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Compute_Point_Dimension( PosiX, PosiY, Dimension )				\
				*pDestination =											\
						Bij11 * GetData( ClipX(PosiX+0), ClipY( PosiY+0 ), Dimension ) + Bij21 * GetData( ClipX(PosiX+1), ClipY( PosiY+0 ), Dimension ) + Bij31 * GetData( ClipX(PosiX+2), ClipY( PosiY+0 ), Dimension )  +  Bij41 * GetData( ClipX(PosiX+3), ClipY( PosiY+0 ), Dimension ) +	\
						Bij12 * GetData( ClipX(PosiX+0), ClipY( PosiY+1 ), Dimension ) + Bij22 * GetData( ClipX(PosiX+1), ClipY( PosiY+1 ), Dimension ) + Bij32 * GetData( ClipX(PosiX+2), ClipY( PosiY+1 ), Dimension )  +  Bij42 * GetData( ClipX(PosiX+3), ClipY( PosiY+1 ), Dimension ) +	\
						Bij13 * GetData( ClipX(PosiX+0), ClipY( PosiY+2 ), Dimension ) + Bij23 * GetData( ClipX(PosiX+1), ClipY( PosiY+2 ), Dimension ) + Bij33 * GetData( ClipX(PosiX+2), ClipY( PosiY+2 ), Dimension )  +  Bij43 * GetData( ClipX(PosiX+3), ClipY( PosiY+2 ), Dimension ) +	\
						Bij14 * GetData( ClipX(PosiX+0), ClipY( PosiY+3 ), Dimension ) + Bij24 * GetData( ClipX(PosiX+1), ClipY( PosiY+3 ), Dimension ) + Bij34 * GetData( ClipX(PosiX+2), ClipY( PosiY+3 ), Dimension )  +  Bij44 * GetData( ClipX(PosiX+3), ClipY( PosiY+3 ), Dimension );	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Set_Blending_Parameters( Matrix, d )							\
				AB ##d ##1 = Matrix[0];  BB ##d ##1 = Matrix[1];  CB ##d ##1 = Matrix[2];  DB ##d ##1 = Matrix[3];\
				AB ##d ##2 = Matrix[4];  BB ##d ##2 = Matrix[5];  CB ##d ##2 = Matrix[6];  DB ##d ##2 = Matrix[7];\
				AB ##d ##3 = Matrix[8];  BB ##d ##3 = Matrix[9];  CB ##d ##3 = Matrix[10]; DB ##d ##3 = Matrix[11];\
				AB ##d ##4 = Matrix[12]; BB ##d ##4 = Matrix[13]; CB ##d ##4 = Matrix[14]; DB ##d ##4 = Matrix[15];\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Set_Blending_Tangent_Parameters( d )							\
				BdB ##d ##1 = 3.0*AB ##d ##1; CdB ##d ##1 = 2.0*BB ##d ##1;  DdB ##d ##1 = CB ##d ##1;\
				BdB ##d ##2 = 3.0*AB ##d ##2; CdB ##d ##2 = 2.0*BB ##d ##2;  DdB ##d ##2 = CB ##d ##2;\
				BdB ##d ##3 = 3.0*AB ##d ##3; CdB ##d ##3 = 2.0*BB ##d ##3;  DdB ##d ##3 = CB ##d ##3;\
				BdB ##d ##4 = 3.0*AB ##d ##4; CdB ##d ##4 = 2.0*BB ##d ##4;  DdB ##d ##4 = CB ##d ##4;\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_Bi_Blending_Function()		\
		Bi1 = ABi1 * u3 + BBi1 * u2 + CBi1 * u + DBi1;\
		Bi2 = ABi2 * u3 + BBi2 * u2 + CBi2 * u + DBi2;\
		Bi3 = ABi3 * u3 + BBi3 * u2 + CBi3 * u + DBi3;\
		Bi4 = ABi4 * u3 + BBi4 * u2 + CBi4 * u + DBi4;\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Compute_dBi_Blending_Function()		\
		dBi1 = BdBi1 * u2 + CdBi1 * u + DdBi1;\
		dBi2 = BdBi2 * u2 + CdBi2 * u + DdBi2;\
		dBi3 = BdBi3 * u2 + CdBi3 * u + DdBi3;\
		dBi4 = BdBi4 * u2 + CdBi4 * u + DdBi4;\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Compute_Bj_Blending_Function()		\
		Bj1 = ABj1 * v3 + BBj1 * v2 + CBj1 * v + DBj1;\
		Bj2 = ABj2 * v3 + BBj2 * v2 + CBj2 * v + DBj2;\
		Bj3 = ABj3 * v3 + BBj3 * v2 + CBj3 * v + DBj3;\
		Bj4 = ABj4 * v3 + BBj4 * v2 + CBj4 * v + DBj4;\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Compute_dBj_Blending_Function()		\
		dBj1 = BdBj1 * v2 + CdBj1 * v + DdBj1;\
		dBj2 = BdBj2 * v2 + CdBj2 * v + DdBj2;\
		dBj3 = BdBj3 * v2 + CdBj3 * v + DdBj3;\
		dBj4 = BdBj4 * v2 + CdBj4 * v + DdBj4;\
//_____________________________________________________________________________



//_____________________________________________________________________________
#define	Compute_Bij_Polynome_Function()															\
					Bij11 = Bi1 * Bj1; Bij12 = Bi1 * Bj2; Bij13 = Bi1 * Bj3; Bij14 = Bi1 * Bj4;	\
					Bij21 = Bi2 * Bj1; Bij22 = Bi2 * Bj2; Bij23 = Bi2 * Bj3; Bij24 = Bi2 * Bj4;	\
					Bij31 = Bi3 * Bj1; Bij32 = Bi3 * Bj2; Bij33 = Bi3 * Bj3; Bij34 = Bi3 * Bj4;	\
					Bij41 = Bi4 * Bj1; Bij42 = Bi4 * Bj2; Bij43 = Bi4 * Bj3; Bij44 = Bi4 * Bj4;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define	Compute_Bdij_Polynome_Function()																\
					Bdij11 = dBi1 * Bj1; Bdij12 = dBi1 * Bj2; Bdij13 = dBi1 * Bj3; Bdij14 = dBi1 * Bj4;	\
					Bdij21 = dBi2 * Bj1; Bdij22 = dBi2 * Bj2; Bdij23 = dBi2 * Bj3; Bdij24 = dBi2 * Bj4;	\
					Bdij31 = dBi3 * Bj1; Bdij32 = dBi3 * Bj2; Bdij33 = dBi3 * Bj3; Bdij34 = dBi3 * Bj4;	\
					Bdij41 = dBi4 * Bj1; Bdij42 = dBi4 * Bj2; Bdij43 = dBi4 * Bj3; Bdij44 = dBi4 * Bj4;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define	Compute_Bidj_Polynome_Function()																\
					Bidj11 = Bi1 * dBj1; Bidj12 = Bi1 * dBj2; Bidj13 = Bi1 * dBj3; Bidj14 = Bi1 * dBj4;	\
					Bidj21 = Bi2 * dBj1; Bidj22 = Bi2 * dBj2; Bidj23 = Bi2 * dBj3; Bidj24 = Bi2 * dBj4;	\
					Bidj31 = Bi3 * dBj1; Bidj32 = Bi3 * dBj2; Bidj33 = Bi3 * dBj3; Bidj34 = Bi3 * dBj4;	\
					Bidj41 = Bi4 * dBj1; Bidj42 = Bi4 * dBj2; Bidj43 = Bi4 * dBj3; Bidj44 = Bi4 * dBj4;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define	Compute_Bdidj_Polynome_Function()																\
					Compute_Bidj_Polynome_Function();													\
					Compute_Bdij_Polynome_Function();													\
//_____________________________________________________________________________



/* Pointer Macro */

//_____________________________________________________________________________
#define Wrap_Around_Control_Point_Pointers_U()											\
				if( pPoint03 >= pLimit_Point_Line ){Sub_Control_Point_Pointers_Column( 3, Line_Source_Length );	Wrap_Flag_U |= 1;}\
				if( pPoint02 >= pLimit_Point_Line ){Sub_Control_Point_Pointers_Column( 2, Line_Source_Length );	Wrap_Flag_U |= 2;}\
				if( pPoint01 >= pLimit_Point_Line ){Sub_Control_Point_Pointers_Column( 1, Line_Source_Length );	Wrap_Flag_U |= 4;}\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Backp_Up_Wrap_Around_Control_Pointer_U()								\
																				\
			switch(Wrap_Flag_U){												\
			case 7:	Add_Control_Point_Pointers_Column( 1, Line_Source_Length );	\
			case 3: Add_Control_Point_Pointers_Column( 2, Line_Source_Length );	\
			case 1:	Add_Control_Point_Pointers_Column( 3, Line_Source_Length ); \
/*//break;																		\
//			case 0: break;														\
//			default : K_ASSERT_MSG_NOW("Something wrong : Wrap_Flag_U");		\
*/																				\
			}																	\
																				\
			Wrap_Flag_U = 0;													\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Set_Control_Point_Pointers( pArray, Count_X, Count_Y, N_Dimension) 										\
																												\
					pPoint00 = &pArray[ (Count_X + (Count_Y * mSource_Point_Count_X) ) * N_Dimension ];			\
					pPoint01 = pPoint00 + N_Dimension;															\
					pPoint02 = pPoint01 + N_Dimension;															\
					pPoint03 = pPoint02 + N_Dimension;															\
																												\
					pPoint10 = pPoint00 + Line_Source_Length;													\
					pPoint11 = pPoint10 + N_Dimension;															\
					pPoint12 = pPoint11 + N_Dimension;															\
					pPoint13 = pPoint12 + N_Dimension;															\
																												\
					pPoint20 = pPoint10 + Line_Source_Length;													\
					pPoint21 = pPoint20 + N_Dimension;															\
					pPoint22 = pPoint21 + N_Dimension;															\
					pPoint23 = pPoint22 + N_Dimension;															\
																												\
					pPoint30 = pPoint20 + Line_Source_Length;													\
					pPoint31 = pPoint30 + N_Dimension;															\
					pPoint32 = pPoint31 + N_Dimension;															\
					pPoint33 = pPoint32 + N_Dimension;															\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Increment_Control_Point_Pointers()								\
					pPoint00++; pPoint01++; pPoint02++; pPoint03++;		\
					pPoint10++; pPoint11++; pPoint12++; pPoint13++;		\
					pPoint20++; pPoint21++; pPoint22++; pPoint23++;		\
					pPoint30++; pPoint31++; pPoint32++; pPoint33++;		\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Decrement_Control_Point_Pointers_By_2()																		\
					pPoint00--;pPoint00--;  pPoint01--;pPoint01--;  pPoint02--;pPoint02--;  pPoint03--;pPoint03--;	\
					pPoint10--;pPoint10--;  pPoint11--;pPoint11--;  pPoint12--;pPoint12--;  pPoint13--;pPoint13--;	\
					pPoint20--;pPoint20--;  pPoint21--;pPoint21--;  pPoint22--;pPoint22--;  pPoint23--;pPoint23--;	\
					pPoint30--;pPoint30--;  pPoint31--;pPoint31--;  pPoint32--;pPoint32--;  pPoint33--;pPoint33--;	\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Sub_Control_Point_Pointers(a)												\
					Sub_Control_Point_Pointers_Row(0,a);							\
					Sub_Control_Point_Pointers_Row(1,a);							\
					Sub_Control_Point_Pointers_Row(2,a);							\
					Sub_Control_Point_Pointers_Row(3,a);							\
//_____________________________________________________________________________


//_____________________________________________________________________________
#define Add_Control_Point_Pointers(a)												\
					Add_Control_Point_Pointers_Row(0,a);							\
					Add_Control_Point_Pointers_Row(1,a);							\
					Add_Control_Point_Pointers_Row(2,a);							\
					Add_Control_Point_Pointers_Row(3,a);							\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Add_Control_Point_Pointers_Column(i,a)  									\
					pPoint0 ##i +=a;												\
					pPoint1 ##i +=a;												\
					pPoint2 ##i +=a;												\
					pPoint3 ##i +=a;												\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Add_Control_Point_Pointers_Row(j,a)											\
					pPoint ##j ## 0 +=a;											\
					pPoint ##j ## 1 +=a;											\
					pPoint ##j ## 2 +=a;											\
					pPoint ##j ## 3 +=a;											\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Sub_Control_Point_Pointers_Column(i,a)  									\
					pPoint0 ## i -=a;												\
					pPoint1 ## i -=a;												\
					pPoint2 ## i -=a;												\
					pPoint3 ## i -=a;												\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define Sub_Control_Point_Pointers_Row(j,a)											\
					pPoint ##j ##0 -=a;												\
					pPoint ##j ##1 -=a;												\
					pPoint ##j ##2 -=a;												\
					pPoint ##j ##3 -=a;												\
//_____________________________________________________________________________


/* Blending Factor Macro */

//_____________________________________________________________________________
#define BuildCVs()																					\
		double *lCVs[16];																			\
			lCVs[0] = pPoint00; lCVs[1] = pPoint01; lCVs[2] = pPoint02; lCVs[3] = pPoint03;			\
			lCVs[4] = pPoint10; lCVs[5] = pPoint11; lCVs[6] = pPoint12; lCVs[7] = pPoint13;			\
			lCVs[8] = pPoint20; lCVs[9] = pPoint21; lCVs[10] = pPoint22; lCVs[11] = pPoint23;		\
			lCVs[12] = pPoint30; lCVs[13] = pPoint31; lCVs[14] = pPoint32; lCVs[15] = pPoint33;		\
//_____________________________________________________________________________

//_____________________________________________________________________________
#define BuildWeigths()																				\
		double lWeights[16];																		\
			lWeights[0] = pPoint00[0]; lWeights[1] = pPoint01[0]; lWeights[2] = pPoint02[0]; lWeights[3] = pPoint03[0];		\
			lWeights[4] = pPoint10[0]; lWeights[5] = pPoint11[0]; lWeights[6] = pPoint12[0]; lWeights[7] = pPoint13[0];		\
			lWeights[8] = pPoint20[0]; lWeights[9] = pPoint21[0]; lWeights[10] = pPoint22[0]; lWeights[11] = pPoint23[0];	\
			lWeights[12] = pPoint30[0]; lWeights[13] = pPoint31[0]; lWeights[14] = pPoint32[0]; lWeights[15] = pPoint33[0];	\
//_____________________________________________________________________________

#include <fbxfilesdk_nsend.h>

#endif /* this must be the last line of this file */


