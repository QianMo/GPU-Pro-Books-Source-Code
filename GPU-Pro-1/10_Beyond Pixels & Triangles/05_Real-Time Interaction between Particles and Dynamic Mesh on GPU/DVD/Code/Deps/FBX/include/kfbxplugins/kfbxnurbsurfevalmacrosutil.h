/*!  \file kfbxnurbsurfevalmacrosutil.h
 */

#ifndef _FBXSDK_NURB_SURFACE_EVALUATOR_MACROS_UTILS_H_
#define _FBXSDK_NURB_SURFACE_EVALUATOR_MACROS_UTILS_H_

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

#include <kfbxplugins/kfbxsurfevalmacrosutil.h>
#include <fbxfilesdk_nsbegin.h>


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij( )							\
				*pDestination =	(											\
						pBij[0]  * (*pPoint00) + pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +  pBij[3] * (*pPoint03) +	\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) +	\
						pBij[12] * (*pPoint30) + pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32) +  pBij[15]* (*pPoint33)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_V0( )							\
				*pDestination =	(											\
						pBij[0]  * (*pPoint00) + pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +  pBij[3] * (*pPoint03) +	\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) 	\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_V0_U0( )							\
				*pDestination =	(											\
						pBij[0]  * (*pPoint00) + pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +	\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) 	\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_V0_U1( )							\
				*pDestination =	(											\
						pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +  pBij[3] * (*pPoint03) +	\
						pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) 	\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_U0( )							\
				*pDestination =	(											\
						pBij[0]  * (*pPoint00) + pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +	\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +	\
						pBij[12] * (*pPoint30) + pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_U1( )							\
				*pDestination =	(											\
						pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +  pBij[3] * (*pPoint03) +	\
						pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) +	\
						pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32) +  pBij[15]* (*pPoint33)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_V1( )							\
				*pDestination =	(											\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) +	\
						pBij[12] * (*pPoint30) + pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32) +  pBij[15]* (*pPoint33)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_V1_U0( )							\
				*pDestination =	(											\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +	\
						pBij[12] * (*pPoint30) + pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij_V1_U1( )							\
				*pDestination =	(											\
						pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) +	\
						pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32) +  pBij[15]* (*pPoint33)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_Dimension_With_pBij( )							\
				*pDestination =	(											\
						pBij[0]  * (*pPoint00) + pBij[1] * (*pPoint01) + pBij[2] * (*pPoint02) +  pBij[3] * (*pPoint03) +	\
						pBij[4]  * (*pPoint10) + pBij[5] * (*pPoint11) + pBij[6] * (*pPoint12) +  pBij[7] * (*pPoint13) +	\
						pBij[8]  * (*pPoint20) + pBij[9] * (*pPoint21) + pBij[10]* (*pPoint22) +  pBij[11]* (*pPoint23) +	\
						pBij[12] * (*pPoint30) + pBij[13]* (*pPoint31) + pBij[14]* (*pPoint32) +  pBij[15]* (*pPoint33)		\
								) /pWeight_Table[0] /*Point_Weight*/;


//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[0] * (*pPoint00) + pBdij[1] * (*pPoint01) + pBdij[2] * (*pPoint02) +  pBdij[3] * (*pPoint03) +\
						pBdij[4] * (*pPoint10) + pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +  pBdij[7] * (*pPoint13) +\
						pBdij[8] * (*pPoint20) + pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +  pBdij[11]* (*pPoint23) +\
						pBdij[12]* (*pPoint30) + pBdij[13]* (*pPoint31) + pBdij[14]* (*pPoint32) +  pBdij[15]* (*pPoint33) 	\
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																							\
						pBidj[0] * (*pPoint00) + pBidj[1] * (*pPoint01) + pBidj[2] * (*pPoint02) +  pBidj[3] * (*pPoint03) +\
						pBidj[4] * (*pPoint10) + pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +  pBidj[7] * (*pPoint13) +\
						pBidj[8] * (*pPoint20) + pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +  pBidj[11]* (*pPoint23) +\
						pBidj[12]* (*pPoint30) + pBidj[13]* (*pPoint31) + pBidj[14]* (*pPoint32) +  pBidj[15]* (*pPoint33) 	\
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\


//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_V0( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[0] * (*pPoint00) + pBdij[1] * (*pPoint01) + pBdij[2] * (*pPoint02) +  pBdij[3] * (*pPoint03) +\
						pBdij[4] * (*pPoint10) + pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +  pBdij[7] * (*pPoint13) +\
						pBdij[8] * (*pPoint20) + pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +  pBdij[11]* (*pPoint23) \
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																							\
						pBidj[0] * (*pPoint00) + pBidj[1] * (*pPoint01) + pBidj[2] * (*pPoint02) +  pBidj[3] * (*pPoint03) +\
						pBidj[4] * (*pPoint10) + pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +  pBidj[7] * (*pPoint13) +\
						pBidj[8] * (*pPoint20) + pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +  pBidj[11]* (*pPoint23) \
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\

//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_V0_U0( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[0] * (*pPoint00) + pBdij[1] * (*pPoint01) + pBdij[2] * (*pPoint02) +\
						pBdij[4] * (*pPoint10) + pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +\
						pBdij[8] * (*pPoint20) + pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) \
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																							\
						pBidj[0] * (*pPoint00) + pBidj[1] * (*pPoint01) + pBidj[2] * (*pPoint02) +\
						pBidj[4] * (*pPoint10) + pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +\
						pBidj[8] * (*pPoint20) + pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) \
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\


//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_V0_U1( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[1] * (*pPoint01) + pBdij[2] * (*pPoint02) +  pBdij[3] * (*pPoint03) +\
						pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +  pBdij[7] * (*pPoint13) +\
						pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +  pBdij[11]* (*pPoint23) \
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																							\
						pBidj[1] * (*pPoint01) + pBidj[2] * (*pPoint02) +  pBidj[3] * (*pPoint03) +\
						pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +  pBidj[7] * (*pPoint13) +\
						pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +  pBidj[11]* (*pPoint23) \
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\

//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_U0( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[0] * (*pPoint00) + pBdij[1] * (*pPoint01) + pBdij[2] * (*pPoint02) +\
						pBdij[4] * (*pPoint10) + pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +\
						pBdij[8] * (*pPoint20) + pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +\
						pBdij[12]* (*pPoint30) + pBdij[13]* (*pPoint31) + pBdij[14]* (*pPoint32) 	\
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																							\
						pBidj[0] * (*pPoint00) + pBidj[1] * (*pPoint01) + pBidj[2] * (*pPoint02) +\
						pBidj[4] * (*pPoint10) + pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +\
						pBidj[8] * (*pPoint20) + pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +\
						pBidj[12]* (*pPoint30) + pBidj[13]* (*pPoint31) + pBidj[14]* (*pPoint32) 	\
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\

//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_U1( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[1] * (*pPoint01) + pBdij[2] * (*pPoint02) +  pBdij[3] * (*pPoint03) +\
						pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +  pBdij[7] * (*pPoint13) +\
						pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +  pBdij[11]* (*pPoint23) +\
						pBdij[13]* (*pPoint31) + pBdij[14]* (*pPoint32) +  pBdij[15]* (*pPoint33) 	\
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																							\
						+ pBidj[1] * (*pPoint01) + pBidj[2] * (*pPoint02) +  pBidj[3] * (*pPoint03) +\
						+ pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +  pBidj[7] * (*pPoint13) +\
						+ pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +  pBidj[11]* (*pPoint23) +\
						+ pBidj[13]* (*pPoint31) + pBidj[14]* (*pPoint32) +  pBidj[15]* (*pPoint33) 	\
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\

//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_V1( )				\
																				\
				*pPointNormalU =( (												\
						pBdij[4] * (*pPoint10) + pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +  pBdij[7] * (*pPoint13) +\
						pBdij[8] * (*pPoint20) + pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +  pBdij[11]* (*pPoint23) +\
						pBdij[12]* (*pPoint30) + pBdij[13]* (*pPoint31) + pBdij[14]* (*pPoint32) +  pBdij[15]* (*pPoint33) 	\
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];										\
																															\
				*pPointNormalV = ((																	 						\
						pBidj[4] * (*pPoint10) + pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +  pBidj[7] * (*pPoint13) +\
						pBidj[8] * (*pPoint20) + pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +  pBidj[11]* (*pPoint23) +\
						pBidj[12]* (*pPoint30) + pBidj[13]* (*pPoint31) + pBidj[14]* (*pPoint32) +  pBidj[15]* (*pPoint33) 	\
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];										\

//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_V1_U0( )								\
																									\
				*pPointNormalU =( (																	\
						pBdij[4] * (*pPoint10) + pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12)+	\
						pBdij[8] * (*pPoint20) + pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22)+	\
						pBdij[12]* (*pPoint30) + pBdij[13]* (*pPoint31) + pBdij[14]* (*pPoint32)	\
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];				\
																									\
				*pPointNormalV = ((																	\
						pBidj[4] * (*pPoint10) + pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12)+	\
						pBidj[8] * (*pPoint20) + pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22)+	\
						pBidj[12]* (*pPoint30) + pBidj[13]* (*pPoint31) + pBidj[14]* (*pPoint32)	\
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];				\

//! Compute Point macro.
#define Compute_NUR_Point_NormalUV_Dimension_With_pBdij_pBidj_V1_U1( )								\
																									\
				*pPointNormalU =( (																	\
						pBdij[5] * (*pPoint11) + pBdij[6] * (*pPoint12) +  pBdij[7] * (*pPoint13) +	\
						pBdij[9] * (*pPoint21) + pBdij[10]* (*pPoint22) +  pBdij[11]* (*pPoint23) +	\
						pBdij[13]* (*pPoint31) + pBdij[14]* (*pPoint32) +  pBdij[15]* (*pPoint33) 	\
						) -  pWeight_Table[1] * (*pDestination)  ) /pWeight_Table[0];				\
																									\
				*pPointNormalV = ((																	\
						pBidj[5] * (*pPoint11) + pBidj[6] * (*pPoint12) +  pBidj[7] * (*pPoint13) +	\
						pBidj[9] * (*pPoint21) + pBidj[10]* (*pPoint22) +  pBidj[11]* (*pPoint23) +	\
						pBidj[13]* (*pPoint31) + pBidj[14]* (*pPoint32) +  pBidj[15]* (*pPoint33) 	\
						) -  pWeight_Table[2] * (*pDestination)  ) /pWeight_Table[0];				\

/*
//! Compute Point macro.
#define Compute_NUR_Point_Dimension( PosiX, PosiY, Dimension )				\
				*pDestination =	(									\
						pBij[0]  * GetWeightedData( ClipX(PosiX+0), ClipY( PosiY+0 ), Dimension ) + pBij[1] * GetWeightedData( ClipX(PosiX+1), ClipY( PosiY+0 ), Dimension ) + pBij[2] * GetWeightedData( ClipX(PosiX+2), ClipY( PosiY+0 ), Dimension ) +  pBij[3] * GetWeightedData( ClipX(PosiX+3), ClipY( PosiY+0 ), Dimension ) +	\
						pBij[4]  * GetWeightedData( ClipX(PosiX+0), ClipY( PosiY+1 ), Dimension ) + pBij[5] * GetWeightedData( ClipX(PosiX+1), ClipY( PosiY+1 ), Dimension ) + pBij[6] * GetWeightedData( ClipX(PosiX+2), ClipY( PosiY+1 ), Dimension ) +  pBij[7] * GetWeightedData( ClipX(PosiX+3), ClipY( PosiY+1 ), Dimension ) +	\
						pBij[8]  * GetWeightedData( ClipX(PosiX+0), ClipY( PosiY+2 ), Dimension ) + pBij[9] * GetWeightedData( ClipX(PosiX+1), ClipY( PosiY+2 ), Dimension ) + pBij[10]* GetWeightedData( ClipX(PosiX+2), ClipY( PosiY+2 ), Dimension ) +  pBij[11]* GetWeightedData( ClipX(PosiX+3), ClipY( PosiY+2 ), Dimension ) +	\
						pBij[12] * GetWeightedData( ClipX(PosiX+0), ClipY( PosiY+3 ), Dimension ) + pBij[13]* GetWeightedData( ClipX(PosiX+1), ClipY( PosiY+3 ), Dimension ) + pBij[14]* GetWeightedData( ClipX(PosiX+2), ClipY( PosiY+3 ), Dimension ) +  pBij[15]* GetWeightedData( ClipX(PosiX+3), ClipY( PosiY+3 ), Dimension )		\
								) / Point_Weight;
*/




//* Table Setup macro util
#define Compute_Blending_FunctionBi()											\
																				\
					Bi1 = Compute_Bik( u, Posi_X  , mOrder_U, mKnot_Vector_U);	\
					Bi2 = Compute_Bik( u, Posi_X+1, mOrder_U, mKnot_Vector_U);	\
					Bi3 = Compute_Bik( u, Posi_X+2, mOrder_U, mKnot_Vector_U);	\
					Bi4 = Compute_Bik( u, Posi_X+3, mOrder_U, mKnot_Vector_U);	\

//* Table Setup macro util
#define Compute_Blending_FunctionBj()											\
																				\
					Bj1 = Compute_Bik( v, Posi_Y  , mOrder_V, mKnot_Vector_V);	\
					Bj2 = Compute_Bik( v, Posi_Y+1, mOrder_V, mKnot_Vector_V);	\
					Bj3 = Compute_Bik( v, Posi_Y+2, mOrder_V, mKnot_Vector_V);	\
					Bj4 = Compute_Bik( v, Posi_Y+3, mOrder_V, mKnot_Vector_V);	\

//* Table Setup macro util
#define Compute_Blending_Tangent_FunctiondBi()										\
																					\
					dBi1 = Compute_dBik( u, Posi_X  , mOrder_U, mKnot_Vector_U);	\
					dBi2 = Compute_dBik( u, Posi_X+1, mOrder_U, mKnot_Vector_U);	\
					dBi3 = Compute_dBik( u, Posi_X+2, mOrder_U, mKnot_Vector_U);	\
					dBi4 = Compute_dBik( u, Posi_X+3, mOrder_U, mKnot_Vector_U);	\

//* Table Setup macro util
#define Compute_Blending_Tangent_FunctiondBj()										\
																					\
					dBj1 = Compute_dBik( v, Posi_Y  , mOrder_V, mKnot_Vector_V);	\
					dBj2 = Compute_dBik( v, Posi_Y+1, mOrder_V, mKnot_Vector_V);	\
					dBj3 = Compute_dBik( v, Posi_Y+2, mOrder_V, mKnot_Vector_V);	\
					dBj4 = Compute_dBik( v, Posi_Y+3, mOrder_V,mKnot_Vector_V);	\

//* Table Setup macro util
#define Compute_Point_Weight( PosiX, PosiY )							\
				Point_Weight =											\
						pBij[0] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+0 ) ) + pBij[1] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+0 ) ) + pBij[2] * GetWeightData( ClipX(PosiX+2), ClipY( PosiY+0 ) ) + pBij[3] * GetWeightData( ClipX(PosiX+3), ClipY( PosiY+0 ) ) +	\
						pBij[4] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+1 ) ) + pBij[5] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+1 ) ) + pBij[6] * GetWeightData( ClipX(PosiX+2), ClipY( PosiY+1 ) ) + pBij[7] * GetWeightData( ClipX(PosiX+3), ClipY( PosiY+1 ) ) +	\
						pBij[8] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+2 ) ) + pBij[9] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+2 ) ) + pBij[10]* GetWeightData( ClipX(PosiX+2), ClipY( PosiY+2 ) ) + pBij[11]* GetWeightData( ClipX(PosiX+3), ClipY( PosiY+2 ) ) +	\
						pBij[12]* GetWeightData( ClipX(PosiX+0), ClipY( PosiY+3 ) ) + pBij[13]* GetWeightData( ClipX(PosiX+1), ClipY( PosiY+3 ) ) + pBij[14]* GetWeightData( ClipX(PosiX+2), ClipY( PosiY+3 ) ) + pBij[15]* GetWeightData( ClipX(PosiX+3), ClipY( PosiY+3 ) );	\

//* Table Setup macro util
#define Compute_Normal_Point_Weight( PosiX, PosiY )						\
				Point_NormalU_Weight =									\
						pBdij[0] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+0 ) ) + pBdij[1] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+0 ) ) + pBdij[2] * GetWeightData( ClipX(PosiX+2), ClipY( PosiY+0 ) ) +  pBdij[3] * GetWeightData( ClipX(PosiX+3), ClipY( PosiY+0 ) ) +	\
						pBdij[4] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+1 ) ) + pBdij[5] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+1 ) ) + pBdij[6] * GetWeightData( ClipX(PosiX+2), ClipY( PosiY+1 ) ) +  pBdij[7] * GetWeightData( ClipX(PosiX+3), ClipY( PosiY+1 ) ) +	\
						pBdij[8] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+2 ) ) + pBdij[9] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+2 ) ) + pBdij[10]* GetWeightData( ClipX(PosiX+2), ClipY( PosiY+2 ) ) +  pBdij[11]* GetWeightData( ClipX(PosiX+3), ClipY( PosiY+2 ) ) +	\
						pBdij[12]* GetWeightData( ClipX(PosiX+0), ClipY( PosiY+3 ) ) + pBdij[13]* GetWeightData( ClipX(PosiX+1), ClipY( PosiY+3 ) ) + pBdij[14]* GetWeightData( ClipX(PosiX+2), ClipY( PosiY+3 ) ) +  pBdij[15]* GetWeightData( ClipX(PosiX+3), ClipY( PosiY+3 ) );		\
																																																																			\
				Point_NormalV_Weight =									\
						pBidj[0] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+0 ) ) + pBidj[1] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+0 ) ) + pBidj[2] * GetWeightData( ClipX(PosiX+2), ClipY( PosiY+0 ) ) +  pBidj[3] * GetWeightData( ClipX(PosiX+3), ClipY( PosiY+0 ) ) +	\
						pBidj[4] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+1 ) ) + pBidj[5] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+1 ) ) + pBidj[6] * GetWeightData( ClipX(PosiX+2), ClipY( PosiY+1 ) ) +  pBidj[7] * GetWeightData( ClipX(PosiX+3), ClipY( PosiY+1 ) ) +	\
						pBidj[8] * GetWeightData( ClipX(PosiX+0), ClipY( PosiY+2 ) ) + pBidj[9] * GetWeightData( ClipX(PosiX+1), ClipY( PosiY+2 ) ) + pBidj[10]* GetWeightData( ClipX(PosiX+2), ClipY( PosiY+2 ) ) +  pBidj[11]* GetWeightData( ClipX(PosiX+3), ClipY( PosiY+2 ) ) +	\
						pBidj[12]* GetWeightData( ClipX(PosiX+0), ClipY( PosiY+3 ) ) + pBidj[13]* GetWeightData( ClipX(PosiX+1), ClipY( PosiY+3 ) ) + pBidj[14]* GetWeightData( ClipX(PosiX+2), ClipY( PosiY+3 ) ) +  pBidj[15]* GetWeightData( ClipX(PosiX+3), ClipY( PosiY+3 ) );		\

#include <fbxfilesdk_nsend.h>

#endif


