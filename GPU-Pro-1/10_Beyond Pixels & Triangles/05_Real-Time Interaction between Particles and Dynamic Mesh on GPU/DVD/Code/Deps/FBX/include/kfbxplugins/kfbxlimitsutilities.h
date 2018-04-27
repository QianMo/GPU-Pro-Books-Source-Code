/*!  \file kfbxlimitsutilities.h
 */

#ifndef _FBXSDK_LIMITS_UTILITIES_H_
#define _FBXSDK_LIMITS_UTILITIES_H_

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
#include <kfbxplugins/kfbxnodelimits.h>
#include <kfbxplugins/kfbxnode.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxLimitsUtilities
{
public:
	typedef enum 
	{
		eT,
		eR,
		eS,
	} ELimitType;

	typedef enum 
	{ 
		eROTATION_TYPE_QUATERNION, 
		eROTATION_TYPE_EULER, 
	} ERotationType ;

	typedef enum 
	{ 
		eROTATION_CLAMP_TYPE_RECTANGULAR, 
		eROTATION_CLAMP_TYPE_ELIPSOID, 
	} ERotationClampType ;


	KFbxLimitsUtilities(KFbxNodeLimits* pLimits);
	KFbxNodeLimits* mLimits;

	void SetAuto(ELimitType pType, bool pAuto);
	bool GetAuto(ELimitType pType);

	void SetEnable(ELimitType pType, bool pEnable);
	bool GetEnable(ELimitType pType);

	void SetDefault(ELimitType pType, KFbxVector4 pDefault);
	KFbxVector4 GetDefault(ELimitType pType);

	void SetMin(ELimitType pType, KFbxVector4 pMin);
	KFbxVector4 GetMin(ELimitType pType);

	void SetMax(ELimitType pType, KFbxVector4 pMax);
	KFbxVector4 GetMax(ELimitType pType);

	void SetRotationType(ERotationType pType);
	ERotationType GetRotationType();

	void SetRotationClampType(ERotationClampType pType);
	ERotationClampType GetRotationClampType();

	void SetRotationAxis(KFbxVector4 pRotationAxis);
	KFbxVector4 GetRotationAxis();

	void SetAxisLength(double pLength);
	double GetAxisLength();

	void UpdateAutomatic(KFbxNode* pNode);
	KFbxVector4 GetEndPointTranslation(KFbxNode* pNode);
	KFbxVector4 GetEndSite(KFbxNode* pNode);

	double mAxisLength; 
};


#include <fbxfilesdk_nsend.h>

#endif // #ifndef _KFbxLimitsUtilites_h



