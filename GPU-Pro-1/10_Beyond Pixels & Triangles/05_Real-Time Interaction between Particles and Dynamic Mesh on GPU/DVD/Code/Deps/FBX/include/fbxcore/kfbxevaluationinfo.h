#ifndef _FBXSDK_EVALUATION_INFO_H_
#define _FBXSDK_EVALUATION_INFO_H_

/**************************************************************************************

 Copyright ?2001 - 2008 Autodesk, Inc. and/or its licensors.
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

#include <kfbxplugins/kfbxplug.h>
#include <kfbxplugins/kfbxtypes.h>

#include <fbxfilesdk_nsbegin.h>

	typedef int kFbxEvaluationId;

	/***********************************************
	  KFbxEvaluationInfo
	************************************************/
/**	\brief This class contains evaluation info.
* \nosubgrouping
*/
class KFBX_DLL KFbxEvaluationInfo {
	// Overridable Test functions
public:
	/**
	* \name Create and Destroy
	*/
	//@{
	/** Create an instance.
	* \return The pointer to the created instance.
	*/
	static inline KFbxEvaluationInfo* Create(KFbxSdkManager * )
	{
		return new KFbxEvaluationInfo();
	}

public:
	//!Destroy an allocated version of the KFbxEvaluationInfo.
	inline void Destroy() 
	{ 
		delete this; 
	}
	//@}

	/**
	* \name Set and Change the evaluation info
	*/
	//@{

	/** Get time
	* \return The time value.
	*/
	inline fbxTime GetTime() const					{ return mTime; }
	/** Set time 
	* \param pTime The given time value .
	*/
	inline void	   SetTime(fbxTime pTime)			{ mTime=pTime; }

	/** Get evaluation ID
	* \return The evaluation ID.
	*/
	inline kFbxEvaluationId GetEvaluationId() const	{ return mEvaluationId; }

	//! Update evaluation ID, the value get one more every time.
	inline void				UpdateEvaluationId()	{ mEvaluationId++; }

	//@}
protected:
	inline KFbxEvaluationInfo()
		: mEvaluationId(0)
	{
	}

	inline ~KFbxEvaluationInfo()
	{
	}


private:
	fbxTime				mTime;
	kFbxEvaluationId	mEvaluationId;
};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_Document_H_


