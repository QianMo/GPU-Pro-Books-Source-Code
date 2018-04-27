/*!  \file kfbxmanipulators.h
 */
#ifndef _FBXSDK_MANIPULATORS_H_
#define _FBXSDK_MANIPULATORS_H_

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

#include <kfbxplugins/kfbxobject.h>
#include <kfbxplugins/kfbxcamera.h>
#include <kfbxmath/kfbxvector2.h>
#include <kfbxmath/kfbxvector4.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxCamManip_state;

/** This class can be used to provide basic camera manipulation in any program using this library.
  * \nosubgrouping
  */
class KFBX_DLL KFbxCameraManipulator : public KFbxObject
{
	KFBXOBJECT_DECLARE(KFbxCameraManipulator, KFbxObject);

public:
    /** Set the camera used for the manipulation.
	*	\param	pCamera				Camera that will be used for the manipulation.
	*	\param	pValidateLookAtPos	If TRUE, LookAt position will be aligned with the camera orientation. */
	void SetCamera(const KFbxCamera& pCamera, bool pValidateLookAtPos);

    /** Set the manipulator up vector relative to the scene.
	*	\param	pUpVector	Vector defining the up direction of the scene. */
	void SetUpVector(const KFbxVector4& pUpVector);

    /** Change camera position and look at to frame all objects.
	*	\param	pScene	The scene containing the elements to frame. */
	void FrameAll(const KFbxScene& pScene);

    /** Change camera position and look at to frame all selected objects.
	*	\param	pScene	The scene containing the elements to frame. */
	void FrameSelected(const KFbxScene& pScene);

    /** Begin orbit manipulation around camera's look at.
	*	\param	pMouseX	Horizontal position of the mouse cursor.
	*	\param	pMouseY	Vertical position of the mouse cursor.
	*	\return			If TRUE, orbit manipulation successfully initialized. */
	bool OrbitBegin(int pMouseX, int pMouseY);

    /** Notify orbit manipulation of latest input.
	*	\param	pMouseX	Horizontal position of the mouse cursor.
	*	\param	pMouseY	Vertical position of the mouse cursor.
	*	\return			TRUE if orbit manipulation was previously initialized successfully. */
	bool OrbitNotify(int pMouseX, int pMouseY);

    /** End orbit manipulation. */
	void OrbitEnd();

    /** Begin dolly manipulation.
	*	\param	pMouseX	Horizontal position of the mouse cursor.
	*	\param	pMouseY	Vertical position of the mouse cursor.
	*	\return			If TRUE, dolly manipulation successfully initialized. */
	bool DollyBegin(int pMouseX, int pMouseY);

    /** Notify dolly manipulation of latest input.
	*	\param	pMouseX	Horizontal position of the mouse cursor.
	*	\param	pMouseY	Vertical position of the mouse cursor.
	*	\return			TRUE if dolly manipulation was previously initialized successfully. */
	bool DollyNotify(int pMouseX, int pMouseY);

    /** End dolly manipulation. */
	void DollyEnd();

    /** Begin pan manipulation.
	*	\param	pMouseX	Horizontal position of the mouse cursor.
	*	\param	pMouseY	Vertical position of the mouse cursor.
	*	\return			If TRUE, pan manipulation successfully initialized. */
	bool PanBegin(int pMouseX, int pMouseY);

    /** Notify pan manipulation of latest input.
	*	\param	pMouseX	Horizontal position of the mouse cursor.
	*	\param	pMouseY	Vertical position of the mouse cursor.
	*	\return			TRUE if pan manipulation was previously initialized successfully. */
	bool PanNotify(int pMouseX, int pMouseY);

    /** End pan manipulation. */
	void PanEnd();

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
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

	//! Assignment operator.
	KFbxCameraManipulator& operator=(KFbxCameraManipulator const& pCamManip);

private:
	KFbxCameraManipulator(KFbxSdkManager& pManager, char const* pName);
	~KFbxCameraManipulator();

	KFbxCamManip_state* mState;
#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS 
};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_MANIPULATORS_H_


