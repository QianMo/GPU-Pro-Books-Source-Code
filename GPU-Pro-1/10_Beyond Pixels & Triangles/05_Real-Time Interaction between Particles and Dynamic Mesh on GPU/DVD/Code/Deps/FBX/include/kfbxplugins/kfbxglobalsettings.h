/*!  \file kfbxglobalsettings.h
 */

#ifndef _KFbxGlobalSettings_h
#define _KFbxGlobalSettings_h

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

#include <klib/kstring.h>
#include <klib/karrayul.h>
#include <klib/ktime.h>
#include <klib/kerror.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <kfbxplugins/kfbxaxissystem.h>
#include <kfbxplugins/kfbxsystemunit.h>
#include <kfbxplugins/kfbxobject.h>

#include <fbxfilesdk_nsbegin.h>


/** \brief This class contains functions for accessing global settings.
  * \nosubgrouping
  */
class KFBX_DLL KFbxGlobalSettings : public KFbxObject
{
	KFBXOBJECT_DECLARE(KFbxGlobalSettings,KFbxObject);

public:
	//! Assignment operator.
	const KFbxGlobalSettings& operator=(const KFbxGlobalSettings& pGlobalSettings);        

    /** 
	  * \name Axis system
	  */
	//@{
    
	/** Set the coordinate system for the scene.
	  * \param pAxisSystem     Coordinate system defined by the class kFbxAxisSystem.
	  */
    void SetAxisSystem(const KFbxAxisSystem& pAxisSystem);
    
	/** Get the scene's coordinate system.
	  * \return     The coordinate system of the current scene, defined by the class kFbxAxisSystem.
	  */
    KFbxAxisSystem GetAxisSystem();
    //@}

    /** 
	  * \name System Units
	  */
	//@{

	/** Set the unit of measurement used by the system.
	  * \param pOther     A unit of measurement defined by the class kFbxSystemUnit. 
	  */
    void SetSystemUnit(const KFbxSystemUnit& pOther);
    
	/** Get the unit of measurement used by the system.
	  * \return     The unit of measurement defined by the class kFbxSystemUnit.     
	  */
    KFbxSystemUnit GetSystemUnit() const;

    //@}

protected:

	/**
	  * \name Properties
	  */
	//@{
		KFbxTypedProperty<fbxInteger1>	UpAxis;
		KFbxTypedProperty<fbxInteger1>	UpAxisSign;

		KFbxTypedProperty<fbxInteger1>	FrontAxis;
		KFbxTypedProperty<fbxInteger1>	FrontAxisSign;

		KFbxTypedProperty<fbxInteger1>	CoordAxis;
		KFbxTypedProperty<fbxInteger1>	CoordAxisSign;

		KFbxTypedProperty<fbxDouble1>	UnitScaleFactor;

	//@}

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

    //
    // KFbxObject
    //
protected:
	
	virtual void Construct(const KFbxGlobalSettings* pFrom);
	virtual bool ConstructProperties(bool pForceSet);
	
private:

	KFbxGlobalSettings(KFbxSdkManager& pManager, char const* pName);

	~KFbxGlobalSettings();

    void AxisSystemToProperties();
    void PropertiesToAxisSystem();

    void Init();

    KFbxAxisSystem mAxisSystem;
    friend class KFbxScene;    

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS 
};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _KFbxGlobalSettings_h


