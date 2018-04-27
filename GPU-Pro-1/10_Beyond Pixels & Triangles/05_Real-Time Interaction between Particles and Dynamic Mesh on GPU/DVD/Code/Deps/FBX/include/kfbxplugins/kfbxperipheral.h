/*!  \file kfbxperipheral.h
 */

#ifndef _FBXSDK_PERIPHERAL_H_
#define _FBXSDK_PERIPHERAL_H_

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

#include <klib/karrayul.h>
#include <kfbxplugins/kfbxobject.h>
#include <fbxfilesdk_nsbegin.h>

/**FBX SDK peripheral class
  * \nosubgrouping
  */

class KFBX_DLL KFbxPeripheral 
{
public:
	/**
	  * \name Constructor and Destructor
	  */
	//@{

	//!Constructor.
			 KFbxPeripheral();

    //!Destructor.
	virtual ~KFbxPeripheral();
	//@}

	/** Reset the peripheral to its initial state.
	  */
	virtual void Reset() = 0;

	/** Unload the content of pObject.
	  * \param pObject                 Object who's content is to be offloaded into 
	  * the peripheral storage area.
	  * \return                        \c true if the object content has been successfully transferred.
	  */
	virtual bool UnloadContentOf(KFbxObject* pObject) = 0;

	/** Load the content of pObject.
	  * \param pObject                 Object who's content is to be loaded from
	  * the peripheral storage area.
	  * \return                        \c true if the object content has been successfully transferred.
	  */
	virtual bool LoadContentOf(KFbxObject* pObject) = 0;

	/** Check if this peripheral can unload the given object content.
	  * \param pObject                 Object who's content has to be transferred.
	  * \return                        \c true if the peripheral can handle this object content AND/OR
	  * has enough space in its storage area.
	  */
	virtual bool CanUnloadContentOf(KFbxObject* pObject) = 0;

    /** Check if this peripheral can load the given object content.
    * \param pObject                  Object who's content has to be transferred.
    * \return                         \c true if the peripheral can handle this object content
    */
    virtual bool CanLoadContentOf(KFbxObject* pObject) = 0;

    /** Initialize the connections of an object
    * \param pObject                  Object on which the request for connection is done
    */
    virtual void InitializeConnectionsOf(KFbxObject* pObject) = 0;

    /** Uninitialize the connections of an object
    * \param pObject                 Object on which the request for deconnection is done
    */
    virtual void UninitializeConnectionsOf(KFbxObject* pObject) = 0;
};


// pre-defined offload peripherals
extern KFBX_DLL KFbxPeripheral* TMPFILE_PERIPHERAL;
extern KFBX_DLL KFbxPeripheral* NULL_PERIPHERAL;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_PERIPHERAL_H_


