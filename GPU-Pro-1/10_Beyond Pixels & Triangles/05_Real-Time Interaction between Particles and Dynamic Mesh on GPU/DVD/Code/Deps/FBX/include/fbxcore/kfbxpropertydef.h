#ifndef _FBXSDK_KFBXPROPERTYDEF_H_
#define _FBXSDK_KFBXPROPERTYDEF_H_

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

#include <fbxfilesdk_nsbegin.h>

	typedef int					kFbxPropertyId;
	const   kFbxPropertyId		kFbxProperyIdNull = -1;
	const   kFbxPropertyId		kFbxProperyIdRoot = 0;

	enum	KFbxInheritType		{ eFbxOverride=0,eFbxInherit=1,eFbxDeleted=2 } ;

    enum kFbxConnectionType
    { 
	    eFbxConnectionNone				= 0,

		// System or user
		eFbxConnectionSystem			= 1 << 0,
	    eFbxConnectionUser				= 1 << 1,
	    eFbxConnectionSystemOrUser		= eFbxConnectionUser | eFbxConnectionSystem,

		// Type of Link
	    eFbxConnectionReference			= 1 << 2,
	    eFbxConnectionContains			= 1 << 3,
	    eFbxConnectionData				= 1 << 4,
	    eFbxConnectionLinkType			= eFbxConnectionReference | eFbxConnectionContains | eFbxConnectionData,

	    eFbxConnectionDefault			= eFbxConnectionUser | eFbxConnectionReference,


	    eFbxConnectionUnidirectional    = 1 << 7
    };

	class FbxPropertyFlags
	{
	public:
			enum eFbxPropertyFlags
			{
				eNO_FLAG    	= 0,
				eANIMATABLE 	= 1, 
				eUSER       	= 1<<1,
				eTEMPORARY  	= 1<<2,  // System property
				ePUBLISHED		= 1<<3, 
				ePSTATIC		= 1<<4, 

				eNOT_SAVABLE	= 1<<5,
				eHIDDEN     	= 1<<6,

				eUI_DISABLED	= 1<<7,  // for dynamic UI
				eUI_GROUP       = 1<<8,  // for dynamic UI
				eUI_BOOLGROUP   = 1<<9,  // for dynamic UI
				eUI_EXPANDED    = 1<<10, // for dynamic UI
				eUI_NOCAPTION   = 1<<11, // for dynamic UI
				eUI_PANEL     = 1<<12  // for dynamic UI

			};

			// VC6 Does not like static variables that are initialized in the header
			// and there is no kfbxpropertydef.cxx file.
			inline static int GetFlagCount() { return 14; }

			inline static eFbxPropertyFlags AllFlags()
			{
				eFbxPropertyFlags lAllFlags = eNO_FLAG;

				for( int i = 0; i < GetFlagCount()-1; ++i )
					lAllFlags = (eFbxPropertyFlags) ( (lAllFlags << 1) | 1 );
				
				return lAllFlags;
			}
	};

	/**************************************************************************
    * Filter management
    **************************************************************************/
	class	KFbxConnectionPoint;
	typedef int kFbxFilterId;

	/**	\brief Class to manage ConnectFilter.
	* \nosubgrouping
	*/
	class KFBX_DLL KFbxConnectionPointFilter
	{
		// CONSTRUCTOR/DESTRUCTOR
		/**
		* \name Constructor and Destructor
		*/
		//@{
	public: 
		//! Constructor
		KFbxConnectionPointFilter() { }
		//! Destructor
		virtual ~KFbxConnectionPointFilter();
		//@}
	public:
		/**
		* \name ConnectFilter management
		*/
		//@{

		//! Return reference ConnectionPoint filter.
		virtual KFbxConnectionPointFilter*		Ref();
		//! Cancel reference
		virtual void							Unref();

		//! Get unique filter ID
		virtual kFbxFilterId					GetUniqueId() const { return 0; }

		//! Judge if the given Connection Point is valid
		virtual bool							IsValid				(KFbxConnectionPoint*	pConnect) const;
		//! Judge if the given Connection Point is a valid connection
		virtual bool							IsValidConnection	(KFbxConnectionPoint*	pConnect,kFbxConnectionType pType) const;
		//! Judge if it is equal with the given  ConnectionPoint filter. 
		virtual bool							IsEqual				(KFbxConnectionPointFilter*	pConnectFilter)	const;

		//@}
	};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_Document_H_




