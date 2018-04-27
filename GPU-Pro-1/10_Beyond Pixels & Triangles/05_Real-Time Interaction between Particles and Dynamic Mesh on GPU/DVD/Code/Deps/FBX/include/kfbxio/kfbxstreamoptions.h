/*!  \file kfbxstreamoptions.h
 */
 
#ifndef _FBXSDK_KFBXSTREAMOPTIONS_H_
#define _FBXSDK_KFBXSTREAMOPTIONS_H_

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

#include <kfbxplugins/kfbxsdkmanager.h>
#include <fbxfilesdk_nsbegin.h>


/**	\brief This class provides the structure to build a KFbx Stream Option.
  *	This class is a composite class that contains stream options management services. 
  * The content of a Kfbx Stream Option is stored in a property (KFbxProperty).
  */

class KFBX_DLL KFbxStreamOptions : public KFbxObject
{
	KFBXOBJECT_DECLARE(KFbxStreamOptions,KFbxObject);

public:


	/** Reset all the options to default value
	*/
	virtual void Reset();

	/** Get a Stream Option by Stream Option Name.
	  * \param pName     The name of the Stream Option
	  * \return          A KFbxProperty if the name is valid.
	  * \remarks         In the last case, an assert is raised
	  */
	KFbxProperty GetOption(KString& pName);


	/** Get a Stream Option by Stream Option Name.
	  * \return     A KFbxProperty if the name is valid.
	  * \remarks    In the last case, an assert is raised
	  */
	KFbxProperty GetOption(const char* pName);
	

	/** Set a Stream Option by Stream Option Name and a Value.
	  * \param pName      Name of the option where a change is needed.
	  * \param pValue     Value to be set.
	  * \return           \c true if the Stream Option was found and the value has been set.
	  */
	template <class T> inline bool  SetOption(KString& pName, T const &pValue )
	{
		 
		KFbxProperty lProperty=this->GetOption(pName);
		if(lProperty.IsValid() && FbxTypeOf(pValue) == lProperty.GetPropertyDataType().GetType())
		{
			lProperty.Set((void*)&pValue, FbxTypeOf(pValue));
			return true;
		}
		return false;
	}

	/** Set a Stream Option by Stream Option Name and a Value.
	  * \param pName     Name of the option where a change is needed.
	  * \param pValue    Value to be set.
	  * \return          \c true if the Stream Option was found and the value has been set.
	  */
	template <class T> inline bool  SetOption(const char* pName, T const &pValue )
	{
		 
		KFbxProperty lProperty=this->GetOption(pName);
		if(lProperty.IsValid() && FbxTypeOf(pValue) == lProperty.GetPropertyDataType().GetType())
		{
			lProperty.Set((void*)&pValue, FbxTypeOf(pValue));
			return true;
		}
		return false;
	}

	/** Set a Stream Option by a Property (KFbxProperty).
	  * \param pProperty     Property containing the value to be set.
	  * \return              \c true if the Property has been set, otherwise \c false.
	  */
	bool SetOption(KFbxProperty pProperty);

	/** Copies the properties of another KFbxStreamOptions.
	  * \param pKFbxStreamOptionsSrc     Contains the properties to be copied
	  */
	bool CopyFrom(const KFbxStreamOptions* pKFbxStreamOptionsSrc);

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS


protected:

	KFbxStreamOptions(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxStreamOptions();
	virtual void Destruct(bool pRecursive, bool pDependents);

public:
	//clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};
#include <fbxfilesdk_nsend.h>
#endif
