/*!  \file kfbxDocumentInfo.h
 */
#ifndef _FBXSDK_DOCUMENT_INFO_H_
#define _FBXSDK_DOCUMENT_INFO_H_

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
#include <kaydara.h>
#ifndef KFBX_DLL 
	#define KFBX_DLL K_DLLIMPORT
#endif

#include <klib/kstring.h>
#include <kfbxplugins/kfbxobject.h>

#ifndef MB_FBXSDK
	#include <kbaselib_nsuse.h>
#endif

class KStreamFbx; 

#include <fbxfilesdk_nsbegin.h>

	class KFbxThumbnail;
	class KFbxSdkManager;

	/** Contains scene thumbnail and user-defined summary data.
	  */
	class KFBX_DLL KFbxDocumentInfo : public KFbxObject
	{
		KFBXOBJECT_DECLARE(KFbxDocumentInfo,KFbxObject);

		/**
		  * \name Public properties
		  */
		//@{
		public:
			KFbxTypedProperty<fbxString>	LastSavedUrl;
			KFbxTypedProperty<fbxString>	Url;

            //! Parent property for all 'creation-related' properties; these properties
            // should be set once, when the file is created, and should be left alone
            // on subsequent save/reload operations.
            // 
            // Below are the default properties, but application vendors can add new
            // properties under this parent property.
            KFbxProperty                    Original;

            KFbxTypedProperty<fbxString>    Original_ApplicationVendor;     // "CompanyName"
            KFbxTypedProperty<fbxString>    Original_ApplicationName;       // "UberGizmo"
            KFbxTypedProperty<fbxString>    Original_ApplicationVersion;    // "2009.10"

            KFbxTypedProperty<fbxString>    Original_FileName;              // "foo.bar"

            //! Date/time should be stored in GMT.
            KFbxTypedProperty<fbxDateTime>  Original_DateTime_GMT;

            //! Parent property for all 'lastmod-related' properties; these properties
            // should be updated everytime a file is saved.
            // 
            // Below are the default properties, but application vendors can add new
            // properties under this parent property.
            // 
            // It is up to the file creator to set both the 'Original' and
            // 'Last Saved' properties.
            KFbxProperty                    LastSaved;

            KFbxTypedProperty<fbxString>    LastSaved_ApplicationVendor;
            KFbxTypedProperty<fbxString>    LastSaved_ApplicationName;
            KFbxTypedProperty<fbxString>    LastSaved_ApplicationVersion;

            //! Date/time should be stored in GMT.
            KFbxTypedProperty<fbxDateTime>  LastSaved_DateTime_GMT;

			/**
			 * This property is set to point to the .fbm folder created when 
			 * reading a .fbx file with embedded data.  It is not saved in 
			 * the .fbx file. 
          */
			KFbxTypedProperty<fbxString>	EmbeddedUrl;
		//@}


		/** User-defined summary data.
		  * These fields are filled by the user to identify/classify
		  * the files.
		  */
		//@{
		public:
			//! Title.
			KString mTitle;

			//! Subject.
			KString mSubject;

			//! Author
			KString mAuthor;

			//! Keywords.
			KString mKeywords;

			//! Revision.
			KString mRevision;

			//! Comment.
			KString mComment;
		//@}

		/** Scene Thumbnail.
		  */
		//@{

			/** Get the scene thumbnail.
			  * \return Pointer to the thumbnail.
			  */
			KFbxThumbnail* GetSceneThumbnail();

			/** Set the scene thumbnail.
			  * \param pSceneThumbnail Pointer to a thumbnail object.
			  */
			void SetSceneThumbnail(KFbxThumbnail* pSceneThumbnail);
		//@}

		/** Clear the content.
		  * Reset all the strings to the empty string and clears 
		  * the pointer to the thumbnail.
		  */
		void Clear();

		//! assignment operator.
		KFbxDocumentInfo& operator=(const KFbxDocumentInfo& pDocumentInfo);

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

			protected:
				//! Object Contructors and Destructor.
				KFbxDocumentInfo(KFbxSdkManager& pManager,char const *pName);
				~KFbxDocumentInfo();
				virtual void Construct	(const KFbxDocumentInfo* pFrom);
				virtual void Destruct	(bool pRecursive, bool pDependents);
				bool	ConstructProperties(bool pForceSet);

				KFbxThumbnail*	mSceneThumbnail;

				friend class KStreamFbx;
			#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS 
	};

	typedef KFbxDocumentInfo* HKFbxDocumentInfo;

	// Backward compatibility
	// --------------------------------------------------------------

//	typedef KFbxDocumentInfo* HKFbxDocumentInfo;
//	typedef KFbxDocumentInfo* HKFbxDocumentInfo;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_SCENE_INFO_H_


