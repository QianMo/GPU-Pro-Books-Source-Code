/*!  \file kfbxlibrary.h
 */

#ifndef _FBXSDK_LIBRARY_H_
#define _FBXSDK_LIBRARY_H_

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

#include <fbxcore/fbxcollection/kfbxdocument.h>
#include <kfbxplugins/kfbxobjectfilter.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxWriterFbx6;
class KFbxLocalizationManager;
class KFbxCriteria;

/** \brief This object represents a shading node library.
  * \nosubgrouping
  */
class KFBX_DLL KFbxLibrary : public KFbxDocument
{
    KFBXOBJECT_DECLARE(KFbxLibrary,KFbxDocument);

    public:

        //! Get a handle on the parent library if exists.
        KFbxLibrary* GetParentLibrary(void) const;

        void SystemLibrary(bool pSystemLibrary);
        bool IsSystemLibrary() const;

        /**
        *  The prefix must not include the dash and language code, nor
        *  must it contain the extension.  But it can contain a folder,
        *  or sub-folder, if you want, such as:
        *
        *  locales/mydocloc
        *
        *  This will be resolved using the XRef Manager, with priority
        *  given to the library's .fbm folder, if it has one.
        */
        void LocalizationBaseNamePrefix(const char* pPrefix);
        KString LocalizationBaseNamePrefix() const;

        // =======================================================================
        //
        // Sub library
        //
        // =======================================================================

        //! Adds a sub Library
        bool AddSubLibrary(KFbxLibrary* pSubLibrary);

        //! Removes a sub Library
        bool RemoveSubLibrary(KFbxLibrary* pSubLibrary);

        //! Gets the total number of sub libraries
        int GetSubLibraryCount(void) const;

        //! Sub library accessor
        KFbxLibrary* GetSubLibrary(int pIndex) const;

        /** Get the number of sub libraries that contains shading node according
          * to their implementations.
          *
          * \param pCriteria filtering criteria that identifies the kind of
          * implementations to take into account.
          *
          * \returns the number of sub-libraries corresponding to the filtering parameters
          */
        int GetSubLibraryCount(
            const KFbxImplementationFilter & pCriteria
        ) const;

        /** Get a handle on the (pIndex)th sub-library that corresponds to the given filtering parameters.
          * \param pIndex
          * \param pCriteria filtering criteria that identifies the kind of
          * implementations to take into account.
          *
          * \returns a handle on the (pIndex)th sub-library that corresponds to the given filtering parameters
          */
        KFbxLibrary* GetSubLibrary(
            int pIndex,
            const KFbxImplementationFilter & pCriteria
        ) const;


        KFbxObject* CloneAsset( KFbxObject* pToClone, KFbxObject* pOptionalDestinationContainer = NULL) const;

        //
        // Returns a criteria filter which can be used to ... filter ... objects
        // when iterating items in the library; only real 'assets' will be returned,
        // rather than FBX support objects.  This includes, at this time,
        // lights, environments, materials and textures (maps)
        //
        // This is typically used to IMPORT from a library.
        //
        static KFbxCriteria GetAssetCriteriaFilter();

        //
        // Returns a filter which should be used when cloning / exporting objects --
        // this filters out objects that should stay in the asset library.
        //
        // This is typically used to EXPORT from a library (or CLONE from a library,
        // although CloneAsset does all the nitty gritty for you)
        //
        static KFbxCriteria GetAssetDependentsFilter();

        /**
         * Transfer ownership from the src library to us for all assets meeting the filter.
         * Name clashing and other details are assumed to have been resolved beforehand.
         *
         * External asset files required by the assets are copied over -- not moved.  It's
         * up to the owner of pSrcLibrary to clean up (maybe they can't; the files may
         * be on a read-only transport).  If this document hasn't been commited yet, then
         * the assets WON'T be copied.
         *
         * Returns true if no assets meeting the filter were skipped.  If there are no
         * assets meeting the filter, then true would be returned, as nothing was skipped.
         *
         * This may leave the source library in an invalid state, if for instance you
         * decide to transfer texture objects to our library, but you keep materials in
         * the source library.
         *
         * To safeguard against this, the transfer will disconnect objects, and you'd thus
         * be left with materials without textures.
         *
         * When transfering an object, all its dependents come with it.  If you move
         * a material, it WILL grab its textures.  Just not the other way around.
         *
         **/
        bool ImportAssets(KFbxLibrary* pSrcLibrary);
        bool ImportAssets(KFbxLibrary* pSrcLibrary, const KFbxCriteria&);


        /** Return a new instance of a member of the library.
          * This instantiates the first object found that matches the filter.
          * \param pFBX_TYPE The type of member
          * \param pFilter A user specified filter
          * \param pRecurse Check sublibraries
          * \param pOptContainer Optional container for the cloned asset
          * \return A new instance of the member. Note that the new member is not inserted into this library.
          */
        template < class T > T* InstantiateMember( T const* pFBX_TYPE, const KFbxObjectFilter& pFilter, bool pRecurse = true, KFbxObject* pOptContainer = NULL);


    // =======================================================================
    //
    // Localization
    //
    // =======================================================================
    /** Get the localization manager for the library.
      */

        KFbxLocalizationManager & GetLocalizationManager() const;

        /** Localization helper function. Calls the FBX SDK manager implementation.
          * sub-classes which manage their own localization could over-ride this.
          * \param pID the identifier for the text to localize.
          * \param pDefault the default text. Uses pID if NULL.
          * \return the potentially localized text. May return the parameter passed in.
        */
        virtual const char * Localize( const char * pID, const char * pDefault = NULL ) const;

    // =======================================================================
    //
    // Shading Node
    //
    // =======================================================================

        //! Adds a shading node
        bool AddShadingObject(KFbxObject* pShadingObject);

        //! Removes a shading node
        bool RemoveShadingObject(KFbxObject* pShadingObject);

        //! Gets the total number of shading nodes
        int GetShadingObjectCount(void) const;

        //! Shading node accessor
        KFbxObject* GetShadingObject(int pIndex) const;

        /** Get the number of shading nodes according to their implementations.
          *
          * \param pCriteria filtering criteria that identifies the kind of
          * implementations to take into account.
          *
          * \returns the number of shading nodes corresponding to the filtering parameters
          */
        int GetShadingObjectCount(
            const KFbxImplementationFilter & pCriteria
        ) const;

        /** Get a handle on the (pIndex)th sub-library that corresponds to the given filtering parameters.
          * \param pIndex
          * \param pCriteria filtering criteria that identifies the kind of
          * implementations to take into account.
          *
          * \returns a handle on the (pIndex)th shading node that corresponds to the given filtering parameters
          */
        KFbxObject* GetShadingObject(
            int pIndex,
            const KFbxImplementationFilter & pCriteria
        ) const;

        ///////////////////////////////////////////////////////////////////////////////
        //
        //  WARNING!
        //
        //  Anything beyond these lines may not be documented accurately and is
        //  subject to change without notice.
        //
        ///////////////////////////////////////////////////////////////////////////////
        #ifndef DOXYGEN_SHOULD_SKIP_THIS

        public:
                // Clone
            virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

        protected:

            // Constructor / Destructor
            KFbxLibrary(KFbxSdkManager& pManager, char const* pName);
            ~KFbxLibrary();
            virtual void Destruct   (bool pRecursive, bool pDependents);

            virtual void Construct(const KFbxLibrary* pFrom);

            mutable KFbxLocalizationManager * mLocalizationManager;

            friend class KFbxWriterFbx6;

        #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
    };


    template < class T > T* KFbxLibrary::InstantiateMember( T const* pFBX_TYPE, const KFbxObjectFilter& pFilter, bool pRecurse, KFbxObject* pOptContainer )
    {
        // first check all materials in the library.
        int i = 0;
        for( i = 0; i < GetMemberCount( FBX_TYPE(T) ); ++i )
        {
            T* lObject = GetMember( FBX_TYPE(T), i );
            if( pFilter.Match(lObject) )
                return KFbxCast<T>(CloneAsset(lObject,pOptContainer));
        }

        if( pRecurse )
        {
            // then check all materials in each sub-library.
            for( i = 0; i < GetMemberCount( FBX_TYPE(KFbxLibrary) ); ++i )
            {
                KFbxLibrary* lLibrary = GetMember( FBX_TYPE(KFbxLibrary), i );
                T* lClonedObject = lLibrary->InstantiateMember( pFBX_TYPE, pFilter, pRecurse, pOptContainer );
                if( lClonedObject )
                    return lClonedObject;
            }
        }

        return NULL;
    }


  /** \class KFbxEventPopulateSystemLibrary
    * \nosubgrouping
    * \brief Library events are triggered when an application requires specific library content
    */
    class KFBX_DLL KFbxEventPopulateSystemLibrary : public KFbxEvent<KFbxEventPopulateSystemLibrary>
    {
      KFBXEVENT_DECLARE(KFbxEventPopulateSystemLibrary)

        public:			
        /**
          *\name Constructor / Destructor
          */
        //@{

			//!Constructor.
            KFbxEventPopulateSystemLibrary(KFbxLibrary *pLibrary) { mLibrary = pLibrary; }

        //@}
       
            public:
		/**
          *\name Member access
          */
        //@{
				//!Retrieve the library.
                inline KFbxLibrary* GetLibrary() const { return mLibrary; }
            private:
                KFbxLibrary*    mLibrary;
        //@}
    };

    /** \class KFbxEventUpdateSystemLibrary
	  * \nosubgrouping
      * \brief This library event is triggered when an application requires an update to a specific library
      */
    class KFBX_DLL KFbxEventUpdateSystemLibrary : public KFbxEvent<KFbxEventUpdateSystemLibrary>
    {
        KFBXEVENT_DECLARE(KFbxEventUpdateSystemLibrary)
        public:
	  /**
        *\name Constructor and Destructor
        */
        //@{

			//!Constructor.
            KFbxEventUpdateSystemLibrary(KFbxLibrary *pLibrary) { mLibrary = pLibrary; }

		//@}

	  /**
        *\name Member access
        */
        //@{

			//! Retrieve the library.
            inline KFbxLibrary* GetLibrary() const { return mLibrary; }

		//@}
        private:
                KFbxLibrary*    mLibrary;
    };

    /**
     * This library event is used by the asset building system to request that a
     * library provides its localization information.  This is used by the asset
     * builder.
     *
     * This is used by libraries that do not want to include their
     * data in the main asset library's xlf file, but rather as a
     * separate file.  These libraries are then responsible for
     * loading their files back.
    */
    class KFBX_DLL KFbxEventWriteLocalization : public KFbxEvent<KFbxEventWriteLocalization>
    {
        KFBXEVENT_DECLARE(KFbxEventWriteLocalization)
        public:
            KFbxEventWriteLocalization(KFbxLibrary *pAssetLibrary) { mAssetLibrary = pAssetLibrary; }

            inline KFbxLibrary* GetLibrary() const { return mAssetLibrary; }

        private:
            KFbxLibrary*    mAssetLibrary;
    };

    ///////////////////////////////////////////////////////////////////////////////
    //
    //  WARNING!
    //
    //  Anything beyond these lines may not be documented accurately and is
    //  subject to change without notice.
    //
    ///////////////////////////////////////////////////////////////////////////////
    #ifndef DOXYGEN_SHOULD_SKIP_THIS

    // Some assets correspond 1:1 with files containing their actual data.
    class KFBX_DLL KFbxEventMapAssetFileToAssetObject : public KFbxEvent<KFbxEventMapAssetFileToAssetObject>
    {
        KFBXEVENT_DECLARE(KFbxEventMapAssetFileToAssetObject)
        public:

            // pFile - The asset file to query
            KFbxEventMapAssetFileToAssetObject(const char* pFile) :
                mFilePath( pFile ), 
                mAsset(NULL)
                {}

            // listeners query this
            inline const char* GetFilePath() const { return mFilePath; }

            // listeners set this
            mutable KFbxObject* mAsset;

        private:
            KString mFilePath;
    };

    #endif //DOXYGEN_SHOULD_SKIP_THIS

#include <fbxfilesdk_nsend.h>

#endif //_FBXSDK_LIBRARY_H_
