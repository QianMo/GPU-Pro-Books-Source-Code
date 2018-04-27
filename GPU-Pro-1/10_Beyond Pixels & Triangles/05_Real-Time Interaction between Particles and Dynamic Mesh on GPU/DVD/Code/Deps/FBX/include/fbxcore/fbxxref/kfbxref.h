#ifndef _KFBXKXREF_H
#define _KFBXKXREF_H

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

#include <fbxcore/fbxxref/fbxxref_def.h>

#include <kfbxplugins/kfbxsdkmanager.h>

namespace FBXXREF_NAMESPACE {

    class KXRefManagerProject;
    class KFbxDocument;


    /** This class manages external references to files.
      */
    class FBXXREF_DLL KFbxXRefManager {
        public:
            KFbxXRefManager();
            virtual ~KFbxXRefManager();

            /**
              * \name Pre-defined Project Types
              */
            //@{

                //! This project represents a Url for storing temporary files.
                static const char* sTemporaryFileProject;

                //! This project represents a Url for configuration files.
                static const char* sConfigurationProject;

                //! This project represents a url for storing localization files (not part of the asset lib)
                static const char* sLocalizationProject;

                /** This project is used for creating the .fbm folders used by
                *   embedded ressources in .fbx files.
                *  
                *   When not set, or if the folder is not writable, the .fbm
                *   folder is created alongside the .fbx file.
                *  
                *   If we cannot write in that folder, we look at the
                *   sTemporaryFileProject location.  If that's not set, or not
                *   writable, we use the operating system's temp folder
                *   location.
                */
                static const char* sEmbeddedFileProject;
            //@}

        public:

            /**
              * \name XRef Url properties
              */
            //@{
                /** Get the number of Urls stored in a property
                  * \return The Url Count
                  */
                static int     GetUrlCount(KFbxProperty const &pProperty);
                static int     GetUrlCount(fbxString const& pUrl);

                /** Return The nth Relative Url stored in the property
                  * upon return the pXRefProject will return the name of the XRef Project closest to the Url of the property
                  * \return The Url if it is valid relative
                  */
                static bool IsRelativeUrl  (KFbxProperty const &pProperty,int pIndex);

                /** Return The nth Url stored in the property
                  * \return The Url
                  */
                static KString GetUrl(KFbxProperty const &pProperty,int pIndex);

                /** Return The nth Relative Url stored in the property
                  * upon return the pXRefProject will return the name of the XRef Project closest to the Url of the property
                  * \return The Url if it is valid relative
                  */
                bool GetResolvedUrl (KFbxProperty const &pProperty,int pIndex,KString & pResolvedPath) const;
            //@}


            bool GetResolvedUrl (char const* pUrl, KFbxDocument* pDoc, KString& pResolvedPath) const;

            /**
             * Looks for the first file matching a specified "pattern",
             * which is built as:
             *
             * if pOptExt is given:         prefix*.ext
             * If pOptExt is NULL:          prefix*
             * if pOptExt is "" or ".":     prefix*.
             *
             * Returns the URL of the first matching pattern.  Cannot be
             * used to resolve folders, only files.
             *
             * If a document is given, we start by looking at the document's
             * fbm folder.
             */
            bool GetFirstMatchingUrl(char const* pPrefix, char const* pOptExt, const KFbxDocument* pDoc, KString& pResolvedPath) const;

            /**
              * \name XRef Resolve Url and Projects
              */
            //@{

                /** Add XRef Projects.
                  * Note that only one Url is associated with a project. Calling 
                  * this on an existing project will replace the project's existing Url.
                  * \param pName The name of the project
                  * \param pUrl The Url to associate with the project
                  * \return true on success, false otherwise.
                 */
                bool        AddXRefProject   (const char *pName,const char *pUrl);
                bool        AddXRefProject   (const char *pName,const char *pExtension,const char *pUrl);
                /** 
                 * Add an XRef project based on a document's EmbeddedUrl 
                 * property, if set, otherwise based on its current Url. 
                 * \param Document to use to name the project, and to specify 
                 *        the Url.
                 * \return true on success, false otherwise.
                 */
                bool        AddXRefProject   (KFbxDocument*);
                bool        RemoveXRefProject(const char *pName);
                bool        RemoveAllXRefProjects();

                int         GetXRefProjectCount() const;
                const char *GetXRefProjectName(int pIndex) const;

                /** Retrieve the base Url for the given project.
                  * \param pName The name of the project
                  * \return The base Url of the project or NULL if the Project is not found.
                  */
                const char* GetXRefProjectUrl(const char* pName);   // FIXME: Should be const, will break AV.
                const char* GetXRefProjectUrl(const char* pName) const;
                const char* GetXRefProjectUrl(int pIndex) const;

                //! Check if a project with the given name is defined in this manager.
                inline bool HasXRefProject( const char* pName ) { return GetXRefProjectUrl(pName) != NULL; }

                /** Return Try to resolve an Relative Url
                  * \return true if the Url is resolved
                  */
                bool GetResolvedUrl (char const *pUrl,KString & pResolvePath) const;

            //@}
        private:
            KArrayTemplate<KXRefManagerProject*>    mProjects;

            static bool UrlExist(const char* pUrl);
    };

}

#endif // #ifndef _FBXKXREF_H


