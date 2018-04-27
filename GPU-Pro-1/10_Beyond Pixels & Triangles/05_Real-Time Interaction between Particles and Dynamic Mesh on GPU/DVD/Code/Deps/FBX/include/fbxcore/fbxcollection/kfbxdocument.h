#ifndef _FBXSDK_DOCUMENT_H_
#define _FBXSDK_DOCUMENT_H_

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

#include <fbxcore/fbxcollection/kfbxcollection.h>

#include <fbxfilesdk_nsbegin.h>

    // Forward declaration
    class KFbxTakeInfo;
    class KFbxPeripheral;
    class KFbxDocumentInfo;

    /** This class contains objects
      * \nosubgrouping
      * \par
      * This class also provides access to take information.
      *
      */
    class KFBX_DLL KFbxDocument : public KFbxCollection
    {
        KFBXOBJECT_DECLARE(KFbxDocument,KFbxCollection);


        /**
          * \name Properties
          */
        //@{
            KFbxTypedProperty<fbxReference*>                    Roots;
        //@}

        /**
        * \name Document Member Manager
        */
        //@{
        public:
            //! Delete all contained objects.
            virtual void    Clear();
            //! Add a member.
            inline  void    AddRootMember   (KFbxObject *pMember)   { AddMember(pMember); Roots.ConnectSrcObject(pMember); }
            //! Remove a member.
            inline  void    RootRootRemoveMember(KFbxObject *pMember)   { RemoveMember(pMember); Roots.DisconnectSrcObject(pMember); }
            //! Find a member.
            template <class T> inline T *       FindRootMember(T const *pfbxType, char *pName) { return Roots.FindSrcObject(pfbxType, pName); }

            //! Return the number of objects in the collection.
            inline int                          GetRootMemberCount () const { return Roots.GetSrcObjectCount(); }
            //! Return the number of objects in the collection.
            template < class T > inline int     GetRootMemberCount (T const *pFBX_TYPE) const { return Roots.GetSrcObjectCount(T::ClassId); }
            int                                 GetRootMemberCount( KFbxCriteria pCriteria ) const;

            //! Return the index'th member of the collection.
            inline KFbxObject*                  GetRootMember (int pIndex=0) const                  { return Roots.GetSrcObject(pIndex); }
            //! Return the index'th member of the collection.
            template < class T > inline T*      GetRootMember (T const *pFBX_TYPE, int pIndex=0) const  { return (T *)Roots.GetSrcObject(T::ClassId,pIndex); }
            KFbxObject*                         GetRootMember (KFbxCriteria pCriteria, int pIndex=0) const;
            //! Is an object part of the collection.

            virtual bool    IsRootMember(KFbxObject *pMember) const;
        //@}


        /**
          * \name Scene information
          */
        //@{
            /** Get the scene information.
              * \return Pointer to the scene information object.
              */
            KFbxDocumentInfo* GetDocumentInfo() const;

            /** Set the scene information.
              * \param pSceneInfo Pointer to the scene information object.
              */
            void SetDocumentInfo(KFbxDocumentInfo* pSceneInfo);
        //@}

        /**
          * \name Offloading management
          *
          * NOTE: The document does not own the peripheral therefore
          * it will not attempt to delete it at destruction time. Also, cloning
          * the document will share the pointer to the peripheral across
          * the cloned objects. And so will do the assignment operator.
          */
        //@{
        public:
            /** Set the current peripheral.
              */
            void SetPeripheral(KFbxPeripheral* pPeripheral);

            /** Retrieve the peripheral of that object.
            * \return Return the current peripheral for that object
            * \remark A peripheral manipulates the content of an object for instance, a peripheral can load the connections of an object on demand.
            */
            virtual KFbxPeripheral* GetPeripheral();

            /** Offload all the unloadable objects contained in the document using the
              * currently set offload peripheral.
              * \return The number of objects that the document have been able to unload.
              * \remark Errors that occured during the operation can be inspected using the
              * GetError() method.
              */
            int UnloadContent();

            /** Load all the objects contained in the document with the data from the
              * currently set offload peripheral.
              * \return The number of objects reloaded.
              * \remark Errors that occured during the operation can be inspected using the
              * GetError() method.
              */
            int LoadContent();

        //@}

        /**
          * \name Referencing management
          */
        //@{

            /**
              * Erase then fills an array of pointers to documents that reference objects in this document.
              *
              * \param pReferencingDocuments array of pointers to documents
              * \returns number of documents that reference objects in this document.
              */
            int GetReferencingDocuments(KArrayTemplate<KFbxDocument*> & pReferencingDocuments) const;

            /**
              * Erase then fills an array of pointers to objects in a given document (pFromDoc)
              * that reference objects in this document.
              *
              * \param pFromDoc pointer to the document containing referencing objects.
              * \param pReferencingObjects array of pointers to referencing objects.
              * \returns number of objects that reference objects in this document.
              */
            int GetReferencingObjects(KFbxDocument const * pFromDoc, KArrayTemplate<KFbxObject*> & pReferencingObjects) const;

            /**
              * Erase then fills an array of pointers to documents that are referenced by objects in this document.
              *
              * \param pReferencedDocuments array of pointers to documents
              * \returns number of documents that are referenced by objects in this document.
              */
            int GetReferencedDocuments(KArrayTemplate<KFbxDocument*> & pReferencedDocuments) const;

            /**
              * Erase then fills an array of pointers to objects in a given document (pToDoc)
              * that are referenced by objects in this document.
              *
              * \param pToDoc pointer to the document containing referenced objects.
              * \param pReferencedObjects array of pointers to referenced objects.
              * \returns number of objects that are referenced by objects in this document.
              */
            int GetReferencedObjects(KFbxDocument const * pToDoc, KArrayTemplate<KFbxObject*> & pReferencedObjects) const;

            // Gets the path string to the root document if any.
            KString GetPathToRootDocument(void) const;

            // Gets the document path to the root document if any.
            void GetDocumentPathToRootDocument(KArrayTemplate<KFbxDocument*> & pDocumentPath, bool pFirstCall = true) const;

            // Tells if this document is a root document.
            bool IsARootDocument(void) { return (NULL == GetDocument()); }
        //@}

        /**
          * \name Take Management
          */
        //@{

            /** Create a take.
              * \param pName Created take name.
              * \return \c true if not a single node, texture or material in the
              * hierarchy had a take with this name before.
              * \return \c false if at least one node, texture or material in the
              * hierarchy had a take with this name before.
              * \return In the last case, KFbxDocument::GetLastErrorID() will return
              * \c eTAKE_ERROR.
              * \remarks This function will create a new take node for every node,
              * texture and material in the hierarchy. It may be more efficient to call
              * KFbxTakeNodeContainer::CreateTakeNode() on the relevant nodes, textures
              * and materials if a take only has a few of them with animation data.
              */
            bool CreateTake(char* pName);

            /** Remove a take.
              * \param pName Name of the take to remove.
              * \return \c true if every node, texture and material in the hierarchy
              * have a take with this name.
              * \return \c false if at least one node, texture or material in the
              * hierarchy don't have a take with this name.
              * \return In the last case, KFbxDocument::GetLastErrorID() will return
              * \c eTAKE_ERROR.
              * \remarks Scans the node hierarchy, the texture list and the material
              * list to remove all take nodes found with that name.
              */
            bool RemoveTake(char* pName);

            /** Set the current take.
              * \param pName Name of the take to set.
              * \return \c true if every node, texture and material in the hierarchy
              * have a take with this name.
              * \return \c false if at least one node, texture or material in the
              * hierarchy don't have a take with this name.
              * \return In the last case, KFbxDocument::GetLastErrorID() will return
              * \c eTAKE_ERROR.
              * \remarks Scans the node hierarchy, the texture list and the material
              * list to set all take nodes found with that name.
              * \remarks All nodes, textures and materials without a take node of the
              * requested name are set to default take node. It means that, if a node,
              * texture or material does not have the requested take, it is assumed
              * that this node is not animated in this take.
              */
            bool SetCurrentTake(char* pName);

            /** Get current take name.
              * \return Current take name.
              * \return An empty string if the document has not been imported from a file
              * and function KFbxDocument::SetCurrentTake() has not been called previously
              * at least once.
              */
            char* GetCurrentTakeName();

            /** Fill a string array with all existing take names.
              * \param pNameArray An array of string objects.
              * \remarks Scans the node hierarchy, the texture list and the material
              * list to find all existing take node names.
              * \remarks The array of string is cleared before scanning the node
              * hierarchy.
              */
            void FillTakeNameArray(KArrayTemplate<KString*>& pNameArray);

        //@}

        /**
          * \name Take Information Management
          */
        //@{

            /** Set take information about an available take.
              * \param pTakeInfo Take information, field KFbxTakeInfo::mName specifies
              * the targeted take.
              * \return \c true if take is found and take information set.
              */
            bool SetTakeInfo(const KFbxTakeInfo& pTakeInfo);

            /** Get take information about an available take.
              * \param pTakeName Take name.
              * \return Pointer to take information or \c NULL if take isn't found or
              *   has no information set.
              */
            KFbxTakeInfo* GetTakeInfo(const KString& pTakeName);

        //@}

        /**
          * \name Error Management
          * The same error object is shared among instances of this class.
          */
        //@{

            /** Retrieve error object.
              * \return Reference to error object.
              */
            KError& GetError();

            /** Error identifiers.
              * Most of these are only used internally.
              */
            typedef enum
            {
                eTAKE_ERROR,
                eKFBX_OBJECT_IS_NULL,
                eKFBX_OBJECT_ALREADY_OWNED,
                eKFBX_OBJECT_UNKNOWN,
                eKFBX_MISSING_PERIPHERAL,
                eKFBX_OBJECT_PERIPHERAL_FAILURE,
                eERROR_COUNT
            } EError;

            /** Get last error code.
              * \return Last error code.
              */
            EError GetLastErrorID() const;

            /** Get last error string.
              * \return Textual description of the last error.
              */
            const char* GetLastErrorString() const;

        //@}

        ///////////////////////////////////////////////////////////////////////////////
        //  WARNING!
        //  Anything beyond these lines may not be Documented accurately and is
        //  subject to change without notice.
        ///////////////////////////////////////////////////////////////////////////////
        #ifndef DOXYGEN_SHOULD_SKIP_THIS
            public:
                virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

            // Constructor / Destructor
            protected:

                KFbxDocument(KFbxSdkManager& pManager, char const* pName);
                ~KFbxDocument();
                virtual void Construct  (const KFbxDocument* pFrom);
                virtual void Destruct   (bool pRecursive, bool pDependents);
                bool    ConstructProperties(bool pForceSet);

                KFbxDocument& operator=(const KFbxDocument& pOther);

            // Notification and connection management
            protected:
                virtual bool    ConnecNotify (KFbxConnectEvent const &pEvent);
                virtual void    SetDocument(KFbxDocument* pDocument);

            // Helper functions
            protected:
                void ConnectVideos();

            // Take management
            protected:
                bool FindTakeName(const KString& pTakeName);

            //
            protected:
                KFbxSdkManager*                     mSdkManager;
                KFbxPeripheral*                     mPeripheral;
                KString                             mCurrentTakeName;
                KArrayTemplate<KFbxTakeInfo *>      mTakeInfoArray;
                KError                              mError;
                KFbxDocumentInfo*                   mDocumentInfo;



            friend class KFbxLayerContainer;
            friend class KFbxNodeFinderDuplicateName;

            friend class KFbxWriterFbx;
            friend class KFbxWriterFbx6;
            friend class KFbxWriterFbx7;
            friend struct KFbxWriterFbx7Impl;
            friend class KFbxReaderFbx;
            friend class KFbxReaderFbx6;
            friend class KFbxReaderFbx7;
            friend struct KFbxReaderFbx7Impl;

        #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_Document_H_


