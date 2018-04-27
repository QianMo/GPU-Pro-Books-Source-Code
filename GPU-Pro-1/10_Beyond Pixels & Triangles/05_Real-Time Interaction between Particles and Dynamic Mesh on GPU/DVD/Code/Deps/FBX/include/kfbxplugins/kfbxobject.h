/*!  \file kfbxobject.h
 */

#ifndef _FBXSDK_OBJECT_H_
#define _FBXSDK_OBJECT_H_

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

// FBX includes
#include <klib/kstring.h>
#include <klib/kstringlist.h>
#include <klib/kname.h>
#include <klib/karrayul.h>
#include <klib/kscopedptr.h>
#include <kfbxplugins/kfbxplug.h>
#include <kfbxplugins/kfbxproperty.h>
#include <kfbxevents/kfbxevents.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif
#include <kbaselib_forward.h>

// FBX namespace
#include <fbxfilesdk_nsbegin.h>

    class KFbxSdkManager;
    class KFbxDocument;
    class KFbxScene;
    class KFbxObject_internal;
    class KFbxProperty_internal;
    class KFbxTakeNodeContainer;
    class KFbxTakeNode;
    class UserDataRecord;
    class KFbxImplementation;
    class KFbxImplementationFilter;
    class KFbxLibrary;
    class KFbxStream;
    class KFbxPeripheral;
    class KFbxObject;
    class KFbxMessage;

    //////   KFbxObject Events  //////////////////////////////////////////////////////////////////
    class KFbxObjectPropertyChanged : public kfbxevents::KFbxEvent<KFbxObjectPropertyChanged>
    {
    public:
       KFBXEVENT_DECLARE(KFbxObjectPropertyChanged)

    public:
        KFbxObjectPropertyChanged(KFbxProperty pProp):mProp(pProp){}
        KFbxProperty mProp;
    };
    ////// End KFbxObject Events  //////////////////////////////////////////////////////////////////

    enum eFbxCompare {
        eFbxCompareProperties
    };

    #define KFBXOBJECT_DECLARE(Class,Parent) \
    private: \
        KFBXPLUG_DECLARE(Class) \
        typedef Parent ParentClass;\
        static Class* Create(KFbxObject *pContainer,  char const *pName); \
        static Class* CreateForClone( KFbxSdkManager *pManager, char const *pName, const Class* pFrom ); \
    public: \
        Class*  TypedClone(KFbxObject* pContainer = NULL, KFbxObject::ECloneType pCloneType = eDEEP_CLONE) const; \
        virtual bool Compare(KFbxObject *pOtherObject,eFbxCompare pCompareMethod=eFbxCompareProperties);\

    #define KFBXOBJECT_DECLARE_ABSTRACT(Class,Parent) \
    private: \
        KFBXPLUG_DECLARE_ABSTRACT(Class) \
        typedef Parent ParentClass; \

    #define KFBXOBJECT_IMPLEMENT(Class) \
        KFBXPLUG_IMPLEMENT(Class) \
        Class* Class::Create(KFbxObject *pContainer, char const *pName) \
        {                                                                   \
          Class* ClassPtr=Class::Create(pContainer->GetFbxSdkManager(),pName); \
            pContainer->ConnectSrcObject(ClassPtr);                         \
            return ClassPtr;                                                \
        } \
        Class* Class::TypedClone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const \
        { \
            KFbxObject* lObjClone = Clone(pContainer, pCloneType); \
            if (lObjClone == NULL) \
            { \
                return NULL; \
            } \
            Class* lTypedClone = KFbxCast<Class>(lObjClone); \
            if (lTypedClone == NULL) \
            { \
                lObjClone->Destroy(); \
            } \
            return lTypedClone; \
        }   \
        Class* Class::CreateForClone(KFbxSdkManager *pManager, char const *pName, const Class* pFrom )  \
        {   \
            return (Class *)pManager->CreateClassFrom(Class::ClassId, pName, pFrom); \
        }

    #define KFBXOBJECT_IMPLEMENT_ABSTRACT(Class) \
        KFBXPLUG_IMPLEMENT_ABSTRACT(Class) \

    typedef size_t KFbxObjectID;

    class _KFbxObjectData;

    typedef int kFbxUpdateId;

    /** \brief Basic class for object type identification and instance naming.
      * \nosubgrouping
      */
    class KFBX_DLL KFbxObject : public KFbxPlug
    {
        public:

            /** Types of clones that can be created for KFbxObjects.
              */
            typedef enum
            {
                eSURFACE_CLONE,     //!<
                eREFERENCE_CLONE,   //!< Changes to original object propagate to clone. Changes to clone do not propagate to original.
                eDEEP_CLONE         //!< A deep copy of the object. Changes to either the original or clone do not propagate to each other.
            } ECloneType;

            KFBXOBJECT_DECLARE(KFbxObject,KFbxPlug);

        /**
          * \name Cloning and references
          */
        //@{
        public:
            // Clone

            /** Creates a clone of this object.
              * \param pContainer The object, typically a document or scene, that will contain the new clone. Can be NULL.
              * \param pCloneType The type of clone to create
              * \return The new clone, or NULL if the specified clone type is not supported.
              */
            virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

            /** Check if this object is a reference clone of another.
              * \return true if this object is a clone of another, false otherwise
              */
            bool        IsAReferenceTo(void) const;

            /** If this object is a reference clone, this method returns the original object from
              * which this one was cloned.
              * \return The original, or NULL if this object is not a reference clone.
              */
            KFbxObject* GetReferenceTo(void) const;

            /** Check if any objects were reference cloned from this one.
              * \return true If objects were cloned from this one, false otherwise.
              */
            bool        IsReferencedBy(void) const;

            /** Get the number of objects that were reference cloned from this one.
              * \return The number of objects cloned from this one.
              */
            int         GetReferencedByCount(void) const;

            /** Get a reference clone of this object.
              * \param pIndex Valid values are [0, GetReferencedByCount())
              * \return The requested clone, or NULL if pIndex is out of range.
              */
            KFbxObject* GetReferencedBy(int pIndex) const;

        protected:
            //KFbxObject*   SetReferenceTo(KFbxObject* pRef);
        //@}
        /**
          * \name Object Name Management
          */
        //@{
        public:

            /** Set the name of the object.
              * \param pName A \c NULL terminated string.
              */
            void SetName(char const* pName);

            /** Return the full name of the object.
              * \return Return a \c NULL terminated string.
              */
            char const* GetName() const;

            /** Return the name of the object without the namespace qualifier.
              * \return Return the name in a temporary string.
              */
            KString GetNameWithoutNameSpacePrefix() const;

            /** Return the name of the object with the namespace qualifier.
              * \return Return the name in a temporary string.
              */
            KString GetNameWithNameSpacePrefix() const;

            /** Set the initial name of the object.
              * \param pName A \c NULL terminated string.
              */
            void SetInitialName(char const* pName);

            /** Return the initial name of the object.
              * \return Return a \c NULL terminated string.
              */
            char const* GetInitialName() const;

            /** Return the namespace of the object.
              * \return Return a \c NULL terminated string.
              */
            KString GetNameSpaceOnly( );

            /** Set the namespace of the object.
              * \return Return a \c NULL terminated string.
              */
            void SetNameSpace(KString pNameSpace);

            /** Return an array of all the namespace of the object
              * \return Return a \c NULL terminated string.
              */
            KArrayTemplate<KString*> GetNameSpaceArray( char identifier );

            /** Get the name only (no namespace or prefix) of the object.
              * \return Return a \c NULL terminated string.
              */
            KString GetNameOnly() const;

            KString GetNameSpacePrefix() const;
            static KString RemovePrefix(char* pName);
            static KString StripPrefix(KString& lName);
            static KString StripPrefix(const char* pName);

            KFbxObjectID const& GetUniqueID() const;
        //@}

        /**
          * \name UpdateId Management
          */
        //@{
        public:
            typedef enum {
                eUpdateId_Object,
                eUpdateId_Dependency
            } eFbxUpdateIdType;

            virtual kFbxUpdateId GetUpdateId(eFbxUpdateIdType pUpdateId=eUpdateId_Object) const;
        protected:
            virtual kFbxUpdateId IncUpdateId(eFbxUpdateIdType pUpdateId=eUpdateId_Object);


        //@}

        /**
          * \name Off-loading Management
          * \remark the unloaded state flag can be modified using the SetObjectFlags()
          *         method. The ContentIsUnloaded() method below (implemented in this class)
          *         is simply a synonym to GetObjectFlags(eCONTENT_UNLOADED_FLAG)
          */
        //@{
        public:
            /** Unload this object content using the offload peripheral currently set in the document
              * then flush it from memory.
              * \return 2 if the object's content is already unloaded or 1 if
              *         this object content has been successfully unloaded to the current
              *         peripheral.
              *
              * \remark If the content is locked more than once or the peripheral cannot handle
              * this object unload or an error occurred, the method will return 0 and the
              * content is not flushed.
              */
            int ContentUnload();

            /** Load this object content using the offload peripheral currently set in the document.
              * \return 1 if this object content has been successfully loaded from the current
              *         peripheral. 2 If the content is already loaded and 0 if an error occurred or
              *         there is a lock on the object content.
              * \remark On a successful Load attempt, the object content is locked.
              */
            int ContentLoad();

            /** Returns true if this object content is currently loaded.
              * \remark An object that has not been filled yet must be considered
              * unloaded.
              */
            bool ContentIsLoaded() const;

            /**  Decrement the content lock count of an object. If the content lock count of an object
              *  is greater than 0, the content of the object is considered locked.
              */
            void ContentDecrementLockCount();

            /** Increment the content lock count of an object. If the content lock count of an object
              * is greater than 0, the content of the object is considered locked.
            */
            void ContentIncrementLockCount();

            /** Returns true if this object content is locked. The content is locked if the content lock count
              * is greater than 0
              * \remark A locked state prevents the object content to be unloaded from memory but
              * does not block the loading.
              */
            bool ContentIsLocked() const;


        protected:
            /** Clear this object content from memory.
              * This method has to be overridden in the derived classes.
              * \remark This method is called by ContentUnload() upon success.
              */
            virtual void ContentClear();

            /** Retrieve the peripheral of that object.
            * \return Return the current peripheral for that object
            * \remark A peripheral manipulates the content of an object for instance, a peripheral can load the connections of an object on demand.
            */
            virtual KFbxPeripheral* GetPeripheral();
        //@}
        public:
        /**
          * \name Off-loading Serialization section
          * The methods in this section are, usually, called by
          * a peripheral.
          */
        //@{
            /** Write the content of the object to the given stream.
              * \param pStream The destination stream.
              * \return True if the content has been successfully processed
              * by the receiving stream.
              */
            virtual bool ContentWriteTo(KFbxStream& pStream) const;

            /** Read the content of the object from the given stream.
              * \param pStream The source streak.
              * \return True if the object has been able to fill itself with the received data
              * from the stream.
              */
            virtual bool ContentReadFrom(const KFbxStream& pStream);
        //@}

        /**
          * \name Selection management
          */
        //@{
        public:
            virtual bool GetSelected();
            virtual void SetSelected(bool pSelected);
        //@}

        /**
          * \name Evaluation Info
          */
        //@{
            virtual bool Evaluate(KFbxProperty & pProperty,KFbxEvaluationInfo const *pEvaluationInfo);
        //@}


        /**
          * \name Properties access
          */
        //@{
        public:
            inline KFbxProperty GetFirstProperty() const
            {
                return RootProperty.GetFirstDescendent();
            }

            inline KFbxProperty GetNextProperty(KFbxProperty const &pProperty) const
            {
                return RootProperty.GetNextDescendent(pProperty);
            }

            /** Find a property using its name and its data type.
              * \param pName The name of the property as a \c NULL terminated string.
			  * \param pCaseSensitive
              * \return A valid KFbxProperty if the property was found, else
              *         an invalid KFbxProperty. See KFbxProperty::IsValid()
              */
            inline KFbxProperty FindProperty(const char* pName, bool pCaseSensitive = true)const
            {
                return RootProperty.Find(pName, pCaseSensitive );
            }

            inline KFbxProperty FindProperty(const char* pName, KFbxDataType const &pDataType, bool pCaseSensitive = true) const
            {
                return RootProperty.Find(pName, pDataType, pCaseSensitive );
            }

            inline KFbxProperty FindPropertyHierarchical(const char* pName, bool pCaseSensitive = true)const
            {
                return RootProperty.FindHierarchical(pName, pCaseSensitive );
            }

            inline KFbxProperty FindPropertyHierarchical(const char* pName, KFbxDataType const &pDataType, bool pCaseSensitive = true) const
            {
                return RootProperty.FindHierarchical(pName, pDataType, pCaseSensitive );
            }

            inline KFbxProperty &GetRootProperty() { return RootProperty; }
            inline const KFbxProperty& GetRootProperty()const{ return RootProperty; }

            KFbxProperty GetClassRootProperty();

        public:
            KFbxProperty RootProperty;

        private:
            void SetClassRootProperty(KFbxProperty &lProperty);

        // property callbacks
        protected:
            typedef enum {
                eFbxProperty_SetRequest,
                eFbxProperty_Set,
                eFbxProperty_Get
            } eFbxPropertyNotify;
            virtual bool PropertyNotify(eFbxPropertyNotify pType, KFbxProperty* pProperty);
        //@}

        /**
          * \name Class based defaults properties
          */
        //@{
        //@}

        /**
          * \name General Object Connection and Relationship Management
          */
        //@{
        public:
            // SrcObjects
            inline bool ConnectSrcObject        (KFbxObject* pObject,kFbxConnectionType pType=eFbxConnectionNone)   { return RootProperty.ConnectSrcObject(pObject,pType); }
            inline bool IsConnectedSrcObject    (const KFbxObject* pObject) const { return RootProperty.IsConnectedSrcObject  (pObject); }
            inline bool DisconnectSrcObject (KFbxObject* pObject)       { return RootProperty.DisconnectSrcObject(pObject); }

            inline bool DisconnectAllSrcObject() { return RootProperty.DisconnectAllSrcObject(); }
            inline bool DisconnectAllSrcObject(KFbxCriteria const &pCriteria) { return RootProperty.DisconnectAllSrcObject(pCriteria); }
            inline bool DisconnectAllSrcObject(kFbxClassId pClassId) { return RootProperty.DisconnectAllSrcObject(pClassId); }
            inline bool DisconnectAllSrcObject(kFbxClassId pClassId,KFbxCriteria const &pCriteria) { return RootProperty.DisconnectAllSrcObject(pClassId,pCriteria); }

            inline int GetSrcObjectCount    () const { return RootProperty.GetSrcObjectCount(); }
            inline int GetSrcObjectCount    (KFbxCriteria const &pCriteria) const { return RootProperty.GetSrcObjectCount(pCriteria); }
            inline int GetSrcObjectCount    (kFbxClassId pClassId) const { return RootProperty.GetSrcObjectCount(pClassId); }
            inline int GetSrcObjectCount    (kFbxClassId pClassId,KFbxCriteria const &pCriteria) const { return RootProperty.GetSrcObjectCount(pClassId,pCriteria); }

            inline KFbxObject*  GetSrcObject (int pIndex=0) const { return RootProperty.GetSrcObject(pIndex); }
            inline KFbxObject*  GetSrcObject (KFbxCriteria const &pCriteria,int pIndex=0) const { return RootProperty.GetSrcObject(pCriteria,pIndex); }
            inline KFbxObject*  GetSrcObject (kFbxClassId pClassId,int pIndex=0) const { return RootProperty.GetSrcObject(pClassId,pIndex); }
            inline KFbxObject*  GetSrcObject (kFbxClassId pClassId,KFbxCriteria const &pCriteria,int pIndex=0) const { return RootProperty.GetSrcObject(pClassId,pCriteria,pIndex); }

            inline KFbxObject*  FindSrcObject (const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcObject(pName,pStartIndex); }
            inline KFbxObject*  FindSrcObject (KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcObject(pCriteria,pName,pStartIndex); }
            inline KFbxObject*  FindSrcObject (kFbxClassId pClassId,const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcObject(pClassId,pName,pStartIndex); }
            inline KFbxObject*  FindSrcObject (kFbxClassId pClassId,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcObject(pClassId,pCriteria,pName,pStartIndex); }

            template < class T > inline bool DisconnectAllSrcObject (T const *pFBX_TYPE) { return RootProperty.DisconnectAllSrcObject(pFBX_TYPE);   }
            template < class T > inline bool DisconnectAllSrcObject (T const *pFBX_TYPE,KFbxCriteria const &pCriteria)  { return RootProperty.DisconnectAllSrcObject(pFBX_TYPE,pCriteria);  }
            template < class T > inline int  GetSrcObjectCount(T const *pFBX_TYPE) const { return RootProperty.GetSrcObjectCount(pFBX_TYPE);    }
            template < class T > inline int  GetSrcObjectCount(T const *pFBX_TYPE,KFbxCriteria const &pCriteria) const { return RootProperty.GetSrcObjectCount(pFBX_TYPE,pCriteria);    }
            template < class T > inline T*   GetSrcObject(T const *pFBX_TYPE,int pIndex=0) const { return RootProperty.GetSrcObject(pFBX_TYPE,pIndex);  }
            template < class T > inline T*   GetSrcObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,int pIndex=0) const { return RootProperty.GetSrcObject(pFBX_TYPE,pCriteria,pIndex);  }
            template < class T > inline T*   FindSrcObject(T const *pFBX_TYPE,const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcObject(pFBX_TYPE,pName,pStartIndex);  }
            template < class T > inline T*   FindSrcObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcObject(pFBX_TYPE,pCriteria,pName,pStartIndex);  }

            // DstObjects
            inline bool ConnectDstObject        (KFbxObject* pObject,kFbxConnectionType pType=eFbxConnectionNone)   { return RootProperty.ConnectDstObject(pObject,pType); }
            inline bool IsConnectedDstObject    (const KFbxObject* pObject) const { return RootProperty.IsConnectedDstObject  (pObject); }
            inline bool DisconnectDstObject (KFbxObject* pObject)       { return RootProperty.DisconnectDstObject(pObject); }

            inline bool DisconnectAllDstObject() { return RootProperty.DisconnectAllDstObject(); }
            inline bool DisconnectAllDstObject(KFbxCriteria const &pCriteria) { return RootProperty.DisconnectAllDstObject(pCriteria); }
            inline bool DisconnectAllDstObject(kFbxClassId pClassId) { return RootProperty.DisconnectAllDstObject(pClassId); }
            inline bool DisconnectAllDstObject(kFbxClassId pClassId,KFbxCriteria const &pCriteria) { return RootProperty.DisconnectAllDstObject(pClassId,pCriteria); }

            inline int GetDstObjectCount    () const { return RootProperty.GetDstObjectCount(); }
            inline int GetDstObjectCount    (KFbxCriteria const &pCriteria) const { return RootProperty.GetDstObjectCount(pCriteria); }
            inline int GetDstObjectCount    (kFbxClassId pClassId) const { return RootProperty.GetDstObjectCount(pClassId); }
            inline int GetDstObjectCount    (kFbxClassId pClassId,KFbxCriteria const &pCriteria) const { return RootProperty.GetDstObjectCount(pClassId,pCriteria); }

            inline KFbxObject*  GetDstObject (int pIndex=0) const { return RootProperty.GetDstObject(pIndex); }
            inline KFbxObject*  GetDstObject (KFbxCriteria const &pCriteria,int pIndex=0) const { return RootProperty.GetDstObject(pCriteria,pIndex); }
            inline KFbxObject*  GetDstObject (kFbxClassId pClassId,int pIndex=0) const { return RootProperty.GetDstObject(pClassId,pIndex); }
            inline KFbxObject*  GetDstObject (kFbxClassId pClassId,KFbxCriteria const &pCriteria,int pIndex=0) const { return RootProperty.GetDstObject(pClassId,pCriteria,pIndex); }

            inline KFbxObject*  FindDstObject (const char *pName,int pStartIndex=0) const { return RootProperty.FindDstObject(pName,pStartIndex); }
            inline KFbxObject*  FindDstObject (KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return RootProperty.FindDstObject(pCriteria,pName,pStartIndex); }
            inline KFbxObject*  FindDstObject (kFbxClassId pClassId,const char *pName,int pStartIndex=0) const { return RootProperty.FindDstObject(pClassId,pName,pStartIndex); }
            inline KFbxObject*  FindDstObject (kFbxClassId pClassId,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return RootProperty.FindDstObject(pClassId,pCriteria,pName,pStartIndex); }

            template < class T > inline bool DisconnectAllDstObject (T const *pFBX_TYPE) { return RootProperty.DisconnectAllDstObject(pFBX_TYPE);   }
            template < class T > inline bool DisconnectAllDstObject (T const *pFBX_TYPE,KFbxCriteria const &pCriteria)  { return RootProperty.DisconnectAllDstObject(pFBX_TYPE,pCriteria);  }
            template < class T > inline int  GetDstObjectCount(T const *pFBX_TYPE) const { return RootProperty.GetDstObjectCount(pFBX_TYPE);    }
            template < class T > inline int  GetDstObjectCount(T const *pFBX_TYPE,KFbxCriteria const &pCriteria) const { return RootProperty.GetDstObjectCount(pFBX_TYPE,pCriteria);    }
            template < class T > inline T*   GetDstObject(T const *pFBX_TYPE,int pIndex=0) const { return RootProperty.GetDstObject(pFBX_TYPE,pIndex);  }
            template < class T > inline T*   GetDstObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,int pIndex=0) const { return RootProperty.GetDstObject(pFBX_TYPE,pCriteria,pIndex);  }
            template < class T > inline T*   FindDstObject(T const *pFBX_TYPE,const char *pName,int pStartIndex=0) const { return RootProperty.FindDstObject(pFBX_TYPE,pName,pStartIndex);  }
            template < class T > inline T*   FindDstObject(T const *pFBX_TYPE,KFbxCriteria const &pCriteria,const char *pName,int pStartIndex=0) const { return RootProperty.FindDstObject(pFBX_TYPE,pCriteria,pName,pStartIndex);  }
        //@}

        // Optimized routine
        KFbxProperty FindProperty(const char* pName, int pStartIndex, int pSearchDomain) const;

        /**
          * \name General Property Connection and Relationship Management
          */
        //@{
            // Properties
            inline bool         ConnectSrcProperty      (KFbxProperty const & pProperty) { return RootProperty.ConnectSrcProperty(pProperty); }
            inline bool         IsConnectedSrcProperty  (KFbxProperty const & pProperty) { return RootProperty.IsConnectedSrcProperty(pProperty); }
            inline bool         DisconnectSrcProperty   (KFbxProperty const & pProperty) { return RootProperty.DisconnectSrcProperty(pProperty); }
            inline int          GetSrcPropertyCount     () const { return RootProperty.GetSrcPropertyCount(); }
            inline KFbxProperty GetSrcProperty          (int pIndex=0) const { return RootProperty.GetSrcProperty(pIndex); }
            inline KFbxProperty FindSrcProperty         (const char *pName,int pStartIndex=0) const { return RootProperty.FindSrcProperty(pName,pStartIndex); }

            inline bool         ConnectDstProperty      (KFbxProperty const & pProperty) { return RootProperty.ConnectDstProperty(pProperty); }
            inline bool         IsConnectedDstProperty  (KFbxProperty const & pProperty) { return RootProperty.IsConnectedDstProperty(pProperty); }
            inline bool         DisconnectDstProperty   (KFbxProperty const & pProperty) { return RootProperty.DisconnectDstProperty(pProperty); }
            inline int          GetDstPropertyCount     () const { return RootProperty.GetDstPropertyCount(); }
            inline KFbxProperty GetDstProperty          (int pIndex=0) const { return RootProperty.GetDstProperty(pIndex); }
            inline KFbxProperty FindDstProperty         (const char *pName,int pStartIndex=0) const { return RootProperty.FindDstProperty(pName,pStartIndex); }
        //@}

        /**
          * \name User data
          */
        //@{
            void        SetUserDataPtr(KFbxObjectID const& pUserID, void* pUserData);
            void*       GetUserDataPtr(KFbxObjectID const& pUserID) const;

            inline void SetUserDataPtr(void* pUserData)     { SetUserDataPtr( GetUniqueID(), pUserData ); }
            inline void* GetUserDataPtr() const             { return GetUserDataPtr( GetUniqueID() ); }
        //@}


        /**
          * \name Document Management
          */
        //@{
            /** Get a pointer to the document containing this object.
              * \return Return a pointer to the document containing this object or \c NULL if the
              * object does not belong to any document.
              */
            KFbxDocument* GetDocument() const;

            /** Get a const pointer to the root document containing this object.
              * \return Return a const pointer to the root document containing this object or \c NULL if the
              * object does not belong to any document.
              */
            KFbxDocument* GetRootDocument() const;

            /** Get a pointer to the scene containing this object.
              * \return Return a pointer to the scene containing this object or \c NULL if the
              * object does not belong to any scene.
              */
            KFbxScene* GetScene() const;
        //@}


        /**
          * \name Logging.
          */
        //@{
            /** Emit a message in all available message emitter in the document or SDK manager.
              * \param pMessage the message to emit. Ownership is transfered, do not delete.
              */
            void EmitMessage(KFbxMessage * pMessage) const;
        //@}

        /**
          * \name Localization helper.
          */
        //@{
            /** Localization helper function. Calls the FBX SDK manager implementation.
              * sub-classes which manage their own localization could over-ride this.
              * \param pID the identifier for the text to localize.
              * \param pDefault the default text. Uses pID if NULL.
              * \return the potentially localized text. May return the parameter passed in.
            */
            virtual const char * Localize( const char * pID, const char * pDefault = NULL ) const;
        //@}

        /**
          * \name Application Implementation Management
          */
        //@{
            //! Get a handle on the parent library if exists.
            KFbxLibrary* GetParentLibrary(void) const;

            /** Adds an implementation.
              *
              * \param pImplementation a handle on an implementation
              *
              * \returns true on success, false otherwise
              *
              * \remarks to succeed this function needs to be called with an
              * implementation that has not already been added to this node.
              */
            bool AddImplementation(KFbxImplementation* pImplementation);

            /** Removes an implementation.
              *
              * \param pImplementation a handle on an implementation
              *
              * \returns true on success, false otherwise
              *
              * \remarks to succeed this function needs to be called with an
              * implementation that has already been added to this node.
              */
            bool RemoveImplementation(KFbxImplementation* pImplementation);

            //! Tells if this shading node has a default implementation
            bool HasDefaultImplementation(void) const;

            //! Returns the default implementation.
            KFbxImplementation* GetDefaultImplementation(void) const;

            /** Sets the default implementation.
              *
              * \param pImplementation a handle on an implementation
              *
              * \returns true on success, false otherwise
              *
              * \remarks to succeed this function needs to be called with an
              * implementation that has already been added to this node.
              */
            bool SetDefaultImplementation(KFbxImplementation* pImplementation);

            /** Returns the number of implementations that correspond to a given criteria
              *
              * \param pCriteria filtering criteria that identifyies the kind of
              * implementations to take into account.
              *
              * \returns the number of implementation(s)
              */
            int GetImplementationCount(
                const KFbxImplementationFilter* pCriteria = NULL
            ) const;

            /** Returns a handle on the (pIndex)th implementation that corresponds
              * to the given criteria
              * \param pIndex
              * \param pCriteria filtering criteria that identifyies the kind of
              * implementations to take into account.
              *
              * \remarks returns NULL if the criteria or the index are invalid
              */
            KFbxImplementation* GetImplementation(
                int                             pIndex,
                const KFbxImplementationFilter* pCriteria = NULL
            ) const;
        //@}

        /**
          * \name Object Storage && Retrieval
          */
        //@{
        public:
            virtual KString GetUrl() const;
            virtual bool    SetUrl(char *pUrl);
            virtual bool    PopulateLoadSettings(KFbxObject *pSettings,char *pFileName=0);
            virtual bool    Load(char *pFileName=0);
            virtual bool    PopulateSaveSettings(KFbxObject *pSettings,char *pFileName=0);
            virtual bool    Save(char *pFileName=0);
        //@}

    ///////////////////////////////////////////////////////////////////////////////
    //
    //  WARNING!
    //
    //  Anything beyond these lines may not be documented accurately and is
    //  subject to change without notice.
    //
    ///////////////////////////////////////////////////////////////////////////////

    #ifndef DOXYGEN_SHOULD_SKIP_THIS

    protected:
        KFbxObject(KFbxSdkManager& pManager, char const* pName);
        virtual ~KFbxObject();

        // Constructs the object. Each subclass should call the base
        // KFbxObject::Construct first. Note that property initialization
        // should be done in the ConstructProperties() method, not this one.
        // pFrom - The object this object should be cloned from. NULL to
        // construct an independent object.
        virtual void Construct(const KFbxObject* pFrom);

        // Initialize the KFbxProperty member variables of this class.
        // Each subclass should call its parent, and initialize its properties
        // in this method.
        // pForceSet - Forces the property values to be set to default values.
        virtual bool ConstructProperties(bool pForceSet);
        virtual void Destruct(bool pRecursive, bool pDependents);

    public:
        virtual KFbxSdkManager* GetFbxSdkManager() const;
        virtual kFbxClassId     GetRuntimeClassId() const;

        typedef enum
        {
            eNONE                   = 0x00000000,
            eSYSTEM_FLAG            = 0x00000001,
            eSAVABLE_FLAG           = 0x00000002,
            eHIDDEN_FLAG            = 0x00000008,

            eSYSTEM_FLAGS           = 0x0000ffff,

            eUSER_FLAGS             = 0x000f0000,

            // These flags are not saved to the fbx file
            eSYSTEM_RUNTIME_FLAGS   = 0x0ff00000,

            eCONTENT_LOADED_FLAG    = 0x00100000,

            eUSER_RUNTIME_FIRST_FLAG= 0x10000000,
            eUSER_RUNTIME_FLAGS     = 0xf0000000,

            eRUNTIME_FLAGS          = 0xfff00000
        } EObjectFlag;

        void                    SetObjectFlags(EObjectFlag pFlags, bool pValue);
        bool                    GetObjectFlags(EObjectFlag pFlags) const;

        // All flags replaced at once. This includes overriding the runtime flags, so
        // most likely you'd want to do something like this:
        //
        // SetObjectFlags(pFlags | (GetObjectFlags() & KFbxObject::eRUNTIME_FLAGS));
        void                    SetObjectFlags(kUInt pFlags);
        kUInt                   GetObjectFlags() const; // All flags at once, as a bitmask

    protected:
        virtual KFbxTakeNodeContainer* GetTakeNodeContainer();

    protected:

        KFbxObject& operator=(KFbxObject const& pObject);

        // Currently not called from operator=; you must call it yourself if you
        // want properties to be copied.  Most likely it will be called automatically
        // by operator=(), at which point this method may become private.
        // At the very least it should be renamed to spot everyone that's currently
        // using it in their operator=, since it would then become irrelevant (and slow)
        void CopyPropertiesFrom(const KFbxObject& pFrom);

        virtual bool            SetRuntimeClassId(kFbxClassId pClassId);
        virtual                 bool ConnecNotify (KFbxConnectEvent const &pEvent);

        virtual KString         GetTypeName() const;
        virtual KStringList     GetTypeFlags() const;

        virtual void            PropertyAdded(KFbxProperty* pProperty);
        virtual void            PropertyRemoved(KFbxProperty* pProperty);

        // Animation Management
        virtual void            AddChannels(KFbxTakeNode *pTakeNode);
        virtual void            UpdateChannelFromProperties(KFbxTakeNode *pTakeNode);
        virtual void            SetDocument(KFbxDocument* pDocument);

        public:
            inline KFbxObjectHandle &GetPropertyHandle() { return RootProperty.mPropertyHandle; }

        private:
            KScopedPtr<_KFbxObjectData> mData;

        // friend classes for sdk access
        friend class KFbxReaderFbx;
        friend class KFbxWriterFbx6;
        friend struct KFbxWriterFbx7Impl;
        friend class KFbxScene;
        friend class KFbxObject_internal;
        friend class KFbxProperty;
    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };

    // need to be declared here instead or we get an INTERNAL COMPILER ERROR with VC6
    inline bool KFbxConnectSrc(KFbxObject *pDstObject,KFbxObject *pSrcObject)   { return (pSrcObject && pDstObject) ? pDstObject->ConnectSrcObject(pSrcObject) : 0; }
    inline bool KFbxConnectDst(KFbxObject *pSrcObject,KFbxObject *pDstObject)   { return (pSrcObject && pDstObject) ? pSrcObject->ConnectDstObject(pDstObject) : 0; }

    typedef KFbxObject* HKFbxObject;

    // template access functions SRC
    template < class T > inline int KFbxGetSrcCount(KFbxObject const *pObject, T const* VC6Dummy = 0)                               { return pObject ? pObject->GetSrcObjectCount(T::ClassId) : 0; }
    template < class T > inline int KFbxGetSrcCount(KFbxObject const *pObject,kFbxClassId pClassId, T const* VC6Dummy = 0)          { return pObject ? pObject->GetSrcObjectCount(pClassId) : 0;     }
    template < class T > inline T* KFbxGetSrc(KFbxObject const *pObject,int pIndex=0)                                               { return pObject ? (T *) pObject->GetSrcObject(T::ClassId,pIndex) : 0; }
    template < class T > inline T* KFbxGetSrc(KFbxObject const *pObject,int pIndex,kFbxClassId pClassId)                            { return pObject ? (T *) pObject->GetSrcObject(pClassId,pIndex) : 0;    }
    template < class T > inline T* KFbxFindSrc(KFbxObject const *pObject,char const *pName,int pIndex=0)                            { return pObject ? (T *) pObject->FindSrcObject(T::ClassId,pName,pIndex) : 0;   }
    template < class T > inline T* KFbxFindSrc(KFbxObject const *pObject,char const *pName,kFbxClassId pClassId,int pIndex=0)       { return pObject ? (T *) pObject->FindSrcObject(pClassId,pName,pIndex) : 0; }

    template < class T > inline bool KFbxDisconnectAllSrc(KFbxObject *pObject,T *VC6Dummy=0)                                        { return pObject->DisconnectAllSrcObject(T::ClassId);   }

    // template access functions DST
    template < class T > inline int KFbxGetDstCount(KFbxObject const *pObject, T const* VC6Dummy = 0)                               { return pObject ? pObject->GetDstObjectCount(T::ClassId) : 0; }
    template < class T > inline int KFbxGetDstCount(KFbxObject const *pObject,kFbxClassId pClassId, T const* VC6Dummy = 0)          { return pObject ? pObject->GetDstObjectCount(pClassId) : 0;     }
    template < class T > inline T* KFbxGetDst(KFbxObject const *pObject,int pIndex=0)                                               { return pObject ? (T *) pObject->GetDstObject(T::ClassId,pIndex) : 0; }
    template < class T > inline T* KFbxGetDst(KFbxObject const *pObject,int pIndex,kFbxClassId pClassId)                            { return pObject ? (T *) pObject->GetDstObject(pClassId,pIndex) : 0;    }
    template < class T > inline T* KFbxFindDst(KFbxObject const *pObject,char const *pName,int pIndex=0)                            { return pObject ? (T *) pObject->FindDstObject(T::ClassId,pName,pIndex) : 0;   }
    template < class T > inline T* KFbxFindDst(KFbxObject const *pObject,char const *pName,kFbxClassId pClassId,int pIndex=0)       { return pObject ? (T *) pObject->FindDstObject(pClassId,pName,pIndex) : 0; }
    template < class T > inline bool KFbxDisconnectAllDst(KFbxObject *pObject,T *VC6Dummy=0)                                        { return pObject->DisconnectAllDstObject(T::ClassId);   }


    /**********************************************************************
    * Object Iterator
    **********************************************************************/
    template<typename KFbxProperty> class KFbxIterator
    {
        public:
            KFbxIterator(KFbxObject const *pObject) : mObject(pObject) {}

            inline KFbxProperty const &GetFirst() { mProperty = mObject->GetFirstProperty(); return mProperty; }
            inline KFbxProperty const &GetNext() { mProperty = mObject->GetNextProperty(mProperty); return mProperty; }

        private:
            KFbxProperty        mProperty;
            KFbxObject const*   mObject;
    };

    class KFbxIteratorSrcBase
    {
    protected:
        KFbxProperty    mProperty;
        kFbxClassId     mClassId;
        int             mSize;
        int             mIndex;
    public:
        inline KFbxIteratorSrcBase(KFbxProperty &pProperty,kFbxClassId pClassId) :
            mClassId(pClassId),
            mProperty(pProperty),
            mSize(0),
            mIndex(-1)
        {
            ResetToBegin();
        }
        inline KFbxIteratorSrcBase(KFbxObject* pObject,kFbxClassId pClassId) :
            mClassId(pClassId),
            mProperty(pObject->RootProperty),
            mSize(0),
            mIndex(-1)
        {
            ResetToBegin();
        }
        inline KFbxObject* GetFirst()
        {
            ResetToBegin();
            return GetNext();
        }
        inline KFbxObject* GetNext()
        {
            mIndex++;
            return ((mIndex>=0) && (mIndex<mSize)) ? mProperty.GetSrcObject(mClassId,mIndex) : NULL;
        }
        inline KFbxObject* GetSafeNext()
        {
            mSize = mProperty.GetSrcObjectCount(mClassId);
            return GetNext();
        }
        inline KFbxObject* GetLast()
        {
            ResetToEnd();
            return GetPrevious();
        }
        inline KFbxObject* GetPrevious()
        {
            mIndex--;
            return ((mIndex>=0) && (mIndex<mSize)) ? mProperty.GetSrcObject(mClassId,mIndex) : NULL;
        }
        inline KFbxObject* GetSafePrevious()
        {
            mSize = mProperty.GetSrcObjectCount(mClassId);
            while (mIndex>mSize) mIndex--;
            return GetPrevious();
        }


    // Internal Access Function
    protected:
        inline void ResetToBegin()
        {
            mSize = mProperty.GetSrcObjectCount(mClassId);
            mIndex = -1;
        }
        inline void ResetToEnd()
        {
            mSize = mProperty.GetSrcObjectCount(mClassId);
            mIndex = mSize;
        }
    };

    template<class Type> class KFbxIteratorSrc : protected KFbxIteratorSrcBase
    {
    public:
        inline KFbxIteratorSrc(KFbxObject* pObject) : KFbxIteratorSrcBase(pObject,Type::ClassId) {}
        inline KFbxIteratorSrc(KFbxProperty& pProperty) : KFbxIteratorSrcBase(pProperty,Type::ClassId) {}
        inline Type *GetFirst()         { return (Type *)KFbxIteratorSrcBase::GetFirst(); }
        inline Type *GetNext()          { return (Type *)KFbxIteratorSrcBase::GetNext(); }
        inline Type *GetSafeNext()      { return (Type *)KFbxIteratorSrcBase::GetSafeNext(); }
        inline Type *GetLast()          { return (Type *)KFbxIteratorSrcBase::GetLast(); }
        inline Type *GetPrevious()      { return (Type *)KFbxIteratorSrcBase::GetPrevious(); }
        inline Type *GetSafePrevious()  { return (Type *)KFbxIteratorSrcBase::GetSafePrevious(); }

    // Internal Access Function
    protected:
    };

    class KFbxIteratorDstBase
    {
    protected:
        KFbxProperty    mProperty;
        kFbxClassId     mClassId;
        int             mSize;
        int             mIndex;
    public:
        inline KFbxIteratorDstBase(KFbxProperty &pProperty,kFbxClassId pClassId) :
            mClassId(pClassId),
            mProperty(pProperty),
            mSize(0),
            mIndex(-1)
        {
            ResetToBegin();
        }
        inline KFbxIteratorDstBase(KFbxObject* pObject,kFbxClassId pClassId) :
            mClassId(pClassId),
            mProperty(pObject->RootProperty),
            mSize(0),
            mIndex(-1)
        {
            ResetToBegin();
        }
        inline KFbxObject* GetFirst()
        {
            ResetToBegin();
            return GetNext();
        }
        inline KFbxObject* GetNext()
        {
            mIndex++;
            return ((mIndex>=0) && (mIndex<mSize)) ? mProperty.GetDstObject(mClassId,mIndex) : NULL;
        }
        inline KFbxObject* GetSafeNext()
        {
            mSize = mProperty.GetDstObjectCount(mClassId);
            return GetNext();
        }
        inline KFbxObject* GetLast()
        {
            ResetToEnd();
            return GetPrevious();
        }
        inline KFbxObject* GetPrevious()
        {
            mIndex--;
            return ((mIndex>=0) && (mIndex<mSize)) ? mProperty.GetDstObject(mClassId,mIndex) : NULL;
        }
        inline KFbxObject* GetSafePrevious()
        {
            mSize = mProperty.GetDstObjectCount(mClassId);
            while (mIndex>mSize) mIndex--;
            return GetPrevious();
        }


    // Internal Access Function
    protected:
        inline void ResetToBegin()
        {
            mSize = mProperty.GetDstObjectCount(mClassId);
            mIndex = -1;
        }
        inline void ResetToEnd()
        {
            mSize = mProperty.GetDstObjectCount(mClassId);
            mIndex = mSize;
        }
    };

    template<class Type> class KFbxIteratorDst : protected KFbxIteratorDstBase
    {
    public:
        inline KFbxIteratorDst(KFbxObject* pObject) : KFbxIteratorDstBase(pObject,Type::ClassId) {}
        inline KFbxIteratorDst(KFbxProperty& pProperty) : KFbxIteratorDstBase(pProperty,Type::ClassId) {}
        inline Type *GetFirst()         { return (Type *)KFbxIteratorDstBase::GetFirst(); }
        inline Type *GetNext()          { return (Type *)KFbxIteratorDstBase::GetNext(); }
        inline Type *GetSafeNext()      { return (Type *)KFbxIteratorDstBase::GetSafeNext(); }
        inline Type *GetLast()          { return (Type *)KFbxIteratorDstBase::GetLast(); }
        inline Type *GetPrevious()      { return (Type *)KFbxIteratorDstBase::GetPrevious(); }
        inline Type *GetSafePrevious()  { return (Type *)KFbxIteratorDstBase::GetSafePrevious(); }
    };

    #define KFbxForEach(Iterator,Object) for ( (Object)=(Iterator).GetFirst(); (Object)!=0; (Object)=(Iterator).GetNext() )
    #define KFbxReverseForEach(Iterator,Object) for ( Object=(Iterator).GetLast(); (Object)!=0;  Object=(Iterator).GetPrevious() )
    #define KFbxForEach_Safe(Iterator,Object) for ( Object=(Iterator).GetFirst(); (Object)!=0; Object=(Iterator).GetSafeNext() )
    #define KFbxReverseForEach_Safe(Iterator,Object) for ( Object=(Iterator).GetLast(); (Object)!=0;  Object=(Iterator).GetSafePrevious() )


#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_OBJECT_H_


