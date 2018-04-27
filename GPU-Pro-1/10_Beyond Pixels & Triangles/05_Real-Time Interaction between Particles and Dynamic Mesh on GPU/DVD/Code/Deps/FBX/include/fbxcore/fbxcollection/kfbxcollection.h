#ifndef _FBXSDK_COLLECTION_H_
#define _FBXSDK_COLLECTION_H_

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

// FBX includes

#include <kfbxplugins/kfbxobject.h>
#include <kfbxplugins/kfbxsdkmanager.h>

#include <kfbxevents/kfbxevents.h>

#include <klib/kerror.h>

#include <fbxfilesdk_nsbegin.h>

    class KFbxCriteria;

    /** This class contains objects.
      * \nosubgrouping
      * This class also provides access to global settings and take information.
      *
      */
    class KFBX_DLL KFbxCollection : public KFbxObject
    {
        KFBXOBJECT_DECLARE(KFbxCollection,KFbxObject);

        /**
        * \name Collection member management
        */
        //@{
        public:
            //! Delete all contained objects.
            virtual void    Clear();
            //! Add a member.
            inline  void    AddMember(KFbxObject *pMember)      { ConnectSrcObject(pMember); }
            //! Remove a member.
            inline  void    RemoveMember(KFbxObject *pMember)   { DisconnectSrcObject(pMember); }
            //! Find a member.
            template <class T> inline T *       FindMember(T const *pfbxType, const char *pName) { return FindSrcObject(pfbxType, pName); }

            //! Return the number of objects in the collection.
            inline int                          GetMemberCount () const { return GetSrcObjectCount(); }
            //! Return the number of objects in the collection.
            template < class T > inline int     GetMemberCount (T const *pFBX_TYPE) const { return GetSrcObjectCount(T::ClassId); }
            int                                 GetMemberCount( KFbxCriteria pCriteria ) const;

            //! Return the index'th member of the collection.
            inline KFbxObject*                  GetMember (int pIndex=0) const                  { return GetSrcObject(pIndex); }
            //! Return the index'th member of the collection.
            template < class T > inline T*      GetMember (T const *pFBX_TYPE, int pIndex=0) const  { return (T *)GetSrcObject(T::ClassId,pIndex); }
            KFbxObject*                         GetMember (KFbxCriteria pCriteria, int pIndex=0) const;
            //! Is an object part of the collection.

            virtual bool    IsMember(KFbxObject *pMember) const;
        //@}

        /**
        * \name Selection managent
        */
        //@{
        public:
            //! Select/Unselect all contained objects.
            virtual void    SetSelectedAll(bool pSelection);
            //! Select/Unselect objects.
            virtual void    SetSelected(KFbxObject* pObject,bool pSelection);
            //! Get Select/Unselect .
            virtual bool    GetSelected(KFbxObject* pSelection);
        //@}

    ///////////////////////////////////////////////////////////////////////////////
    //  WARNING!
    //  Anything beyond these lines may not be Collectioned accurately and is
    //  subject to change without notice.
    ///////////////////////////////////////////////////////////////////////////////
    #ifndef DOXYGEN_SHOULD_SKIP_THIS

        // Constructor / Destructor
        protected:
            virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;
            KFbxCollection(KFbxSdkManager& pManager, char const* pName);
            ~KFbxCollection();
            virtual void Construct  (const KFbxCollection* pFrom);
            virtual void Destruct   (bool pRecursive, bool pDependents);


    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_Collection_H_


