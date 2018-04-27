/*!  \file kfbxclonemanager.h
 */

#ifndef _FBXSDK_CLONE_MANAGER_H_
#define _FBXSDK_CLONE_MANAGER_H_

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
#include <kfbxplugins/kfbxobject.h>
#include <fbxcore/kfbxquery.h>
#include <fbxcore/fbxcollection/kfbxpropertymap.h>
#include <klib/karrayul.h>

#include <fbxfilesdk_nsbegin.h>



/** \brief The clone manager is a utility for cloning entire networks of KFbxObjects.
  *        Options are availible for specifying how the clones inherit the connections
  *        of the original.
  *
  * \nosubgrouping
  */
class KFBX_DLL KFbxCloneManager
{
public:

    //! Maximum depth to clone dependents.
    static const int sMaximumCloneDepth;

    //! connect to objects that are connected to original object
    static const int sConnectToOriginal;

    /** connect to clones of objects that are connected to original object
      * (only if those original objects are also in the clone set)
      */
    static const int sConnectToClone;

    /** This represents an element in a set of objects to be cloned
      */
    struct KFBX_DLL CloneSetElement
    {
    public:
        CloneSetElement( int pSrcPolicy = 0,
                         int pExternalDstPolicy = 0,
                         KFbxObject::ECloneType pCloneType = KFbxObject::eREFERENCE_CLONE );

        //! the type of cloning to perform
        KFbxObject::ECloneType mType;

        /** Policy on how to handle source connections on the original object. Valid values are 0
          * or any bitwise OR'd combination of sConnectToOriginal, and sConnectToClone.
          */
        int mSrcPolicy;

        /** policy on how to handle destination connections on the original object to
          * objects NOT in the clone set. (Destination connections to objects in the set
          * are handled by that object's source policy) Valid values are 0 or sConnectToOriginal.
          */
        int mExternalDstPolicy;

        /** This is a pointer to the newly created clone.
          * It is set after the call to KFbxCloneManager::Clone()
          */
        KFbxObject* mObjectClone;
    };

    /** Functor to compare object pointers
      */
    class KFBX_DLL KFbxObjectCompare {
        public:
        inline int operator()(KFbxObject* const& pKeyA, KFbxObject* const& pKeyB) const
        {
            return (pKeyA < pKeyB) ? -1 : ((pKeyB < pKeyA) ? 1 : 0);
        }
    };

    /** The CloneSet is a collection of pointers to objects that will be cloned in Clone()
      * Attached to each object is a CloneSetElement. Its member variables dictate how
      * the corresponding object will be cloned, and how it will inherit connections
      * on the original object.
      */
    typedef KMap<KFbxObject*,CloneSetElement,KFbxObjectCompare> CloneSet;

    /** Constructor
      */
    KFbxCloneManager();

    /** Destructor
      */
    virtual ~KFbxCloneManager();

    /** Clone all objects in the set using the given policies for duplication
      * of connections. Each CloneSetElement in the set will have its mObjectClone
      * pointer set to the newly created clone.
      * \param pSet Set of objects to clone
      * \param pContainer This object (typically a scene or document) will contain the new clones
      * \return true if all objects were cloned, false otherwise.
      */
    virtual bool Clone( CloneSet& pSet, KFbxObject* pContainer = NULL ) const;

    /** Add all dependents of the given object to the CloneSet.
      * Dependents of items already in the set are ignored to prevent
      * infinite recursion on cyclic dependencies.
      * \param pSet The set to add items.
      * \param pObject Object to add dependents to
	  * \param pCloneOptions  
      * \param pTypes Types of dependent objects to consider
      * \param pDepth Maximum recursive depth. Valid range is [0,sMaximumCloneDepth]
        */
    virtual void AddDependents( CloneSet& pSet,
                        const KFbxObject* pObject,
                        const CloneSetElement& pCloneOptions = CloneSetElement(),
                        KFbxCriteria pTypes = KFbxCriteria::ObjectType(KFbxObject::ClassId),
                        int pDepth = sMaximumCloneDepth ) const;


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
        bool CloneConnections( CloneSet::RecordType* pIterator, const CloneSet& pSet ) const;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

#include <fbxfilesdk_nsend.h>

#endif //_FBXSDK_CLONE_MANAGER_H_
