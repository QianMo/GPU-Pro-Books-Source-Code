/*!  \file kfbxcameraswitcher.h
 */

#ifndef _FBXSDK_CAMERA_SWITCHER_H_
#define _FBXSDK_CAMERA_SWITCHER_H_

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

#include <klib/karrayul.h>

#include <kfbxplugins/kfbxnodeattribute.h>

#include <fbxfilesdk_nsbegin.h>

    class KFbxTakeNode;
    class KFbxSdkManager;

    struct KFbxReaderFbx7Impl;

    /** \brief This node attribute contains methods for accessing the properties of a camera switcher.
      * \nosubgrouping
      */
    class KFBX_DLL KFbxCameraSwitcher : public KFbxNodeAttribute
    {
        KFBXOBJECT_DECLARE(KFbxCameraSwitcher,KFbxNodeAttribute);

        public:
        /**
          * \name Properties
          */
        //@{
            KFbxPropertyInteger1        CameraIndex;
        //@}

        //! Return the type of node attribute which is EAttributeType::eCAMERA_SWITCHER.
    virtual EAttributeType GetAttributeType() const;

        /**
          * \name Default Animation Values
          * These functions provides direct access to default animation values specific to a camera switcher. The default animation
          * values are found in the default take node of the associated node. These functions only work if the camera switcher has been
          * associated with a node.
          *
          * Camera indices start at 1. Out of range indices are clamped between 1 and the number of cameras in the scene. The index of a
          * camera refers to its order of appearance when searching the node tree depth first.
          */
        //@{

        /** Get default camera index.
          * \return Camera index. The return value is an integer between 1 and the number
          * of cameras in the scene, or 0 if there are no default camera set in the camera switcher.
          */
        int GetDefaultCameraIndex();

        /** Set default camera index.
          * \param pIndex Id of the camera to set as default. This parameter has an integer
          * scale from 1 to the number of cameras in the scene. Its default value is 1 if
          * there is at least one camera in the camera switcher, 0 if there are none.
          * No validation checks are made.
          */
        void SetDefaultCameraIndex(int pIndex);

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

    public:

        // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

    protected:

        KFbxCameraSwitcher(KFbxSdkManager& pManager, char const* pName);
        virtual ~KFbxCameraSwitcher();

    //! Assignment operator.
    KFbxCameraSwitcher& operator=(KFbxCameraSwitcher const& pCameraIndex);

        virtual bool ConstructProperties(bool pForceSet);

        virtual KString     GetTypeName() const;
        virtual KStringList GetTypeFlags() const;

        void AddCameraName(char* pCameraName);
        char* GetCameraName(kUInt pIndex);
        kUInt GetCameraNameCount();

        KArrayTemplate<KString*> mCameraNameList;

        friend class KFbxWriterFbx6;
        friend class KFbxReaderFbx;
        friend class KFbxReaderFbx6;

        friend struct KFbxReaderFbx7Impl;
        friend struct KFbxWriterFbx7Impl;
    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

    };

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_CAMERA_SWITCHER_H_


