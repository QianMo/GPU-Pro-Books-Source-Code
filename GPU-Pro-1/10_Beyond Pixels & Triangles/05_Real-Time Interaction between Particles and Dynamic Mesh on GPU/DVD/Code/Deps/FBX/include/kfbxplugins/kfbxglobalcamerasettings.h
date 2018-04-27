/*!  \file kfbxglobalcamerasettings.h
 */

#ifndef _FBXSDK_GLOBAL_CAMERA_SETTINGS_H_
#define _FBXSDK_GLOBAL_CAMERA_SETTINGS_H_

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

#include <klib/kstring.h>
#include <klib/kerror.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxNode;
class KFbxSdkManager;
class KFbxScene;
class KFbxCamera;
class KFbxCameraSwitcher;
class KFbxGlobalCameraSettingsProperties;

#define PRODUCER_PERSPECTIVE "Producer Perspective"
#define PRODUCER_TOP "Producer Top"
#define PRODUCER_FRONT "Producer Front"
#define PRODUCER_BACK "Producer Back"
#define PRODUCER_RIGHT "Producer Right"
#define PRODUCER_LEFT "Producer Left"
#define PRODUCER_BOTTOM "Producer Bottom"
#define CAMERA_SWITCHER "Camera Switcher"


/** This class contains the global camera settings.
  * \nosubgrouping
  */
class KFBX_DLL KFbxGlobalCameraSettings
{

    /**
      * \name Default viewing settings Camera
      */
    //@{
    public:
        /** Set the default camera
          * \param pCameraName     Name of the default camera.
          * \return                \c true if camera name is valid, \c false otherwise.
          * \remarks               A valid camera name is either one of the defined tokens (PRODUCER_PERSPECTIVE,
          *                        PRODUCER_TOP, PRODUCER_FRONT, PRODUCER_RIGHT, CAMERA_SWITCHER) or the name
          *                        of a camera inserted in the node tree under the scene's root node.
          */
        bool SetDefaultCamera(char* pCameraName);

        /** Get default camera name
          * \return     The default camera name, or an empty string if the camera name has not been set
          */
        char* GetDefaultCamera();

        //! Restore default settings.
        void RestoreDefaultSettings();

        /** \enum EViewingMode Viewing modes.
          * - \e eSTANDARD
          * - \e eXRAY
          * - \e eMODELS_ONLY
          */
        typedef enum
        {
            eSTANDARD,
            eXRAY,
            eMODELS_ONLY
        } EViewingMode;

        /** Set default viewing mode.
          * \param pViewingMode     Set default viewing mode to either eSTANDARD, eXRAY or eMODELS_ONLY.
          */
        void SetDefaultViewingMode(EViewingMode pViewingMode);

        /** Get default viewing mode.
          * \return     The currently set Viewing mode.
          */
        EViewingMode GetDefaultViewingMode();

    //@}

    /**
      * \name Producer Cameras
      * Producer cameras are global cameras used in MotionBuilder to view the scene.
      * They are not animatable but their default position can be set.
      */
    //@{

        /** Create the default producer cameras.
          */
        void CreateProducerCameras();

        /** Destroy the default producer cameras.
          */
        void DestroyProducerCameras();

        /** Check if the camera is one of the producer cameras
          * \return     true if it is a producer camera false if not
          */
        bool IsProducerCamera(KFbxCamera* pCamera);

        /** Get the camera switcher node.
          * \return Pointer to the camera switcher
          * \remarks This node has a node attribute of type \c KFbxNodeAttribute::eCAMERA_SWITCHER.
          * This node isn't saved if the scene contains no camera.
          * Nodes inserted below are never saved.
          *
          * Camera indices start at 1. Out of range indices are clamped between 1 and the
          * number of cameras in the scene. The index of a camera refers to its order of
          * appearance when searching the node tree depth first.
          *
          * Use function KFbxTakeNode::GetCameraIndex() to get and set the camera index.
          * If a camera is added or removed after camera indices have been set, the camera
          * indices must be updated. It's much simpler to set the camera indices once all
          * cameras have been set.
          *
          * Camera index keys must be set with constant interpolation to make sure camera
          * switches occur exaclty at key time.
          */
        KFbxCameraSwitcher* GetCameraSwitcher();

        /** Set the camera the camera switcher.
          * \return     The reference to the internal Perspective camera.
          */
        void                SetCameraSwitcher(KFbxCameraSwitcher* pSwitcher);

        /** Get a reference to producer perspective camera.
          * \return     The reference to the internal Perspective camera.
          */
        KFbxCamera* GetCameraProducerPerspective();

        /** Get a reference to producer top camera.
          * \return     The reference to the internal Top camera.
          */
        KFbxCamera* GetCameraProducerTop();

        /** Get a reference to producer bottom camera.
          * \return     The reference to the internal Bottom camera.
          */
        KFbxCamera* GetCameraProducerBottom();

        /** Get reference to producer front camera.
          * \return     The reference to the internal Front camera.
          */
        KFbxCamera* GetCameraProducerFront();

        /** Get reference to producer back camera.
          * \return     The reference to the internal Back camera.
          */
        KFbxCamera* GetCameraProducerBack();

        /** Get reference to producer right camera.
          * \return     The reference to the internal Right camera.
          */
        KFbxCamera* GetCameraProducerRight();

        /** Get reference to producer left camera.
          * \return     The reference to the internal Left camera.
          */
        KFbxCamera* GetCameraProducerLeft();

        //@}

        //! Assignment operator.
        const KFbxGlobalCameraSettings& operator=(const KFbxGlobalCameraSettings& pGlobalCameraSettings);

        /**
          * \name Error Management
          * The same error object is shared among instances of this class.
          */
        //@{

        /** Retrieve error object.
         *  \return     Reference to error object.
         */
        KError& GetError();

        /** \enum EError Error identifiers Most of these are only used internally.
          * - \e eNULL_PARAMETER
          * - \e eUNKNOWN_CAMERA_NAME
          * - \e eERROR_COUNT
          */
        typedef enum
        {
            eNULL_PARAMETER,
            eUNKNOWN_CAMERA_NAME,
            eERROR_COUNT
        } EError;

        /** Get last error code.
         *  \return     Last error code.
         */
        EError GetLastErrorID() const;

        /** Get last error string.
         *  \return     Textual description of the last error.
         */
        const char* GetLastErrorString() const;

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
        bool CopyProducerCamera(KString pCameraName, const KFbxCamera* pCamera);
        int  GetProducerCamerasCount() { return 7; }
    private:
        KFbxGlobalCameraSettings(KFbxSdkManager& pManager, KFbxScene& pScene);
        ~KFbxGlobalCameraSettings();

        KFbxGlobalCameraSettingsProperties* mPH;

        friend class KFbxScene;

    #endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_GLOBAL_CAMERA_SETTINGS_H_


