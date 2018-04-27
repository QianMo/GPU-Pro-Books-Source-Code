/*!  \file kfbxcamera.h
 */

#ifndef _FBXSDK_CAMERA_H_
#define _FBXSDK_CAMERA_H_

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

#include <kfbxplugins/kfbxnodeattribute.h>
#include <kfbxplugins/kfbxcolor.h>

#include <kfbxmath/kfbxvector4.h>

#include <klib/kstring.h>

#ifdef KARCH_DEV_MACOSX_CFM
    #include <CFURL.h>
    #include <Files.h>
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxTexture;
class KFbxSdkManager;
class KFbxMatrix;
class KFbxXMatrix;
class KTime;

/** \brief This node attribute contains methods for accessing the properties of a camera.
  * \nosubgrouping
  * A camera can be set to automatically point at and follow
  * another node in the hierarchy. To do this, the focus source
  * must be set to ECameraFocusSource::eCAMERA_INTEREST and the
  * followed node associated with function KFbxNode::SetTarget().
  */
class KFBX_DLL KFbxCamera : public KFbxNodeAttribute
{
    KFBXOBJECT_DECLARE(KFbxCamera,KFbxNodeAttribute);

public:
    //! Return the type of node attribute which is EAttributeType::eCAMERA.
    virtual EAttributeType GetAttributeType() const;

    //! Reset the camera to default values.
    void Reset();

    /**
      * \name Camera Position and Orientation Functions
      */
    //@{

    /** Set the default XYZ camera position.
      * \param pPosition     X, Y and Z values of the camera position, expressed as a vector.
      * \remarks             If this attribute is not yet attached to a node, this method does nothing.
      *
      * \remarks This function is deprecated. Use property Position.Set(pPosition) instead.
      *
      */
    K_DEPRECATED void SetPosition(const KFbxVector4& pPosition);

    /** Get the default position of the camera.
      * \return      The X, Y and Z values of the default position of the camera.
      * \remarks     If this attribute is not attached to a node yet, this method will return 0,0,0.
      *
      * \remarks This function is deprecated. Use property Position.Get() instead.
      *
      */
    K_DEPRECATED KFbxVector4 GetPosition() const;

    /** Set the camera's Up vector.
      * \param pVector     The X, Y and Z values for the Up vector.
      *
      * \remarks This function is deprecated. Use property UpVector.Set(pVector) instead.
      *
      */
    K_DEPRECATED void SetUpVector(const KFbxVector4& pVector);

    /** Get the current Up vector.
      * \return     The X, Y and Z values of the currently set Up vector.
      *
      * \remarks This function is deprecated. Use property UpVector.Get() instead.
      *
      */
    K_DEPRECATED KFbxVector4 GetUpVector() const;

    /** Set the default point the camera is looking at.
      * \param pPosition     X, Y and Z values of the camera interest point.
      * \remarks             During the computations of the camera position and orientation,
      *                      the default camera interest position is overridden by the position of a
      *                      valid target in the parent node.
      *
      * \remarks This function is deprecated. Use property InterestPosition.Set(pPosition) instead.
      *
      */
    K_DEPRECATED void SetDefaultCameraInterestPosition(const KFbxVector4& pPosition);

    /** Return the camera interest to default coordinates.
      * \return      The X,Y and Z values of the default interest point.
      * \remarks     During the computations of the camera position and orientation,
      *              the default camera interest position is overridden by the position of a
      *              valid target in the parent node.
      *
      * \remarks This function is deprecated. Use property InterestPosition.Get() instead.
      *
      */
    K_DEPRECATED KFbxVector4 GetDefaultCameraInterestPosition() const;

    /** Set the camera roll angle.
      * \param pRoll     The roll angle in degrees.
      *
      * \remarks This function is deprecated. Use property Roll.Set(pRoll) instead.
      *
      */
    K_DEPRECATED void SetRoll(double pRoll);

    /** Get the camera roll angle.
      * \return     The roll angle in degrees.
      *
      * \remarks This function is deprecated. Use property Roll.Get() instead.
      *
      */
    K_DEPRECATED double GetRoll() const;

    /** Set the camera turntable angle.
      * \param pTurnTable     The turntable angle in degrees.
      *
      * \remarks This function is deprecated. Use property TurnTable.Set(pTurnTable) instead.
      *
      */
    K_DEPRECATED void SetTurnTable(double pTurnTable);

    /** Get the camera turntable angle.
      * \return     The turntable angle in degrees.
      *
      * \remarks This function is deprecated. Use property TurnTable.Get() instead.
      *
      */
    K_DEPRECATED double GetTurnTable() const;

    /** Camera projection types.
      * \enum ECameraProjectionType Camera projection types.
      * - \e ePERSPECTIVE
      * - \e eORTHOGONAL
      * \remarks     By default, the camera projection type is set to ePERSPECTIVE.
      *              If the camera projection type is set to eORTHOGONAL, the following options
      *              are not relevant:
      *                   - aperture format
      *                   - aperture mode
      *                   - aperture width and height
      *                   - angle of view/focal length
      *                   - squeeze ratio
      */
    typedef enum
    {
        ePERSPECTIVE,
        eORTHOGONAL
    } ECameraProjectionType;

    /** Set the camera projection type.
      * \param pProjectionType     The camera projection identifier.
      *
      * \remarks This function is deprecated. Use property ProjectionType.Set(pProjectionType) instead.
      *
      */
    K_DEPRECATED void SetProjectionType(ECameraProjectionType pProjectionType);

    /** Get the camera projection type.
      * \return     The camera's current projection identifier.
      *
      * \remarks This function is deprecated. Use property ProjectionType.Get() instead.
      *
      */
    K_DEPRECATED ECameraProjectionType GetProjectionType() const;

    //@}

    /**
      * \name Viewing Area Functions
      */
    //@{

    /** \enum ECameraFormat Camera formats.
      * - \e eCUSTOM_FORMAT
      * - \e eD1_NTSC
      * - \e eNTSC
      * - \e ePAL
      * - \e eD1_PAL
      * - \e eHD
      * - \e e640x480
      * - \e e320x200
      * - \e e320x240
      * - \e e128x128
      * - \e eFULL_SCREEN
      */
    typedef enum
    {
        eCUSTOM_FORMAT,
        eD1_NTSC,
        eNTSC,
        ePAL,
        eD1_PAL,
        eHD,
        e640x480,
        e320x200,
        e320x240,
        e128x128,
        eFULL_SCREEN
    } ECameraFormat;

    /** Set the camera format.
      * \param pFormat     The camera format identifier.
      * \remarks           Changing the camera format sets the camera aspect
      *                    ratio mode to eFIXED_RESOLUTION and modifies the aspect width
      *                    size, height size, and pixel ratio accordingly.
      */
    void SetFormat(ECameraFormat pFormat);

    /** Get the camera format.
      * \return     The current camera format identifier.
      */
    ECameraFormat GetFormat() const;

    /** \enum ECameraAspectRatioMode Camera aspect ratio modes.
      * - \e eWINDOW_SIZE
      * - \e eFIXED_RATIO
      * - \e eFIXED_RESOLUTION
      * - \e eFIXED_WIDTH
      * - \e eFIXED_HEIGHT
      */
    typedef enum
    {
        eWINDOW_SIZE,
        eFIXED_RATIO,
        eFIXED_RESOLUTION,
        eFIXED_WIDTH,
        eFIXED_HEIGHT
    } ECameraAspectRatioMode;

    /** Set the camera aspect.
      * \param pRatioMode     Camera aspect ratio mode.
      * \param pWidth         Camera aspect width, must be a positive value.
      * \param pHeight        Camera aspect height, must be a positive value.
      * \remarks              Changing the camera aspect sets the camera format to eCustom.
      *                            - If the ratio mode is eWINDOW_SIZE, both width and height values aren't relevant.
      *                            - If the ratio mode is eFIXED_RATIO, the height value is set to 1.0 and the width value is relative to the height value.
      *                            - If the ratio mode is eFIXED_RESOLUTION, both width and height values are in pixels.
      *                            - If the ratio mode is eFIXED_WIDTH, the width value is in pixels and the height value is relative to the width value.
      *                            - If the ratio mode is eFIXED_HEIGHT, the height value is in pixels and the width value is relative to the height value.
      */
    void SetAspect(ECameraAspectRatioMode pRatioMode, double pWidth, double pHeight);

    /** Get the camera aspect ratio mode.
      * \return     The current aspect ratio identifier.
      */
    ECameraAspectRatioMode GetAspectRatioMode() const;

    /** Get the aspect width.
      * \return     The aspect width value or an undefined value if aspect ratio mode is set to eWINDOW_SIZE.
      *
      * \remarks This function is deprecated. Use property AspectWidth.Get() instead.
      *
      */
    K_DEPRECATED double GetAspectWidth() const;

    /** Get the aspect height.
      * \return     The aspect height value or an undefined value if aspect ratio mode is set to eWINDOW_SIZE.
      *
      * \remarks This function is deprecated. Use property AspectHeight.Get() instead.
      *
      */
    K_DEPRECATED double GetAspectHeight() const;

    /** Set the pixel ratio.
      * \param pRatio     The pixel ratio value.
      * \remarks          The value must be a positive number. Comprised between 0.05 and 20.0. Values
      *                   outside these limits will be clamped. Changing the pixel ratio sets the camera format to eCUSTOM_FORMAT.
      */
    void SetPixelRatio(double pRatio);

    /** Get the pixel ratio.
      * \return     The current camera's pixel ratio value.
      */
    double GetPixelRatio() const;

    /** Set the near plane distance from the camera.
      * The near plane is the minimum distance to render a scene on the camera display.
      * \param pDistance     The near plane distance value.
      * \remarks             The near plane value is limited to the range [0.001, 600000.0] and
      *                      must be inferior to the far plane value.
      */
    void SetNearPlane(double pDistance);

    /** Get the near plane distance from the camera.
      * The near plane is the minimum distance to render a scene on the camera display.
      * \return     The near plane value.
      */
    double GetNearPlane() const;

    /** Set the far plane distance from camera.
      * The far plane is the maximum distance to render a scene on the camera display.
      * \param pDistance     The far plane distance value.
      * \remarks             The far plane value is limited to the range [0.001, 600000.0] and
      *                      must be superior to the near plane value.
      */
    void SetFarPlane(double pDistance);

    /** Get the far plane distance from camera.
      * The far plane is the maximum distance to render a scene on the camera display.
      * \return     The far plane value.
      */
    double GetFarPlane() const;

    /** Set interactive camera lock flag.
      * \param pMouseLock     If \c true, disable modifications of the view area controls using mouse and keyboard commands.
      * \remarks              It is the responsibility of the client application to perform the required tasks according to the state
      *                       of this flag.
      *
      * \remarks This function is deprecated. Use property LockMode.Set(pMouseLock) instead.
      *
      */
    K_DEPRECATED void SetMouseLock(bool pMouseLock);

    /** Get the camera lock flag.
      * \return     \c true If modifications of the view area controls using mouse and keyboard commands are disabled.
      * \remark     It is the responsibility of the client application to perform the required tasks according to the state
      *             of this flag.
      *
      * \remarks This function is deprecated. Use property LockMode.Get() instead.
      *
      */
    K_DEPRECATED bool GetMouseLock() const;

    //@}

    /**
      * \name Aperture and Film Functions
      * The aperture mode determines which values drive the camera aperture. When the aperture mode is \e eHORIZONTAL_AND_VERTICAL,
      * \e eHORIZONTAL or \e eVERTICAL, the field of view is used. When the aperture mode is \e eFOCAL_LENGTH, the focal length is used.
      *
      * It is possible to convert the aperture mode into field of view or vice versa using functions ComputeFieldOfView and
      * ComputeFocalLength. These functions use the camera aperture width and height for their computation.
      */
    //@{

    /** \enum ECameraApertureFormat Camera aperture formats.
      * - \e eCUSTOM_APERTURE_FORMAT
      * - \e e16MM_THEATRICAL
      * - \e eSUPER_16MM
      * - \e e35MM_ACADEMY
      * - \e e35MM_TV_PROJECTION
      * - \e e35MM_FULL_APERTURE
      * - \e e35MM_185_PROJECTION
      * - \e e35MM_ANAMORPHIC
      * - \e e70MM_PROJECTION
      * - \e eVISTAVISION
      * - \e eDYNAVISION
      * - \e eIMAX
      */
    typedef enum
    {
        eCUSTOM_APERTURE_FORMAT = 0,
        e16MM_THEATRICAL,
        eSUPER_16MM,
        e35MM_ACADEMY,
        e35MM_TV_PROJECTION,
        e35MM_FULL_APERTURE,
        e35MM_185_PROJECTION,
        e35MM_ANAMORPHIC,
        e70MM_PROJECTION,
        eVISTAVISION,
        eDYNAVISION,
        eIMAX
    } ECameraApertureFormat;

    /** Set the camera aperture format.
      * \param pFormat     The camera aperture format identifier.
      * \remarks           Changing the aperture format modifies the aperture width, height, and squeeze ratio accordingly.
      */
    void SetApertureFormat(ECameraApertureFormat pFormat);

    /** Get the camera aperture format.
      * \return     The camera's current aperture format identifier.
      */
    ECameraApertureFormat GetApertureFormat() const;

    /** \enum ECameraApertureMode
      * Camera aperture modes. The aperture mode determines which values drive the camera aperture. If the aperture mode is \e eHORIZONTAL_AND_VERTICAL,
      * \e eHORIZONTAL, or \e eVERTICAL, then the field of view is used. If the aperture mode is \e eFOCAL_LENGTH, then the focal length is used.
      * - \e eHORIZONTAL_AND_VERTICAL
      * - \e eHORIZONTAL
      * - \e eVERTICAL
      * - \e eFOCAL_LENGTH
      */
    typedef enum
    {
        eHORIZONTAL_AND_VERTICAL,
        eHORIZONTAL,
        eVERTICAL,
        eFOCAL_LENGTH
    } ECameraApertureMode;

    /** Set the camera aperture mode.
      * \param pMode     The camera aperture mode identifier.
      */
    void SetApertureMode(ECameraApertureMode pMode);

    /** Get the camera aperture mode.
      * \return     The camera's current aperture mode identifier.
      */
    ECameraApertureMode GetApertureMode() const;

    /** Set the camera aperture width in inches.
      * \param pWidth     The aperture width value.
      * \remarks          Must be a positive value. The minimum accepted value is 0.0001.
      *                   Changing the aperture width sets the camera aperture format to eCUSTOM_FORMAT.
      */
    void SetApertureWidth(double pWidth);

    /** Get the camera aperture width in inches.
      * \return     The camera's current aperture width value in inches.
      */
    double GetApertureWidth() const;

    /** Set the camera aperture height in inches.
      * \param pHeight     The aperture height value.
      * \remarks           Must be a positive value. The minimum accepted value is 0.0001.
      *                    Changing the aperture height sets the camera aperture format to eCUSTOM_FORMAT.
      */
    void SetApertureHeight(double pHeight);

    /** Get the camera aperture height in inches.
      * \return     The camera's current aperture height value in inches.
      */
    double GetApertureHeight() const;

    /** Set the squeeze ratio.
      * \param pRatio      The sqeeze ratio value.
      * \remarks           Must be a positive value. The minimum accepted value is 0.0001.
      *                    Changing the squeeze ratio sets the camera aperture format to eCUSTOM_FORMAT.
      */
    void SetSqueezeRatio(double pRatio);

    /** Get the camera squeeze ratio.
      * \return     The camera's current squeeze ratio value.
      */
    double GetSqueezeRatio() const;

    /** Set the camera ortographic zoom
      * \param pOrthoZoom     This parameter's default value is 1.0.
      * \remarks              This parameter is not used if the camera is not orthographic.
      *
      * \remarks This function is deprecated. Use property OrthoZoom.Set(pOrthoZoom) instead.
      *
      */
    K_DEPRECATED void SetOrthoZoom(const double& pOrthoZoom);

    /** Get the camera ortographic zoom.
      * \return     The camera's current orthographic zoom.
      *
      * \remarks This function is deprecated. Use property OrthoZoom.Get() instead.
      *
      */
    K_DEPRECATED double GetOrthoZoom() const;

    /** \enum ECameraGateFit
      * Camera gate fit modes.
      * - \e eNO_FIT            No resoluton gate fit.
      * - \e eVERTICAL_FIT      Fit the resolution gate vertically within the film gate.
      * - \e eHORIZONTAL_FIT    Fit the resolution gate horizontally within the film gate.
      * - \e eFILL_FIT          Fit the resolution gate within the film gate.
      * - \e eOVERSCAN_FIT      Fit the film gate within the resolution gate.
      * - \e eSTRETCH_FIT       Fit the resolution gate to the film gate.
      */
    typedef enum
    {
        eNO_FIT,
        eVERTICAL_FIT,
        eHORIZONTAL_FIT,
        eFILL_FIT,
        eOVERSCAN_FIT,
        eSTRETCH_FIT
    } ECameraGateFit;

    /** Set the camera gate fit.
      * \param pGateFit     This parameter's default value is eNO_FIT.
      *
      * \remarks This function is deprecated. Use property GateFit.Set(pGateFit) instead.
      *
      */
    K_DEPRECATED void SetGateFit(const ECameraGateFit pGateFit);

    /** Get the camera gate fit.
      * \return     The camera's current gate fit.
      *
      * \remarks This function is deprecated. Use property GateFit.Get() instead.
      *
      */
    K_DEPRECATED ECameraGateFit GetGateFit() const;

    /** Compute the angle of view based on the given focal length, the aperture width, and aperture height.
      * \param pFocalLength     The focal length in millimeters
      * \return                 The computed angle of view in degrees
      */
    double ComputeFieldOfView(double pFocalLength) const;

    /** Compute the focal length based on the given angle of view, the aperture width, and aperture height.
      * \param pAngleOfView     The angle of view in degrees
      * \return                 The computed focal length in millimeters
      */
    double ComputeFocalLength(double pAngleOfView);
    //@}

    /**
      * \name Background Functions
      */
    //@{

    /** Set the associated background image file.
      * \param pFileName     The path of the background image file.
      * \remarks             The background image file name must be valid.
      */
    void SetBackgroundFileName(const char* pFileName);

#ifdef KARCH_DEV_MACOSX_CFM
    bool SetBackgroundFile(const FSSpec &pMacFileSpec);
    bool SetBackgroundFile(const FSRef &pMacFileRef);
    bool SetBackgroundFile(const CFURLRef &pMacURL);
#endif

    /** Get the background image file name.
      * \return     Pointer to the background filename string or \c NULL if not set.
      */
    char const* GetBackgroundFileName() const;

#ifdef KARCH_DEV_MACOSX_CFM
    bool GetBackgroundFile(FSSpec &pMacFileSpec) const;
    bool GetBackgroundFile(FSRef &pMacFileRef) const;
    bool GetBackgroundFile(CFURLRef &pMacURL) const;
#endif

    /** Set the media name associated to the background image file.
      * \param pFileName     The media name of the background image file.
      * \remarks             The media name is a unique name used to identify the background image file.
      */
    void SetBackgroundMediaName(const char* pFileName);

    /** Get the media name associated to the background image file.
      * \return     Pointer to the media name string or \c NULL if not set.
      */
    char const* GetBackgroundMediaName() const;

    /** \enum ECameraBackgroundDisplayMode Background display modes.
      * - \e eDISABLED
      * - \e eALWAYS
      * - \e eWHEN_MEDIA
      */
    typedef enum
    {
        eDISABLED,
        eALWAYS,
        eWHEN_MEDIA
    } ECameraBackgroundDisplayMode;

    /** Set the background display mode.
      * \param pMode     The background display mode identifier.
      *
      * \remarks This function is deprecated. Use property ViewFrustumBackPlaneMode.Set(pMode) instead.
      *
      */
    K_DEPRECATED void SetBackgroundDisplayMode(ECameraBackgroundDisplayMode pMode);

    /** Get the background display mode.
      * \return     The currently set background display mode identifier.
      *
      * \remarks This function is deprecated. Use property ViewFrustumBackPlaneMode.Get() instead.
      *
      */
    K_DEPRECATED ECameraBackgroundDisplayMode GetBackgroundDisplayMode() const;

    /** \enum ECameraBackgroundDrawingMode Background drawing modes.
      * - \e eBACKGROUND                 Image is drawn behind models.
      * - \e eFOREGROUND                 Image is drawn in front of models based on alpha channel.
      * - \e eBACKGROUND_AND_FOREGROUND  Image is drawn behind and in front of models depending on alpha channel.
      */
    typedef enum
    {
        eBACKGROUND,
        eFOREGROUND,
        eBACKGROUND_AND_FOREGROUND
    } ECameraBackgroundDrawingMode;

    /** Set the background drawing mode.
      * \param pMode     The background drawing mode identifier.
      *
      * \remarks This function is deprecated. Use property BackgroundMode.Set(pMode) instead.
      *
      */
    K_DEPRECATED void SetBackgroundDrawingMode(ECameraBackgroundDrawingMode pMode);

    /** Get the background drawing mode.
      * \return The currently set background drawing mode identifier.
      *
      * \remarks This function is deprecated. Use property BackgroundMode.Get() instead.
      *
      */
    K_DEPRECATED ECameraBackgroundDrawingMode GetBackgroundDrawingMode() const;

    /** Set the foreground matte threshold flag.
      * \param pEnable     If \c true enable foreground matte threshold.
      * \remarks           It is the responsibility of the client application to perform the required tasks according to the state
      *                    of this flag.
      *
      * \remarks This function is deprecated. Use property ForegroundTransparent.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetForegroundMatteThresholdEnable(bool pEnable);

    /** Get the foreground matte threshold flag.
      * \return            \c true if foreground matte threshold is enabled, otherwise \c false.
      * \remark            It is the responsibility of the client application to perform the required tasks according to the state
      *                    of this flag.
      *
      * \remarks This function is deprecated. Use property ForegroundTransparent.Get() instead.
      *
      */
    K_DEPRECATED bool GetForegroundMatteThresholdEnable() const;

    /** Set foreground matte threshold.
      * \param pThreshold     Threshold value on a range from 0.0 to 1.0.
      * \remarks              This option is only relevant if the background drawing mode is set to eFOREGROUND or eBACKGROUND_AND_FOREGROUND.
      *
      * \remarks This function is deprecated. Use property BackgroundAlphaTreshold.Set(pThreshold) instead.
      *
      */
    K_DEPRECATED void SetForegroundMatteThreshold(double pThreshold);

    /** Get foreground matte threshold.
      * \return      Threshold value on a range from 0.0 to 1.0.
      * \remarks     This option is only relevant if the background drawing mode is set to eFOREGROUND or eBACKGROUND_AND_FOREGROUND.
      *
      * \remarks This function is deprecated. Use property BackgroundAlphaTreshold.Get() instead.
      *
      */
    K_DEPRECATED double GetForegroundMatteThreshold() const;

    /** \enum ECameraBackgroundPlacementOptions Background placement options.
      * - \e eFIT
      * - \e eCENTER
      * - \e eKEEP_RATIO
      * - \e eCROP
      */
    typedef enum
    {
        eFIT = 1<<0,
        eCENTER = 1<<1,
        eKEEP_RATIO = 1<<2,
        eCROP = 1<<3
    } ECameraBackgroundPlacementOptions;

    /** Set background placement options.
      * \param pOptions     Bitwise concatenation of one or more background placement options.
      */
    K_DEPRECATED void SetBackgroundPlacementOptions(kUInt pOptions);

    /** Get background placement options.
      * \return     The bitwise concatenation of the currently set background placement options.
      */
    kUInt GetBackgroundPlacementOptions() const;

    /** ECamerabackgroundDistanceMode Background distance modes.
      * - \e eRELATIVE_TO_INTEREST
      * - \e eABSOLUTE_FROM_CAMERA
      */
    typedef enum
    {
        eRELATIVE_TO_INTEREST,
        eABSOLUTE_FROM_CAMERA
    } ECameraBackgroundDistanceMode;

    /** Set the background distance mode.
      * \param pMode     The background distance mode identifier.
      *
      * \remarks This function is deprecated. Use property BackPlaneDistanceMode.Set(pMode) instead.
      *
      */
    K_DEPRECATED void SetBackgroundDistanceMode(ECameraBackgroundDistanceMode pMode);

    /** Get the background distance mode.
      * \return     Return the background distance mode identifier.
      *
      * \remarks This function is deprecated. Use property BackPlaneDistanceMode.Get() instead.
      *
      */
    K_DEPRECATED ECameraBackgroundDistanceMode GetBackgroundDistanceMode() const;

    /** Set the background distance.
      * \param pDistance     Distance of the background plane. This value can be either relative to the camera interest point or
      *                      absolute from the camera position.
      *
      * \remarks This function is deprecated. Use property BackPlaneDistance.Set(pDistance) instead.
      *
      */
    K_DEPRECATED void SetBackgroundDistance(double pDistance);

    /** Get the background distance.
      * \return     The distance of the background plane.
      *
      * \remarks This function is deprecated. Use property BackPlaneDistance.Get() instead.
      *
      */
    K_DEPRECATED double GetBackgroundDistance() const;

    //@}

    /**
      * \name Camera View Functions
      * It is the responsibility of the client application to perform the required tasks according to the state
      * of the options that are either set or returned by these methods.
      */
    //@{

    /** Change the camera interest visibility flag.
      * \param pEnable     Set to \c true if the camera interest is shown.
      *
      * \remarks This function is deprecated. Use property ViewCameraToLookAt.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetViewCameraInterest(bool pEnable);

    /** Get current visibility state of the camera interest.
      * \return     \c true if the camera interest is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ViewCameraToLookAt.Get() instead.
      *
      */
    K_DEPRECATED bool GetViewCameraInterest() const;

    /** Change the camera near and far planes visibility flag.
      * \param pEnable     Set to \c true if the near and far planes are shown.
      *
      * \remarks This function is deprecated. Use property ViewFrustumNearFarPlane.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetViewNearFarPlanes(bool pEnable);

    /** Get current visibility state of the camera near and far planes.
      * \return     \c true if the near and far planes are shown.
      *
      * \remarks This function is deprecated. Use property ViewFrustumNearFarPlane.Get() instead.
      *
      */
    K_DEPRECATED bool GetViewNearFarPlanes() const;

    /** Change the draw floor grid flag.
      * \param pEnable     Set to \c true if the floor grid is shown.
      *
      * \remarks This function is deprecated. Use property ShowGrid.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetShowGrid(bool pEnable);

    /** Get current floor grid draw state.
      * \return    \c true if the floor grid is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ShowGrid.Get() instead.
      *
      */
    K_DEPRECATED bool GetShowGrid() const;

    /** Change the draw system axis flag.
      * \param pEnable     Set to \c true if the system axis is shown.
      *
      * \remarks This function is deprecated. Use property ShowAzimut.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetShowAxis(bool pEnable);

    /** Get current system axis draw state.
      * \return     \c true if the system axis is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ShowAzimut.Get() instead.
      *
      */
    K_DEPRECATED bool GetShowAxis() const;

    /** Change the show camera name flag.
      * \param pEnable     Set to \c true if the camera name is shown.
      *
      * \remarks This function is deprecated. Use property ShowName.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetShowName(bool pEnable);

    /** Get current camera name show state.
      * \return     \c true if the camera name is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ShowName.Get() instead.
      *
      */
    K_DEPRECATED bool GetShowName() const;

    /** Change the show info on moving flag.
      * \param pEnable     Set to \c true if info on moving is shown.
      *
      * \remarks This function is deprecated. Use property ShowInfoOnMoving.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetShowInfoOnMoving(bool pEnable);

    /** Get current info on moving show state.
      * \return     \c true if info on moving is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ShowInfoOnMoving.Get() instead.
      *
      */
    K_DEPRECATED bool GetShowInfoOnMoving() const;

    /** Change the timecode show flag.
      * \param pEnable     Set to \c true if the timecode is shown.
      *
      * \remarks This function is deprecated. Use property ShowTimeCode.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetShowTimeCode(bool pEnable);

    /** Get current timecode show state.
      * \return     \c true if the timecode is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ShowTimeCode.Get() instead.
      *
      */
    K_DEPRECATED bool GetShowTimeCode() const;

    /** Change the display camera safe area flag.
      * \param pEnable     Set to \c true if the safe area is shown.
      *
      * \remarks This function is deprecated. Use property DisplaySafeArea.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetDisplaySafeArea(bool pEnable);

    /** Get current safe area display state.
      * \return     \c true if safe area is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property DisplaySafeArea.Get() instead.
      *
      */
    K_DEPRECATED bool GetDisplaySafeArea() const;

    /** Change the display of the camera's safe area on render flag.
      * \param pEnable     Set to \c true if safe area is shown on render.
      *
      * \remarks This function is deprecated. Use property DisplaySafeAreaOnRender.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetDisplaySafeAreaOnRender(bool pEnable);

    /** Get current safe area on render display state.
      * \return     \c true if the safe area is shown on render, or \c false if it is hidden on render.
      *
      * \remarks This function is deprecated. Use property DisplaySafeAreaOnRender.Get() instead.
      *
      */
    K_DEPRECATED bool GetDisplaySafeAreaOnRender() const;

    /** \enum ECameraSafeAreaStyle Camera safe area display styles.
      * - \e eROUND
      * - \e eSQUARE
      */
    typedef enum
    {
        eROUND = 0,
        eSQUARE = 1
    } ECameraSafeAreaStyle;

    /** Set the safe area style.
      * \param pStyle    Safe area style identifier.
      *
      * \remarks This function is deprecated. Use property SafeAreaDisplayStyle.Set(pStyle) instead.
      *
      */
    K_DEPRECATED void SetSafeAreaStyle(ECameraSafeAreaStyle pStyle);

    /** Get the currently set safe area style.
      * \return     Safe area style identifier.
      *
      * \remarks This function is deprecated. Use property SafeAreaDisplayStyle.Get() instead.
      *
      */
    K_DEPRECATED ECameraSafeAreaStyle GetSafeAreaStyle() const;

    /** Change the show audio flag.
      * \param pEnable     Set to \c true if audio waveform is shown.
      *
      * \remarks This function is deprecated. Use property ShowAudio.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetShowAudio(bool pEnable);

    /** Get current audio show state.
      * \return     \c true if audio is shown, or \c false if hidden.
      *
      * \remarks This function is deprecated. Use property ShowAudio.Get() instead.
      *
      */
    K_DEPRECATED bool GetShowAudio() const;

    /** Set audio color.
      * \param pColor     RGB values for the audio waveform color.
      *
      * \remarks This function is deprecated. Use property AudioColor.Set(pColor) instead.
      *
      */
    K_DEPRECATED void SetAudioColor(const KFbxColor& pColor);

    /** Get audio color.
      * \return     Currently set audio waveform color.
      *
      * \remarks This function is deprecated. Use property AudioColor.Get() instead.
      *
      */
    K_DEPRECATED KFbxColor GetAudioColor() const;

    /** Change the use frame color flag.
      * \param pEnable     Set to \c true if the frame color is used.
      *
      * \remarks This function is deprecated. Use property UseFrameColor.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetUseFrameColor(bool pEnable);

    /** Get the use frame color state.
      * \return     \c true if the frame color is used, or \c false otherwise.
      *
      * \remarks This function is deprecated. Use property UseFrameColor.Get() instead.
      *
      */
    K_DEPRECATED bool GetUseFrameColor() const;

    /** Set frame color.
      * \param pColor     RGB values for the frame color.
      *
      * \remarks This function is deprecated. Use property FrameColor.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetFrameColor(const KFbxColor& pColor);

    /** Get frame color.
      * \return     Currently set frame color.
      *
      * \remarks This function is deprecated. Use property FrameColor.Set(pEnable) instead.
      *
      */
    K_DEPRECATED KFbxColor GetFrameColor() const;

    //@}

    /**
      * \name Render Functions
      * It is the responsibility of the client application to perform the required tasks according to the state
      * of the options that are either set or returned by these methods.
      */
    //@{

    /** \enum ECameraRenderOptionsUsageTime Render options usage time.
      * - \e eINTERACTIVE
      * - \e eAT_RENDER
      */
    typedef enum
    {
        eINTERACTIVE,
        eAT_RENDER
    } ECameraRenderOptionsUsageTime;

    /** Set the render options usage time.
      * \param pUsageTime     The render options usage time identifier.
      *
      * \remarks This function is deprecated. Use property UseRealTimeDOFAndAA.Set(pEnable) instead.
      * pEnable == true <=> pUsageTime == eINTERACTIVE
      * pEnable == false <=> pUsageTime == eAT_RENDER
      *
      */
    K_DEPRECATED void SetRenderOptionsUsageTime(ECameraRenderOptionsUsageTime pUsageTime);

    /** Get the render options usage time.
      * \return     Render options usage time identifier.
      *
      * \remarks This function is deprecated. Use property UseRealTimeDOFAndAA.Get() instead.
      * true <=> eINTERACTIVE
      * false <=> eAT_RENDER
      *
      */
    K_DEPRECATED ECameraRenderOptionsUsageTime GetRenderOptionsUsageTime() const;

    /** Change the use antialiasing flag.
      * \param pEnable    Set to \c true if antialiasing is enabled.
      *
      * \remarks This function is deprecated. Use property UseRealTimeDOFAndAA.Set(pEnable) instead.
      *
      */
    K_DEPRECATED void SetUseAntialiasing(bool pEnable);

    /** Get the use antialiasing state.
      * \return     \c true if antialiasing is enabled, or \c false if disabled.
      *
      * \remarks This function is deprecated. Use property UseRealTimeDOFAndAA.Get() instead.
      *
      */
    K_DEPRECATED bool GetUseAntialiasing() const;

    /** Set antialiasing intensity.
      * \param pIntensity     Antialiasing intensity value.
      *
      * \remarks This function is deprecated. Use property AntialiasingIntensity.Set(pIntensity) instead.
      *
      */
    K_DEPRECATED void SetAntialiasingIntensity(double pIntensity);

    /** Get the antialiasing intensity.
      * \return     Return the current antialiasing intensity value.
      *
      * \remarks This function is deprecated. Use property AntialiasingIntensity.Get() instead.
      *
      */
    K_DEPRECATED double GetAntialiasingIntensity() const;

    /** \enum ECameraAntialiasingMethod Antialiasing methods.
      * - \e eOVERSAMPLING_ANTIALIASING
      * - \e eHARDWARE_ANTIALIASING
      */
    typedef enum
    {
        eOVERSAMPLING_ANTIALIASING,
        eHARDWARE_ANTIALIASING
    } ECameraAntialiasingMethod;

    /** Set antialiasing method.
      * \param pMethod     The antialiasing method identifier.
      *
      * \remarks This function is deprecated. Use property AntialiasingMethod.Set(pMethod) instead.
      *
      */
    K_DEPRECATED void SetAntialiasingMethod(ECameraAntialiasingMethod pMethod);

    /** Get antialiasing method.
      * \return     The current antialiasing method identifier.
      *
      * \remarks This function is deprecated. Use property AntialiasingMethod.Get() instead.
      *
      */
    K_DEPRECATED ECameraAntialiasingMethod GetAntialiasingMethod() const;

    /** Set the number of samples used to process oversampling.
      * \param pNumberOfSamples     Number of samples used to process oversampling.
      * \remarks                    This option is only relevant if antialiasing method is set to eOVERSAMPLING_ANTIALIASING.
      *
      * \remarks This function is deprecated. Use property FrameSamplingCount.Set(pNumberOfSamples) instead.
      *
      */
    K_DEPRECATED void SetNumberOfSamples(int pNumberOfSamples);

    /** Get the number of samples used to process oversampling.
      * \return      The current number of samples used to process oversampling.
      * \remarks     This option is only relevant if antialiasing method is set to eOVERSAMPLING_ANTIALIASING.
      *
      * \remarks This function is deprecated. Use property FrameSamplingCount.Get() instead.
      *
      */
    K_DEPRECATED int GetNumberOfSamples() const;

    /** \enum ECameraSamplingType Oversampling types.
      * - \e eUNIFORM
      * - \e eSTOCHASTIC
      */
    typedef enum
    {
        eUNIFORM,
        eSTOCHASTIC
    } ECameraSamplingType;

    /** Set sampling type.
      * \param pType     Sampling type identifier.
      * \remarks         This option is only relevant if antialiasing type is set to eOVERSAMPLING_ANTIALIASING.
      *
      * \remarks This function is deprecated. Use property FrameSamplingType.Set(pType) instead.
      *
      */
    K_DEPRECATED void SetSamplingType(ECameraSamplingType pType);

    /** Get sampling type.
      * \return      The current sampling type identifier.
      * \remarks     This option is only relevant if antialiasing type is set to eOVERSAMPLING_ANTIALIASING.
      *
      * \remarks This function is deprecated. Use property FrameSamplingType.Get() instead.
      *
      */
    K_DEPRECATED ECameraSamplingType GetSamplingType() const;

    /** Change the use accumulation buffer flag.
      * \param pUseAccumulationBuffer     Set to \c true to enable use of the accumulation buffer.
      *
      * \remarks This function is deprecated. Use property UseAccumulationBuffer.Set(pUseAccumulationBuffer) instead.
      *
      */
    K_DEPRECATED void SetUseAccumulationBuffer(bool pUseAccumulationBuffer);

    /** Get the state of the use accumulation buffer flag.
      * \return     \c true if the use accumulation buffer flag is enabled, \c false otherwise.
      *
      * \remarks This function is deprecated. Use property UseAccumulationBuffer.Get() instead.
      *
      */
    K_DEPRECATED bool GetUseAccumulationBuffer() const;

    /** Change use depth of field flag.
      * \param pUseDepthOfField     Set to \c true if depth of field is used.
      *
      * \remarks This function is deprecated. Use property UseDepthOfField.Set(pUseDepthOfField) instead.
      *
      */
    K_DEPRECATED void SetUseDepthOfField(bool pUseDepthOfField);

    /** Get use depth of field state.
      * \return     \c true if depth of field is used, \c false otherwise.
      *
      * \remarks This function is deprecated. Use property UseDepthOfField.Get() instead.
      *
      */
    K_DEPRECATED bool GetUseDepthOfField() const;

    /** \enum ECameraFocusDistanceSource Camera focus sources.
      * - \e eCAMERA_INTEREST
      * - \e eSPECIFIC_DISTANCE
      */
    typedef enum
    {
        eCAMERA_INTEREST,
        eSPECIFIC_DISTANCE
    } ECameraFocusDistanceSource;

    /** Set source of camera focus distance.
      * \param pSource     Focus distance source identifier.
      *
      * \remarks This function is deprecated. Use property FocusSource.Set(pSource) instead.
      *
      */
    K_DEPRECATED void SetFocusDistanceSource(ECameraFocusDistanceSource pSource);

    /** Get source of camera focus distance.
      * \return     Focus distance source identifier.
      *
      * \remarks This function is deprecated. Use property FocusSource.Get() instead.
      *
      */
    K_DEPRECATED ECameraFocusDistanceSource GetFocusDistanceSource() const;

    /** Set the focus distance of the lens in millimiters.
      * \param pDistance     Focus distance value.
      * \remarks             This option is only relevant if focus distance source is set to eSPECIFIC_DISTANCE.
      *
      * \remarks This function is deprecated. Use property FocusDistance.Set(pDistance) instead.
      *
      */
    K_DEPRECATED void SetSpecificDistance(double pDistance);

    /** Get the focus distance of the lens in millimiters.
      * \return      Focus distance value.
      * \remarks     This option is only relevant if focus distance source is set to eSPECIFIC_DISTANCE.
      *
      * \remarks This function is deprecated. Use property FocusDistance.Get() instead.
      *
      */
    K_DEPRECATED double GetSpecificDistance() const;

    /** Set the focus angle in degrees.
      * \param pAngle     Focus angle value.
      *
      * \remarks This function is deprecated. Use property FocusAngle.Set(pAngle) instead.
      *
      */
    K_DEPRECATED void SetFocusAngle(double pAngle);

    /** Get the focus angle in degrees.
      * \return     Focus angle value.
      *
      * \remarks This function is deprecated. Use property FocusAngle.Get() instead.
      *
      */
    K_DEPRECATED double GetFocusAngle() const;

    //@}

    /**
      * \name Default Animation Values
      * These functions provide direct access to default animation values specific to a camera.
      * Since the default animation values are found in the default take node of the associated node,
      * these functions only work if the camera has been associated with a node.
      */
    //@{

    /** Set default field of view in degrees.
      * Use this function to set the default field of view value when the camera aperture mode is set to either \e eHORIZONTAL or \e eVERTICAL.
      * When the camera aperture mode is set to \e eHORIZONTAL, this function sets the horizontal field of view in degrees and the vertical field of
      * view is adjusted accordingly. When the camera aperture mode is set to \e eVERTICAL, this function sets the vertical field of view in
      * degrees and the horizontal field of view is adjusted accordingly.
      * \param pFieldOfView     Field of view value.
      * \remarks                This function has no effect when the camera aperture mode is set to either \e eHORIZONTAL_AND_VERTICAL or \e eFOCAL_LENGTH.
      *                         The default field of view value is 25.115.
      *
      * \remarks This function is deprecated. Use property FieldOfView.Set(pFieldOfView) instead.
      *
      */
    K_DEPRECATED void SetDefaultFieldOfView(double pFieldOfView);

    /** Get default field of view.
      * Use this function to get the default field of view value when the camera aperture mode is set to either \e eHORIZONTAL or \e eVERTICAL.
      * \return     If the camera aperture mode is set to either \e eHORIZONTAL or \e eVERTICAL, this function returns either the horizontal or vertical
      *             field of view value in degrees. If the camera aperture mode is set to either \e eHORIZONTAL_AND_VERTICAL or \e eFOCAL_LENGTH, this
      *             function has no effect and returns 0.
      * \remarks    The default field of view value is 25.115.
      *
      * \remarks This function is deprecated. Use property FieldOfView.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultFieldOfView() const;

    /** Set default field of view X.
      * Use this function to set the default field of view horizontal value when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * This function sets the horizontal field of view in degrees.
      * \param pFieldOfViewX     Field of view value.
      * \remarks                 This function has no effect if the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH.
      *                          The default field of view horizontal value is 40.
      *
      * \remarks This function is deprecated. Use property FieldOfViewX.Set(pFieldOfViewX) instead.
      *
      */
    K_DEPRECATED void SetDefaultFieldOfViewX(double pFieldOfViewX);

    /** Get default field of view X.
      * Use this function to get the default field of view horizontal value when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \return      If the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL, return the current field of view horizontal value in degrees.
      *              If the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH, this function has no effect and returns 0.
      * \remarks     The default field of view X value is 40 degrees.
      *
      * \remarks This function is deprecated. Use property FieldOfViewX.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultFieldOfViewX() const;

    /** Set default field of view Y.
      * Use this function to set the default field of view vertical value when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \param pFieldOfViewY     Field of view value.
      * \remarks                 This function has no effect if the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH.
      *                          The default field of view horizontal value is 40.
      *
      * \remarks This function is deprecated. Use property FieldOfViewY.Set(pFieldOfViewY) instead.
      *
      */
    K_DEPRECATED void SetDefaultFieldOfViewY(double pFieldOfViewY);

    /** Get default field of view Y.
      * Use this function to get the default field of view vertical value when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \return      If the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL, return the current field of view vertical value in degrees.
      *              If the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH, this function has no effect and returns 0.
      * \remarks     The default field of view Y value is 40 degrees.
      *
      * \remarks This function is deprecated. Use property FieldOfViewY.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultFieldOfViewY() const;

    /** Set default optical center X, in pixels.
      * Use this function to set the default optical center horizontal value when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \param pOpticalCenterX     Optical center offset.
      * \remarks                   This function has no effect if the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH.
      *                            The default optical center horizontal offset is 0.
      *
      * \remarks This function is deprecated. Use property OpticalCenterX.Set(pOpticalCenterX) instead.
      *
      */
    K_DEPRECATED void SetDefaultOpticalCenterX(double pOpticalCenterX);

    /** Get default optical center X, in pixels.
      * Use this function to get the default optical center horizontal offset when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \return      If the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL, return the current optical center horizontal offset.
      *              If the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH, this function has no effect and returns 0.
      * \remarks     The default optical center X offset is 0.
      * \remarks This function is deprecated. Use property OpticalCenterX.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultOpticalCenterX() const;

    /** Set default optical center Y, in pixels.
      * Use this function to set the default optical center vertical offset when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \param pOpticalCenterY     Optical center offset.
      * \remarks                   This function has no effect if the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH.
      *                            The default optical center vertical offset is 0.
      * \remarks This function is deprecated. Use property OpticalCenterY.Set(pOpticalCenterY) instead.
      *
      */
    K_DEPRECATED void SetDefaultOpticalCenterY(double pOpticalCenterY);

    /** Get default optical center Y, in pixels.
      * Use this function to get the default optical center vertical offset when the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL.
      * \return      If the camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL, return the current optical center vertical offset.
      *              If the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eFOCAL_LENGTH, this function has no effect and returns 0.
      * \remarks     The default optical center X offset is 0.
      *
      * \remarks This function is deprecated. Use property OpticalCenterY.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultOpticalCenterY() const;

    /** Set default focal length, in millimeters
      * Use this function to set the default focal length when the camera aperture mode is set to \e eFOCAL_LENGTH.
      * \param pFocalLength     Focal length value.
      * \remarks                This function has no effect if the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eHORIZONTAL_AND_VERTICAL.
      *                         The default focal length is 0.
      *
      * \remarks This function is deprecated. Use property FocalLength.Set(pFocalLength) instead.
      *
      */
    K_DEPRECATED void SetDefaultFocalLength(double pFocalLength);

    /** Get default focal length, in millimeters
      * Use this function to get the default focal length when the camera aperture mode is set to \e eFOCAL_LENGTH.
      * \return      If the camera aperture mode is set to \e eFOCAL_LENGTH, return the current default focal length.
      *              If the camera aperture mode is set to \e eHORIZONTAL, \e eVERTICAL, or \e eHORIZONTAL_AND_VERTICAL, this function has no effect and
      *              returns 0.
      * \remarks     The default focal length is 0.
      *
      * \remarks This function is deprecated. Use property FocalLength.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultFocalLength() const;

    /** Set default camera roll in degrees.
      * \param pRoll     Roll value.
      * \remarks         The default roll value is 0.
      *
      * \remarks This function is deprecated. Use property Roll.Set(pRoll) instead.
      *
      */
    K_DEPRECATED void SetDefaultRoll(double pRoll);

    /** Get default camera roll in degrees.
      * \return     Current roll value.
      * \remarks    The default roll value is 0.
      *
      * \remarks This function is deprecated. Use property Roll.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultRoll() const;

    /** Set default turntable value in degrees.
      * \param pTurnTable     Turntable value.
      * \remarks              The default turntable value is 0.
      *
      * \remarks This function is deprecated. Use property TurnTable.Set(pTurnTable) instead.
      *
      */
    K_DEPRECATED void SetDefaultTurnTable(double pTurnTable);

    /** Get default turntable in degrees.
      * \return     Current turntable value.
      * \remarks    The default turntable value is 0.
      *
      * \remarks This function is deprecated. Use property TurnTable.Get() instead.
      *
      */
    K_DEPRECATED double GetDefaultTurnTable() const;

    /** Set default background color.
      * \param pColor     RGB values of the background color.
      * \remarks          The default background color is black.
      *
      * \remarks This function is deprecated. Use property BackgroundColor.Set(pColor) instead.
      *
      */
    K_DEPRECATED void SetDefaultBackgroundColor(const KFbxColor& pColor);

    /** Get default background color.
      * \return      Current background color.
      * \remarks     The default background color is black.
      *
      * \remarks This function is deprecated. Use property BackgroundColor.Get() instead.
      *
      */
    K_DEPRECATED KFbxColor GetDefaultBackgroundColor() const;

    //@}

    /**
      * \name Obsolete Functions
      * These functions are obsolete since animated background color, animated field of view, and animated focal length are now supported.
      */
    //@{

    /** Set background color. This method is replaced by the SetDefaultBackgroundColor when setting a non-animated value. For animated values,
      * the client application must access the BackgroundColor fcurves in the take.
      * \remarks This function is deprecated. Use property BackgroundColor.Set(pColor) instead.
      *
      */
    K_DEPRECATED void SetBackgroundColor(const KFbxColor& pColor);

    /** Get background color. This method is replaced by the GetDefaultBackgroundColor when getting a non-animated value. For animated values,
      * the client application must access the BackgroundColor fcurves in the take.
      * \remarks This function is deprecated. Use property BackgroundColor.Get() instead.
      *
      */
    K_DEPRECATED KFbxColor GetBackgroundColor() const;

    /** Set the camera angle of view in degrees.
      * \param pAngleOfView     The camera angle of view value in degrees. This value is limited to the range [1.0, 179.0].
      * \warning                Modifying the angle of view will automatically change the focal length.
      * \remarks This function is deprecated. Use SetDefaultFieldOfView(pAngleOfView), SetDefaultFieldOfViewX(pAngleOfView) or
      * SetDefaultFieldOfViewY(pAngleOfView) instead.
      */
    K_DEPRECATED void SetAngleOfView(double pAngleOfView);

    /** Get the camera angle of view in degrees.
      * \return      The camera's current angle of view value in degrees.
      * \remarks This function is deprecated. Use GetDefaultFieldOfView(), GetDefaultFieldOfViewX() or
      * GetDefaultFieldOfViewY() instead.
      */
    K_DEPRECATED double GetAngleOfView() const;

    /** Set the focal length of the camera in millimeters.
      * \param pFocalLength     The focal length in mm.
      * \warning                Modifying the focal length will automatically change the angle of view.
      * \remarks This function is deprecated. Use SetDefaultFocalLength(pFocalLength) instead.
      */
    K_DEPRECATED void SetFocalLength(double pFocalLength);

    /** Get the camera focal length in millimeters.
      * \return      The camera's current focal length value.
      * \remarks This function is deprecated. Use GetDefaultFocalLength() instead.
      */
    K_DEPRECATED double GetFocalLength() const;

    //@}

    //The background texture is now in the property called BackgroundTexture
    K_DEPRECATED void SetBackgroundTexture(KFbxTexture* pTexture);

    /**
      * \name Utility Functions.
      */
    //@{

    /** Determine if the given bounding box is in the camera's view. 
      * The input points do not need to be ordered in any particular way.
      * \param pWorldToScreen The world to screen transformation. See ComputeWorldToScreen.
      * \param pWorldToCamera The world to camera transformation. 
               Inverse matrix returned from KFbxNode::GetGlobalFromCurrentTake is suitable.
               See KFbxNodeAttribute::GetNode() and KFbxNode::GetGlobalFromCurrentTake().
      * \param pPoints 8 corners of the bounding box.
      * \return true if any of the given points are in the camera's view, false otherwise.
      */
    bool IsBoundingBoxInView( const KFbxMatrix& pWorldToScreen, 
                             const KFbxMatrix& pWorldToCamera, 
                             const KFbxVector4 pPoints[8] ) const;

    /** Determine if the given 3d point is in the camera's view. 
      * \param pWorldToScreen The world to screen transformation. See ComputeWorldToScreen.
      * \param pWorldToCamera The world to camera transformation. 
               Inverse matrix returned from KFbxNode::GetGlobalFromCurrentTake is suitable.
               See KFbxNodeAttribute::GetNode() and KFbxNode::GetGlobalFromCurrentTake().
      * \param pPoint World-space point to test.
      * \return true if the given point is in the camera's view, false otherwise.
      */
    bool IsPointInView( const KFbxMatrix& pWorldToScreen, const KFbxMatrix& pWorldToCamera, const KFbxVector4& pPoint ) const;

    /** Compute world space to screen space transformation matrix.
      * \param pPixelHeight The pixel height of the output image.
      * \param pPixelWidth The pixel height of the output image.
      * \param pWorldToCamera The world to camera affine transformation matrix.
      * \return The world to screen space matrix, or the identity matrix on error.
      */
    KFbxMatrix ComputeWorldToScreen(int pPixelWidth, int pPixelHeight, const KFbxXMatrix& pWorldToCamera) const;

    /** Compute the perspective matrix for this camera. 
      * Suitable for transforming camera space to normalized device coordinate space.
      * Also suitable for use as an OpenGL projection matrix. Note this fails if the
      * ProjectionType is not ePERSPECTIVE. 
      * \param pPixelHeight The pixel height of the output image.
      * \param pPixelWidth The pixel height of the output image.
      * \param pIncludePostPerspective Indicate that post-projection transformations (offset, roll) 
      *        be included in the output matrix.
      * \return A perspective matrix, or the identity matrix on error.
      */
    KFbxMatrix ComputePerspective( int pPixelWidth, int pPixelHeight, bool pIncludePostPerspective ) const;

    //@}

    //////////////////////////////////////////////////////////////////////////
    //
    // Properties
    //
    //////////////////////////////////////////////////////////////////////////

    // -----------------------------------------------------------------------
    // Geometrical
    // -----------------------------------------------------------------------

    /** This property handles the camera position (XYZ coordinates).
      *
      * To access this property do: Position.Get().
      * To set this property do: Position.Set(fbxDouble3).
      *
      * \remarks Default Value is (0.0, 0.0, 0.0)
      */
    KFbxTypedProperty<fbxDouble3>                       Position;

    /** This property handles the camera Up Vector (XYZ coordinates).
      *
      * To access this property do: UpVector.Get().
      * To set this property do: UpVector.Set(fbxDouble3).
      *
      * \remarks Default Value is (0.0, 1.0, 0.0)
      */
    KFbxTypedProperty<fbxDouble3>                       UpVector;

    /** This property handles the default point (XYZ coordinates) the camera is looking at.
      *
      * To access this property do: InterestPosition.Get().
      * To set this property do: InterestPosition.Set(fbxDouble3).
      *
      * \remarks During the computations of the camera position
      * and orientation, this property is overridden by the
      * position of a valid target in the parent node.
      *
      * \remarks Default Value is (0.0, 0.0, 0.0)
      */
    KFbxTypedProperty<fbxDouble3>                       InterestPosition;

    /** This property handles the camera roll angle in degree(s).
      *
      * To access this property do: InterestPosition.Get().
      * To set this property do: InterestPosition.Set(fbxDouble1).
      *
      * Default value is 0.
      */
    KFbxTypedProperty<fbxDouble1>                       Roll;

    /** This property handles the camera optical center X, in pixels.
      * It parameter sets the optical center horizontal offset when the
      * camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL. It
      * has no effect otherwise.
      *
      * To access this property do: OpticalCenterX.Get().
      * To set this property do: OpticalCenterX.Set(fbxDouble1).
      *
      * Default value is 0.
      */
    KFbxTypedProperty<fbxDouble1>                       OpticalCenterX;

    /** This property handles the camera optical center Y, in pixels.
      * It sets the optical center horizontal offset when the
      * camera aperture mode is set to \e eHORIZONTAL_AND_VERTICAL. This
      * parameter has no effect otherwise.
      *
      * To access this property do: OpticalCenterY.Get().
      * To set this property do: OpticalCenterY.Set(fbxDouble1).
      *
      * Default value is 0.
      */
    KFbxTypedProperty<fbxDouble1>                       OpticalCenterY;

    /** This property handles the camera RGB values of the background color.
      *
      * To access this property do: BackgroundColor.Get().
      * To set this property do: BackgroundColor.Set(fbxDouble3).
      *
      * Default value is black (0, 0, 0)
      */
    KFbxTypedProperty<fbxDouble3>                       BackgroundColor;

    /** This property handles the camera turn table angle in degree(s)
      *
      * To access this property do: TurnTable.Get().
      * To set this property do: TurnTable.Set(fbxDouble1).
      *
      * Default value is 0.
      */
    KFbxTypedProperty<fbxDouble1>                       TurnTable;

    /** This property handles a flags that indicates if the camera displays the
      * Turn Table icon or not.
      *
      * To access this property do: DisplayTurnTableIcon.Get().
      * To set this property do: DisplayTurnTableIcon.Set(fbxBool1).
      *
      * Default value is false (no display).
      */
    KFbxTypedProperty<fbxBool1>                         DisplayTurnTableIcon;

    // -----------------------------------------------------------------------
    // Motion Blur
    // -----------------------------------------------------------------------

    /** This property handles a flags that indicates if the camera uses
      * motion blur or not.
      *
      * To access this property do: UseMotionBlur.Get().
      * To set this property do: UseMotionBlur.Set(fbxBool1).
      *
      * Default value is false (do not use motion blur).
      */
    KFbxTypedProperty<fbxBool1>                         UseMotionBlur;

    /** This property handles a flags that indicates if the camera uses
      * real time motion blur or not.
      *
      * To access this property do: UseRealTimeMotionBlur.Get().
      * To set this property do: UseRealTimeMotionBlur.Set(fbxBool1).
      *
      * Default value is false (use real time motion blur).
      */
    KFbxTypedProperty<fbxBool1>                         UseRealTimeMotionBlur;

    /** This property handles the camera motion blur intensity (in pixels).
      *
      * To access this property do: MotionBlurIntensity.Get().
      * To set this property do: MotionBlurIntensity.Set(fbxDouble1).
      *
      * Default value is 1.
      */
    KFbxTypedProperty<fbxDouble1>                       MotionBlurIntensity;

    // -----------------------------------------------------------------------
    // Optical
    // -----------------------------------------------------------------------

    /** This property handles the camera aspect ratio mode.
      *
      * \remarks This Property is in a Read Only mode.
      * \remarks Please use function SetAspect() if you want to change its value.
      *
      * Default value is eWINDOW_SIZE.
      *
      */
    KFbxTypedProperty<ECameraAspectRatioMode>           AspectRatioMode;

    /** This property handles the camera aspect width.
      *
      * \remarks This Property is in a Read Only mode.
      * \remarks Please use function SetAspect() if you want to change its value.
      *
      * Default value is 320.
      */
    KFbxTypedProperty<fbxDouble1>                       AspectWidth;

    /** This property handles the camera aspect height.
      *
      * \remarks This Property is in a Read Only mode.
      * \remarks Please use function SetAspect() if you want to change its value.
      *
      * Default value is 200.
      */
    KFbxTypedProperty<fbxDouble1>                       AspectHeight;

    /** This property handles the pixel aspect ratio.
      *
      * \remarks This Property is in a Read Only mode.
      * \remarks Please use function SetPixelRatio() if you want to change its value.
      * Default value is 1.
      * \remarks Value range is [0.050, 20.0].
      */
    KFbxTypedProperty<fbxDouble1>                       PixelAspectRatio;

    /** This property handles the aperture mode.
      *
      * Default value is eVERTICAL.
      */
    KFbxTypedProperty<ECameraApertureMode>              ApertureMode;

    /** This property handles the gate fit mode.
      *
      * To access this property do: GateFit.Get().
      * To set this property do: GateFit.Set(ECameraGateFit).
      *
      * Default value is eNO_FIT.
      */
    KFbxTypedProperty<ECameraGateFit>                   GateFit;

    /** This property handles the field of view in degrees.
      *
      * To access this property do: FieldOfView.Get().
      * To set this property do: FieldOfView.Set(fbxDouble1).
      *
      * \remarks This property has meaning only when
      * property ApertureMode equals eHORIZONTAL or eVERTICAL.
      *
      * \remarks Default vaule is 40.
      * \remarks Value range is [1.0, 179.0].
      */
    KFbxTypedProperty<fbxDouble1>                       FieldOfView;


    /** This property handles the X (horizontal) field of view in degrees.
      *
      * To access this property do: FieldOfViewX.Get().
      * To set this property do: FieldOfViewX.Set(fbxDouble1).
      *
      * \remarks This property has meaning only when
      * property ApertureMode equals eHORIZONTAL or eVERTICAL.
      *
      * Default value is 1.
      * \remarks Value range is [1.0, 179.0].
      */
    KFbxTypedProperty<fbxDouble1>                       FieldOfViewX;

    /** This property handles the Y (vertical) field of view in degrees.
      *
      * To access this property do: FieldOfViewY.Get().
      * To set this property do: FieldOfViewY.Set(fbxDouble1).
      *
      * \remarks This property has meaning only when
      * property ApertureMode equals eHORIZONTAL or eVERTICAL.
      *
      * \remarks Default vaule is 1.
      * \remarks Value range is [1.0, 179.0].
      */
    KFbxTypedProperty<fbxDouble1>                       FieldOfViewY;

    /** This property handles the focal length (in millimeters).
      *
      * To access this property do: FocalLength.Get().
      * To set this property do: FocalLength.Set(fbxDouble1).
      *
      * Default value is the result of ComputeFocalLength(40.0).
      */
    KFbxTypedProperty<fbxDouble1>                       FocalLength;

    /** This property handles the camera format.
      *
      * To access this property do: CameraFormat.Get().
      * To set this property do: CameraFormat.Set(ECameraFormat).
      *
      * \remarks This Property is in a Read Only mode.
      * \remarks Please use function SetFormat() if you want to change its value.
      * Default value is eCUSTOM_FORMAT.
      */
    KFbxTypedProperty<ECameraFormat>                    CameraFormat;

    // -----------------------------------------------------------------------
    // Frame
    // -----------------------------------------------------------------------

    /** This property stores a flag that indicates to use or not a color for
      * the frame.
      *
      * To access this property do: UseFrameColor.Get().
      * To set this property do: UseFrameColor.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1>                         UseFrameColor;

    /** This property handles the fame color
      *
      * To access this property do: FrameColor.Get().
      * To set this property do: FrameColor.Set(fbxDouble3).
      *
      * Default value is (0.3, 0.3, 0.3).
      */
    KFbxTypedProperty<fbxDouble3>                       FrameColor;

    // -----------------------------------------------------------------------
    // On Screen Display
    // -----------------------------------------------------------------------

    /** This property handles the show name flag.
      *
      * To access this property do: ShowName.Get().
      * To set this property do: ShowName.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         ShowName;

    /** This property handles the show info on moving flag.
      *
      * To access this property do: ShowInfoOnMoving.Get().
      * To set this property do: ShowInfoOnMoving.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         ShowInfoOnMoving;

    /** This property handles the draw floor grid flag
      *
      * To access this property do: ShowGrid.Get().
      * To set this property do: ShowGrid.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         ShowGrid;

    /** This property handles the show optical center flag
      *
      * To access this property do: ShowOpticalCenter.Get().
      * To set this property do: ShowOpticalCenter.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1>                         ShowOpticalCenter;

    /** This property handles the show axis flag
      *
      * To access this property do: ShowAzimut.Get().
      * To set this property do: ShowAzimut.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         ShowAzimut;

    /** This property handles the show time code flag
      *
      * To access this property do: ShowTimeCode.Get().
      * To set this property do: ShowTimeCode.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         ShowTimeCode;

    /** This property handles the show audio flag
      *
      * To access this property do: ShowAudio.Get().
      * To set this property do: ShowAudio.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1>                         ShowAudio;

    /** This property handles the show audio flag
      *
      * To access this property do: AudioColor.Get().
      * To set this property do: AudioColor.Set(fbxDouble3).
      *
      * Default value is (0.0, 1.0, 0.0)
      */
    KFbxTypedProperty<fbxDouble3>                       AudioColor;

    // -----------------------------------------------------------------------
    // Clipping Planes
    // -----------------------------------------------------------------------

    /** This property handles the near plane distance.
      *
      * \remarks This Property is in a Read Only mode.
      * \remarks Please use function SetNearPlane() if you want to change its value.
      * Default value is 10.
      * \remarks Value range is [0.001, 600000.0].
      */
    KFbxTypedProperty<fbxDouble1>                       NearPlane;

    /** This property handles the far plane distance.
      *
      * \remarks This Property is in a Read Only mode
      * \remarks Please use function SetPixelRatio() if you want to change its value
      * Default value is 4000
      * \remarks Value range is [0.001, 600000.0]
      */
    KFbxTypedProperty<fbxDouble1>                       FarPlane;


    /** This property indicates that the clip planes should be automatically computed.
      *
      * To access this property do: AutoComputeClipPlanes.Get().
      * To set this property do: AutoComputeClipPlanes.Set(fbxBool1).
      *
      * When this property is set to true, the NearPlane and FarPlane values are
      * ignored. Note that not all applications support this flag.
      */
    KFbxTypedProperty<fbxBool1>                         AutoComputeClipPlanes;


    // -----------------------------------------------------------------------
    // Camera Film Setting
    // -----------------------------------------------------------------------

    /** This property handles the film aperture width (in inches).
      *
      * \remarks This Property is in a Read Only mode
      * \remarks Please use function SetApertureWidth()
      * or SetApertureFormat() if you want to change its value
      * Default value is 0.8160
      * \remarks Value range is [0.0001, +inf[
      */
    KFbxTypedProperty<fbxDouble1>                       FilmWidth;

    /** This property handles the film aperture height (in inches).
      *
      * \remarks This Property is in a Read Only mode
      * \remarks Please use function SetApertureHeight()
      * or SetApertureFormat() if you want to change its value
      * Default value is 0.6120
      * \remarks Value range is [0.0001, +inf[
      */
    KFbxTypedProperty<fbxDouble1>                       FilmHeight;

    /** This property handles the film aperture aspect ratio.
      *
      * \remarks This Property is in a Read Only mode
      * \remarks Please use function SetApertureFormat() if you want to change its value
      * Default value is (FilmWidth / FilmHeight)
      * \remarks Value range is [0.0001, +inf[
      */
    KFbxTypedProperty<fbxDouble1>                       FilmAspectRatio;

    /** This property handles the film aperture squeeze ratio.
      *
      * \remarks This Property is in a Read Only mode
      * \remarks Please use function SetSqueezeRatio()
      * or SetApertureFormat() if you want to change its value
      * Default value is 1.0
      * \remarks Value range is [0.0001, +inf[
      */
    KFbxTypedProperty<fbxDouble1>                       FilmSqueezeRatio;

    /** This property handles the film aperture format.
      *
      * \remarks This Property is in a Read Only mode
      * \remarks Please use function SetApertureFormat()
      * if you want to change its value
      * Default value is eCUSTOM_APERTURE_FORMAT
      */
    KFbxTypedProperty<ECameraApertureFormat>            FilmFormat;

    /** This property handles the offset from the center of the film aperture,
      * defined by the film height and film width. The offset is measured
      * in inches.
      *
      * To access this property do: FilmOffset.Get().
      * To set this property do: FilmOffset.Set(fbxDouble2).
      *
      */
    KFbxTypedProperty<fbxDouble2>                       FilmOffset;


    // -----------------------------------------------------------------------
    // Camera View Widget Option
    // -----------------------------------------------------------------------

    /** This property handles the view frustrum flag.
      *
      * To access this property do: ViewFrustum.Get().
      * To set this property do: ViewFrustum.Set(fbxBool1).
      *
      * Default value is true
      */
    KFbxTypedProperty<fbxBool1>                         ViewFrustum;

    /** This property handles the view frustrum near and far plane flag.
      *
      * To access this property do: ViewFrustumNearFarPlane.Get().
      * To set this property do: ViewFrustumNearFarPlane.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         ViewFrustumNearFarPlane;

    /** This property handles the view frustrum back plane mode.
      *
      * To access this property do: ViewFrustumBackPlaneMode.Get().
      * To set this property do: ViewFrustumBackPlaneMode.Set(ECameraBackgroundDisplayMode).
      *
      * Default value is eWHEN_MEDIA
      */
    KFbxTypedProperty<ECameraBackgroundDisplayMode>     ViewFrustumBackPlaneMode;

    /** This property handles the view frustrum back plane distance.
      *
      * To access this property do: BackPlaneDistance.Get().
      * To set this property do: BackPlaneDistance.Set(fbxDouble1).
      *
      * Default value is 100.0
      */
    KFbxTypedProperty<fbxDouble1>                       BackPlaneDistance;

    /** This property handles the view frustrum back plane distance mode.
      *
      * To access this property do: BackPlaneDistanceMode.Get().
      * To set this property do: BackPlaneDistanceMode.Set(ECameraBackgroundDistanceMode).
      *
      * Default value is eRELATIVE_TO_INTEREST
      */
    KFbxTypedProperty<ECameraBackgroundDistanceMode>    BackPlaneDistanceMode;

    /** This property handles the view camera to look at flag.
      *
      * To access this property do: ViewCameraToLookAt.Get().
      * To set this property do: ViewCameraToLookAt.Set(fbxBool1).
      *
      * Default value is true
      */
    KFbxTypedProperty<fbxBool1>                         ViewCameraToLookAt;

    // -----------------------------------------------------------------------
    // Camera Lock Mode
    // -----------------------------------------------------------------------

    /** This property handles the lock mode.
      *
      * To access this property do: LockMode.Get().
      * To set this property do: LockMode.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         LockMode;

    /** This property handles the lock interest navigation flag.
      *
      * To access this property do: LockInterestNavigation.Get().
      * To set this property do: LockInterestNavigation.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         LockInterestNavigation;

    // -----------------------------------------------------------------------
    // Background Image Display Options
    // -----------------------------------------------------------------------

    /** This property handles the fit image flag.
      *
      * To access this property do: FitImage.Get().
      * To set this property do: FitImage.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1>                         FitImage;

    /** This property handles the crop flag.
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1>                         Crop;

    /** This property handles the center flag.
      *
      * To access this property do: Center.Get().
      * To set this property do: Center.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         Center;

    /** This property handles the keep ratio flag.
      *
      * To access this property do: KeepRatio.Get().
      * To set this property do: KeepRatio.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1>                         KeepRatio;

    /** This property handles the background mode flag.
      *
      * To access this property do: BackgroundMode.Get().
      * To set this property do: BackgroundMode.Set(ECameraBackgroundDrawingMode).
      *
      * Default value is eBACKGROUND.
      */
    KFbxTypedProperty<ECameraBackgroundDrawingMode>     BackgroundMode;

    /** This property handles the background alpha threshold value.
      *
      * To access this property do: BackgroundAlphaTreshold.Get().
      * To set this property do: BackgroundAlphaTreshold.Set(fbxDouble1).
      *
      * Default value is 0.5.
      */
    KFbxTypedProperty<fbxDouble1>                       BackgroundAlphaTreshold;

    // -----------------------------------------------------------------------
    // Foreground Image Display Options
    // -----------------------------------------------------------------------

    /** This property handles the fit image for front plate flag.
      *
      * To access this property do: FrontPlateFitImage.Get().
      * To set this property do: FrontPlateFitImage.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1> FrontPlateFitImage;

    /** This property handles the front plane crop flag.
      *
      * To access this property do: FrontPlateCrop.Get().
      * To set this property do: FrontPlateCrop.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1> FrontPlateCrop;

    /** This property handles the front plane center flag.
      *
      * To access this property do: FrontPlateCenter.Get().
      * To set this property do: FrontPlateCenter.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1> FrontPlateCenter;

    /** This property handles the front plane keep ratio flag.
      *
      * To access this property do: FrontPlateKeepRatio.Get().
      * To set this property do: FrontPlateKeepRatio.Set(fbxBool1).
      *
      * Default value is true.
      */
    KFbxTypedProperty<fbxBool1> FrontPlateKeepRatio;


    /** This property handles the front plane show flag.
      *
      * To access this property do: ShowFrontPlate.Get().
      * To set this property do: ShowFrontPlate.Set(fbxBool1).
      *
      * Default value is false.
      * \remarks this replaces ForegroundTransparent 
      */
    KFbxTypedProperty<fbxBool1> ShowFrontPlate;

    /** This property handles the view frustrum front plane mode.
      *
      * To access this property do: ViewFrustumFrontPlaneMode.Get().
      * To set this property do: ViewFrustumFrontPlaneMode.Set(ECameraBackgroundDisplayMode).
      *
      * Default value is eWHEN_MEDIA
      */
    KFbxTypedProperty<ECameraBackgroundDisplayMode>     ViewFrustumFrontPlaneMode;

    /** This property handles the view frustrum front plane distance.
      *
      * To access this property do: FrontPlaneDistance.Get().
      * To set this property do: FrontPlaneDistance.Set(fbxDouble1).
      *
      * Default value is 100.0
      */
    KFbxTypedProperty<fbxDouble1>                       FrontPlaneDistance;

    /** This property handles the view frustrum front plane distance mode.
      *
      * To access this property do: FrontPlaneDistanceMode.Get().
      * To set this property do: FrontPlaneDistanceMode.Set(ECameraBackgroundDistanceMode).
      *
      * Default value is eRELATIVE_TO_INTEREST
      */
    KFbxTypedProperty<ECameraBackgroundDistanceMode>    FrontPlaneDistanceMode;

    /** This property handles the foreground alpha value.
      *
      * To access this property do: ForegroundAlpha.Get().
      * To set this property do: ForegroundAlpha.Set(fbxDouble1).
      *
      * Default value is 0.5.
      */
    KFbxTypedProperty<fbxDouble1> ForegroundAlpha;


    /** This property has the foreground textures connected to it.
      *
      * To access this property do: ForegroundTexture.Get().
      * To set this property do: ForegroundTexture.Set(fbxReference).
      *
      * \remarks they are connected as source objects
      */
    KFbxTypedProperty<fbxReference> ForegroundTexture;

    /** This property has the background textures connected to it.
      *
      * To access this property do: BackgroundTexture.Get().
      * To set this property do: BackgroundTexture.Set(fbxReference).
      *
      * \remarks they are connected as source objects
      */
    KFbxTypedProperty<fbxReference> BackgroundTexture;


    // -----------------------------------------------------------------------
    // Safe Area
    // -----------------------------------------------------------------------

    /** This property handles the display safe area flag.
      *
      * To access this property do: DisplaySafeArea.Get().
      * To set this property do: DisplaySafeArea.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         DisplaySafeArea;

    /** This property handles the display safe area on render flag.
      *
      * To access this property do: DisplaySafeAreaOnRender.Get().
      * To set this property do: DisplaySafeAreaOnRender.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         DisplaySafeAreaOnRender;

    /** This property handles the display safe area display style.
      *
      * To access this property do: SafeAreaDisplayStyle.Get().
      * To set this property do: SafeAreaDisplayStyle.Set(ECameraSafeAreaStyle).
      *
      * Default value is eSQUARE
      */
    KFbxTypedProperty<ECameraSafeAreaStyle>             SafeAreaDisplayStyle;

    /** This property handles the display safe area aspect ratio.
      *
      * To access this property do: SafeAreaDisplayStyle.Get().
      * To set this property do: SafeAreaAspectRatio.Set(fbxDouble1).
      *
      * Default value is 1.33333333333333
      */
    KFbxTypedProperty<fbxDouble1>                       SafeAreaAspectRatio;

    // -----------------------------------------------------------------------
    // 2D Magnifier
    // -----------------------------------------------------------------------

    /** This property handles the use 2d magnifier zoom flag.
      *
      * To access this property do: Use2DMagnifierZoom.Get().
      * To set this property do: Use2DMagnifierZoom.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         Use2DMagnifierZoom;

    /** This property handles the 2d magnifier zoom value.
      *
      * To access this property do: _2DMagnifierZoom.Get().
      * To set this property do: _2DMagnifierZoom.Set(fbxDouble1).
      *
      * Default value is 100.0
      */
    KFbxTypedProperty<fbxDouble1>                       _2DMagnifierZoom;

    /** This property handles the 2d magnifier X value.
      *
      * To access this property do: _2DMagnifierX.Get().
      * To set this property do: _2DMagnifierX.Set(fbxDouble1).
      *
      * Default value is 50.0
      */
    KFbxTypedProperty<fbxDouble1>                       _2DMagnifierX;

    /** This property handles the 2d magnifier Y value.
      *
      * To access this property do: _2DMagnifierY.Get().
      * To set this property do: _2DMagnifierY.Set(fbxDouble1).
      *
      * Default value is 50.0
      */
    KFbxTypedProperty<fbxDouble1>                       _2DMagnifierY;

    // -----------------------------------------------------------------------
    // Projection Type: Ortho, Perspective
    // -----------------------------------------------------------------------

    /** This property handles the projection type
      *
      * To access this property do: ProjectionType.Get().
      * To set this property do: ProjectionType.Set(ECameraProjectionType).
      *
      * Default value is ePERSPECTIVE.
      */
    KFbxTypedProperty<ECameraProjectionType>            ProjectionType;

    /** This property handles the otho zoom
      *
      * To access this property do: OrthoZoom.Get().
      * To set this property do: OrthoZoom.Set(fbxDouble1).
      *
      * Default value is 1.0.
      */
    KFbxTypedProperty<fbxDouble1>                       OrthoZoom;

    // -----------------------------------------------------------------------
    // Depth Of Field & Anti Aliasing
    // -----------------------------------------------------------------------

    /** This property handles the use real time DOF and AA flag
      *
      * To access this property do: UseRealTimeDOFAndAA.Get().
      * To set this property do: UseRealTimeDOFAndAA.Set(fbxBool1).
      *
      * Default value is false.
      */
    KFbxTypedProperty<fbxBool1>                         UseRealTimeDOFAndAA;

    /** This property handles the use depth of field flag
      *
      * To access this property do: UseDepthOfField.Get().
      * To set this property do: UseDepthOfField.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         UseDepthOfField;

    /** This property handles the focus source
      *
      * To access this property do: FocusSource.Get().
      * To set this property do: FocusSource.Set(ECameraFocusDistanceSource).
      *
      * Default value is eCAMERA_INTEREST
      */
    KFbxTypedProperty<ECameraFocusDistanceSource>       FocusSource;

    /** This property handles the focus angle (in degrees)
      *
      * To access this property do: FocusAngle.Get().
      * To set this property do: FocusAngle.Set(fbxDouble1).
      *
      * Default value is 3.5
      */
    KFbxTypedProperty<fbxDouble1>                       FocusAngle;

    /** This property handles the focus distance
      *
      * To access this property do: FocusDistance.Get().
      * To set this property do: FocusDistance.Set(fbxDouble1).
      *
      * Default value is 200.0
      */
    KFbxTypedProperty<fbxDouble1>                       FocusDistance;

    /** This property handles the use anti aliasing flag
      *
      * To access this property do: UseAntialiasing.Get().
      * To set this property do: UseAntialiasing.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         UseAntialiasing;

    /** This property handles the anti aliasing intensity
      *
      * To access this property do: AntialiasingIntensity.Get().
      * To set this property do: AntialiasingIntensity.Set(fbxDouble1).
      *
      * Default value is 0.77777
      */
    KFbxTypedProperty<fbxDouble1>                       AntialiasingIntensity;

    /** This property handles the anti aliasing method
      *
      * To access this property do: AntialiasingMethod.Get().
      * To set this property do: AntialiasingMethod.Set(ECameraAntialiasingMethod).
      *
      * Default value is eOVERSAMPLING_ANTIALIASING
      */
    KFbxTypedProperty<ECameraAntialiasingMethod>        AntialiasingMethod;

    // -----------------------------------------------------------------------
    // Accumulation Buffer
    // -----------------------------------------------------------------------

    /** This property handles the use accumulation buffer flag
      *
      * To access this property do: UseAccumulationBuffer.Get().
      * To set this property do: UseAccumulationBuffer.Set(fbxBool1).
      *
      * Default value is false
      */
    KFbxTypedProperty<fbxBool1>                         UseAccumulationBuffer;

    /** This property handles the frame sampling count
      *
      * To access this property do: FrameSamplingCount.Get().
      * To set this property do: FrameSamplingCount.Set(fbxInteger1).
      *
      * Default value is 7
      */
    KFbxTypedProperty<fbxInteger1>                      FrameSamplingCount;

    /** This property handles the frame sampling type
      *
      * To access this property do: FrameSamplingType.Get().
      * To set this property do: FrameSamplingType.Set(ECameraSamplingType).
      *
      * Default value is eSTOCHASTIC
      */
    KFbxTypedProperty<ECameraSamplingType>              FrameSamplingType;

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    friend class KFbxGlobalCameraSettings;

public:

    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

    //! Assignment operator.
    KFbxCamera& operator=(KFbxCamera const& pCamera);

    KFbxCamera(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxCamera ();

    virtual void Construct(const KFbxCamera* pFrom);
    virtual bool ConstructProperties(bool pForceSet);
    virtual void Destruct(bool pRecursive, bool pDependents);

    /**
      * Used to retrieve the KProperty list from an attribute
      */
    virtual KString     GetTypeName() const;
    virtual KStringList GetTypeFlags() const;

private:

    double ComputePixelRatio(kUInt pWidth, kUInt pHeight, double pScreenRatio = 1.3333333333);

    // Background Properties
    KString mBackgroundMediaName;
    KString mBackgroundFileName;

    friend class KFbxNode;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxCamera* HKFbxCamera;

inline EFbxType FbxTypeOf( KFbxCamera::ECameraAntialiasingMethod const &pItem )         { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraApertureFormat const &pItem )             { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraApertureMode const &pItem )               { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraAspectRatioMode const &pItem )            { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraBackgroundDisplayMode const &pItem )      { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraBackgroundDistanceMode const &pItem )     { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraBackgroundDrawingMode const &pItem )      { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraBackgroundPlacementOptions const &pItem ) { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraFocusDistanceSource const &pItem )        { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraFormat const &pItem )                     { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraGateFit const &pItem )                    { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraProjectionType const &pItem )             { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraRenderOptionsUsageTime const &pItem )     { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraSafeAreaStyle const &pItem )              { return eENUM; }
inline EFbxType FbxTypeOf( KFbxCamera::ECameraSamplingType const &pItem )               { return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_CAMERA_H_


