/*!  \file kfbxgloballightsettings.h
 */

#ifndef _FBXSDK_GLOBAL_LIGHT_SETTINGS_H_
#define _FBXSDK_GLOBAL_LIGHT_SETTINGS_H_

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

#include <kfbxplugins/kfbxcolor.h>

#include <kfbxmath/kfbxvector4.h>

#include <klib/karrayul.h>
#include <klib/kerror.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxGlobalLightSettingsProperties;

/** This class contains functions for accessing global light settings.
  * \nosubgrouping
  */
class KFBX_DLL KFbxGlobalLightSettings
{

public:

    /**
      * \name Ambient Color
      */
    //@{

    /** Set ambient color.
      * \param pAmbientColor     The ambient color to set.
      * \remarks                 Only the RGB channels are used.
      */
    void SetAmbientColor(KFbxColor pAmbientColor);

    /** Get ambient color.
      * \return     The ambient color.
      */
    KFbxColor GetAmbientColor();

    //@}

    /**
      * \name Fog Option
      */
    //@{

    /** Enable or disable the fog.
      * \param pEnable     Set to \c true to enable the fog option. \c false disables the fog option.
      */
    void SetFogEnable(bool pEnable);

    /** Get the current state of the fog option.
      * \return     \c true if fog is enabled, \c false otherwise.
      */
    bool GetFogEnable();

    /** Set the fog color.
      * \param pColor     The fog color.
      * \remarks          Only the RGB channels are used.
      */
    void SetFogColor(KFbxColor pColor);

    /** Get the fog color.
      * \return      The fog color.
      * \remarks     Only the RGB channels are used.
      */
    KFbxColor GetFogColor();

    /** \enum EFogMode Fog types.
      * - \e eLINEAR
        - \e eEXPONENTIAL
        - \e eSQUAREROOT_EXPONENTIAL
      */
    typedef enum
    {
        eLINEAR,
        eEXPONENTIAL,
        eSQUAREROOT_EXPONENTIAL
    } EFogMode;

    /** Set the fog mode.
      * \param pMode     The fog type.
      */
    void SetFogMode(EFogMode pMode);

    /** Get the fog mode.
      * \return     The currently set fog mode.
      */
    EFogMode GetFogMode();

    /** Set the fog density.
      * \param pDensity     The density of the fog. Can be any double value, however it is
      *                     possible that other sections of FBX SDK may clamp values to reasonable values.
      * \remarks            Only use this function when the fog mode is exponential or squareroot exponential.
      */
    void SetFogDensity(double pDensity);

    /** Get the fog density.
      * \return      The currently set fog density.
      * \remarks     Only use this function when the fog mode is exponential or squareroot exponential.
      */
    double GetFogDensity();

    /** Set the distance from the view where the fog starts.
      * \param pStart     Distance where the fog starts.
      * \remarks          Only use this function when the fog mode is linear. The new value is clamped to fit inside the interval [0, FogEnd()].
      */
    void SetFogStart(double pStart);

    /** Get the distance from the view where the fog starts.
      * \return      The distance from the view where the fog starts.
      * \remarks     Only use this function when the fog mode is linear.
      */
    double GetFogStart();

    /** Set the distance from the view where the fog ends.
      * \param pEnd     Distance where the fog ends.
      * \remarks        Only use this function when the fog mode is linear. The new value is adjusted to fit within the interval [FogStart(), inf).
      */
    void SetFogEnd(double pEnd);

    /** Get the distance from the view where the fog ends.
      * \return      The distance from the view where the fog ends.
      * \remarks     Only use this function when the fog mode is linear.
      */
    double GetFogEnd();

    //@}

    /**
      * \name Shadow Planes
      * The functions in this section are supported by FiLMBOX 2.7 and previous versions only.
      * FiLMBOX 3.0 supports shadow planes within a specific shader, which is not supported by the FBX SDK.
      */
    //@{

    struct KFbxShadowPlane
    {
        KFbxShadowPlane();

        bool mEnable; //! Enable flag.
        KFbxVector4 mOrigin; //! Origin point.
        KFbxVector4 mNormal; //! Normal vector.
    };

    /** Enable or disable the shadow planes display.
      * \param pShadowEnable     Set to \c true to display shadow planes in the scene.
      */
    void SetShadowEnable(bool pShadowEnable);

    /** Get the current state of the ShadowEnable flag.
      * \return     \c true if shadow planes are set to be displayed in the scene.
      */
    bool GetShadowEnable();

    /** Set the shadow intensity applied to all shadow planes.
      * \param pShadowIntensity     Intensity applied to all the shadow planes.
      * \remarks                    Range is from 0 to 300.
      */
    void SetShadowIntensity(double pShadowIntensity);

    /** Get the shadow intensity applied to all shadow planes.
      * \return      The intensity applied to all shadow planes in the scene.
      * \remarks     Range is from 0 to 300.
      */
    double GetShadowIntensity();

    /** Get the number of shadow planes.
      * \return     Number of shadow planes.
      */
    int GetShadowPlaneCount();

    /** Get a shadow plane.
      * \param pIndex     Index of shadow plane.
      * \return           Pointer the shadow plane, or \c NULL if the index is out of range.
      * \remarks          To identify the error, call KFbxGlobalLightSettings::GetLastErrorID() which returns eINDEX_OUT_OF_RANGE.
      */
    KFbxShadowPlane* GetShadowPlane(int pIndex);

    /** Add a shadow plane.
      * \param pShadowPlane     The shadow plane to add.
      */
    void AddShadowPlane(KFbxShadowPlane pShadowPlane);

    //! Remove all shadow planes.
    void RemoveAllShadowPlanes();

    //@}

    /**
      * \name Error Management
      */
    //@{

    /** Retrieve error object.
     *  \return     Reference to error object.
     */
    KError& GetError();

    /** \enum EError Error identification.
     *  - \e eINDEX_OUT_OF_RANGE
     *  - \e eERROR_COUNT
     */
    typedef enum
    {
        eINDEX_OUT_OF_RANGE,
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

    //! Restore default settings.
    void RestoreDefaultSettings();

    //! Assignment operator.
    const KFbxGlobalLightSettings& operator=(const KFbxGlobalLightSettings& pGlobalLightSettings);


///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:

    KFbxGlobalLightSettings();
    ~KFbxGlobalLightSettings();

    KFbxGlobalLightSettingsProperties* mPH;

    friend class KFbxScene;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_GLOBAL_LIGHT_SETTINGS_H_


