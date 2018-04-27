/*!  \file kfbxglobaltimesettings.h
 */

#ifndef _FBXSDK_GLOBAL_TIME_SETTINGS_H_
#define _FBXSDK_GLOBAL_TIME_SETTINGS_H_

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
#include <klib/karrayul.h>
#include <klib/ktime.h>
#include <klib/kerror.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxGlobalTimeSettingsProperties;

/** This class contains functions for accessing global time settings.
  * \nosubgrouping
  */
class KFBX_DLL KFbxGlobalTimeSettings
{

public:

    //! Restore default settings.
    void RestoreDefaultSettings();

    /** Set time mode.
      * \param pTimeMode     One of the defined modes in class KTime.
      */
    void SetTimeMode(KTime::ETimeMode pTimeMode);

    /** Get time mode.
      * \return     The currently set TimeMode.
      */
    KTime::ETimeMode GetTimeMode();

    /** Set time protocol.
      * \param pTimeProtocol     One of the defined protocols in class KTime.
      */
    void SetTimeProtocol(KTime::ETimeProtocol pTimeProtocol);

    /** Get time protocol.
      * \return     The currently set TimeProtocol.
      */
    KTime::ETimeProtocol GetTimeProtocol();

    /** \enum ESnapOnFrameMode Snap on frame mode
      * - \e eNO_SNAP
      * - \e eSNAP_ON_FRAME
      * - \e ePLAY_ON_FRAME
      * - \e eSNAP_PLAY_ON_FRAME
      */
    typedef enum
    {
        eNO_SNAP,
        eSNAP_ON_FRAME,
        ePLAY_ON_FRAME,
        eSNAP_PLAY_ON_FRAME
    } ESnapOnFrameMode;

    /** Set snap on frame mode.
      * \param pSnapOnFrameMode     One of the following values: eNO_SNAP, eSNAP_ON_FRAME, ePLAY_ON_FRAME, or eSNAP_PLAY_ON_FRAME.
      */
    void SetSnapOnFrameMode(ESnapOnFrameMode pSnapOnFrameMode);

    /** Get snap on frame mode.
      * \return     The currently set FrameMode.
      */
    ESnapOnFrameMode GetSnapOnFrameMode();

    /**
      * \name Timeline Time span
      */
    //@{

    /** Set Timeline default time span
      * \param pTimeSpan The time span of the time line.
      */
    void SetTimelineDefautTimeSpan(const KTimeSpan& pTimeSpan);

    /** Get Timeline default time span
      * \param pTimeSpan return the default time span for the time line.
      */
    void GetTimelineDefautTimeSpan(KTimeSpan& pTimeSpan) const;

    //@}

    /**
      * \name Time Markers
      */
    //@{

    struct KFbxTimeMarker
    {
        KFbxTimeMarker();
        KFbxTimeMarker(const KFbxTimeMarker& pTimeMarker);
        KFbxTimeMarker& operator=(const KFbxTimeMarker& pTimeMarker);

        KString mName; //! Marker name.
        KTime mTime; //! Marker time.
        bool mLoop; //! Loop flag.
    };

    /** Get number of time markers.
      * \return     The number of time markers.
      */
    int GetTimeMarkerCount();

    /** Set current time marker index.
      * \param pIndex     Current time marker index.
      * \return           \c true if successful, or \c false if pIndex is invalid.
      * \remarks          If pIndex is invalid, KFbxGlobalTimeSettings::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    bool SetCurrentTimeMarker(int pIndex);

    /** Get current time marker index.
      * \return     Current time marker index, or -1 if the current time marker has not been set.
      */
    int GetCurrentTimeMarker();

    /** Get time marker at given index.
      * \param pIndex     Time marker index.
      * \return           Pointer to the time marker at pIndex, or \c NULL if the index is out of range.
      * \remarks          If pIndex is out of range, KFbxGlobalTimeSettings::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    KFbxTimeMarker* GetTimeMarker(int pIndex);

    /** Add a time marker.
      * \param     pTimeMarker New time marker.
      */
    void AddTimeMarker(KFbxTimeMarker pTimeMarker);

    //! Remove all time markers and set current time marker to -1.
    void RemoveAllTimeMarkers();

    //@}

    //! Assignment operator.
    const KFbxGlobalTimeSettings& operator=(const KFbxGlobalTimeSettings& pGlobalTimeSettings);

    /**
      * \name Error Management
      */
    //@{

    /** Retrieve error object.
     *  \return     Reference to error object.
     */
    KError& GetError();

    /** Error identifiers.
      * Most of these are only used internally.
      * - \e eINDEX_OUT_OF_RANGE
      * - \e eERROR_COUNT
      */
    typedef enum
    {
        eINDEX_OUT_OF_RANGE,
        eERROR_COUNT
    } EError;

    /** Get last error code.
     *  \return   Last error code.
     */
    EError GetLastErrorID() const;

    /** Get last error string.
     *  \return   Textual description of the last error.
     */
    const char* GetLastErrorString() const;

    //@}

    /**
      * \name Obsolete Functions
      * These functions still work but are no longer relevant.
      */
    //@{

    /** Set snap on frame flag.
      * \param pSnapOnFrame     If \c true, snap on frame mode is set to eSNAP_ON_FRAME. If \c false, snap on frame mode is set to \c eNO_SNAP.
      * \remarks                This function is replaced by KFbxGlobalTimeSettings::SetSnapOnFrameMode().
      */
    void SetSnapOnFrame(bool pSnapOnFrame);

    /** Get snap on frame flag
      * \return      \c true if snap on frame mode is set to either eSNAP_ON_FRAME or ePLAY_ON_FRAME. \c false if snap on frame mode is set to \c eNO_SNAP.
      * \remarks     This function is replaced by KFbxGlobalTimeSettings::GetSnapOnFrameMode().
      */
    bool GetSnapOnFrame();

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

private:

    KFbxGlobalTimeSettings();
    ~KFbxGlobalTimeSettings();

    KFbxGlobalTimeSettingsProperties* mPH;

    friend class KFbxScene;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_GLOBAL_TIME_SETTINGS_H_


