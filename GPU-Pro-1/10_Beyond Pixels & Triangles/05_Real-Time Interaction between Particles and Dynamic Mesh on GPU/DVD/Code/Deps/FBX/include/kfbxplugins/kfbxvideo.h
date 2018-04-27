/*!  \file kfbxvideo.h
 */

#ifndef _FBXSDK_VIDEO_H_
#define _FBXSDK_VIDEO_H_

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

#include <klib/kerror.h>
#ifdef KARCH_DEV_MACOSX_CFM
    #include <CFURL.h>
    #include <Files.h>
#endif

#include <kfbxmath/kfbxvector2.h>
#include <kfbxplugins/kfbxgroupname.h>
#include <kfbxplugins/kfbxtakenodecontainer.h>

#include <fbxfilesdk_nsbegin.h>


/**	FBX SDK video class.
  * \nosubgrouping
  */
class KFBX_DLL KFbxVideo : public KFbxTakeNodeContainer
{
    KFBXOBJECT_DECLARE(KFbxVideo,KFbxTakeNodeContainer);
  public:
	/**
	  *\name Reset vedio
	  */
	//@{

    //! Reset the video to default values.
    void Reset();
	//@}

    /**
      * \name Video attributes Management
      */
    //@{

    /** Set the use of MipMap on the video.
      * \param pUseMipMap If \c true, use MipMap on the video.
      */
    void ImageTextureSetMipMap(bool pUseMipMap);

    /** Retrieve use MipMap state.
      * \return          MipMap flag state.
      */
    bool ImageTextureGetMipMap();

    /** Specify the Video full filename.
      * \param pName     Video full filename.
	  * \return          \c True,if update successfully, \c false otherwise.
      * \remarks         Update the texture filename if the connection exists.
      */
    bool SetFileName(char const* pName);

    /** Retrieve the Video full filename.
      * \return          Video full filename.
      */
    KString GetFileName ();

    /** Specify the Video relative filename.
      * \param pName     Video relative filename.
	  * \return          \c True, if update successfully, \c false otherwise.
      * \remark          Update the texture filename if the connection exists.
      */
    bool SetRelativeFileName(char const* pName);

    /** Retrieve the Video relative filename.
      * \return         Video relative filename.
      */
    char* GetRelativeFileName();

    /** Retrieve the Frame rate of the video clip.
      * \return        Frame rate.
      */
    double GetFrameRate();

    /** Retrieve the last frame of the video clip.
      * \return       Last frame number.
      */
    int GetLastFrame();

    /** Retrieve the clip width.
      * \return      Video image width.
      */
    int GetWidth();

    /** Retrieve the clip height.
      * \return      Video image height.
      */
    int GetHeight();

    /** Set the start frame of the video clip.
      * \param pStartFrame     Start frame number.
      * \remarks               The parameter value is not checked. It is the responsibility
      *                        of the caller to deal with bad frame numbers.
      */
    void SetStartFrame(int pStartFrame);

    /** Retrieve the start frame of the video clip.
      * \return     Start frame number.
      */
    int GetStartFrame();

    /** Set the stop frame of the video clip.
      * \param pStopFrame     Stop frame number.
      * \remarks              The parameter value is not checked. It is the responsibility
      *                       of the caller to deal with bad frame numbers.
      */
    void SetStopFrame(int pStopFrame);

    /** Retrieve the stop frame of the video clip.
      * \return     Stop frame number.
      */
    int GetStopFrame();

    /** Set the play speed of the video clip.
      * \param pPlaySpeed     Playback speed of the clip.
      * \remarks             The parameter value is not checked. It is the responsibility
      *                      of the caller to deal with bad playback speed values.
      */
    void SetPlaySpeed(double pPlaySpeed);

    /** Retrieve the play speed of the video clip.
      * \return Playback     speed.
      */
    double GetPlaySpeed();

    /** Set the time offset.
      * The offset can be used to shift the playback start time of the clip.
      * \param pTime     Time offset of the clip.
      */
    void SetOffset(KTime pTime);

    /* Retrieve the time offset.
     * \return     The current time shift.
     */
    KTime GetOffset();

    /** Set the Free Running state of the video clip.
      * The Free Running flag can be used by a client application to implement a
      * playback scheme that is independent of the main timeline.
      * \param pState     State of the Free running flag.
      */
    void SetFreeRunning(bool pState);

    /** Retrieve the Free Running state.
      * \return     Current free running flag.
      */
    bool GetFreeRunning();


    /** Set the Loop state of the video clip.
      * The Loop flag can be used by a client application to implement the loop
      * playback of the video clip.
      * \param pLoop     State of the loop flag.
      */
    void SetLoop(bool pLoop);

    /** Retrieve the Loop state.
      * \return     Current loop flag.
      */
    bool GetLoop();


    /** \enum EInterlaceMode Video interlace modes.
      * - \e Node
      * - \e Fields
      * - \e HalfEven
      * - \e HalfOdd
      * - \e FullEven
      * - \e FullOdd
      * - \e FullEvenOdd
      * - \e FullOddEven
      */
    typedef enum
    {
        None,       // Progressive frame (full frame)
        Fields,     // Alternate even/odd fields
        HalfEven,   // Half of a frame, even fields only
        HalfOdd,    // Half of a frame, odd fields only
        FullEven,   // Extract and use the even field of a full frame
        FullOdd,    // Extract and use the odd field of a full frame
        FullEvenOdd, // Extract Fields and make full frame with each one beginning with Even (60fps)
        FullOddEven
    } EInterlaceMode;

    /** Set the Interlace mode.
      * \param pInterlaceMode     Interlace mode identifier.
      */
    void SetInterlaceMode(EInterlaceMode pInterlaceMode);

    /** Retrieve the Interlace mode
      * \return     Interlace mode identifier.
      */
    EInterlaceMode GetInterlaceMode();


    /** \enum EAccessMode Video clip access mode.
      * - \e Disk
      * - \e Memory
      * - \e DiskAsync
      */
    typedef enum
    {
        Disk,
        Memory,
        DiskAsync
    } EAccessMode;

    /** Set the clip Access Mode.
      * \param pAccessMode     Clip access mode identifier.
      */
    void SetAccessMode(EAccessMode pAccessMode);

    /** Retrieve the clip Access Mode.
      * \return     Clip access mode identifier.
      */
    EAccessMode GetAccessMode();
    //@}
    

    /**
      * \name Error Management
      */
    //@{

    /** Retrieve error object.
      * \return     Reference to error object.
      */
    KError& GetError();

    /** \enum EError Error identifiers.
      * - \e eTAKE_NODE_ERROR
      * - \e eERROR_COUNT
      */
    typedef enum
    {
        eTAKE_NODE_ERROR,
        eERROR_COUNT
    } EError;

    /** Get last error code.
      * \return     Last error code.
      */
    EError GetLastErrorID() const;

    /** Get last error string.
      * \return     Textual description of the last error.
      */
    const char* GetLastErrorString() const;

    //@}

#ifndef DOXYGEN_SHOULD_SKIP_THIS

    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

  protected:
    KFbxVideo(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxVideo();

    virtual void Construct(const KFbxVideo* pFrom);
    virtual bool ConstructProperties(bool pForceSet);
    virtual void Destruct(bool pRecursive, bool pDependents);
    virtual                 bool ConnecNotify (KFbxConnectEvent const &pEvent);

    virtual KString     GetTypeName() const;
    virtual KStringList GetTypeFlags() const;

    void SetOriginalFormat(bool pState);
    bool GetOriginalFormat();
    void SetOriginalFilename(const char* pOriginalFilename);
    const char* GetOriginalFilename() const;

    void Init();

    static KError smError;

    KFbxTypedProperty<fbxDouble1> FrameRate;
    KFbxTypedProperty<fbxInteger1> LastFrame;
    KFbxTypedProperty<fbxInteger1> Width;
    KFbxTypedProperty<fbxInteger1> Height;
    KFbxTypedProperty<fbxString> Path;
    KFbxTypedProperty<fbxInteger1> StartFrame;
    KFbxTypedProperty<fbxInteger1> StopFrame;
    KFbxTypedProperty<fbxDouble1> PlaySpeed;
    KFbxTypedProperty<fbxTime> Offset;
    KFbxTypedProperty<EInterlaceMode> InterlaceMode;
    KFbxTypedProperty<fbxBool1> FreeRunning;
    KFbxTypedProperty<fbxBool1> Loop;
    KFbxTypedProperty<EAccessMode> AccessMode;

    bool                   mUseMipMap;
    bool                   mOriginalFormat;
    KString                mOriginalFilename;

    // Temporary values
    KString                mRelativeFilename;

    friend class KFbxWriterFbx6;
    friend struct KFbxWriterFbx7Impl;
    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;
    friend class KFbxReaderFbx7;
    friend struct KFbxReaderFbx7Impl;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

typedef KFbxVideo* HKFbxVideo;

inline EFbxType FbxTypeOf( KFbxVideo::EInterlaceMode const &pItem ) { return eENUM; }
inline EFbxType FbxTypeOf( KFbxVideo::EAccessMode const &pItem ) { return eENUM; }

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_VIDEO_H_


