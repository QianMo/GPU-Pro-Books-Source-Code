/*!  \file kfbxio.h
 */

#ifndef _FBXSDK_IO_H_
#define _FBXSDK_IO_H_

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
#include <klib/kstring.h>

#include <kbaselib_forward.h>
#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <kfbxplugins/kfbxobject.h>

#ifdef KARCH_DEV_MACOSX_CFM
    #include <CFURL.h>
    #include <Files.h>
#endif

#include <fbxfilesdk_nsbegin.h>

class KFbxSdkManager;
class KFbxStreamOptions;

#define KFBXIO_END_NODE_STR ("_End")

/** \brief Base class for FBX file import and export.
  * \nosubgrouping
  */
class KFBX_DLL KFbxIO : public KFbxObject

{
    KFBXOBJECT_DECLARE(KFbxIO,KFbxObject);

public:

    /** Get the FBX version number for this version of the FBX SDK.
      * FBX version numbers start at 5.0.0.
      * \param pMajor        Version major number.
      * \param pMinor        Version minor number.
      * \param pRevision     Version revision number.
      */
    static void GetCurrentVersion(int& pMajor, int& pMinor, int& pRevision);

    /** Initialize object.
      * \param pFileName     Name of file to access.
      * \return              \c true if successful, \c false otherwise.
      * \remarks             To identify the error, call KFbxIO::GetLastErrorID().
      */
    virtual bool Initialize(const char *pFileName);

#ifdef KARCH_DEV_MACOSX_CFM
    virtual bool Initialize(const FSSpec &pMacFileSpec);
    virtual bool Initialize(const FSRef &pMacFileRef);
    virtual bool Initialize(const CFURLRef &pMacURL);
#endif

    /** Get the file name.
       * \return     Filename or an empty string if the filename has not been set.
       */
    virtual KString GetFileName();

    /** Progress update function.
      * \param pTitle           Title of status box.
      * \param pMessage         Description of current file read/write step.
      * \param pDetail          Additional string appended to previous parameter.
      * \param pPercentDone     Finished percent of current file read/write.
      * \remarks                Overload this function to receive an update of current file read/write.
      */
    virtual void ProgressUpdate(char* pTitle, char* pMessage, char* pDetail, float pPercentDone);

    /**
      * \name Error Management
      */
    //@{

    /** Retrieve error object.
      * \return     Reference to error object.
      */
    KError& GetError();

    /** \enum EError Error identifiers.
      * - \e eFILE_CORRUPTED
      * - \e eFILE_VERSION_NOT_SUPPORTED_YET
      * - \e eFILE_VERSION_NOT_SUPPORTED_ANYMORE
      * - \e eFILE_NOT_OPENED
      * - \e eFILE_NOT_CREATED
      * - \e eOUT_OF_DISK_SPACE
      * - \e eUNINITIALIZED_FILENAME
      * - \e eUNIDENTIFIED_ERROR
      * - \e eINDEX_OUT_OF_RANGE
      * - \e ePASSWORD_ERROR
      * - \e eSTREAM_OPTIONS_NOT_SET
      * - \e eEMBEDDED_OUT_OF_SPACE
      */
    typedef enum
    {
        eFILE_CORRUPTED,
        eFILE_VERSION_NOT_SUPPORTED_YET,
        eFILE_VERSION_NOT_SUPPORTED_ANYMORE,
        eFILE_NOT_OPENED,
        eFILE_NOT_CREATED,
        eOUT_OF_DISK_SPACE,
        eUNINITIALIZED_FILENAME,
        eUNIDENTIFIED_ERROR,
        eINDEX_OUT_OF_RANGE,
        ePASSWORD_ERROR,
        eSTREAM_OPTIONS_NOT_SET,
        eEMBEDDED_OUT_OF_SPACE,
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

    /** Get warning message from file reader/writer.
      * \param pMessage     Warning message
      */
    void GetMessage(KString& pMessage) const;

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

    KFbxIO(KFbxSdkManager& pManager,char const* pName);
    virtual ~KFbxIO();

    KError mError;
    KString mFilename;
    KFbxSdkManager* mManager;
    KFbxStreamOptions* mStreamOptions;
    KString mMessage;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_IO_H_


