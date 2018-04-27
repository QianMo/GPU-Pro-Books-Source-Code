/*!  \file kfbxcache.h
 */

#ifndef _FBXSDK_CACHE_H_
#define _FBXSDK_CACHE_H_

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
#include <kfbxplugins/kfbxgroupname.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxCache_internal;
class KFbxVertexCacheDeformer;

/**
* \brief This object contains methods for accessing point animation in a cache file.
* 
* The FBX SDK supports two point cache file formats :
*    - \e ePC2: the 3ds Max Point Cache 2 file format.
*    - \e eMC: the Maya Cache file format.
*
* Accessing cache data using these formats differ significantly. To address this difference, two sets of methods have been created.
* Use the GetCacheFileFormat() function to determine which set of methods to use.
*/

class KFBX_DLL KFbxCache : public KFbxObject
{
    KFBXOBJECT_DECLARE(KFbxCache,KFbxObject);

public:
    /**
      * \name Format Independant Functions.
      */
    //@{

    /** \enum EFileFormat Supported cache file formats.
      * - \e eUNKNOWN
      * - \e ePC2      3ds Max Point Cache 2 file format
      * - \e eMC       Maya Cache file format
      */
    typedef enum
    {
        eUNKNOWN,
        ePC2,
        eMC
    } EFileFormat;

    /** Set the cache file format.
      * \param pFileFormat     Valid values are \e ePC2 or \e eMC.
      */
    void SetCacheFileFormat(EFileFormat pFileFormat);

    /** Get the cache file format.
      * \return     The current cache file format, or \e eUNKNOWN if it is not set.
      */
    EFileFormat GetCacheFileFormat();

    /** Set the cache file name.
      * \param pRelativeFileName     The point cache file, relative to the FBX file name.
      * \param pAbsoluteFileName     The point cache file absolute path.
      */
    void SetCacheFileName(const char* pRelativeFileName, const char* pAbsoluteFileName);

    /** Get the cache file name.
      * \param pRelativeFileName     Return the point cache file name, relative to the FBX File name.
      * \param pAbsoluteFileName     Return the point cache file absolute path.
      */
    void GetCacheFileName(KString& pRelativeFileName, KString& pAbsoluteFileName);

    /** Open the cache file for reading.
      * \return     \c true if the file is successfully opened, \c false otherwise. See the error management
      *             functions for error details.
      */
    bool OpenFileForRead();

    /** Get the open state of the cache file.
      * \return     \c true if the cache file is currently open, \c false otherwise.
      */
    bool IsOpen();

    /** Close the cache file.
      * \return     \c true if the cache file is closed successfully, \c false otherwise.
      */
    bool CloseFile();

    /** Get the sampling frame rate of the cache file.
      * \return     The sampling frame rate of the cache file, in frames per second.
      */
    double GetSamplingFrameRate();

    /** Get the sampling frame rate of the cache file, as a KTime object.
      * \return     The sampling frame rate of the cache file.
      */
    KTime GetCacheTimePerFrame();

    //@}

    /**
      * \name eMC Format Specific Functions.
      */
    //@{

    /** \enum EMCFileCount Number of files used to store the animation.
      *	- \e eMC_ONE_FILE            One file is used for all the frames of animation
      *	- \e eMC_ONE_FILE_PER_FRAME  One file is used for every frame of animation
      */
    typedef enum
    {
        eMC_ONE_FILE,
        eMC_ONE_FILE_PER_FRAME
    } EMCFileCount;

    /** \enum EMCDataType cache datatypes formats.
      * - \e kUnknownData
      * - \e kDouble is a double
      * - \e kDoubleArray would be double*
      * - \e kDoubleVectorArray is double* [3]
      * - \e kInt32Array is int*
      * - \e kFloatVectorArray is float* [3]
      */
    typedef enum
    {
        kUnknownData,
        kDouble,             //kDouble
        kDoubleArray,         //KArrayTemplate<kDouble>
        kDoubleVectorArray,  //KArrayTemplate<fbxDouble3> 
        kInt32Array,         //KArrayTemplate<kInt>
        kFloatVectorArray    //KArrayTemplate<fbxFloat3>
    }EMCDataType;

    /** Open a cache file for writing.
      * \param pFileCount             Create one file for each frame of animation, or one file for all the frames.
      * \param pSamplingFrameRate     Number of frames per second.
      * \param pChannelName           The name of the channel of animation to create.
      * \param pMCDataType            The MC DataType of the cache.
      */
    bool OpenFileForWrite(EMCFileCount pFileCount, double pSamplingFrameRate, const char* pChannelName, EMCDataType pMCDataType = kDoubleVectorArray );     

    /** Get the number of channels in the cache file.
      * \return     The number of animation channels in the cache file.
      */
    int  GetChannelCount();

    /** Get the channel name for a specific channel index.
      * \param pChannelIndex     The index of the animation channel, between 0 and GetChannelCount().
      * \param pChannelName      Returns the name of the requested channel.
      * \return                  \c true if successful, \c false otherwise. See the error management
      *                          functions for error details.
      */
    bool GetChannelName(int pChannelIndex, KString& pChannelName);

    /** Get the DataType of the specified channel.
      * \param pChannelIndex    The index of the channel.
      * \param pChannelType     The channel's datatype
      * \return                 \c true if successful, \c false otherwise. See the error management
      *                         functions for error details.
      */
    bool GetChannelDataType(int pChannelIndex, EMCDataType& pChannelType);


    /** Get the index of the specified channel.
      * \param pChannelName     The name of the channel.
      * \return                 The index of the channel in the cache file, or -1 if an error occured. See the error management
      *                         functions for error details.
      */
    int  GetChannelIndex(const char* pChannelName);

    /** Read a sample at a given time.
      * \param pChannelIndex     The index of the animation channel, between 0 and GetChannelCount().
      * \param pTime             Time at which the point animation must be evaluated.
      * \param pBuffer           The place where the point value will be copied. This buffer must be
      *                          of size 3*pPointCount.
      * \param pPointCount       The number of points to read from the point cache file.
      * \return                  \c true if successful, \c false otherwise. See the error management
      *                          functions for error details.
      */
    bool Read(int pChannelIndex, KTime& pTime, double* pBuffer, unsigned int pPointCount);

    /** Read a sample at a given time.
      * \param pChannelIndex     The index of the animation channel, between 0 and GetChannelCount().
      * \param pTime             Time at which the point animation must be evaluated.
      * \param pBuffer           The place where the point value will be copied. This buffer must be
      *                          of size 3*pPointCount.
      * \param pPointCount       The number of points to read from the point cache file.
      * \return                  \c true if successful, \c false otherwise. See the error management
      *                          functions for error details.
      */
    bool Read(int pChannelIndex, KTime& pTime, float* pBuffer, unsigned int pPointCount);

	/** Write a sample at a given time.
	  * \param pChannelIndex   The index of the animation channel, between 0 and GetChannelCount().
	  * \param pTime           Time at which the point animation must be inserted.
	  * \param pBuffer         Point to the values to be copied. This buffer must be
	  *                        of size 3*pPointCount.
	  * \param pPointCount     The number of points to write in the point cache file.
	  * \return                \c true if successful, \c false otherwise. See the error management
	  *                        functions for error details.
	  */
    bool Write(int pChannelIndex, KTime& pTime, double* pBuffer, unsigned int pPointCount);

    /** Write a sample at a given time.
      * \param pChannelIndex   The index of the animation channel, between 0 and GetChannelCount().
      * \param pTime           Time at which the point animation must be inserted.
      * \param pBuffer         Point to the values to be copied. This buffer must be
      *                        of size 3*pPointCount.
      * \param pPointCount     The number of points to write in the point cache file.
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      */
    bool Write(int pChannelIndex, KTime& pTime, float* pBuffer, unsigned int pPointCount);

    /** Get the Animation Range of the specified channel.
      * \param pChannelIndex    The index of the channel.
      * \param pTimeStart       The Channel's Animation Start Time
      * \param pTimeEnd         The Channel's Animation End Time
      * \return                 \c true if successful, \c false otherwise. See the error management
      *                         functions for error details.
      */
    bool GetAnimationRange(int pChannelIndex, KTime &pTimeStart, KTime &pTimeEnd);

    /** Get the cache type
      * \param pFileCount       The cache type.
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      */
    bool GetCacheType(EMCFileCount& pFileCount);

    /** Get the cache channel interpretation.
      * \param pChannelIndex    The index of the animation channel, between 0 and GetChannelCount().
      * \param pInterpretation  The channel interpretation; user-defined.
      * \return                    \c true if successful, \c false otherwise. See the error management
      *                               functions for error details.
      */
    bool GetChannelInterpretation(int pChannelIndex, KString& pInterpretation);

    /** \enum EMCSamplingType cache channel sampling types.
      * - \e eSAMPLING_REGULAR      Regular sampling
      * - \e eSAMPLING_IRREGULAR    Irregular sampling
      */
    typedef enum
    {
        eSAMPLING_REGULAR,
        eSAMPLING_IRREGULAR
    }EMCSamplingType;

    /** Get the cache channel sampling type.
      * \param pChannelIndex    The index of the animation channel, between 0 and GetChannelCount().
      * \param pSamplingType    The sampling type of the channel.
      * \return                    \c true if successful, \c false otherwise. See the error management
      *                               functions for error details.
      */

    bool GetChannelSamplingType(int pChannelIndex, EMCSamplingType& pSamplingType);

    /** Get the cache channel sampling rate, in frames per second.
      * \param pChannelIndex   The index of the animation channel, between 0 and GetChannelCount().
      * \param pSamplingRate   The sampling rate of the channel.  The channel must have a regular
      *                        sampling type.
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      */
    bool GetChannelSamplingRate(int pChannelIndex, KTime& pSamplingRate);

    /** Get the number of data points for a channel.
      * \param pChannelIndex   The index of the animation channel, between 0 and GetChannelCount().
      * \param pSampleCount    Number of available samples.
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      */
    bool GetChannelSampleCount(int pChannelIndex, unsigned int& pSampleCount);

    /** Get the number of points animated in the cache file, for a channel, for a given time.
      * \param pChannelIndex   The index of the animation channel, between 0 and GetChannelCount().
      * \param pTime           Reference time; must be within the boundaries of the animation.
      * \param pPointCount     Number of available points.
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      */
    bool GetChannelPointCount(int pChannelIndex, KTime pTime, unsigned int& pPointCount);

    /** Returns the number of cache data files.
      * \return     The count returned does not include the main cache file, and depends on the
      *             cache type.  WIll return -1 point cache support is not enabled.
      */
    int  GetCacheDataFileCount();

    /** Get the nth cache file name.
      * \param pIndex                Index of the cache file to return; index is zero-based, and must be
      *                                 < GetCacheDataFileCount().
      * \param pRelativeFileName     Return the point cache file name, relative to the FBX File name.
      * \param pAbsoluteFileName     Return the point cache file absolute path.
      * \return                      \c true if successful, \c false otherwise. See the error management
      *                         functions for error details.
      */
    bool GetCacheDataFileName(int pIndex, KString& pRelativeFileName, KString& pAbsoluteFileName);

    /** Enable multi-channel fetching.
      * \param pMultiChannelFetching Enable/disable multi-channel fetching.  When multi-channel is enabled,
      *                              any load of data on a channel at a specific time will prefetch data
      *                              from all channels, for that specific time.  This can reduce disk
      *                              access, and increase performance (but requires more memory).
      * \return                      \c true if successful, \c false otherwise. See the error management
      *                              functions for error details.
      */
    bool EnableMultiChannelFetching(bool pMultiChannelFetching);

    //@}

    /**
      * \name ePC2 Format Specific Functions.
      */
    //@{

    /** Open a cache file for writing.
      * \param pFrameStartOffset      Start time of the animation, in frames.
      * \param pSamplingFrameRate     Number of frames per second.
      * \param pSampleCount           The number of samples to write to the file.
      * \param pPointCount            The number of points to write in the point cache file.
      * \return                       \c true if successful, \c false otherwise. See the error management
      *                               functions for error details.
      */
    bool OpenFileForWrite(double pFrameStartOffset, double pSamplingFrameRate, unsigned int pSampleCount, unsigned int pPointCount);

    /** Get the number of frames of animation found in the point cache file.
      * \return     The number of frames of animation.
      */
    unsigned int GetSampleCount();

    /** Get the number of points animated in the cache file.
      * \return     The number of points.
      */
    unsigned int GetPointCount();

    /** Get the start time of the animation
      * \return     The start time of the animation, in frames.
      */
    double GetFrameStartOffset();


    /** Get the sampling frame rate of the cache file.
      * \return     The sampling frame rate of the cache file, in frames per second.
      */

    /** Read a sample at a given frame index.
      * \param pFrameIndex     The index of the animation frame, between 0 and GetSampleCount().
      * \param pBuffer         The place where the point value will be copied. This buffer must be
      *                        of size 3*pPointCount.
      * \param pPointCount     The number of points to read from the point cache file.
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      */
    bool Read(unsigned int pFrameIndex, double* pBuffer, unsigned int pPointCount);

    /** Write a sample at a given frame index.
      * \param pFrameIndex     The index of the animation frame.
      * \param pBuffer         Point to the values to be copied. This buffer must be
      *                        of size 3*pPointCount, as passed to the function OpenFileForWrite().
      * \return                \c true if successful, \c false otherwise. See the error management
      *                        functions for error details.
      * \remarks               Successive calls to Write() must use successive index.
      */
    bool Write(unsigned int pFrameIndex, double* pBuffer);

    //@}

    /**
      * \name File conversion Functions.
      */
    //@{

    /** Create an \e eMC cache file from an ePC2 cache file.
      * \param pFileCount             Create one file for each frame of animation, or one file for all the frames.
      * \param pSamplingFrameRate     Number of frames per second used to resample the point animation.
      * \return                       \c true if successful, \c false otherwise. See the error management
      *                               functions for error details.
      * \remarks                      The created point cache file will be located in the _fpc folder associate with the FBX file.
      */
    bool ConvertFromPC2ToMC(EMCFileCount pFileCount, double pSamplingFrameRate);

    /** Create an \e ePC2 cache file from an eMC cache file.
      * \param pSamplingFrameRate     Number of frames per second to resample the point animation.
      * \param pChannelIndex          Index of the channel of animation to read from.
      * \return                       \c true if successful, \c false otherwise. See the error management
      *                               functions for error details.
      * \remarks                      The created point cache file will be located in the _fpc folder associate with the FBX file.
      */
    bool ConvertFromMCToPC2(double pSamplingFrameRate, unsigned int pChannelIndex);

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
      * - \e eUNSUPPORTED_ARCHITECTURE
      * - \e eINVALID_ABSOLUTE_PATH
      * - \e eINVALID_SAMPLING_RATE
      * - \e eINVALID_CACHE_FORMAT
      * - \e eUNSUPPORTED_FILE_VERSION
      * - \e eCONVERSION_FROM_PC2_FAILED
      * - \e eCONVERSION_FROM_MC_FAILED
      * - \e eCACHE_FILE_NOT_FOUND
      * - \e eCACHE_FILE_NOT_OPENED
      * - \e eCACHE_FILE_NOT_CREATED
      * - \e eINVALID_OPEN_FLAG
      * - \e eERROR_WRITING_SAMPLE
      * - \e eERROR_READING_SAMPLE
      * - \e eERROR_DATATYPE
      * - \e eERROR_INVALIDCHANNELINDEX
      * - \e eERROR_IRREGULARCHANNELSAMPLING
      * - \e eERROR_CHANNELINTERPRETATION
      * - \e eERROR_CHANNELSAMPLING
      * - \e eERROR_INVALID_FILEINDEX
      * - \e eERROR_CACHEDATAFILENAME
      * - \e eERROR_COUNT
      * - \e eERROR_CHANNELSTARTTIME
      * - \e eERROR_CHANNELPOINTCOUNT
      * - \e eERROR_INVALIDTIME
      */
    typedef enum
    {
        eUNSUPPORTED_ARCHITECTURE,
        eINVALID_ABSOLUTE_PATH,
        eINVALID_SAMPLING_RATE,
        eINVALID_CACHE_FORMAT,
        eUNSUPPORTED_FILE_VERSION,
        eCONVERSION_FROM_PC2_FAILED,
        eCONVERSION_FROM_MC_FAILED,
        eCACHE_FILE_NOT_FOUND,
        eCACHE_FILE_NOT_OPENED,
        eCACHE_FILE_NOT_CREATED,
        eINVALID_OPEN_FLAG,
        eERROR_WRITING_SAMPLE,
        eERROR_READING_SAMPLE,
        eERROR_DATATYPE,
        eERROR_INVALIDCHANNELINDEX,
        eERROR_IRREGULARCHANNELSAMPLING,
        eERROR_CHANNELINTERPRETATION,
        eERROR_CHANNELSAMPLING,
        eERROR_INVALID_FILEINDEX,
        eERROR_CACHEDATAFILENAME,
        eERROR_CHANNELSTARTTIME,
        eERROR_CHANNELPOINTCOUNT,
        eERROR_INVALIDTIME,
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

    //! Assignment operator.
    KFbxCache& operator=( const KFbxCache& pOther );



///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////
#ifndef DOXYGEN_SHOULD_SKIP_THIS
    static const char* CacheFilePropertyName;
    static const char* CacheFileAbsolutePathPropertyName;
    static const char* CacheFileTypePropertyName;

    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

    typedef enum
    {
        _O_RDONLY,
        _O_WRONLY
    } EOpenFlag;

protected:
    bool OpenFile(EOpenFlag pFlag, EMCFileCount pFileCount, double pSamplingFrameRate, const char* pChannelName, unsigned int pSampleCount, unsigned int pPointCount, double pFrameStartOffset, EMCDataType pMCDataType = kDoubleVectorArray); 


    KFbxCache(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxCache();

    virtual void Construct( const KFbxCache* pFrom );
    virtual bool ConstructProperties( bool pForceSet );
    virtual void Destruct(bool pRecursive, bool pDependents);

    // Cache
    KFbxCache_internal* mData;

    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;
    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS
};

typedef KFbxCache* HKFbxCache;

inline EFbxType FbxTypeOf( KFbxCache::EFileFormat const &pItem )            { return eENUM; }


#include <fbxfilesdk_nsend.h>

#endif //_FBXSDK_CACHE_H_
