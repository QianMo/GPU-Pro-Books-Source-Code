/*!  \file kfbxkfcurvefilters.h
 */

#ifndef _FBXSDK_KFBXKFCURVE_FILTERS_H_
#define _FBXSDK_KFBXKFCURVE_FILTERS_H_

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

#include <kfbxplugins/kfbxobject.h>
#include <klib/ktime.h>
#include <klib/kerror.h>
#include <kfbxmath/kfbxxmatrix.h>


#include <fbxfilesdk_nsbegin.h>
class KFCurve;
class KFCurveNode;
class KFCurveFilterConstantKeyReducer;
class KFCurveFilterMatrixConverter;
class KFCurveFilterResample;
class KFCurveFilterUnroll;
class KFCurveFilter;

/** \brief Base class for KFCurveNode and KFCurve filtering.
* \nosubgrouping
* A class is necessary to hold the parameters of a filtering
* algorithm.  Independent UI can then be attached to those
* parameters.
*/
class KFBX_DLL KFbxKFCurveFilters : public KFbxObject
{
    KFBXOBJECT_DECLARE(KFbxKFCurveFilters,KFbxObject);

public:
    /** Get the Name of the Filter
    * \return     Pointer to name.
    */
    virtual const char* GetName() {return NULL;}

    /** Get the Start Time
    * \return     The time expressed as KTime.
    */
    virtual KTime& GetStartTime() {return mTime;}

    /** Set the Start Time
    * \param pTime     The time to be set.
    */
    virtual void SetStartTime(KTime& pTime){return;}

    /** Get the Stop Time
    * \return     The time expressed as KTime.
    */
    virtual KTime& GetStopTime(){return mTime;}

    /** Set the Stop Time
    * \param pTime     The time to be set.
    */
    virtual void SetStopTime(KTime& pTime){return ;}

    /** Get the Start Key
    * \param pCurve     Curve on which we want to retrieve the start key
    * \return           The position of the start key
    */
    virtual int GetStartKey(KFCurve& pCurve){return 0;}

    /** Get the Stop Key
    * \param pCurve     Curve on which we want to retrieve the stop key
    * \return           The position of the stop key
    */
    virtual int GetStopKey(KFCurve& pCurve){return 0;}

    /** Check if the KFCurveNode need an application of the filter.
    * \param pCurveNode     Curve to test if it needs application of filter
    * \param pRecursive     Check recursively through the Curve
    * \return               \c true if the KFCurveNode need an application of the filter.
    */
    virtual bool NeedApply(KFCurveNode& pCurveNode, bool pRecursive = true){return false;}

    /** Check if one KFCurve in an array needs an application of the filter.
    * \param pCurve     Array of Curves to test if it needs application of filter
    * \param pCount     Number of Curves in array to test
    * \return           \c true if one KFCurve in an array need an application of the filter.
    */
    virtual bool NeedApply(KFCurve** pCurve, int pCount){return false;}

    /** Check if a KFCurve need an application of the filter.
    * \param pCurve     Curve to test if it needs application of filter
    * \return           \c true if the KFCurve need an application of the filter.
    */
    virtual bool NeedApply(KFCurve& pCurve){return false;}

    /** Apply filter on a KFCurveNode.
    * \param pCurveNode     Curve to apply the filter
    * \param pRecursive     Apply recursively through the Curve
    * \return               \c true if successful, \c false otherwise.
    */
    virtual bool Apply(KFCurveNode& pCurveNode, bool pRecursive = true){return false;}

    /** Apply filter on a number of KFCurve.
    * \param pCurve     Array of curves to apply the filter
    * \param pCount     Number of curves in array to apply the filter
    * \return           \c true if successful, \c false otherwise.
    */
    virtual bool Apply(KFCurve** pCurve, int pCount){return false;}

    /** Apply filter on a KFCurve.
    * \param pCurve         Curve to apply the filter
    * \return               \c true if successful, \c false otherwise.
    */
    virtual bool Apply(KFCurve& pCurve){return false;}

    /** Reset default parameters.
    */
    virtual void Reset(){return ;}

    /** Retrieve error object.
    * \return     Error object.
    */
    virtual KError& GetError(){return GetError();}

    /** Get last error ID.
    * \return     Last error ID.
    */
    virtual int GetLastErrorID(){return -1;}

    /** Get last error name.
    * \return     Last error name.
    */
    virtual char* GetLastErrorString(){return NULL;}
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
    KFbxKFCurveFilters(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxKFCurveFilters();
    virtual void Destruct(bool pRecursive, bool pDependents);
    KTime mTime;
#endif
};




/** \brief Key reducing filter.
  * \nosubgrouping
  * Filter to test if each key is really necessary to define the curve
  * at a definite degree of precision. It filters recursively from the
  * strongest difference first. All useless keys are eliminated.
  */
class KFBX_DLL KFbxKFCurveFilterConstantKeyReducer : public KFbxKFCurveFilters
{
    KFBXOBJECT_DECLARE(KFbxKFCurveFilterConstantKeyReducer,KFbxKFCurveFilters);

public:


    /** Get the Name of the Filter
      * \return     Pointer to name
      */
    const char* GetName();

    /** Get the Start Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStartTime();

    /** Set the Start Time
      * \param pTime     The time to be set
      */
    void SetStartTime(KTime& pTime);

    /** Get the Stop Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStopTime();

    /** Set the Stoping Time
      * \param pTime     The time to be set
      */
    void SetStopTime(KTime& pTime);

    /** Get the Start Key
      * \param pCurve     Curve on which we want to retrieve the start key
      * \return           The position of the start key
      */
    int GetStartKey(KFCurve& pCurve);

    /** Get the Stop Key
      * \param pCurve     Curve on which we want to retrieve the stop key
      * \return           The position of the stop key
      */
    int GetStopKey(KFCurve& pCurve);

    /** Check if the KFCurveNode needs an application of the filter.
      * \param pCurveNode     Curve to test if it needs application of filter
      * \param pRecursive     Check recursively through the Curve
      * \return               \c true if the KFCurveNode need an application of the filter.
      */
    bool NeedApply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Check if one KFCurve in an array needs an application of the filter.
      * \param pCurve     Array of Curves to test if it needs application of filter
      * \param pCount     Number of Curves in array to test
      * \return           \c true if one KFCurve in an array need an application of the filter.
      */
    bool NeedApply(KFCurve** pCurve, int pCount);

    /** Check if a KFCurve need an application of the filter.
      * \param pCurve     Curve to test if it needs application of filter
      * \return           \c true if the KFCurve need an application of the filter.
      */
    bool NeedApply(KFCurve& pCurve);

    /** Retrieve error object.
      * \return     Error object.
      */
    KError& GetError();

    /** Get last error ID.
      * \return     Last error ID.
      */
    int GetLastErrorID() const;

    /** Get last error name.
      * \return     Last error name.
      */
    const char* GetLastErrorString() const;

    /** Apply filter on a KFCurveNode.
      * \param pCurveNode     Curve to apply the filter
      * \param pRecursive     Apply recursively through the Curve
      * \return               \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Apply filter on a number of KFCurve.
      * \param pCurve     Array of Curve to apply the filter
      * \param pCount     Number of Curves in array to apply the filter
      * \return           \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve** pCurve, int pCount);

    /** Apply filter on a KFCurve.
      * \param pCurve     Curve to apply the filter
      * \return           \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve& pCurve);

    /** Reset default parameters.
      */
    void Reset();

    /** Get the derivative tolerance.
      * \return     The value of the derivative tolerance.
      */
    double GetDerivativeTolerance();

    /** Set the derivative tolerance.
      * \param pValue     Value derivative tolerance.
      */
    void SetDerivativeTolerance(double pValue);

    /** Get the tolerance value.
      * \return     The tolerance value.
      */
    double GetValueTolerance();

    /** Set the tolerance value.
      * \param pValue     Tolerance value.
      */
    void SetValueTolerance(double pValue);

    /** Get the state of the KeepFirstAndLastKeys flag.
      * \return      \c true if the filter keeps the first and last keys.
      */
    bool GetKeepFirstAndLastKeys();

    /** Set the state of the KeepFirstAndLastKeys flag.
      * \param pKeepFirstAndLastKeys     Set to \c true if you want the filter to keep the first and last keys.
      */
    void SetKeepFirstAndLastKeys( bool pKeepFirstAndLastKeys );

    /** Get the state of the KeepOneKey flag.
      * \return     \c true if the filter keeps one keys.
      */
    bool GetKeepOneKey();

    /** Set the state of the KeepOneKey flag.
      * \param pKeepOneKey     Set to \c true if you want the filter to keep one key.
      */
    void SetKeepOneKey( bool pKeepOneKey );

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef DOXYGEN_SHOULD_SKIP_THIS
    //
    //  If ValueTolerance is default, we use the thresholds here, otherwise
    //  it is the ValueTolerance that is used. (Mainly for backward compatibility)
    //
    void SetTranslationThreshold    ( double pTranslationThreshold );
    void SetRotationThreshold       ( double pRotationThreshold );
    void SetScalingThreshold        ( double pScalingThreshold );
    void SetDefaultThreshold        ( double pDefaultThreshold );
protected:
    //! Constructor.
    KFbxKFCurveFilterConstantKeyReducer(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxKFCurveFilterConstantKeyReducer();
    KFCurveFilterConstantKeyReducer *mDataCurveFilter;
    virtual void Destruct(bool pRecursive, bool pDependents);

#endif
};




/** Matrix conversion filter.
  * \nosubgrouping
  */
class KFBX_DLL KFbxKFCurveFilterMatrixConverter : public KFbxKFCurveFilters
{
    KFBXOBJECT_DECLARE(KFbxKFCurveFilterMatrixConverter,KFbxKFCurveFilters);

public:

    /** Get the Name of the Filter
      * \return     Pointer to name
      */
    const char* GetName();

    /** Get the Start Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStartTime();

    /** Set the Start Time
      * \param pTime     the time to be set
      */
    void SetStartTime(KTime& pTime);

    /** Get the Stop Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStopTime();

    /** Set the Stoping Time
      * \param pTime     the time to be set
      */
    void SetStopTime(KTime& pTime);

    /** Get the Start Key
      * \param pCurve     Curve on which we want to retrieve the start key
      * \return           The position of the start key
      */
    int GetStartKey(KFCurve& pCurve);

    /** Get the Stop Key
      * \param pCurve     Curve on which we want to retrieve the stop key
      * \return           The position of the stop key
      */
    int GetStopKey(KFCurve& pCurve);

    /** Check if the KFCurveNode needs an application of the filter.
      * \param pCurveNode     Curve to test if it needs application of filter
      * \param pRecursive     Check recursively through the Curve
      * \return               \c true if the KFCurveNode needs an application of the filter.
      */
    bool NeedApply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Check if one KFCurve in an array needs an application of the filter.
      * \param pCurve     Array of Curves to test if it needs application of filter
      * \param pCount     Number of Curves in array to test
      * \return           \c true if one KFCurve in an array need an application of the filter.
      */
    bool NeedApply(KFCurve** pCurve, int pCount);

    /** Check if a KFCurve need an application of the filter.
      * \param pCurve     Curve to test if it needs application of filter
      * \return           \c true if the KFCurve need an application of the filter.
      */
    bool NeedApply(KFCurve& pCurve);

    /** Retrieve error object.
      * \return     Error object.
      */
    KError& GetError();

    /** Get last error ID.
      * \return     Last error ID.
      */
    int GetLastErrorID() const;

    /** Get last error name.
      * \return     Last error name.
      */
    const char* GetLastErrorString() const;

    /** Apply filter on a KFCurveNode.
      * \param pCurveNode     Curve to apply the filter
      * \param pRecursive     Apply recursively through the Curve
      * \return               \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Apply filter on a number of KFCurve.
      * \param pCurve     Array of Curve to apply the filter
      * \param pCount     Number of Curves in array to apply the filter
      * \return           \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve** pCurve, int pCount);

    /** Apply filter on a KFCurve.
      * \param pCurve     Curve to apply the filter
      * \return           \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve& pCurve);

    /** Reset default parameters.
      */
    void Reset();

    /** \enum EMatrixID Matrix ID
      * - \e ePreGlobal
      * - \e ePreTranslate
      * - \e ePostTranslate
      * - \e ePreRotate
      * - \e ePreScale
      * - \e ePostGlobal
      * - \e eScaleOffset
      * - \e eInactivePre
      * - \e eInactivePost
      * - \e eRotationPivot
      * - \e eScalingPivot
      * - \e eMatrixCount
      */
    enum EMatrixID
    {
        ePreGlobal,
        ePreTranslate,
        ePostTranslate,
        ePreRotate,
        ePostRotate,
        ePreScale,
        ePostScale,
        ePostGlobal,
        eScaleOffset,
        eInactivePre,
        eInactivePost,
        eRotationPivot,
        eScalingPivot,
        eMatrixCount
    };

    /** Get the Translation Rotation Scaling source matrix
      * \param pIndex      The matrix ID.
      * \param pMatrix     The matrix used to receive the source matrix.
      */
    void GetSourceMatrix(EMatrixID pIndex, KFbxXMatrix& pMatrix);

    /** Set the Translation Rotation Scaling source matrix.
      * \param pIndex      The matrix ID.
      * \param pMatrix     The matrix used to set the source matrix.
      */
    void SetSourceMatrix(EMatrixID pIndex, KFbxXMatrix& pMatrix);

    /** Get the Translation Rotation Scaling destination matrix.
      * \param pIndex      The matrix ID.
      * \param pMatrix     The matrix used to receive the destination matrix.
      */
    void GetDestMatrix(EMatrixID pIndex, KFbxXMatrix& pMatrix);

    /** Set the Translation Rotation Scaling destination matrix.
      * \param pIndex      The matrix ID.
      * \param pMatrix     The matrix used to set the destination matrix.
      */
    void SetDestMatrix(EMatrixID pIndex, KFbxXMatrix& pMatrix);

    /** Get the Resampling Period.
      * \return     the Resampling Period.
      */
    KTime GetResamplingPeriod ();

    /** Set the Resampling period.
      * \param pResamplingPeriod     The Resampling Period to be set.
      */
    void SetResamplingPeriod (KTime& pResamplingPeriod);

    /** Check if the last key is exactly at the end time.
      * \return     \c true if last key is set exactly at end time.
      */
    bool GetGenerateLastKeyExactlyAtEndTime();

    /** Set the last key to be is exactly at end time or not
      * \param pFlag     value to set if last key is set exactly at end time.
      */
    void SetGenerateLastKeyExactlyAtEndTime(bool pFlag);

    /** Check if resampling is on frame rate multiple
      * \return     \c true if resampling is on a frame rate multiple.
      */
    bool GetResamplingOnFrameRateMultiple();

    /** Set the resample on a frame rate multiple.
      * \param pFlag     The value to be set
      * \remarks         It might be necessary that the starting time of the converted
      *                  animation starts at an multiple of frame period starting from time 0.
      *                  Most softwares play their animation at a definite frame rate, starting
      *                  from time 0.  As resampling occurs when we can't garantee interpolation,
      *                  keys must match with the moment when the curve is evaluated.
      */
    void SetResamplingOnFrameRateMultiple(bool pFlag);

    /** Get if Apply Unroll is used
      * \return     \c true if unroll is applied.
      */
    bool GetApplyUnroll();

    /** Set if Apply Unroll is used
      * \param pFlag     Value to set
      */
    void SetApplyUnroll(bool pFlag);

    /** Get if constant key reducer is used
      * \return     \c true if constant key reducer is applied.
      */
    bool GetApplyConstantKeyReducer();

    /** Set if constant key reducer is used
      * \param pFlag     value to set
      */
    void SetApplyConstantKeyReducer(bool pFlag);

    /** Get if the Resample Translation is used
      * \return      \c true if translation data is resampled upon conversion.
      * \remarks     If this flag isn't set, translation data must be calculated
      *              after the conversion process, overriding the resampling process.
      */
    bool GetResampleTranslation();

    /** Set the resample translation data.
      * \param pFlag     Value to be set.
      * \remarks         If this flag isn't set, translation data must be calculated
      *                  after the conversion process, overriding the resampling process.
      */
    void SetResampleTranslation(bool pFlag);

    /** Set the Rotation Order of the Source
      * \param pOrder     the order to be set
      */
    void SetSrcRotateOrder(int pOrder);

    /** Set the Rotation Order of the Destination
      * \param pOrder     the order to be set
      */
    void SetDestRotateOrder(int pOrder);

    /** Set to force apply even if source and destination matrices are equivalent
      * \param pVal     If the forces apply is to be used
      */
    void SetForceApply(bool pVal);

    /** Get if the force apply is used
      * \return     \c true if the force apply is used
      */
    bool GetForceApply();
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
    //! Constructor.
    KFbxKFCurveFilterMatrixConverter(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxKFCurveFilterMatrixConverter();
    KFCurveFilterMatrixConverter *mDataCurveFilter;
    virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};




/** Resampling filter.
* \nosubgrouping
*/
class KFBX_DLL KFbxKFCurveFilterResample : public KFbxKFCurveFilters
{
    KFBXOBJECT_DECLARE(KFbxKFCurveFilterResample,KFbxKFCurveFilters);

public:
    /** Get the Name of the Filter
      * \return     Pointer to name
      */
    const char* GetName();

    /** Get the Start Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStartTime();

    /** Set the Start Time
      * \param pTime     The time to be set
      */
    void SetStartTime(KTime& pTime);

    /** Get the Stop Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStopTime();

    /** Set the Stoping Time
      * \param pTime     The time to be set
      */
    void SetStopTime(KTime& pTime);

    /** Get the Start Key
      * \param pCurve     Curve on which we want to retrieve the start key
      * \return           The position of the start key
      */
    int GetStartKey(KFCurve& pCurve);

    /** Get the Stop Key
      * \param pCurve     Curve on which we want to retrieve the stop key
      * \return           The position of the stop key
      */
    int GetStopKey(KFCurve& pCurve);

    /** Check if the KFCurveNode need an application of the filter.
      * \param pCurveNode     Curve to test if it needs application of filter
      * \param pRecursive     Check recursively through the Curve
      * \return               \c true if the KFCurveNode need an application of the filter.
      */
    bool NeedApply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Check if one KFCurve in an array need an application of the filter.
      * \param pCurve     Array of Curves to test if it needs application of filter
      * \param pCount     Number of Curves in array to test
      * \return           \c true if one KFCurve in an array need an application of the filter.
      */
    bool NeedApply(KFCurve** pCurve, int pCount);

    /** Check if a KFCurve need an application of the filter.
      * \param pCurve     Curve to test if it needs application of filter
      * \return           \c true if the KFCurve need an application of the filter.
      */
    bool NeedApply(KFCurve& pCurve);

    /** Retrieve error object.
      * \return     Error object.
      */
    KError& GetError();

    /** Get last error ID.
      * \return     Last error ID.
      */
    int GetLastErrorID() const;

    /** Get last error name.
      * \return     Last error name.
      */
    const char* GetLastErrorString() const;

    /** Apply filter on a KFCurveNode.
      * \param pCurveNode     Curve to apply the filter
      * \param pRecursive     Apply recursively through the Curve
      * \return               \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Apply filter on a number of KFCurve.
      * \param pCurve     Array of Curve to apply the filter
      * \param pCount     Number of Curves in array to apply the filter
      * \return           \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve** pCurve, int pCount);

     /** Apply filter on a KFCurve.
      * \param pCurve     Curve to apply the filter
      * \return           \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve& pCurve);

    /** Reset default parameters.
      */
    void Reset();

    /** Set if the keys are on frame
      * \param pKeysOnFrame     value if keys are set on frame multiples.
      */
    void SetKeysOnFrame(bool pKeysOnFrame);

    /** Get if the keys are on frame
      * \return     Value if keys are on frame multiples.
      */
    bool GetKeysOnFrame();

    /** Get the Resampling period
      * \return     The Resampling period
      */
    KTime GetPeriodTime();

    /** Set the Resampling Period
      * \param pPeriod     The Resampling Period to be set
      */
    void SetPeriodTime(KTime &pPeriod);


    /** Get the Intelligent Mode
      * \return     the Intelligent Mode
      */
    bool  GetIntelligentMode();

    /** Set the Intelligent Mode
      * \param pIntelligent     the Intelligent Mode to be set
      */
    void  SetIntelligentMode( bool pIntelligent );

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
    //! Constructor.
    KFbxKFCurveFilterResample(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxKFCurveFilterResample();
    KFCurveFilterResample *mDataCurveFilter;
    virtual void Destruct(bool pRecursive, bool pDependents);
#endif

};

/**Unroll filter
  *\nosubgrouping
  */
class KFBX_DLL KFbxKFCurveFilterUnroll : public KFbxKFCurveFilters
{
    KFBXOBJECT_DECLARE(KFbxKFCurveFilterUnroll,KFbxKFCurveFilters);

public:

    /** Get the Name of the Filter
      * \return     Pointer to name
      */
    const char* GetName();

    /** Get the Start Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStartTime();
    /** Set the Start Time
      * \param pTime     The time to be set
      */
    void SetStartTime(KTime& pTime);
    /** Get the Stop Time
      * \return     The time expressed as KTime.
      */
    KTime& GetStopTime();

    /** Set the Stoping Time
      * \param pTime     The time to be set
      */
    void SetStopTime(KTime& pTime);

    /** Get the Start Key
      * \param pCurve     Curve on which we want to retrieve the start key
      * \return           The position of the start key
      */
    int GetStartKey(KFCurve& pCurve);
    /** Get the Stop Key
      * \param pCurve     Curve on which we want to retrieve the stop key
      * \return           The position of the stop key
      */
    int GetStopKey(KFCurve& pCurve);

    /** Check if the KFCurveNode need an application of the filter.
      * \param pCurveNode     Curve to test if it needs application of filter
      * \param pRecursive     Recursive check recursively through the Curve
      * \return               \c true if the KFCurveNode need an application of the filter.
      */
    bool NeedApply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Check if one KFCurve in an array need an application of the filter.
      * \param pCurve         Array of Curves to test if it needs application of filter
      * \param pCount         Number of Curves in array to test
      * \return               \c true if one KFCurve in an array need an application of the filter.
      */
    bool NeedApply(KFCurve** pCurve, int pCount);

    /** Check if a KFCurve need an application of the filter.
      * \param pCurve         Curve to test if it needs application of filter
      * \return               \c true if the KFCurve need an application of the filter.
      */
    bool NeedApply(KFCurve& pCurve);

    /** Retrieve error object.
      * \return     Error object.
      */
    KError& GetError();
    /** Get last error ID.
      * \return     Last error ID.
      */
    int GetLastErrorID() const;
    /** Get last error name.
      * \return     Last error name.
      */
    const char* GetLastErrorString() const;

    /** Apply filter on a KFCurveNode.
      * \param pCurveNode     Curve to apply the filter
      * \param pRecursive      Apply recursively through the Curve
      * \return               \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurveNode& pCurveNode, bool pRecursive = true);

    /** Apply filter on a number of KFCurve.
      * \param pCurve         Array of Curve to apply the filter
      * \param pCount         Number of Curves in array to apply the filter
      * \return               \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve** pCurve, int pCount);

    /** Apply filter on a KFCurve.
      * \param pCurve         Curve to apply the filter
      * \return               \c true if successful, \c false otherwise.
      */
    bool Apply(KFCurve& pCurve);

    /** Reset default parameters.
      */
    void Reset();

    /** Get quality tolerance.
    * \return     The Quality Tolerance
    */
    double GetQualityTolerance();

    /** Set quality tolerance.
      * \param pQualityTolerance     Value to be set.
      */
    void SetQualityTolerance(double pQualityTolerance);

    /** Get if the test path is enabled
      * \return     \c true if test for path is enabled.
      */
    bool GetTestForPath();

    /** Set if the test path is enabled
      * \param pTestForPath     Value to set if test for path is to be enabled.
      */
    void SetTestForPath(bool pTestForPath);

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
    //! Constructor.
    KFbxKFCurveFilterUnroll(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxKFCurveFilterUnroll();
    KFCurveFilterUnroll *mDataCurveFilter;
    virtual void Destruct(bool pRecursive, bool pDependents);
#endif
};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_KFXKFCURVE_FILTERS_H_
