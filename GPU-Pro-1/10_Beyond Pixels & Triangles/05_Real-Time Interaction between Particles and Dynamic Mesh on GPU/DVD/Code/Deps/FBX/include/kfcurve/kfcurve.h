/*!  \file kfcurve.h
 */

#ifndef _FBXSDK_KFCURVE_H_
#define _FBXSDK_KFCURVE_H_

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
#include <kfcurve/kfcurve_h.h>

#include <klib/karrayul.h>
#include <klib/ktime.h>
#include <object/e/keventbase.h>


#ifndef K_PLUGIN
	#include <object/i/iobject.h>
	#include <object/i/ifbobjectholder.h>
#endif

#ifndef K_PLUGIN
	#include <klib/kdebug.h>
#endif

#include <kbaselib_forward.h>
#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#ifdef K_PLUGIN
	#define KFCURVE_INLINE
#else
	#define KFCURVE_INLINE inline
#endif

#include <kfcurve/kfcurve_nsbegin.h>

	#define KFCURVE_FLOAT
	#ifdef KFCURVE_FLOAT
		typedef float kFCurveDouble;
	#else
		typedef double kFCurveDouble;
	#endif


	K_FORWARD(KFCurve);

	#define IKFCurveID 43763635

	typedef HKFCurve HIKFCurve;
	typedef class KFCURVE_DLL KArrayTemplate< KFCurve * > KArrayKFCurve;

	// Recording memory functions declaration
	KFCURVE_DLL kULong GetRecordingMemory();
	KFCURVE_DLL void WatchFree(void* pPtr, kULong pSize);
	KFCURVE_DLL void* WatchMalloc(kULong pSize);

	//! Key interpolation type.
	enum
	{
		KFCURVE_INTERPOLATION_CONSTANT    = 0x00000002,		//! Constant value until next key.
		KFCURVE_INTERPOLATION_LINEAR	  = 0x00000004,		//! Linear progression to next key.
		KFCURVE_INTERPOLATION_CUBIC		  = 0x00000008,		//! Cubic progression to next key.
		KFCURVE_INTERPOLATION_ALL		  =	KFCURVE_INTERPOLATION_CONSTANT|KFCURVE_INTERPOLATION_LINEAR|KFCURVE_INTERPOLATION_CUBIC,
		KFCURVE_INTERPOLATION_COUNT		  = 3
	};												  

	//! Key constant mode.
	enum
	{
		KFCURVE_CONSTANT_STANDARD		  =	0x00000000,
		KFCURVE_CONSTANT_NEXT			  =	0x00000100,
		KFCURVE_CONSTANT_ALL			  =	KFCURVE_CONSTANT_STANDARD | KFCURVE_CONSTANT_NEXT,
		KFCURVE_CONSTANT_COUNT			  = 2
	};

	//! Key tangent mode for cubic interpolation.
	enum
	{
		KFCURVE_TANGEANT_AUTO			  =	0x00000100, 	//! Spline cardinal.
		KFCURVE_TANGEANT_TCB			  =	0x00000200,		//! Spline TCB.
		KFCURVE_TANGEANT_USER			  =	0x00000400, 	//! Slope at the left equal to slope at the right.
		KFCURVE_GENERIC_BREAK			  =	0x00000800, 	//! Independent left and right slopes.
	KFCURVE_GENERIC_CLAMP			  =	0x00001000, 	//! Auto key should be flat if next or prev keys have same value
		KFCURVE_TANGEANT_BREAK			  = KFCURVE_TANGEANT_USER|KFCURVE_GENERIC_BREAK,
		KFCURVE_TANGEANT_AUTO_BREAK	  = KFCURVE_TANGEANT_AUTO|KFCURVE_GENERIC_BREAK,
	KFCURVE_TANGEANT_ALL			  = KFCURVE_TANGEANT_AUTO|KFCURVE_TANGEANT_TCB|KFCURVE_TANGEANT_USER|KFCURVE_GENERIC_BREAK|KFCURVE_GENERIC_CLAMP,
	KFCURVE_TANGEANT_TYPE_MASK 		  = KFCURVE_TANGEANT_AUTO|KFCURVE_TANGEANT_TCB|KFCURVE_TANGEANT_USER|KFCURVE_TANGEANT_BREAK, // Break is part of the modes for historic reasons, should be part of overrides
	KFCURVE_TANGEANT_OVERRIDES_MASK   = KFCURVE_GENERIC_CLAMP
	// KFCURVE_TANGEANT_COUNT			  = 4
	};

	//! Selection mode.
	enum 
	{
		KFCURVE_SELECT_POINT			  =	0x00010000, 
		KFCURVE_SELECT_LEFT				  =	0x00020000, 
		KFCURVE_SELECT_RIGHT			  =	0x00040000, 
		KFCURVE_SELECT_ALL				  =	KFCURVE_SELECT_POINT|KFCURVE_SELECT_LEFT|KFCURVE_SELECT_RIGHT
	};

	//! Manipulation flag
	enum
	{
		KFCURVE_MARKED_FOR_MANIP          = 0x00080000,
		KFCURVE_MARKED_ALL                = KFCURVE_MARKED_FOR_MANIP
	};

	//! Tangent visibility.
	enum 
	{
		KFCURVE_TANGEANT_SHOW_NONE		  = 0x00000000, 
		KFCURVE_TANGEANT_SHOW_LEFT		  = 0x00100000, 
		KFCURVE_TANGEANT_SHOW_RIGHT		  = 0x00200000, 
		KFCURVE_TANGEANT_SHOW_BOTH		  = KFCURVE_TANGEANT_SHOW_LEFT|KFCURVE_TANGEANT_SHOW_RIGHT
	};

//! Continuity flag
enum
{
    KFCURVE_CONTINUITY				  = 0x00000000,
    KFCURVE_CONTINUITY_FLAT           = 0x00100000,
    KFCURVE_CONTINUITY_BREAK          = 0x00200000,
    KFCURVE_CONTINUITY_INSERT         = 0x00400000   // Used to prevent the curve shape from changing when inserting a key
};

	//! Weighted mode.
	enum 
	{
		KFCURVE_WEIGHTED_NONE			  =	0x00000000, 
		KFCURVE_WEIGHTED_RIGHT			  =	0x01000000, 
		KFCURVE_WEIGHTED_NEXT_LEFT		  =	0x02000000, 
		KFCURVE_WEIGHTED_ALL			  =	KFCURVE_WEIGHTED_RIGHT|KFCURVE_WEIGHTED_NEXT_LEFT
	};

	// !Velocity mode
	enum
	{
		KFCURVE_VELOCITY_NONE			  = 0x00000000,
		KFCURVE_VELOCITY_RIGHT			  = 0x10000000,
		KFCURVE_VELOCITY_NEXT_LEFT		  = 0x20000000,
		KFCURVE_VELOCITY_ALL			  = KFCURVE_VELOCITY_RIGHT | KFCURVE_VELOCITY_NEXT_LEFT
	};


	#ifndef DOXYGEN_SHOULD_SKIP_THIS

	#define KFCURVE_WEIGHT_DIVIDER       9999       // precise enough and can be divided by 3 without error
	#define KFCURVE_DEFAULT_WEIGHT       ((kFCurveDouble)(1.0/3.0))
	#define KFCURVE_MIN_WEIGHT           ((kFCurveDouble)(1.0/KFCURVE_WEIGHT_DIVIDER))
	#define KFCURVE_MAX_WEIGHT           ((kFCurveDouble)0.99)
	#define KFCURVE_DEFAULT_VELOCITY	 0.0 

	#endif // DOXYGEN_SHOULD_SKIP_THIS


	//! KFCurveKey data indices for cubic interpolation tangent information.
	enum EKFCurveDataIndex
	{
		// User and Break tangent mode (data are doubles).
		KFCURVEKEY_RIGHT_SLOPE			= 0, 
		KFCURVEKEY_NEXT_LEFT_SLOPE		= 1, 

		// User and Break tangent break mode (data are kInt16 thken from mwight and converted to doubles).
		KFCURVEKEY_WEIGHTS				= 2, 
		KFCURVEKEY_RIGHT_WEIGHT			= 2, 
		KFCURVEKEY_NEXT_LEFT_WEIGHT		= 3, 

		// Velocity mode
		KFCURVEKEY_VELOCITY				= 4,
		KFCURVEKEY_RIGHT_VELOCITY		= 4,
		KFCURVEKEY_NEXT_LEFT_VELOCITY	= 5, 

		// TCB tangent mode (data are floats).
		KFCURVEKEY_TCB_TENSION			= 0, 
		KFCURVEKEY_TCB_CONTINUITY		= 1, 
		KFCURVEKEY_TCB_BIAS				= 2,

		KFCURVEKEY_RIGHT_AUTO			= 0,
		KFCURVEKEY_NEXT_LEFT_AUTO		= 1
	};

	//! Extrapolation mode for function curve extremities.
	enum 
	{
		KFCURVE_EXTRAPOLATION_CONST				= 1, 
		KFCURVE_EXTRAPOLATION_REPETITION		= 2, 
		KFCURVE_EXTRAPOLATION_MIRROR_REPETITION	= 3, 
		KFCURVE_EXTRAPOLATION_KEEP_SLOPE		= 4
	};

	enum 
	{
		KFCURVE_BEZIER	= 0, 
		KFCURVE_SAMPLE	= 1, 
		KFCURVE_ISO		= 2
	};

	typedef kUInt kFCurveInterpolation;
	typedef kUInt kFCurveConstantMode;
	typedef kUInt kFCurveTangeantMode;
	typedef kUInt kFCurveTangeantWeightMode;
	typedef kUInt kFCurveTangeantVelocityMode;
	typedef kUInt kFCurveExtrapolationMode;
	typedef kUInt kFCurveTangeantVisibility;
	typedef int kFCurveIndex;

	enum 
	{
		KFCURVEEVENT_NONE		=0, // default event value
		KFCURVEEVENT_CANDIDATE	=1 << 0, // curve value (not candidate) changed
		KFCURVEEVENT_UNUSED1    =1 << 1,
		KFCURVEEVENT_UNUSED2    =1 << 2,
		KFCURVEEVENT_UNUSED3    =1 << 3,
		KFCURVEEVENT_KEY		=1 << 4, // key changed (add, removed, edited); see bits 11-15 for precisions
		KFCURVEEVENT_DEPRECATED5 =1 << 5,
		KFCURVEEVENT_UNUSED6    =1 << 6,
		KFCURVEEVENT_UNUSED7    =1 << 7,
		KFCURVEEVENT_SELECTION	=1 << 8, // key selection changed
		KFCURVEEVENT_DESTROY	=1 << 9, // fcurve destruction
		KFCURVEEVENT_DEPRECATED10 =1 << 10,
		KFCURVEEVENT_KEYADD     =1 << 11,
		KFCURVEEVENT_KEYREMOVE  =1 << 12,
		KFCURVEEVENT_EDITVALUE  =1 << 13,
		KFCURVEEVENT_EDITTIME   =1 << 14,
		KFCURVEEVENT_EDITOTHER  =1 << 15,
	};


	// Curve event class.
	class KFCURVE_DLL KFCurveEvent : public KEventBase
	{
	public:
		// Curve event type, the enum stated above allow composition of type (bitfield). 
		// Stored in mType

		// Start key index.
		int mKeyIndexStart; 

		//	Stop key index.
		int mKeyIndexStop; 

		// Count of events.
		int mEventCount;

		// Clear curve event.
		KFCURVE_INLINE void Clear (); 
		
		// Add a curve event of type pWhat to a curve event object.
		KFCURVE_INLINE void Add (int pWhat, int pIndex);
	};

	typedef void (*kFCurveCallback) (KFCurve *pFCurve, KFCurveEvent *FCurveEvent, void* pObject);

	/** Defines a tangent derivative and weight
	*	\remarks Implementation was made for performance.
	* \nosubgrouping
	*/
	class KFCURVE_DLL KFCurveTangeantInfo 
	{
	public:
		KFCURVE_INLINE KFCurveTangeantInfo();

		kFCurveDouble mDerivative;
		kFCurveDouble mWeight;
		bool         mWeighted;
		kFCurveDouble mVelocity;
		bool		  mHasVelocity;
		kFCurveDouble mAuto;  // The auto parameter!
	};

	/** Defines a key within a function curve.
	*	\remarks Implementation was made for performance.
	*	Keep in mind that there is no check for consistency and memory 
	* management ever made throughout the methods' code. This class must be 
	* used with a good understanding of its interface.
	* Default constructor is used, which does not initialize data 
	* member. If an instance has to be initialized, use function KFCurveKey::Set().
	* \nosubgrouping
	*/
	class KFCURVE_DLL KFCurveKey 
	{
	public:
		KFCurveKey()
		{
			Init();
		}

	public:
		
		/** Set a key.
		*	Use SetTCB() to set a key with cubic interpolation and TCB tangent type.
		*	\param pTime			Key time.
		*	\param pValue			Key value.
		*	\param pInterpolation	Key interpolation type.	Interpolation types are: 
		*							KFCURVE_INTERPOLATION_CONSTANT, 
		*							KFCURVE_INTERPOLATION_LINEAR,
		*							KFCURVE_INTERPOLATION_CUBIC
		*	\param pTangentMode		Key tangent mode (meaningful for cubic 
		*							interpolation only). Tangent modes are: 
		*							KFCURVE_TANGEANT_AUTO,
		*							KFCURVE_TANGEANT_USER,
		*							KFCURVE_TANGEANT_BREAK
		*	\param pData0			Right slope.
		*	\param pData1			Next left slope.
		*	\param pTangentWeightMode	Weight mode if used
		*								KFCURVE_WEIGHTED_NONE
		*								KFCURVE_WEIGHTED_RIGHT
		*								KFCURVE_WEIGHTED_NEXT_LEFT
		*								KFCURVE_WEIGHTED_ALL
		*	\param pWeight0				Right slope weight.
		*	\param pWeight1				Next left slope weight.
		*	\param pVelocity0			Right velocity.
		*	\param pVelocity1			Next left velocity.
		*/
		KFCURVE_INLINE void Set 
		(
			KTime pTime, 
			kFCurveDouble pValue, 
			kFCurveInterpolation pInterpolation = KFCURVE_INTERPOLATION_CUBIC, 
			kFCurveTangeantMode pTangentMode = KFCURVE_TANGEANT_AUTO, 
			kFCurveDouble pData0 = 0.0,
			kFCurveDouble pData1 = 0.0,
			kFCurveTangeantWeightMode pTangentWeightMode = KFCURVE_WEIGHTED_NONE, 
			kFCurveDouble pWeight0                             = KFCURVE_DEFAULT_WEIGHT,
			kFCurveDouble pWeight1                             = KFCURVE_DEFAULT_WEIGHT,
			kFCurveDouble pVelocity0 = KFCURVE_DEFAULT_VELOCITY,
			kFCurveDouble pVelocity1 = KFCURVE_DEFAULT_VELOCITY
		);
		
		/**	Set a key with cubic interpolation, TCB tangent mode.
		*	\param pTime	Key time.
		*	\param pValue	Key value.
		*	\param pData0	Tension.
		*	\param pData1	Continuity.
		*	\param pData2	Bias.
		*/
		KFCURVE_INLINE void SetTCB 
		(
			KTime pTime, 
			kFCurveDouble pValue, 
			float pData0 = 0.0f, 
			float pData1 = 0.0f, 
			float pData2 = 0.0f
		);
		
		/** Key assignment.
		*	\param pSource	Source key to be copied.
		*/
		KFCURVE_INLINE void Set(KFCurveKey& pSource);
		
		/** Get key interpolation type.
		*	Interpolation types are: KFCURVE_INTERPOLATION_CONSTANT, 
		*							 KFCURVE_INTERPOLATION_LINEAR,
		*							 KFCURVE_INTERPOLATION_CUBIC
		*/
		KFCURVE_INLINE kFCurveInterpolation GetInterpolation();
		
		/** Set key interpolation type.
		*	\param pInterpolation Key interpolation type.
		*	Interpolation types are: KFCURVE_INTERPOLATION_CONSTANT, 
		*							 KFCURVE_INTERPOLATION_LINEAR,
		*							 KFCURVE_INTERPOLATION_CUBIC
		*/
		KFCURVE_INLINE void SetInterpolation(kFCurveInterpolation pInterpolation);

		/** Get key constant mode.
		*	Warning: This method is meaningful for constant interpolation only.
		*			 Using this method for non constant interpolated key will return unpredicted value.
		* Constant modes are:		KFCURVE_CONSTANT_STANDARD
		*							KFCURVE_CONSTANT_NEXT
		*	\return Key constant mode.
		*/
		KFCURVE_INLINE kFCurveConstantMode GetConstantMode();

		/** Get key tangent mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*			 Using this method for non cubic interpolated key will return unpredicted value.
		*	Tangent modes are: KFCURVE_TANGEANT_AUTO,
		*					   KFCURVE_TANGEANT_AUTO_BREAK
		*					   KFCURVE_TANGEANT_TCB,
		*					   KFCURVE_TANGEANT_USER,
		*					   KFCURVE_TANGEANT_BREAK
		*	\return Key tangent mode.
		*/
	KFCURVE_INLINE kFCurveTangeantMode GetTangeantMode( bool pIncludeOverrides = false );

		/** Get key tangent weight mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	Tangent weight modes are:	KFCURVE_WEIGHTED_NONE,
		*								KFCURVE_WEIGHTED_RIGHT,
		*								KFCURVE_WEIGHTED_NEXT_LEFT,
		*								KFCURVE_WEIGHTED_ALL
		*/
		KFCURVE_INLINE kFCurveTangeantWeightMode GetTangeantWeightMode();

		/** Get key tangent velocity mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	Tangent weight modes are:	KFCURVE_VELOCITY_NONE,
		*								KFCURVE_VELOCITY_RIGHT,
		*								KFCURVE_VELOCITY_NEXT_LEFT,
		*								KFCURVE_VELOCITY_ALL
		*/
		KFCURVE_INLINE kFCurveTangeantVelocityMode GetTangeantVelocityMode();

		/** Set key constant mode.
		*	Warning: This method is meaningful for constant interpolation only.
		*	\param pMode Key consant mode.
		*	Constant modes are:		KFCURVE_CONSTANT_STANDARD
		*							KFCURVE_CONSTANT_NEXT
		*/
		KFCURVE_INLINE void SetConstantMode(kFCurveConstantMode pMode);

		/** Set key tangent mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	\param pTangent Key tangent mode.
		*	Tangent modes are: KFCURVE_TANGEANT_AUTO,
		*					   KFCURVE_TANGEANT_AUTO_BREAK
		*					   KFCURVE_TANGEANT_TCB,
		*					   KFCURVE_TANGEANT_USER,
		* 				   KFCURVE_TANGEANT_BREAK
		*/
		KFCURVE_INLINE void SetTangeantMode(kFCurveTangeantMode pTangent);
			
		/** Set key tangent weight mode as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: This method is meaningful for cubic interpolation only.
		*	\param pTangentWeightMode	Weight mode
		*								KFCURVE_WEIGHTED_NONE
		*								KFCURVE_WEIGHTED_RIGHT
		*								KFCURVE_WEIGHTED_NEXT_LEFT
		*								KFCURVE_WEIGHTED_ALL
		*	\param pMask				Used to select the affected tangents
		*								KFCURVE_WEIGHTED_RIGHT
		*								KFCURVE_WEIGHTED_NEXT_LEFT
		*								KFCURVE_WEIGHTED_ALL
		*/

		KFCURVE_INLINE void SetTangeantWeightMode(kFCurveTangeantWeightMode pTangentWeightMode, kFCurveTangeantWeightMode pMask = KFCURVE_WEIGHTED_ALL );

			/** Set key tangent velocity mode as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: This method is meaningful for cubic interpolation only.
		*	\param pTangentVelocityMode	Weight mode
		*								KFCURVE_VELOCITY_NONE
		*								KFCURVE_VELOCITY_RIGHT
		*								KFCURVE_VELOCITY_NEXT_LEFT
		*								KFCURVE_VELOCITY_ALL
		*	\param pMask				Used to select the affected tangents
		*								KFCURVE_VELOCITY_RIGHT
		*								KFCURVE_VELOCITY_NEXT_LEFT
		*								KFCURVE_VELOCITY_ALL
		*/

		KFCURVE_INLINE void SetTangeantVelocityMode(kFCurveTangeantVelocityMode pTangentVelocityMode, kFCurveTangeantVelocityMode pMask = KFCURVE_VELOCITY_ALL );

			
		/** Get key data as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: Using this method for other than cubic interpolated 
		*			 key (linear, constant) will return unpredicted values.
		*	Warning: Slope data is inconsistent for automatic tangent mode.
		*			 Use KFCurve::EvaluateLeftDerivative() and 
		*			 KFCurve::EvaluateRightDerivative() to find
		*			 slope values.
		*	Warning: Using this method for TCB tangent mode key will return 
		*			 unpredicted values. Use KFCurve::GetDataFloat() instead.
		*	\param pIndex Data index, either	KFCURVEKEY_RIGHT_SLOPE,
		*										KFCURVEKEY_NEXT_LEFT_SLOPE.
		*										KFCURVEKEY_NEXT_RIGHT_WEIGHT.
		*										KFCURVEKEY_NEXT_LEFT_WEIGHT
		*/
		KFCURVE_INLINE kFCurveDouble GetDataDouble(EKFCurveDataIndex pIndex);
		
		/**	Set data as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: Using this method for other than cubic interpolated 
		*			 key (linear, constant) is irrelevant.
		*	Warning: Slope data is inconsistent for automatic tangent mode.
		*			 Therefore, it is irrelevant to use this method on automatic 
		*			 tangent mode keys.
		*	Warning: Using this method for a TCB tangent mode key will result
		*			 in unpredictable curve behavior for this key. Use KFCurve::SetDataFloat() 
		*			 instead.
		*	\param pIndex Data index, either	KFCURVEKEY_RIGHT_SLOPE,
		*										KFCURVEKEY_NEXT_LEFT_SLOPE.
		*										KFCURVEKEY_NEXT_RIGHT_WEIGHT.
		*										KFCURVEKEY_NEXT_LEFT_WEIGHT
		*	\param pValue	The data value to set (a slope or a weight).
		*/
		KFCURVE_INLINE void SetDataDouble(EKFCurveDataIndex pIndex, kFCurveDouble pValue);
		
		/** Get key data as float value (cubic interpolation, TCB tangent mode).
		*	Warning: Using this method for any key but a cubic interpolated,
		*			 in TCB tangent mode, will return unpredicted values.
		*	\param pIndex	Data index, either KFCURVEKEY_TCB_TENSION, KFCURVEKEY_TCB_CONTINUITY or KFCURVEKEY_TCB_BIAS.
		*/	
		KFCURVE_INLINE float GetDataFloat(EKFCurveDataIndex pIndex);

		/**	Set data as float value (cubic interpolation, TCB tangent mode).
		*	Warning: Using this method for any key but a cubic interpolated,
		*			 in TCB tangent mode, will return unpredicted values.
		*	\param pIndex	Data index, either KFCURVEKEY_TCB_TENSION, KFCURVEKEY_TCB_CONTINUITY or KFCURVEKEY_TCB_BIAS.
		*	\param pValue	The data value to set.
		*/
		KFCURVE_INLINE void SetDataFloat(EKFCurveDataIndex pIndex, float pValue);

		/**	Get key data as a pointer
		*	Warning: not supported in 'double' mode.
		*/
		KFCURVE_INLINE float* GetDataPtr();

		//!	Get key value.
		KFCURVE_INLINE kFCurveDouble GetValue();

		//! Set key value.
		KFCURVE_INLINE void SetValue(kFCurveDouble pValue);

		/** Increment key value.
		*	\param pValue Value by which key value is incremented.
		*/
		KFCURVE_INLINE void IncValue(kFCurveDouble pValue);

		/** Multiply key value.
		*	\param pValue Value by which the key value is multiplied.
		*/
		KFCURVE_INLINE void MultValue(kFCurveDouble pValue);

		/** Multiply key tangents.
		*	Note: When multiplying a key value, tangents must be
		*         multiplied to conserve the same topology.
		*	\param pValue Value by which key tangents are multiplied.
		*/
		KFCURVE_INLINE void MultTangeant(kFCurveDouble pValue);

		/** Get key time
		*	\return Key time (time at which this key is occurring).
		*/
		KFCURVE_INLINE KTime GetTime();

		/** Set key time.
		*	\param pTime Key time (time at which this key is occurring).
		*/
		KFCURVE_INLINE void SetTime(KTime pTime);

		/** Increment key time.
		*	\param pTime Time value by which the key time is incremented.
		*/
		KFCURVE_INLINE void IncTime(KTime pTime);

		/** Set if key is currently selected.
		*	\param pSelected Selection flag.
		*/
		KFCURVE_INLINE void SetSelected(bool pSelected);	

		/** Return if key is currently selected.
		*	\return Selection flag.
		*/
		KFCURVE_INLINE bool GetSelected();

		/** Set if key is currently marked for manipulation.
		*	\param pMark Mark flag.
		*/
		KFCURVE_INLINE void SetMarkedForManipulation(bool pMark);	

		/** Return if key is currently marked for manipulation.
		*	\return Mark flag.
		*/
		KFCURVE_INLINE bool GetMarkedForManipulation();

		/** Set tangent visibility mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	\param pVisibility	Tangent visibility mode.
		*	Tangent visibility modes are: KFCURVE_TANGEANT_SHOW_NONE
		*						          KFCURVE_TANGEANT_SHOW_LEFT
		*						          KFCURVE_TANGEANT_SHOW_RIGHT
		*/
		KFCURVE_INLINE void	SetTangeantVisibility (kFCurveTangeantVisibility pVisibility);	
		
		/** Return tangent visibility mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	\return Tangent visibility mode.
		*	Tangent visibility modes are: KFCURVE_TANGEANT_SHOW_NONE
		*			                      KFCURVE_TANGEANT_SHOW_LEFT
		*			                      KFCURVE_TANGEANT_SHOW_RIGHT
		*/
		KFCURVE_INLINE kFCurveTangeantVisibility GetTangeantVisibility ();

		/** Set/Unset Break tangent
		* Only valid for User and Auto keys
		*/
		KFCURVE_INLINE void SetBreak(bool pVal); 

		/** Get if tangent is break
		* Only valid for User and Auto keys
		*/
		KFCURVE_INLINE bool GetBreak(); 




	///////////////////////////////////////////////////////////////////////////////
	//
	//  WARNING!
	//
	//	Anything beyond these lines may not be documented accurately and is 
	// 	subject to change without notice.
	//
	///////////////////////////////////////////////////////////////////////////////

		KFCURVE_INLINE void Init();

	#ifndef DOXYGEN_SHOULD_SKIP_THIS

	private:

		kFCurveDouble mValue;		
		KTime mTime;	
		kUInt mFlags;

	#ifdef KFCURVE_FLOAT
		float  mData[4];
	#else 
		double	mData[2];
		kInt16	mWeight[2];
		kInt16	mVelocity[2];
	#endif 	

	#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

	};


	const int KEY_BLOCK_SIZE	= 1024;
	const int KEY_BLOCK_COUNT	= KEY_BLOCK_SIZE/sizeof (KFCurveKey);

	const int KEY_LIST_BLOCK_SIZE	= 256;
	const int KEY_LIST_BLOCK_COUNT	= KEY_LIST_BLOCK_SIZE/sizeof (KFCurveKey *);


	/** Function curve class. 
	* A function curve is basically a collection of keys (see class KFCurveKey) 
	* sorted in time order. Since it is a function, only one key per time is
	* allowed. 
	* \nosubgrouping
	*/
	#ifdef K_PLUGIN
	class KFCURVE_DLL KFCurve
	#else 
	class KFCURVE_DLL KFCurve : public IFBObjectHolder
	#endif
	{

	public:

		/**
		* \name Constructor and Destructor
		*/
		//@{

		//! Constructor.
		KFCurve();

		//! Destructor.
		virtual ~KFCurve();

		#ifdef K_PLUGIN
			void Destroy(int Local=0);
		#else
			IObject_Declare(Implementation)
		#endif

		//@}

		/**	Get function curve color.
		*	\return Pointer to an array of 3 elements: RGB values on a scale from 0 to 1.
		*/
		float* GetColor();
		
		/** Set function curve color.
		*	\param pColor Pointer to an array of 3 elements: RGB values on a scale from 0 to 1.
		*/
		void SetColor(float *pColor);

		/** Set default value.
		* Default value is used when there is no key in the function curve.
		*	\param pValue Default value.
		*/
		void SetValue(kFCurveDouble pValue);

		/** Get default value.
		* Default value is used when there is no key in the function curve.
		*	\return Default value.
		*/
		KFCURVE_INLINE kFCurveDouble GetValue() const;

		/**
		* \name Key Management
		*/
		//@{

		/** Resize fcurve buffer to hold a certain number of key.
		* \param pKeyCount Number of key the function curve will eventually hold.
		*/
		void ResizeKeyBuffer(int pKeyCount);

		/** Call this function prior to modifying the keys of a function curve.
		* Call function KFCurve::KeyModifyEnd() after modification of the keys
		* are completed.
		*/
		void KeyModifyBegin ();
		
		/** Call this function after modification of the keys of a function curve.
		* Call function KFCurve::KeyModifyBegin() prior to modifying the keys.
		*/
		void KeyModifyEnd ();

		//! Get the number of keys.
		int KeyGetCount ();

		//! Get the number of selected keys.
		int KeyGetSelectionCount ();

		// Select all keys.
		void KeySelectAll ();
		
		// Unselect all keys.
		void KeyUnselectAll ();

		/** Get key at given index.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		KFCurveKey KeyGet(kFCurveIndex pIndex);
		
		//! Remove all the keys and free buffer memory.
		void KeyClear ();
		
		//! Minimize use of buffer memory.
		void KeyShrink();

		/** Set key at given index.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* \return true if key time is superior to previous key 
		* and inferior to next key.
		*/
		bool	KeySet(kFCurveIndex pIndex, KFCurveKey& pKey);
		KFCURVE_INLINE bool	KeySet(kFCurveIndex pIndex, KFCurve* pSourceCurve, int pSourceIndex);
		
		/** Change time of key found at given index.
		*	\param pIndex Index of key to move.
		*	\param pTime Destination time.
		*	\return New index of moved key.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		int KeyMove(kFCurveIndex pIndex, KTime pTime);
		
		/** Add time and value offsets to keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pDeltaTime Time offset added to keys.
		*	\param pDeltaValue Value offset added to keys.
		*	\return true on success.
		*/
		bool KeyMoveOf (bool pSelectedOnly, KTime pDeltaTime, kFCurveDouble pDeltaValue);
		
		/** Set value of keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pValue Value set to keys.
		*	\return true on success.
		*/
		bool KeyMoveValueTo (bool pSelectedOnly, kFCurveDouble pValue);
		
		/** Scale value of keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pMultValue Scale applied on key values.
		*	\return true on success.
		*/
		bool KeyScaleValue (bool pSelectedOnly, kFCurveDouble pMultValue);

		/** Scale tangent of keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pMultValue Scale applied on key tangents.
		*	\return true on success.
		*/
		bool KeyScaleTangeant (bool pSelectedOnly, kFCurveDouble pMultValue);

		/** Scale value and tangent of keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pMultValue Scale applied on key values and tangents.
		*	\return true on success.
		*/
		bool KeyScaleValueAndTangeant (bool pSelectedOnly, kFCurveDouble pMultValue);

		/** Remove key at given index.
		*	\param pIndex Index of key to remove.
		*	\return true on success.
		*/
		bool KeyRemove(kFCurveIndex pIndex);

		/** Insert a key at given time.
		*	This function SHOULD be used instead of KFCurve::KeyAdd() if the key 
		* is to be added in the curve and not at the end. It inserts the key in 
		* respect to the interpolation type and tangents of the neighboring keys. 
		* If there is already a key a the given time, the key is modified and no 
		* new key is added.
		*	\param pTime Time to insert the key.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Index of the key at given time, no matter if it was inserted 
		* or already present.
		* \remarks Key value must be set explicitly afterwards. The 
		* interpolation type and tangent mode are copied from the previous key.
		*/
		int KeyInsert ( KTime pTime, kFCurveIndex* pLast = NULL );

		/** Add a key at given time.
		*	Function KFCurve::KeyInsert() SHOULD be used instead if the key 
		* is to be added in the curve and not at the end. This function does not
		* respect the interpolation type and tangents of the neighboring keys. 
		* If there is already a key at the given time, the key is modified and no 
		* new key is added.
		*	\param pTime Time to add the key.
		* \param pKey Key to add.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Index of the key at given time, no matter if it was added 
		* or already present.
		* \remarks Key value, interpolation type and tangent mode must be set 
		* explicitly afterwards.
		*/
		int KeyAdd (KTime pTime, KFCurveKey& pKey, kFCurveIndex* pLast = NULL);
		int KeyAdd(KTime pTime, KFCurve* pSourceCurve, int pSourceIndex, kFCurveIndex* pLast = NULL);
		
		/** Add a key at given time.
		*	Function KFCurve::KeyInsert() SHOULD be used instead if the key 
		* is to be added in the curve and not at the end. This function does not
		* respect of the interpolation type and tangents of the neighboring keys. 
		* If there is already a key a the given time, no key is added.
		*	\param pTime Time to add the key.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Index of the key at given time, no matter if it was added 
		* or already present.
		* \remarks Key value, interpolation type and tangent mode must be set 
		* explicitely afterwards.
		*/
		int KeyAdd (KTime pTime, kFCurveIndex* pLast = NULL);

		/** Append a key at the end of the function curve.
		* \param pAtTime Time of appended key, must be superior to the 
		* last key time.
		* \param pSourceCurve Source curve.
		* \param pSourceIndex Index of the source key in the source curve.
		* \return Index of appended key.
		*/
		int KeyAppend(KTime pAtTime, KFCurve* pSourceCurve, int pSourceIndex);

		/** Append a key at the end of the function curve.
		* \param pTime Time of appended key, must be superior to the 
		* last key time.
		* \param pValue Value of appended key.
		* \return Index of appended key.
		* \remarks Interpolation type of the appended key is set to 
		* KFCURVE_INTERPOLATION_CUBIC and tangent mode is set to 
		* KFCURVE_TANGEANT_AUTO.
		*/
		int KeyAppendFast( KTime pTime, kFCurveDouble pValue );
		
		/** Find key index for a given time.
		*	\param pTime Time of the key looked for.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Key index. The integer part of the key index gives the 
		* index of the closest key with a smaller time. The decimals give
		* the relative position of given time compared to previous and next
		* key times. Returns -1 if function curve has no key.
		*/
		double KeyFind (KTime pTime, kFCurveIndex* pLast = NULL);	

		//@}

		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/


		/**
		* \name Key Manipulation
		*/
		//@{

    		/** Set a key.
		*	Use SetTCB() to set a key with cubic interpolation and TCB tangent type.
		*   \param pKeyIndex        Key index
		*	\param pTime			Key time.
		*	\param pValue			Key value.
		*	\param pInterpolation	Key interpolation type.	Interpolation types are: 
		*							KFCURVE_INTERPOLATION_CONSTANT, 
		*							KFCURVE_INTERPOLATION_LINEAR,
		*							KFCURVE_INTERPOLATION_CUBIC
		*	\param pTangentMode		Key tangent mode (meaningful for cubic 
		*							interpolation only). Tangent modes are: 
		*							KFCURVE_TANGEANT_AUTO,
		*							KFCURVE_TANGEANT_USER,
		*							KFCURVE_TANGEANT_BREAK
		*	\param pData0			Right slope.
		*	\param pData1			Next left slope.
		*	\param pTangentWeightMode	Weight mode if used
		*								KFCURVE_WEIGHTED_NONE
		*								KFCURVE_WEIGHTED_RIGHT
		*								KFCURVE_WEIGHTED_NEXT_LEFT
		*								KFCURVE_WEIGHTED_ALL
		*	\param pWeight0				Right slope weight.
		*	\param pWeight1				Next left slope weight.
		*	\param pVelocity0			Right velocity.
		*	\param pVelocity1			Next left velocity.
		*/
		KFCURVE_INLINE void KeySet 
		(
			kFCurveIndex pKeyIndex,
			KTime pTime, 
			kFCurveDouble pValue, 
			kFCurveInterpolation pInterpolation = KFCURVE_INTERPOLATION_CUBIC, 
			kFCurveTangeantMode pTangentMode = KFCURVE_TANGEANT_AUTO, 
			kFCurveDouble pData0 = 0.0,
			kFCurveDouble pData1 = 0.0,
			kFCurveTangeantWeightMode pTangentWeightMode = KFCURVE_WEIGHTED_NONE, 
			kFCurveDouble pWeight0                             = KFCURVE_DEFAULT_WEIGHT,
			kFCurveDouble pWeight1                             = KFCURVE_DEFAULT_WEIGHT,
			kFCurveDouble pVelocity0 = KFCURVE_DEFAULT_VELOCITY,
			kFCurveDouble pVelocity1 = KFCURVE_DEFAULT_VELOCITY
		);
		
		/**	Set a key with cubic interpolation, TCB tangent mode.
		*   \param pKeyIndex  Key index
		*	\param pTime	Key time.
		*	\param pValue	Key value.
		*	\param pData0	Tension.
		*	\param pData1	Continuity.
		*	\param pData2	Bias.
		*/
		KFCURVE_INLINE void KeySetTCB 
		(
			kFCurveIndex pKeyIndex,
			KTime pTime, 
			kFCurveDouble pValue, 
			float pData0 = 0.0f, 
			float pData1 = 0.0f, 
			float pData2 = 0.0f
		);
			
		/** Get key interpolation type.
		*	Interpolation types are: KFCURVE_INTERPOLATION_CONSTANT, 
		*							 KFCURVE_INTERPOLATION_LINEAR,
		*							 KFCURVE_INTERPOLATION_CUBIC
		*   \param pKeyIndex         Key index
		*   \return                  Key interpolation type
		*/
		KFCURVE_INLINE kFCurveInterpolation KeyGetInterpolation(kFCurveIndex pKeyIndex);
		
		/** Set key interpolation type.
		*   \param pKeyIndex      Key index
		*	\param pInterpolation Key interpolation type.
		*	Interpolation types are: KFCURVE_INTERPOLATION_CONSTANT, 
		*							 KFCURVE_INTERPOLATION_LINEAR,
		*							 KFCURVE_INTERPOLATION_CUBIC
		*/
		KFCURVE_INLINE void KeySetInterpolation(kFCurveIndex pKeyIndex, kFCurveInterpolation pInterpolation);

		/** Get key constant mode.
		*	Warning: This method is meaningful for constant interpolation only.
		*			 Using this method for non constant interpolated key will return unpredicted value.
		* Constant modes are:		KFCURVE_CONSTANT_STANDARD
		*							KFCURVE_CONSTANT_NEXT
		*   \param pKeyIndex      Key index
		*	\return Key constant mode.
		*/
		KFCURVE_INLINE kFCurveConstantMode KeyGetConstantMode(kFCurveIndex pKeyIndex);

		/** Get key tangent mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*			 Using this method for non cubic interpolated key will return unpredicted value.
		*	Tangent modes are: KFCURVE_TANGEANT_AUTO,
		*					   KFCURVE_TANGEANT_AUTO_BREAK
		*					   KFCURVE_TANGEANT_TCB,
		*					   KFCURVE_TANGEANT_USER,
		*					   KFCURVE_TANGEANT_BREAK
		*	\return Key tangent mode.
		*/
	KFCURVE_INLINE kFCurveTangeantMode KeyGetTangeantMode(kFCurveIndex pKeyIndex, bool pIncludeOverrides = false );

		/** Get key tangent weight mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	Tangent weight modes are:	KFCURVE_WEIGHTED_NONE,
		*								KFCURVE_WEIGHTED_RIGHT,
		*								KFCURVE_WEIGHTED_NEXT_LEFT,
		*								KFCURVE_WEIGHTED_ALL
		*/
		KFCURVE_INLINE kFCurveTangeantWeightMode KeyGetTangeantWeightMode(kFCurveIndex pKeyIndex);

		/** Get key tangent velocity mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	Tangent weight modes are:	KFCURVE_VELOCITY_NONE,
		*								KFCURVE_VELOCITY_RIGHT,
		*								KFCURVE_VELOCITY_NEXT_LEFT,
		*								KFCURVE_VELOCITY_ALL
		*/
		KFCURVE_INLINE kFCurveTangeantVelocityMode KeyGetTangeantVelocityMode(kFCurveIndex pKeyIndex);

		/** Set key constant mode.
		*	Warning: This method is meaningful for constant interpolation only.
		*   \param pKeyIndex            Key index
		*	\param pMode Key consant mode.
		*	Constant modes are:		KFCURVE_CONSTANT_STANDARD
		*							KFCURVE_CONSTANT_NEXT
		*/
		KFCURVE_INLINE void KeySetConstantMode(kFCurveIndex pKeyIndex, kFCurveConstantMode pMode);

		/** Set key tangent mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*   \param pKeyIndex   Key index
		*	\param pTangent Key tangent mode.
		*	Tangent modes are: KFCURVE_TANGEANT_AUTO,
		*					   KFCURVE_TANGEANT_AUTO_BREAK
		*					   KFCURVE_TANGEANT_TCB,
		*					   KFCURVE_TANGEANT_USER,
		* 				   KFCURVE_TANGEANT_BREAK
		*/
		KFCURVE_INLINE void KeySetTangeantMode(kFCurveIndex pKeyIndex, kFCurveTangeantMode pTangent);
			
		/** Set key tengent weight mode as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: This method is meaningful for cubic interpolation only.
        *   \param pKeyIndex   Key index
		*	\param pTangentWeightMode	Weight mode
		*								KFCURVE_WEIGHTED_NONE
		*								KFCURVE_WEIGHTED_RIGHT
		*								KFCURVE_WEIGHTED_NEXT_LEFT
		*								KFCURVE_WEIGHTED_ALL
		*	\param pMask				Used to select the affected tangents
		*								KFCURVE_WEIGHTED_RIGHT
		*								KFCURVE_WEIGHTED_NEXT_LEFT
		*								KFCURVE_WEIGHTED_ALL
		*/

		KFCURVE_INLINE void KeySetTangeantWeightMode(kFCurveIndex pKeyIndex, kFCurveTangeantWeightMode pTangentWeightMode, kFCurveTangeantWeightMode pMask = KFCURVE_WEIGHTED_ALL );

			/** Set key tengent velocity mode as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: This method is meaningful for cubic interpolation only.
		*   \param pKeyIndex   Key index
		*	\param pTangentVelocityMode	Weight mode
		*								KFCURVE_VELOCITY_NONE
		*								KFCURVE_VELOCITY_RIGHT
		*								KFCURVE_VELOCITY_NEXT_LEFT
		*								KFCURVE_VELOCITY_ALL
		*	\param pMask				Used to select the affected tangents
		*								KFCURVE_VELOCITY_RIGHT
		*								KFCURVE_VELOCITY_NEXT_LEFT
		*								KFCURVE_VELOCITY_ALL
		*/

		KFCURVE_INLINE void KeySetTangeantVelocityMode(kFCurveIndex pKeyIndex, kFCurveTangeantVelocityMode pTangentVelocityMode, kFCurveTangeantVelocityMode pMask = KFCURVE_VELOCITY_ALL );

			
		/** Get key data as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: Using this method for other than cubic interpolated 
		*			 key (linear, constant) will return unpredicted values.
		*	Warning: Slope data is inconsistent for automatic tangent mode.
		*			 Use KFCurve::EvaluateLeftDerivative() and 
		*			 KFCurve::EvaluateRightDerivative() to find
		*			 slope values.
		*	Warning: Using this method for TCB tangent mode key will return 
		*			 unpredicted values. Use KFCurve::GetDataFloat() instead.
		*   \param pKeyIndex   Key index
		*	\param pIndex Data index, either	KFCURVEKEY_RIGHT_SLOPE,
		*										KFCURVEKEY_NEXT_LEFT_SLOPE.
		*										KFCURVEKEY_NEXT_RIGHT_WEIGHT.
		*										KFCURVEKEY_NEXT_LEFT_WEIGHT
		*/
		KFCURVE_INLINE kFCurveDouble KeyGetDataDouble(kFCurveIndex pKeyIndex, EKFCurveDataIndex pIndex);
		
		/**	Set data as double value (cubic interpolation, non TCB tangent mode).
		*	Warning: Using this method for other than cubic interpolated 
		*			 key (linear, constant) is irrelevant.
		*	Warning: Slope data is inconsistent for automatic tangent mode.
		*			 Therefore, it is irrelevant to use this method on automatic 
		*			 tangent mode keys.
		*	Warning: Using this method for a TCB tangeant mode key will result
		*			 in unpredicted curve behavior for this key. Use KFCurve::SetDataFloat() 
		*			 instead.
		*   \param pKeyIndex   Key index
		*	\param pIndex Data index, either	KFCURVEKEY_RIGHT_SLOPE,
		*										KFCURVEKEY_NEXT_LEFT_SLOPE.
		*										KFCURVEKEY_NEXT_RIGHT_WEIGHT.
		*										KFCURVEKEY_NEXT_LEFT_WEIGHT
		*	\param pValue	The data value to set (a slope or a weight).
		*/
		KFCURVE_INLINE void KeySetDataDouble(kFCurveIndex pKeyIndex, EKFCurveDataIndex pIndex, kFCurveDouble pValue);
		
		/** Get key data as float value (cubic interpolation, TCB tangent mode).
		*	Warning: Using this method for any key but a cubic interpolated,
		*			 in TCB tangent mode, will return unpredicted values.
		*   \param pKeyIndex   Key index
		*	\param pIndex	Data index, either KFCURVEKEY_TCB_TENSION, KFCURVEKEY_TCB_CONTINUITY or KFCURVEKEY_TCB_BIAS.
		*/	
		KFCURVE_INLINE float KeyGetDataFloat(kFCurveIndex pKeyIndex, EKFCurveDataIndex pIndex);

		/**	Set data as float value (cubic interpolation, TCB tangent mode).
		*	Warning: Using this method for any key but a cubic interpolated,
		*			 in TCB tangent mode, will return unpredicted values.
		*   \param pKeyIndex   Key index
		*	\param pIndex	Data index, either KFCURVEKEY_TCB_TENSION, KFCURVEKEY_TCB_CONTINUITY or KFCURVEKEY_TCB_BIAS.
		*	\param pValue	The data value to set.
		*/
		KFCURVE_INLINE void KeySetDataFloat(kFCurveIndex pKeyIndex, EKFCurveDataIndex pIndex, float pValue);

		/**	Get key data as a pointer
		*	Warning: not supported in 'double' mode.
		*/
		KFCURVE_INLINE const float* KeyGetDataPtr(kFCurveIndex pKeyIndex);

		//!	Get key value.
		KFCURVE_INLINE kFCurveDouble KeyGetValue(kFCurveIndex pKeyIndex);

		//! Set key value.
		KFCURVE_INLINE void KeySetValue(kFCurveIndex pKeyIndex, kFCurveDouble pValue);

		/** Increment key value.
		*   \param pKeyIndex   Key index
		*	\param pValue Value by which key value is incremented.
		*/
		KFCURVE_INLINE void KeyIncValue(kFCurveIndex pKeyIndex, kFCurveDouble pValue);

		/** Multiply key value.
		*   \param pKeyIndex   Key index
		*	\param pValue Value by which the key value is multiplied.
		*/
		KFCURVE_INLINE void KeyMultValue(kFCurveIndex pKeyIndex, kFCurveDouble pValue);

		/** Multiply key tangents.
		*	Note: When multiplying a key value, tangents must be
		*         multiplied to conserve the same topology.
		*   \param pKeyIndex   Key index
		*	\param pValue Value by which key tangents are multiplied.
		*/
		KFCURVE_INLINE void KeyMultTangeant(kFCurveIndex pKeyIndex, kFCurveDouble pValue);

		/** Get key time
		*   \param pKeyIndex   Key index
		*	\return Key time (time at which this key is occuring).
		*/
		KFCURVE_INLINE KTime KeyGetTime(kFCurveIndex pKeyIndex);

		/** Set key time.
		*   \param pKeyIndex   Key index
		*	\param pTime Key time (time at which this key is occuring).
		*/
		KFCURVE_INLINE void KeySetTime(kFCurveIndex pKeyIndex, KTime pTime);

		/** Increment key time.
		*   \param pKeyIndex   Key index
		*	\param pTime Time value by which the key time is incremented.
		*/
		KFCURVE_INLINE void KeyIncTime(kFCurveIndex pKeyIndex, KTime pTime);

		/** Set if key is currently selected.
		*   \param pKeyIndex   Key index
		*	\param pSelected Selection flag.
		*/
		KFCURVE_INLINE void KeySetSelected(kFCurveIndex pKeyIndex, bool pSelected);	

		/** Return if key is currently selected.
		*	\return Selection flag.
		*/
		KFCURVE_INLINE bool KeyGetSelected(kFCurveIndex pKeyIndex);

		/** Set if key is currently marked for manipulation.
		*   \param pKeyIndex   Key index
		*	\param pMark Mark flag.
		*/
		KFCURVE_INLINE void KeySetMarkedForManipulation(kFCurveIndex pKeyIndex, bool pMark);	

		/** Return if key is currently marked for manipulation.
		*	\return Mark flag.
		*/
		KFCURVE_INLINE bool KeyGetMarkedForManipulation(kFCurveIndex pKeyIndex);

		/** Set tangent visibility mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*   \param pKeyIndex   Key index
		*	\param pVisibility	Tangent visibility mode.
		*	Tangent visibility modes are: KFCURVE_TANGEANT_SHOW_NONE
		*						          KFCURVE_TANGEANT_SHOW_LEFT
		*						          KFCURVE_TANGEANT_SHOW_RIGHT
		*/
		KFCURVE_INLINE void	KeySetTangeantVisibility (kFCurveIndex pKeyIndex, kFCurveTangeantVisibility pVisibility);	
		
		/** Return tangent visibility mode.
		*	Warning: This method is meaningful for cubic interpolation only.
		*	\return Tangent visibility mode.
		*	Tangent visibility modes are: KFCURVE_TANGEANT_SHOW_NONE
		*			                      KFCURVE_TANGEANT_SHOW_LEFT
		*			                      KFCURVE_TANGEANT_SHOW_RIGHT
		*/
		KFCURVE_INLINE kFCurveTangeantVisibility KeyGetTangeantVisibility (kFCurveIndex pKeyIndex);

		/** Set/Unset Break tangeant
		* Only valid for User and Auto keys
		*/
		KFCURVE_INLINE void KeySetBreak(kFCurveIndex pKeyIndex, bool pVal); 

		/** Get if tangeant is break
		* Only valid for User and Auto keys
		*/
		KFCURVE_INLINE bool KeyGetBreak(kFCurveIndex pKeyIndex); 

		//@}

		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/
		/************************************************************************************************/

		/**
		* \name Key Tangent Management
		*/
		//@{

		/** Set interpolation type on keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pInterpolation Interpolation type.
		*/
		void KeyTangeantSetInterpolation(bool pSelectedOnly, kFCurveInterpolation pInterpolation);
		
		/** Set tangent mode on keys, all or selected only.
		*	\param pSelectedOnly If set to \c true, only selected keys are affected.
		* Otherwise, all keys are affected.
		*	\param pTangentMode Tangent mode.
		* \remarks Tangent mode is only relevant on keys with a cubic interpolation type.
		*/
		void KeyTangeantSetMode(bool pSelectedOnly, kFCurveTangeantMode pTangentMode);
		
		/** Get the left derivative of a key.
		*	\param pIndex Index of key.
		*	\return Left derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		kFCurveDouble KeyGetLeftDerivative(kFCurveIndex pIndex);
		
		/** Set the left derivative of a key.
		*	\param pIndex Index of key.
		*	\param pValue Left derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER, KFCURVE_TANGEANT_BREAK or KFCURVE_TANGEANT_AUTO. 
		*/
		void KeySetLeftDerivative(kFCurveIndex pIndex, kFCurveDouble pValue);

		/** Get the left auto parametric of a key.
		*	\param pIndex Index of key.
	  *	\param pApplyOvershootProtection Clamp is taking into account.
		*	\return left auto parametric.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
	kFCurveDouble KeyGetLeftAuto(kFCurveIndex pIndex, bool pApplyOvershootProtection = false);

		/** Set the left auto parametric  of a key.
		*	\param pIndex Index of key.
		*	\param pValue Left auto parametric .
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER, KFCURVE_TANGEANT_BREAK or KFCURVE_TANGEANT_AUTO.
		*/
		void KeySetLeftAuto(kFCurveIndex pIndex, kFCurveDouble pValue);	
		
		/** Get the left derivative info of a key.
		*	\param pIndex Index of key.
		*	\return Left derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		KFCurveTangeantInfo KeyGetLeftDerivativeInfo(kFCurveIndex pIndex);
		
		/** Set the left derivative info of a key.
		*	\param pIndex Index of key.
		*	\param pValue Left derivative.
		*   \param pForceDerivative
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/

	void KeySetLeftDerivativeInfo(kFCurveIndex pIndex, KFCurveTangeantInfo pValue, bool pForceDerivative = false);


		/** Increment the left derivative of a key.
		*	\param pIndex Index of key.
		*	\param pInc Increment to left derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void KeyIncLeftDerivative(kFCurveIndex pIndex, kFCurveDouble pInc);
		
		/** Get the right derivative of a key.
		*	\param pIndex Index of key.
		*	\return Right derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		kFCurveDouble KeyGetRightDerivative(kFCurveIndex pIndex);
		
		/** Set the right derivative of a key.
		*	\param pIndex Index of key.
		*	\param pValue Right derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER, KFCURVE_TANGEANT_BREAK or KFCURVE_TANGEANT_AUTO.
		*/
		void KeySetRightDerivative(kFCurveIndex pIndex, kFCurveDouble pValue);

		/** Get the right auto parametric of a key.
		*	\param pIndex Index of key.
	  *	\param pApplyOvershootProtection Clamp is taking into account.
		*	\return Right auto parametric.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
	kFCurveDouble KeyGetRightAuto(kFCurveIndex pIndex, bool pApplyOvershootProtection = false);
		
		/** Set the right auto parametric  of a key.
		*	\param pIndex Index of key.
		*	\param pValue Right auto parametric .
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER, KFCURVE_TANGEANT_BREAK or KFCURVE_TANGEANT_AUTO.
		*/
		void KeySetRightAuto(kFCurveIndex pIndex, kFCurveDouble pValue);
		
		
		/** Get the right derivative info of a key.
		*	\param pIndex Index of key.
		*	\return Right derivative info.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		KFCurveTangeantInfo KeyGetRightDerivativeInfo(kFCurveIndex pIndex);
		
		/** Set the right derivative info of a key.
		*	\param pIndex Index of key.
		*	\param pValue Right derivative.
		*   \param pForceDerivative
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
	void KeySetRightDerivativeInfo(kFCurveIndex pIndex, KFCurveTangeantInfo pValue, bool pForceDerivative = false);


		/** Increment the right derivative of a key.
		*	\param pIndex Index of key.
		*	\param pInc Increment to right derivative.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void KeyIncRightDerivative(kFCurveIndex pIndex, kFCurveDouble pInc);
		
		//! This function is disabled and always return 0.
		kFCurveDouble KeyGetRightBezierTangeant(kFCurveIndex pIndex);
		
		/** Set the left derivative of a key as a Bezier tangent.
		*	\param pIndex Index of key.
		*	\param pValue Left derivative as a Bezier tangent.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void KeySetLeftBezierTangeant(kFCurveIndex pIndex, kFCurveDouble pValue);

		//! This function is disabled and always returns 0.
		kFCurveDouble KeyGetLeftBezierTangeant(kFCurveIndex pIndex);
		
		/** Set the right derivative of a key as a Bezier tangent.
		*	\param pIndex Index of key.
		*	\param pValue Right derivative as a Bezier tangent.
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void KeySetRightBezierTangeant(kFCurveIndex pIndex, kFCurveDouble pValue);


		/** Multiply the Derivative of a key.
		*	\param pIndex Index of key.
		*	\param pMultValue Value that multiply Derivative
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		* This function is only relevant if key interpolation is 
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void KeyMultDerivative(kFCurveIndex pIndex, kFCurveDouble pMultValue);

		/** Get the left tangent weight mode of a key
		*	\param pIndex Index of key.
		*	\return true if the key is weighted
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		bool KeyIsLeftTangeantWeighted(kFCurveIndex pIndex);

		/** Get the right tangent weight mode of a key
		*	\param pIndex Index of key.
		*	\return true if the key is weighted
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		bool KeyIsRightTangeantWeighted(kFCurveIndex pIndex);
		
		/** Set the left tangent weight mode of a key
		*	\param pIndex Index of key.
		*	\param pWeighted Weighted state of the tangent
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void   KeySetLeftTangeantWeightedMode( kFCurveIndex pIndex, bool pWeighted );

		/** Set the right tangent weight mode of a key
		*	\param pIndex Index of key.
		*	\param pWeighted Weighted state of the tangent
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void   KeySetRightTangeantWeightedMode( kFCurveIndex pIndex, bool pWeighted );

		/** Get the weight value component of the left tangent of a key
		*	\param pIndex Index of key.
		*	\return right tangen weight
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC
		*/
		kFCurveDouble KeyGetLeftTangeantWeight(kFCurveIndex pIndex);

		/** Get the weight value component of the right tangent of a key
		*	\param pIndex Index of key.
		*	\return right tangen weight
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC
		*/		
		kFCurveDouble KeyGetRightTangeantWeight(kFCurveIndex pIndex);

		/** Set the left tangent weight of a key
		*	\param pIndex Index of key.
		*	\param pWeight Weight
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK. The tangent is 
		* automatically set in weighted mode.
		*/
		void   KeySetLeftTangeantWeight( kFCurveIndex pIndex, kFCurveDouble pWeight );

		/** Set the right tangent weight of a key
		*	\param pIndex Index of key.
		*	\param pWeight Weight
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK. The tangent is 
		* automatically set in weighted mode.
		*/
		void   KeySetRightTangeantWeight( kFCurveIndex pIndex, kFCurveDouble pWeight );

		/** Get the left tangent velocity mode of a key
		*	\param pIndex Index of key.
		*	\return true if the key has velocity
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		bool KeyIsLeftTangeantVelocity(kFCurveIndex pIndex);

		/** Get the right tangent velocity mode of a key
		*	\param pIndex Index of key.
		*	\return true if the key has velocity
		* \remarks Result is undetermined if function curve has no key or index 
		* is out of bounds.
		*/
		bool KeyIsRightTangeantVelocity(kFCurveIndex pIndex);
		
		/** Set the left tangent velocity mode of a key
		*	\param pIndex Index of key.
		*	\param pVelocity Velocity state of the tangent
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void   KeySetLeftTangeantVelocityMode( kFCurveIndex pIndex, bool pVelocity );

		/** Set the right tangent velocity mode of a key
		*	\param pIndex Index of key.
		*	\param pVelocity Velocity state of the tangent
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK.
		*/
		void   KeySetRightTangeantVelocityMode( kFCurveIndex pIndex, bool pVelocity);

		/** Get the velocity value component of the left tangent of a key
		*	\param pIndex Index of key.
		*	\return right tangen velocity
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC
		*/
		kFCurveDouble KeyGetLeftTangeantVelocity(kFCurveIndex pIndex);

		/** Get the velocity value component of the right tangent of a key
		*	\param pIndex Index of key.
		*	\return right tangen velocity
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC
		*/		
		kFCurveDouble KeyGetRightTangeantVelocity(kFCurveIndex pIndex);

		/** Set the left tangent velocity of a key
		*	\param pIndex Index of key.
		*	\param pVelocity Velocity
		* This function is only relevant if previous key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK. The tangent is 
		* automatically set in velocity mode.
		*/
		void   KeySetLeftTangeantVelocity( kFCurveIndex pIndex, kFCurveDouble pVelocity );

		/** Set the right tangent velocity of a key
		*	\param pIndex Index of key.
		*	\param pVelocity Velocity
		* This function is only relevant if key interpolation
		* type is KFCURVE_INTERPOLATION_CUBIC and tangent mode is
		* KFCURVE_TANGEANT_USER or KFCURVE_TANGEANT_BREAK. The tangent is 
		* automatically set in velocity mode.
		*/
		void   KeySetRightTangeantVelocity( kFCurveIndex pIndex, kFCurveDouble pVelocity );
		
		//@}

		/**
		* \name Extrapolation 
		* Extrapolation defines the function curve value before and after the keys.
		* Pre-extrapolation defines the function curve value before first key.
		* Post-extrapolation defines the function curve value after last key.
		* <ul><li>KFCURVE_EXTRAPOLATION_CONST means a constant value matching the first/last key
		*	    <li>KFCURVE_EXTRAPOLATION_REPETITION means the entire function curve is looped
		*		<li>KFCURVE_EXTRAPOLATION_MIRROR_REPETITION means the entire function curve is looped once backward, once forward and so on 
		*		<li>KFCURVE_EXTRAPOLATION_KEEP_SLOPE means a linear function with a slope matching the first/last key</ul>
		*/
		//@{

		//! Set pre-extrapolation mode.
		KFCURVE_INLINE void SetPreExtrapolation(kFCurveExtrapolationMode pExtrapolation);
			
		//! Get pre-extrapolation mode.
		KFCURVE_INLINE kFCurveExtrapolationMode GetPreExtrapolation();
		
		/** Set pre-extrapolation count.
		*	\param pCount Number of repetitions if pre-extrapolation mode is
		* KFCURVE_EXTRAPOLATION_REPETITION or KFCURVE_EXTRAPOLATION_MIRROR_REPETITION.
		*/
		KFCURVE_INLINE void SetPreExtrapolationCount(kULong pCount);
		
		/** Get pre-extrapolation count.
		*	\return Number of repetitions if pre-extrapolation mode is
		* KFCURVE_EXTRAPOLATION_REPETITION or KFCURVE_EXTRAPOLATION_MIRROR_REPETITION.
		*/
		KFCURVE_INLINE kULong GetPreExtrapolationCount();
		
		//! Set post-extrapolation mode.
		KFCURVE_INLINE void SetPostExtrapolation(kFCurveExtrapolationMode pExtrapolation);
		
		//! Get post-extrapolation mode.
		KFCURVE_INLINE kFCurveExtrapolationMode GetPostExtrapolation();
		
		/** Set post-extrapolation count.
		*	\param pCount Number of repetitions if post-extrapolation mode is
		* KFCURVE_EXTRAPOLATION_REPETITION or KFCURVE_EXTRAPOLATION_MIRROR_REPETITION.
		*/
		KFCURVE_INLINE void SetPostExtrapolationCount(kULong pCount);
			
		/** Get post-extrapolation count.
		*	\return Number of repetitions if post-extrapolation mode is
		* KFCURVE_EXTRAPOLATION_REPETITION or KFCURVE_EXTRAPOLATION_MIRROR_REPETITION.
		*/
		KFCURVE_INLINE kULong GetPostExtrapolationCount();

		/** Get total number of keys taking extrapolation into account.
		* The total number of keys includes repetitions of the function 
		* curve if pre-extrapolation and/or post-extrapolation are of
		* mode KFCURVE_EXTRAPOLATION_REPETITION or KFCURVE_EXTRAPOLATION_MIRROR_REPETITION.
		*	\return Total number of keys taking extrapolation into account.
		*/
		int KeyGetCountAll();
		
		/** Find key index for a given time taking extrapolation into account.
		*	\param pTime Time of the key looked for.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Key index between 0 and KFCurve::KeyGetCount() - 1.The 
		* integer part of the key index gives the index of the closest key 
		* with a smaller time. The decimals give the relative position of 
		* given time compared to previous and next key times. Return -1 if 
		* function curve has no key.
		*/
		double KeyFindAll(KTime pTime, kFCurveIndex* pLast = NULL);

		//@}

		/**
		* \name Evaluation and Analysis
		*/
		//@{
	  	
		/**	Evaluate function curve value at a given time.
		*	\param pTime Time of evaluation.
		* If time falls between two keys, function curve value is 
		* interpolated according to previous key interpolation type and
		* tangent mode if relevant.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Function curve value or default value if function curve
		* has no key.
		* \remarks This function takes extrapolation into account.
		*/
  		kFCurveDouble Evaluate (KTime pTime, kFCurveIndex* pLast = NULL);

		/**	Evaluate function curve value at a given key index.
		*	\param pIndex Any value between 0 and KFCurve::KeyGetCount() - 1.
		* If key index is not an integer value, function curve value is 
		* interpolated according to previous key interpolation type and
		* tangent mode if relevant.
		*	\return Function curve value or default value if function curve
		* has no key.
		* \remarks This function does not take extrapolation into account.
		*/
		kFCurveDouble EvaluateIndex( double pIndex);
		
		/**	Evaluate function left derivative at given time.
		*	\param pTime Time of evaluation.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Left derivative at given time.
		* \remarks This function does not take extrapolation into account.
		*/
  		kFCurveDouble EvaluateLeftDerivative (KTime pTime, kFCurveIndex* pLast = NULL);
		
		/**	Evaluate function right derivative at given time.
		*	\param pTime Time of evaluation.
		* \param pLast Function curve index to speed up search. If this 
		* function is called in a loop, initialize this value to 0 and let it 
		* be updated by each call.
		*	\return Right derivative at given time.
		* \remarks This function does not take extrapolation into account.
		*/
  		kFCurveDouble EvaluateRightDerivative (KTime pTime, kFCurveIndex* pLast = NULL);

		/**	Find the peaks time between 2 keys (a local minimum and/or maximum).
		*	\param pLeftKeyIndex Left key index (there must be a right key).
		*	\param pPeakTime1 First peak time.
		*	\param pPeakTime2 Second peak time.
		*	\return Number of peaks found.
		* \remarks Result is undetermined if function curve has no key or 
		* index is out of bounds.
		*/
		int FindPeaks(kFCurveIndex pLeftKeyIndex, KTime& pPeakTime1, KTime& pPeakTime2);

		/**	Find the peaks value between 2 keys (a local minimum and/or maximum).
		*	\param pLeftKeyIndex Left key index (there must be a right key).
		*	\param pPeak1 First peak value.
		*	\param pPeak2 Second peak value.
		*	\return Number of peaks found.
		* \remarks Result is undetermined if function curve has no key or 
		* index is out of bounds.
		*/
		int FindPeaks(kFCurveIndex pLeftKeyIndex, kFCurveDouble& pPeak1, kFCurveDouble& pPeak2);

		/**	Find the peaks time and value between 2 keys (a local minimum and/or maximum).
		*	\param pLeftKeyIndex Left key index (there must be a right key).
		*	\param pPeakTime1 First peak time.
		*	\param pPeak1 First peak value.
		*	\param pPeakTime2 Second peak time.
		*	\param pPeak2 Second peak value.
		*	\return Number of peaks found.
		* \remarks Result is undetermined if function curve has no key or 
		* index is out of bounds.
		*/
		int FindPeaks(kFCurveIndex pLeftKeyIndex, KTime& pPeakTime1, kFCurveDouble& pPeak1, KTime& pPeakTime2, kFCurveDouble& pPeak2);

		/** Get key period statistics. If pAveragePeriod == pMinPeriod, we have iso-sampled data.
		*	\param pAveragePeriod Average key period.
		*	\param pMinPeriod Minimum period found.
		*	\param pMaxPeriod Maximum period found.
		*/
		void KeyGetPeriods(KTime& pAveragePeriod, KTime& pMinPeriod, KTime& pMaxPeriod);
		
		//@}

		/**
		* \name Copy, Insert, Replace and Delete Functions
		*/
		//@{

		/** Create a new function curve and copy keys found between a given time range.
		* Time range is inclusive.
		*	\param pStart Start of time range.
		*	\param pStop End of time range.
		*	\return Created function curve.
		* \remarks 
		*/
		HKFCurve Copy(KTime pStart = KTIME_MINUS_INFINITE, KTime pStop = KTIME_INFINITE);

		/** Copy a function curve content into current function curve.
		*	\param pSource Source function curve.
		*	\param pWithKeys If \c true, clear keys in current function curve and copy
		* keys from source function curve. If \c false, keys in current function curve
		* are left as is.
		*/
		void CopyFrom(KFCurve& pSource, bool pWithKeys = true);

		/**	Replace keys within a range in current function curve with keys found in a source function curve.
		* \param pSource Source function curve.
		* \param	pStart Start of time range.
		* \param	pStop End of time range.
		* \param pUseExactGivenSpan false = original behavior where time of first and last key was used
		* \param pKeyStartEndOnNoKey Inserts a key at the beginning and at the end of the range if there is no key to insert.
		* \param pTimeSpanOffset
		*/
	void Replace(HKFCurve pSource, KTime pStart = KTIME_MINUS_INFINITE, KTime pStop = KTIME_INFINITE, bool pUseExactGivenSpan = false, bool pKeyStartEndOnNoKey = true, KTime pTimeSpanOffset = KTIME_ZERO );

		/**	Replace keys within a range in current function curve with keys found in a source function curve.
		* The copied keys have their value scaled with a factor varying 
		* linearly in time within the given time range.
		* \param pSource Source function curve.
		* \param pStart Start of time range.
		* \param pStop End of time range.
		* \param pScaleStart Scale factor applied at start of time range. 
		* \param pScaleStop Scale factor applied at end of time range. 
		* \param pUseExactGivenSpan false = original behavior where time of first and last key was used
		* \param pKeyStartEndOnNoKey Inserts a key at the beginning and at the end of the range if there is no key to insert.
		* \param pTimeSpanOffset
		*/
	void ReplaceForQuaternion(HKFCurve pSource, KTime pStart, KTime pStop, kFCurveDouble pScaleStart, kFCurveDouble pScaleStop, bool pUseExactGivenSpan = false, bool pKeyStartEndOnNoKey = true, KTime pTimeSpanOffset = KTIME_ZERO );

		/**	Replace keys within a range in current function curve with keys found in a source function curve.
		* \param pSource Source function curve.
		* \param pStart Start of time range.
		* \param pStop End of time range.
		* \param pAddFromStart Offset applied to copied key values within the time range.
		* \param pAddAfterStop Offset applied to key values after the time range.
		* \param pValueSubOffsetAfterStart If \c true, copied key values within 
		* the time range are substracted from time offset specified by parameter
		* \c pAddFromStart. If \c false, copied key values within the time range are 
		* added to time offset specified by parameter \c pAddFromStart. 
		* \param pValueSubOffsetAfterStop If \c true, key values after 
		* the time range are substracted from time offset specified by parameter
		* \c pAddAfterStop. If \c false, key values after the time range are 
		* added to time offset specified by parameter \c pAddAfterStop. 
		* \param pUseExactGivenSpan false = original behavior where time of first and last key was used
		* \param pKeyStartEndOnNoKey Inserts a key at the beginning and at the end of the range if there is no key to insert
		* \param pTimeSpanOffset
		*/
	void ReplaceForEulerXYZ(HKFCurve pSource, KTime pStart, KTime pStop, kFCurveDouble pAddFromStart, kFCurveDouble pAddAfterStop, bool pValueSubOffsetAfterStart, bool pValueSubOffsetAfterStop, bool pUseExactGivenSpan = false, bool pKeyStartEndOnNoKey = true, KTime pTimeSpanOffset = KTIME_ZERO );	

		/**	Insert all keys found in a source function curve in current function curve.
		* A time offset is added to copied keys so that the first copied key occurs 
		* at the given insertion time. Keys from the source function curve are merged into 
		* the current function curve. In other words, no existing key in the current function
		* curve is destroyed unless there is an overlap with a copied key.
		* \param pSource Source function curve.
		* \param pInsertTime Insert time of the first key found in the source function curve.
		* \param pFirstKeyLeftDerivative First key left derivative.
		* \param pFirstKeyIsWeighted  First key left weighted state (true if weighted).
		* \param pFirstKeyWeight First key left weight
		*/

		void Insert(HKFCurve pSource, KTime pInsertTime, kFCurveDouble pFirstKeyLeftDerivative, bool pFirstKeyIsWeighted = false, kFCurveDouble pFirstKeyWeight = KFCURVE_DEFAULT_WEIGHT);

		/**	Insert all keys found in a source function curve in current function curve.
		* A time offset is added to copied keys so that the first copied key occurs 
		* at the given insertion time. Keys from the source function curve are merged into 
		* the current function curve. In other words, no existing key in the current function
		* curve is destroyed unless there is an overlap with a copied key.
		* \param pSource Source function curve.
		* \param pInsertTime Insert time of the first key found in the source function curve.
		* \param pFirstKeyLeftDerivative First key left derivative info.
		*/
		void Insert(HKFCurve pSource, KTime pInsertTime, KFCurveTangeantInfo pFirstKeyLeftDerivative );

		/** Delete keys within an index range.
		* Index range is inclusive.
		* This function is much faster than multiple removes.
		*	\param pStartIndex Index of first deleted key.
		*	\param pStopIndex Index of last deleted key.
		*	\return \c true if the function curve contains keys, \c false otherwise.
		* \remarks Result is undetermined if function curve has keys but an 
		* index is out of bounds.
		*/
		bool Delete(kFCurveIndex pStartIndex , kFCurveIndex pStopIndex);									
		
		/** Delete keys within a time range.
		* Time range is inclusive.
		* This function is much faster than multiple removes.
		*	\param pStart Start of time range.
		*	\param pStop End of time range.
		*	\return \c true if the function curve contains keys, \c false otherwise.
		*/	
		bool Delete (KTime pStart = KTIME_MINUS_INFINITE, KTime pStop = KTIME_INFINITE);

		/** Get if interpolation is cubic and that the tangents and weightings are untouched.
		*	\param	pKeyIndex	Index of the key to test.
		*	\return				Returns true if the interpolation is a pure cubic auto.
		*/
		bool IsKeyInterpolationPureCubicAuto(kFCurveIndex pKeyIndex);

	#ifndef K_PLUGIN
		/** Extract All Keys in the Given Selection Span
		*	\param	pArray	    Array where to Stored Found Keys.
		*	\param	pMinIndex	Index where to start the Search.
		*	\param	pMaxIndex	Index where to stop the Search (the last index is the limit, the Key at this index is not tested).
		*	\param	pMinValue	Minimal Value to Consider the Key.
		*	\param	pMaxValue	Maximal Value to Consider the Key.
		*/
		void ExtractKeysIndex( KArraykInt &pArray, int pMinIndex, int pMaxIndex, double pMinValue =  -K_DOUBLE_MAX, double pMaxValue =  K_DOUBLE_MAX);
	#endif

		//@}

	///////////////////////////////////////////////////////////////////////////////
	//
	//  WARNING!
	//
	//	Anything beyond these lines may not be documented accurately and is 
	// 	subject to change without notice.
	//
	///////////////////////////////////////////////////////////////////////////////

	#ifndef DOXYGEN_SHOULD_SKIP_THIS

		bool	FbxStore (KFbx* pFbx, bool pOnlyDefaults = false, bool pColor = true, bool pIsVersion5 = false );
		bool	FbxRetrieve (KFbx* pFbx, bool pOnlyDefaults = false, bool pColor = false );
		bool	FbxInternalRetrieve (KFbx* pFbx, bool pOnlyDefaults = false, bool pColor = false );

		double CandidateEvaluate (KTime pTime, kFCurveIndex* pLast = NULL);
		bool CandidateClear ();
		bool CandidateSet (KTime pTime, double pValue);
		bool IsCandidate ();
		double CandidateGet ();
		KTime CandidateGetTime ();
		
		bool CandidateKey
		(
			kFCurveIndex	*pLast				= NULL, 
			int	pInterpolation = KFCURVE_INTERPOLATION_CUBIC, 
			int	pTanMode = KFCURVE_TANGEANT_USER, 
		int pContinuity = KFCURVE_CONTINUITY,
			bool			pTangeantOverride	= true,
			KTime			pCandidateTime		= KTIME_INFINITE,
			double			pKeyIndexTolerance  = 0.0
		);

		bool NormalsSeemsToComeFromAPlot();

  		void SetWasData (int pType);
  		int GetWasData ();
		int GuessWasData (KTime* pStart = NULL, KTime* pStep = NULL);

		void KeyTangeantHide ();

		int GetUpdateId ();
		int GetValuesUpdateId ();

		void CallbackRegister (kFCurveCallback pCallback, void* pObject);
		void CallbackUnregister (kFCurveCallback pCallback, void* pObject);
		void CallbackEnable (bool pEnable);
		void CallbackClear ();

	private:
		void IncrementUpdateId(int pInc);
		void CallbackAddEvent (int pWhat, int pIndexStart);

		int MapIndexAll (int pIndex, int &pWhere);
		void InitBuffers (int pKeyCount);

		bool CheckCurve();
	void IsClamped( int pIndex, bool &pLeftClamped, bool &pRightClamped ); // Return true if the specified key is an auto clamp that is currently clamped

		float mColor[3];

		kFCurveDouble mValue;

		int mUpdateId;
		bool mCallbackEnable;
		bool mInternalCallbackEnable; // Internal use, to replace many callback by one
		int mKeyModifyGuard;

		KFCurveKey** mFCurveKeysList;

		int mFCurveKeyCount;	
		int mFCurveKeySize;	
		int mFCurveLastBlockIndex;	


		kUInt mPreExtrapolation;
		kULong mPreExtrapolationCount;
		kUInt mPostExtrapolation;
		kULong mPostExtrapolationCount;

		int mWasType;

		kFCurveIndex mLastSearchIndex;

		KTime mCandidateTime;
		kFCurveDouble mCandidateValue;

		KFCurveEvent mEvent;
		KArrayUL mCallbackFunctions;   // no delete on object must use array ul
		KArrayUL mCallbackObjects;	   // no delete on object must use array ul

		// FBObjectHolder for FBSDK reference
		#ifndef K_PLUGIN
			KFBObjectHolder mFBObjectHolder;
			KFCURVE_INLINE KFBObjectHolder& GetFBHolder ();
		#endif

		KFCURVE_INLINE KFCurveKey* InternalKeyGetPtr(kFCurveIndex pIndex);

	#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

	};


	#ifndef K_PLUGIN
		#include <kfcurve/kfcurveinline.h>
	#endif


	/**	Create a function curve.
	*/
	KFCURVE_DLL HKFCurve KFCurveCreate();

	// Create a function curve, FBX SDK internal use only.
	KFCURVE_DLL HKFCurve KFCurveCreate(KFbx* pFbx, bool pOnlyDefaults = false, bool pColor = false);

	// Create a function curve, FBX SDK internal use only.
	KFCURVE_DLL HKFCurve KFCurveCreate(KFbx* pFbx, HKFCurve pCurve, bool pOnlyDefaults = false, bool pColor = false);

#include <kfcurve/kfcurve_nsend.h>


#endif // #ifndef _FBXSDK_KFCURVE_H_


