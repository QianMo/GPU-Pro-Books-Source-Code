/*!  \file kfbxnurbscurve.h
 */

#ifndef _FBXSDK_NURBS_CURVE_H_
#define _FBXSDK_NURBS_CURVE_H_
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

#include <kfbxplugins/kfbxgeometry.h>

#include <fbxfilesdk_nsbegin.h>

/**
    A Non-Uniform Rational B-Spline (Nurbs) curve is a type of parametric geometry. A Nurbs
    curve is defined by the degree, form, knot vector and control points. 

	Let M be the degree of the curve.
	Let N be the number of control points of the curve.

	The form of the curve can be open, closed or periodic. A curve with end points
	that do not meet is defined as an open curve. The number of knots in an open curve
	is defined as N+(M+1). 
	
	A closed curve simply has its last control point equal to its first control point. 
	Note that this does not imply tangent continuity at the end point.  The curve may 
	have a kink at this point.  In FBX the last control point is not specified by the user
	in the InitControlPoints() method. For example, if there are to be 10 control points in
	total, and the curve is to be closed, than only 9 control points need to be passed 
	into the InitControlPoints() method. The last control point is implied to be equal
	to the first control point. Thus N represents the number of unique CVs. 

	A periodic curve has its last M control points equal to its first M control points. 
	A periodic curve is tangent continuous at the ends. Similiar to a closed curve,
	when creating a periodic curve, only the unique control points need to be set. For
	example a periodic curve of degree 3 with 10 control points requires only 7 CVs to 
	be specified in the InitControlPoints() method. The last 3 CVs, which are the same as
	the first 3, are not included. 

	The calculation of the number of knots in closed and periodic curves is more complex. 
	Since we have excluded one CV in N in a closed curve, the number of knots is N+(M+1)+1. 
	Similiarly, we excluded M CVs in periodic curves so the number of knots is N+(M+1)+M. 

	Note that FBX stores one extra knot at the beginning and and end of the knot vector,
	compared to some other graphics applications such as Maya. The two knots are not 
	used in calculation, but they are included so that no data is lost when converting
	from file formats that do store the extra knots.

  * \nosubgrouping
  */
class KFBX_DLL KFbxNurbsCurve : public KFbxGeometry 
{
	KFBXOBJECT_DECLARE(KFbxNurbsCurve,KFbxGeometry);
public:
	// inhierited from KFbxNodeAttribute
	virtual KFbxNodeAttribute::EAttributeType GetAttributeType() const;

	/** \enum EDimension  The dimension of the CVs
	  * - \e e2D The CVs are two dimensional points
	  * - \e e3D The CVs are three dimensional points
	  */
	enum EDimension
	{
		e2D = 2,
		e3D,
		eDIMENSIONS_COUNT = 2
	};

	/** \enum EType The form of the curve
	  * - \e eOPEN
	  * - \e eCLOSED
	  * - \e ePERIODIC
	  */
	enum EType
	{
		eOPEN,
		eCLOSED,
		ePERIODIC,
		eTYPE_COUNT
	}; 

	/** Allocate memory space for the array of control points as well as the knot 
	  * vector.
      * \param pCount Number of control points.
      * \param pVType Nurb type in V direction.
	  * \remarks This function should always be called after KFbxNurb::SetOrder(). 
      */
	void InitControlPoints( int pCount, EType pVType );

	/** Get knot vector.
	  * \return Pointer to the array of knots.
	  */
	inline double* GetKnotVector() const { return mKnotVector; }

	/** Get the number of elements in the knot vector.
	  * \return The number of knots. See KFbxNurbsCurve description for more details. 
	  */
	int GetKnotCount() const;

	// Sets the order of the curve
	// Must be set before InitControlPoints() is called. 
	inline void SetOrder( int pOrder ) { mOrder = pOrder; }

	/** Get nurb curve order.
	  * \return Order value.
	  */
	inline int GetOrder() const { return mOrder; }

	/** Sets the dimension of the CVs
	  * For 3D curves: control point = ( x, y, z, w ), where w is the weight
	  * For 2D curves: control point = ( x, y, 0, w ), where the z component is unused, and w is the weight. 
	  * \param pDimension - the dimension of the control points. (3d or 2d)
	  */
	inline void SetDimension( EDimension pDimension ) { mDimension = pDimension; }

	/** Gets the dimension of the control points.
	  * \return The dimension of the curve
	  */
	inline EDimension GetDimension() const { return mDimension; }

	/** Determines if the curve is rational or not
	  * \return True if the curve is rational, false otherwise
	  */
	bool IsRational(); 

	/** Calculates the number of spans in the curve using the following:
	  * Where
	  * S = Number of spans
	  * N = Number of CVs
	  * M = Order of the curve
	  *
	  * S = N + M + 1;
	  *
	  * In this calculation N includes the duplicate CVs for closed and periodic curves. 
	  * 
	  * \return The number of spans if the curve has been initialized, -1 otherwise.
	  */
	int GetSpanCount();

	/** Get nurb type.
	  * \return Nurb curve type identifier.
	  */
	inline EType GetType() const { return mNurbType; }

	/** Checks if the curve is a poly line. (A polyline is a 
	  * linear nurbs curve )
	  *
	  * \return \c true if curve is a poly line, false otherwise.
	  */
	inline bool IsPolyline() const { return ( GetOrder() == 2 ); }

	/** Bezier curves are a special case of nurbs curve. This function
	  * determines if this nurbs curve is a Bezier curve.
	  *
	  * \return \c true if curve is a Bezier curve, false otherwise.
	  */
	bool IsBezier();

	// step? 
	// Need to know multiplicity?

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

public:
	// Clone
	virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType = eDEEP_CLONE) const;

	bool FullMultiplicity();

protected:

    // Error identifiers, these are only used internally.
	typedef enum 
	{
		eNurbCurveTypeUnknown,
		eWeightTooSmall,
		eKnotVectorError,
        eWrongNumberOfControlPoint,

		/*
        
        eUMultiplicityVectorError,
        eVMultiplicityVectorError,
        ,
        eVKnotVectorError, */
		eErrorCount
	} EError;

	
	KFbxNurbsCurve(KFbxSdkManager& pManager, char const* pName);
	virtual ~KFbxNurbsCurve();
	
	//! Assignment operator.
    KFbxNurbsCurve& operator=(KFbxNurbsCurve const& pNurb);

	void Reset();

	virtual KString GetTypeName() const;
	virtual KStringList GetTypeFlags() const;

	virtual void Destruct(bool pRecursive, bool pDependents);

private:
	bool mIsRational;
	double* mKnotVector;
	EType mNurbType;
	int mOrder;
	EDimension mDimension; 

public:
	friend class KFbxReaderFbx;
	friend class KFbxReaderFbx6;
	friend class KFbxWriterFbx;
	friend class KFbxWriterFbx6;
	friend class KFbxReaderFbx7;
	friend struct KFbxReaderFbx7Impl;
};

typedef KFbxNurbsCurve* HKFbxNurbsCurve;
typedef class KFBX_DLL KArrayTemplate< KFbxNurbsCurve* > KArrayKFbxNurbsCurve;


#include <fbxfilesdk_nsend.h>

#endif //_FBXSDK_NURBS_CURVE_H_


