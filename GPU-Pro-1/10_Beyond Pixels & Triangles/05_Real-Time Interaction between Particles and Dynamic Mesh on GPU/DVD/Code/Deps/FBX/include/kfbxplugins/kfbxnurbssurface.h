/*!  \file kfbxnurbssurface.h
 */

#ifndef _FBXSDK_NURBS_SURFACE_H_
#define _FBXSDK_NURBS_SURFACE_H_

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

#include <kfbxplugins/kfbxnode.h>
#include <kfbxplugins/kfbxgeometry.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxPatch;
class KFbxSdkManager;

/** A Nurbs surface is a type of parametric geometry. A Nurbs surface is defined by the
    degree, form, knot vector and control points in the U and V directions.

    For more information on the meaning of the form, knot vector and control points,
    see the documentation for the KFbxNurbsCurve. The same concepts for Nurbs curves
    apply to Nurbs surfaces. Nurbs surfaces simply have two dimensions (U and V).

  * \nosubgrouping
  */
class KFBX_DLL KFbxNurbsSurface : public KFbxGeometry
{
    KFBXOBJECT_DECLARE(KFbxNurbsSurface,KFbxGeometry);
public:
    //! Return the type of node attribute which is EAttributeType::eNURBS_SURFACE.
    virtual EAttributeType GetAttributeType() const { return KFbxNodeAttribute::eNURBS_SURFACE; }

    //! Reset the nurb to default values.
    void Reset();

    /**
      * \name Nurb Properties
      */
    //@{

    /** Set surface mode.
      * \param pMode Surface mode identifier (see class KfbxGeometry)
      */
    void SetSurfaceMode(KFbxGeometry::ESurfaceMode pMode);

    /** Get surface mode.
      * \return Currently set surface mode identifier.
      */
    inline ESurfaceMode GetSurfaceMode() const {return mSurfaceMode;}

    /** \enum ENurbType Nurb types.
      * - \e ePERIODIC
      * - \e eCLOSED
      * - \e eOPEN
      */
    typedef enum
    {
        ePERIODIC,
        eCLOSED,
        eOPEN
    } ENurbType;

    /** Allocate memory space for the array of control points as well as the knot
      * and multiplicity vectors.
      * \param pUCount Number of control points in U direction.
      * \param pUType Nurb type in U direction.
      * \param pVCount Number of control points in V direction.
      * \param pVType Nurb type in V direction.
      * \remarks This function should always be called after KFbxNurb::SetOrder().
      */
    void InitControlPoints(int pUCount, ENurbType pUType, int pVCount, ENurbType pVType);

    /** Get number of control points in U direction.
      * \return Number of control points in U.
      */
    inline int GetUCount() const {return mUCount;}

    /** Get number of control points in V direction.
      * \return Number of control points in V.
      */
    inline int GetVCount() const {return mVCount;}

    /** Get nurb type in U direction.
      * \return Nurb type identifier.
      */
    inline ENurbType GetNurbUType() const {return mUType;}

    /** Get nurb type in V direction.
      * \return Nurb type identifier.
      */
    inline ENurbType GetNurbVType() const {return mVType;}

    /** Get the number of elements in the knot vector in U direction. See KFbxNurbsCurve for more information.
      * \return The number of control points in U direction.
      */
    int GetUKnotCount() const;

    /** Get knot vector in U direction.
      * \return Pointer to the array of knots.
      */
    double* GetUKnotVector() const;

    /** Get the number of elements in the knot vector in V direction. See KFbxNurbsCurve for more information.
      * \returns The number of control points in V direction. Nurb order in V
      */
    int GetVKnotCount() const;

    /** Get knot vector in V direction.
      * \return Pointer to the array of knots.
      */
    double* GetVKnotVector() const;

    /** Get multiplicity control points in U direction.
      * \return Pointer to the array of multiplicity values.
      * \remarks The length of this vector is equal to U count and
      * its elements are set to 1 by default.
      */
    //int* GetUMultiplicityVector();

    /** Get multiplicity control points in V.
      * \return Pointer to the array of multiplicity values.
      * \remarks The length of this vector is equal to V count and
      * its elements are set to 1 by default.
      */
    //int* GetVMultiplicityVector();

    /** Set order.
      * \param pUOrder Nurb order in U direction.
      * \param pVOrder Nurb order in V direction.
      */
    void SetOrder(kUInt pUOrder, kUInt pVOrder);

    /** Get nurb order in U direction.
      * \return U order value.
      */
    inline int GetUOrder() const {return mUOrder;}

    /** Get nurb order in V direction.
      * \return V order value.
      */
    inline int GetVOrder() const {return mVOrder;}

    /** Set step.
      * The step is the number of divisions between adjacent control points.
      * \param pUStep Steps in U direction.
      * \param pVStep Steps in V direction.
      */
    void SetStep(int pUStep, int pVStep);

    /** Get the number of divisions between adjacent control points in U direction.
      * \return Step value in U direction.
      */
    inline int GetUStep() const {return mUStep;}

    /** Get the number of divisions between adjacent control points in V direction.
      * \return Step value in V direction.
      */
    inline int GetVStep() const {return mVStep;}

    /* Calculates the number of spans in the surface in the U direction.
     * See KFbxNurbsCurve::GetSpanCount() for more information.
     * \returns The number of spans in U if the surface has been initialized, -1 otherwise.
     */
    int GetUSpanCount() const;

    /* Calculates the number of spans in the surface in the V direction.
     * See KFbxNurbsCurve::GetSpanCount() for more information.
     * \returns The number of spans in V if the surface has been initialized, -1 otherwise.
     */
    int GetVSpanCount() const;

    //@}

    /**
      * \name Nurb Export Flags
      */
    //@{

    /** Set the flag inducing UV flipping at export time.
      * \param pFlag If \c true UV flipping will occur.
      */
    void SetApplyFlipUV(bool pFlag);

    /** Get the flag inducing UV flipping at export time.
      * \return Current state of the UV flip flag.
      */
    bool GetApplyFlipUV() const;

    /** Set the flag inducing link flipping at export time.
      * \param pFlag If \c true the links control points indices will be flipped.
      */
    void SetApplyFlipLinks(bool pFlag);

    /** Get the flag inducing link flipping at export time.
      * \return Current state of the link flip flag.
      */
    bool GetApplyFlipLinks() const;

    /** Get flip flags state.
      * \return \c true if we need to flip either the UV or the links.
      */
    bool GetApplyFlip() const { return GetApplyFlipUV() || GetApplyFlipLinks(); }

    /** Add curve on surface
      * Adds a 2d, parameter space curve to this surface
      * \param pCurve The curve to add to the surface
      */
    void AddCurveOnSurface( KFbxNode* pCurve );

    /* Retrieves a curve on this surface
     * \param pIndex Index of the curve to retrieve. Valid range is 0 to GetCurveOnSurfaceCount() - 1
     * \return The curve at the specified index, or NULL if pIndex is out of range.
     */
    KFbxNode* GetCurveOnSurface( int pIndex );
    KFbxNode const* GetCurveOnSurface( int pIndex ) const;

    /* \return The number of curves on this surface
     */
    int GetCurveOnSurfaceCount() const;

    /* Removes a curve from this surface.
     * \param pCurve The curve to remove
     * \return True if the curve was removed, false otherwise.
     */
    bool RemoveCurveOnSurface( KFbxNode* pCurve );

    //@}

    /** Check if the surface has all rational control points.
      * \return true if rational, false otherwise
      */
    bool IsRational();


///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//  Anything beyond these lines may not be documented accurately and is
//  subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////


#ifndef DOXYGEN_SHOULD_SKIP_THIS

public:

    void SetFlipNormals(bool pFlipNormals);

    bool GetFlipNormals() const;


    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType = eDEEP_CLONE) const;

protected:


    KFbxNurbsSurface(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxNurbsSurface();

    virtual void Destruct(bool pRecursive, bool pDependents);

    //! Assignment operator.
    KFbxNurbsSurface& operator=(KFbxNurbsSurface const& pNurb);

    virtual KString     GetTypeName() const;
    virtual KStringList GetTypeFlags() const;

    kUInt mUOrder, mVOrder;
    int mUCount, mVCount;
    int mUStep, mVStep;
    ENurbType mUType, mVType;

    double* mUKnotVector;
    double* mVKnotVector;

    //int* mUMultiplicityVector;
    //int* mVMultiplicityVector;

    ESurfaceMode mSurfaceMode;

    // Export flags.
    bool mApplyFlipUV;
    bool mApplyFlipLinks;

    bool mFlipNormals;

    // Error identifiers, these are only used internally.
    typedef enum
    {
        eNurbTypeUnknown,
        eWrongNumberOfControlPoint,
        eWeightTooSmall,
        //eUMultiplicityVectorError,
        //eVMultiplicityVectorError,
        eUKnotVectorError,
        eVKnotVectorError,
        eErrorCount
    } EError;

    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;
    friend struct KFbxReaderFbx7Impl;
    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;
    friend struct KFbxWriterFbx7Impl;
    friend class KFbxGeometryConverter;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxNurbsSurface* HKFbxNurbsSurface;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_NURBS_SURFACE_H_


