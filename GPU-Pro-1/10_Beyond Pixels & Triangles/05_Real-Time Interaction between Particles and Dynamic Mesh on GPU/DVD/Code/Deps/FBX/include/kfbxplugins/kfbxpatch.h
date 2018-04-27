/*!  \file kfbxpatch.h
 */

#ifndef _FBXSDK_PATCH_H_
#define _FBXSDK_PATCH_H_

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

class KFbxSdkManager;

/** \brief A patch is a type of parametric geometry node attribute.
  * \nosubgrouping
  */
class KFBX_DLL KFbxPatch : public KFbxGeometry
{
    KFBXOBJECT_DECLARE(KFbxPatch,KFbxGeometry);
public:
    //! Return the type of node attribute which is EAttributeType::ePATCH.
    virtual EAttributeType GetAttributeType() const;

    //! Reset the patch to default values.
    void Reset();

    /**
      * \name Patch Properties
      */
    //@{

    /** Set surface mode.
      * \param pMode     Surface mode identifier (see Class KFbxGeometry)
      */
    void SetSurfaceMode(KFbxGeometry::ESurfaceMode pMode);

    /** Get surface mode.
      * \return     Currently set surface mode identifier.
      */
    inline KFbxGeometry::ESurfaceMode GetSurfaceMode() const {return mSurfaceMode;}

    /** \enum EPatchType Patch types.
      * - \e eBEZIER
      * - \e eBEZIER_QUADRIC
      * - \e eCARDINAL
      * - \e eBSPLINE
      * - \e eLINEAR
      */
    typedef enum
    {
        eBEZIER         = 0,
        eBEZIER_QUADRIC = 1,
        eCARDINAL       = 2,
        eBSPLINE        = 3,
        eLINEAR         = 4
    } EPatchType;

    /** Allocate memory space for the array of control points.
      * \param pUCount     Number of control points in U direction.
      * \param pUType      Patch type in U direction.
      * \param pVCount     Number of control points in V direction.
      * \param pVType      Patch type in V direction.
      */
    void InitControlPoints(int pUCount, EPatchType pUType, int pVCount, EPatchType pVType);

    /** Get number of control points in U direction.
      * \return     Number of control points in U.
      */
    inline int GetUCount() const {return mUCount;}

    /** Get number of control points in V direction.
      * \return     Number of control points in V.
      */
    inline int GetVCount() const {return mVCount;}

    /** Get patch type in U direction.
      * \return     Patch type identifier.
      */
    inline EPatchType GetPatchUType() const {return mUType;}

    /** Get patch type in V direction.
      * \return     Patch type identifier.
      */
    inline EPatchType GetPatchVType () const {return mVType;}

    /** Set step.
      * The step is the number of divisions between adjacent control points.
      * \param pUStep     Steps in U direction.
      * \param pVStep     Steps in V direction.
      */
    void SetStep(int pUStep, int pVStep);

    /** Get the number of divisions between adjacent control points in U direction.
      * \return     Step value in U direction.
      */
    inline int GetUStep() const {return mUStep;}

    /** Get the number of divisions between adjacent control points in V direction.
      * \return     Step value in V direction.
      */
    inline int GetVStep() const {return mVStep;}

    /** Set closed flags.
      * \param pU     Set to \c true if the patch is closed in U direction.
      * \param pV     Set to \c true if the patch is closed in V direction.
      */
    void SetClosed(bool pU, bool pV);

    /** Get state of the U closed flag.
      * \return     \c true if the patch is closed in U direction.
      */
    inline bool GetUClosed() const {return mUClosed;}

    /** Get state of the V closed flag.
      * \return     \c true if the patch is closed in V direction.
      */
    inline bool GetVClosed() const {return mVClosed;}

    /** Set U capped flags.
      * \param pUBottom     Set to \c true if the patch is capped at the bottom in the U direction.
      * \param pUTop \c     Set to \c true if the patch is capped at the top in the U direction.
      * \remarks            Capping options are saved but not loaded by Motionbuilder because they
      *                     are computed from the patch topography.
      */
    void SetUCapped(bool pUBottom, bool pUTop);

    /** Get U capped bottom flag state.
      * \return     \c true if the patch is capped at the bottom.
      */
    inline bool GetUCappedBottom() const {return mUCappedBottom;}

    /** Get U capped top flag state.
      * \return     \c true if the patch is capped at the top.
      */
    inline bool GetUCappedTop() const {return mUCappedTop;}

    /** Set V capped flags.
      * \param pVBottom     Set to \c true if the patch is capped at the bottom in the V direction.
      * \param pVTop        Set to \c true if the patch is capped at the top in the V direction.
      * \remarks            Capping options are saved but not loaded by Motionbuilder because they
      *                     are computed from the patch topography.
      */
    void SetVCapped(bool pVBottom, bool pVTop);

    /** Get V capped bottom flag state.
      * \return     \c true if the patch is capped at the bottom.
      */
    inline bool GetVCappedBottom() const {return mVCappedBottom;}

    /** Get V capped top flag state.
      * \return     \c true if the patch is capped at the top.
      */
    inline bool GetVCappedTop() const {return mVCappedTop;}

    //@}

    /**
      * \name Off-loading Serialization section
      */
    //@{
        virtual bool ContentWriteTo(KFbxStream& pStream) const;
        virtual bool ContentReadFrom(const KFbxStream& pStream);
    //@}

#ifdef _DEBUG
        virtual int MemoryUsage() const;
#endif

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

    // Clone
    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

protected:

    KFbxPatch(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxPatch();

    virtual void Destruct(bool pRecursive, bool pDependents);

    //! Assignment operator.
    KFbxPatch& operator=(KFbxPatch const& pPatch);

    virtual KString     GetTypeName() const;
    virtual KStringList GetTypeFlags() const;

    EPatchType mUType, mVType;
    int mUCount, mVCount;
    int mUStep, mVStep;
    bool mUClosed, mVClosed;
    bool mUCappedBottom, mUCappedTop;
    bool mVCappedBottom, mVCappedTop;

    KFbxGeometry::ESurfaceMode mSurfaceMode;

    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;
    friend struct KFbxReaderFbx7Impl;
    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;
    friend struct KFbxWriterFbx7Impl;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

typedef KFbxPatch* HKFbxPatch;

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_PATCH_H_


