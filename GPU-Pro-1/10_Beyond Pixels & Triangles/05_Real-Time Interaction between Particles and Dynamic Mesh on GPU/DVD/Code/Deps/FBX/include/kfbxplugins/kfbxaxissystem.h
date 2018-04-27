/*!  \file kfbxaxissystem.h
 */

#ifndef _KFbxAxisSystem_h
#define _KFbxAxisSystem_h

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

#include <klib/karrayul.h>
#include <klib/kstring.h>

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif

#include <kfcurve/kfcurve_forward.h>
#ifndef MB_FBXSDK
#include <kfcurve/kfcurve_nsuse.h>
#endif

#include <kfbxmath/kfbxmatrix.h>

#include <fbxfilesdk_nsbegin.h>

class KFbxNode;
class KFbxScene;

/** \brief This class represents the coordinate system of the scene, and can convert scenes from
    its coordinate system to other coordinate systems.
  * \nosubgrouping
  */
class KFBX_DLL KFbxAxisSystem
{
public:

	/** \enum eUpVector Specifies which canonical axis represents up in the system. Typically Y or Z. 
	  * - \e XAxis
	  * - \e YAxis
	  * - \e ZAxis
	  */
    enum eUpVector {
        XAxis =	1,
	    YAxis =	2,
	    ZAxis =	3        
    };
    
    /* \enum eFrontVector. Vector with origin at the screen pointing toward the camera.
     * This is a subset of enum eUpVector because axis cannot be repeated.
     * - \e ParityEven
     * - \e ParityOdd
     */
    enum eFrontVector {
	    ParityEven = 1,
	    ParityOdd  = 2
    };

	/* \enum eCoorSystem.
	* - \e RightHanded
	* - \e LeftHanded
	*/
    enum eCoorSystem {
        RightHanded = 0,
        LeftHanded  = 1
    };

	/* \enum ePreDefinedAxisSystem.
	* - \e eMayaZUp
	* - \e eMayaYUp
	* - \e eMax
	* - \e eMotionBuilder
	* - \e eOpenGL
	* - \e eDirectX
	* - \e eLightwave
	*/
	enum ePreDefinedAxisSystem {
		eMayaZUp = 0,
		eMayaYUp,
		eMax,
		eMotionBuilder,
		eOpenGL,
		eDirectX,
		eLightwave
	};

    KFbxAxisSystem(eUpVector pUpVector, eFrontVector pFrontVector, eCoorSystem pCoorSystem);
    KFbxAxisSystem(const KFbxAxisSystem& pAxisSystem);
	KFbxAxisSystem(const ePreDefinedAxisSystem pAxisSystem);
    virtual ~KFbxAxisSystem();

    bool operator==(const KFbxAxisSystem& pAxisSystem)const;
    bool operator!=(const KFbxAxisSystem& pAxisSystem)const;
	KFbxAxisSystem& operator=(const KFbxAxisSystem& pAxisSystem);

    static const KFbxAxisSystem MayaZUp;
    static const KFbxAxisSystem MayaYUp;
    static const KFbxAxisSystem Max;
    static const KFbxAxisSystem Motionbuilder;
    static const KFbxAxisSystem OpenGL;
    static const KFbxAxisSystem DirectX;
    static const KFbxAxisSystem Lightwave;

    /** Convert a scene to this axis system. Sets the axis system of the
	  * scene to this system unit. 
	  * \param pScene     The scene to convert
	  */
    void ConvertScene(KFbxScene* pScene) const;

	/** Convert a scene to this axis system by using the specified
	  * node as an Fbx_Root. This is provided for backwards compatibility
	  * only and ConvertScene(KFbxScene* pScene) should be used instead
	  * when possible.
	  * \param pScene       The scene to convert
	  * \param pFbxRoot     The Fbx_Root node that will be transformed.
	  */
	void ConvertScene(KFbxScene* pScene, KFbxNode* pFbxRoot) const;

	/** Returns the eUpVector this axis system and get the sign of the axis.
	  * \param pSign     The sign of the axis, 1 if up, -1 is down.
	  */
	eUpVector GetUpVector( int & pSign ) const;

	/** Returns the eCoorSystem this axis system.
	  */
	eCoorSystem GetCoorSystem() const;

	/** Converts the children of the given node to this axis system.
	  * Unlike the ConvertScene() method, this method does not set the axis system 
	  * of the scene that the pRoot node belongs, nor does it adjust KFbxPoses
	  * as they are not stored under the scene, and not under a particular node.
	  */
	void ConvertChildren(KFbxNode* pRoot, const KFbxAxisSystem& pSrcSystem) const;

///////////////////////////////////////////////////////////////////////////////
//
//  WARNING!
//
//	Anything beyond these lines may not be documented accurately and is 
// 	subject to change without notice.
//
///////////////////////////////////////////////////////////////////////////////

protected:

    class KFbxAxis
    {
    public:
        enum eAxis {
            eXAxis = 0, eYAxis, eZAxis
        };

        eAxis mAxis;
        int   mSign;

        bool operator==(const KFbxAxis& pAxis)const
        {
            return mAxis == pAxis.mAxis && mSign == pAxis.mSign;
        }
    };

    KFbxAxis mUpVector;
    KFbxAxis mFrontVector;
    KFbxAxis mCoorSystem;

protected:
    /////////////////////////////////////////////////////////////////////////////////
    // Conversion engine, here you will find all the step involed into the conversion

    // Convert Translation and Rotation FCurve.
	void ConvertTProperty(KArrayTemplate<KFbxNode*>& pNodes, const KFbxAxisSystem& pFrom) const;
    void ConvertFCurve(KArrayTemplate<KFCurve*>& pFCurves, const KFbxAxisSystem& pFrom)const;

    // Adjust Pre rotation to orient the node and childs correctly.
    void AdjustPreRotation(KFbxNode* pNode, const KFbxMatrix& pConversionRM)const;

    void AdjustPivots(KFbxNode* pNode, const KFbxMatrix& pConversionRM)const;

	void GetConversionMatrix(const KFbxAxisSystem& pFrom, KFbxMatrix& pConversionRM)const;

	void AdjustLimits(KFbxNode* pNode, const KFbxMatrix& pConversionRM)const;

	void AdjustPoses(KFbxScene* pScene, const KFbxMatrix& pConversionRM) const;

	void AdjustCamera(KFbxNode* pNode, const KFbxMatrix& pConversionRM ) const;

	void AdjustCluster(KFbxNode* pNode, const KFbxMatrix& pConversionRM) const;

	void ConvertChildren(KFbxNode* pRoot, const KFbxAxisSystem& pSrcSystem, bool pSubChildrenOnly) const;

    friend class KFbxReaderFbx;
	friend class KFbxReaderFbx6;
	friend class KFbxWriterFbx;
	friend class KFbxWriterFbx6;

    friend class KFbxGlobalSettings;
};

#include <fbxfilesdk_nsend.h>


#endif // _KFbxAxisSystem_h
