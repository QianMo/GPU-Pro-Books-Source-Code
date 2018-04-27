/*!  \file kfbxgeometry.h
 */

#ifndef _FBXSDK_GEOMETRY_H_
#define _FBXSDK_GEOMETRY_H_

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

#include <kfbxplugins/kfbxgeometrybase.h>
#include <kfbxplugins/kfbxdeformer.h>

#include <klib/karrayul.h>
#include <klib/kerror.h>

#include <kfcurve/kfcurve_forward.h>
#ifndef MB_FBXSDK
#include <kfcurve/kfcurve_nsuse.h>
#endif

#include <kfbxmath/kfbxxmatrix.h>
#include <fbxfilesdk_nsbegin.h>

class KFbxGeometryWeightedMap;
class KFbxNode;
class KFbxShape;
class KFbxTexture;
class KFbxSdkManager;
class KFbxVector4;
class KFbxTakeNode;
class KFbxScene;
class KFbxCluster;
typedef class KFbxCluster KFbxLink;

/** Contains common properties for mesh, nurb, and patch node attributes.
  * \nosubgrouping
  * A geometry node attribute has arrays of links, shapes, materials and
  * textures. It also has arrays for control points, normals, material indices,
  * texture indices, and texture UV coordinates. Some of these are only used
  * in mesh node attributes.
  */
class KFBX_DLL KFbxGeometry : public KFbxGeometryBase
{
    KFBXOBJECT_DECLARE(KFbxGeometry,KFbxGeometryBase);

public:
    /** Return the type of node attribute.
      * This class is pure virtual.
      */
    virtual EAttributeType GetAttributeType() const;

    /**
      * \name Deformer Management
      */
    //@{

    /** Add a deformer.
      * \param pDeformer     Pointer to the deformer object to add.
      * \return              Index of added deformer.
      */
    int AddDeformer(KFbxDeformer* pDeformer);

    /** Get the number of deformers.
      * \return     Number of deformers that have been added to this object.
      */
    int GetDeformerCount() const;

    /** Get deformer at given index.
      * \param pIndex     Index of deformer.
      * \return           Pointer to deformer or \c NULL if pIndex is out of range. In this case,
      *                   KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    KFbxDeformer* GetDeformer(int pIndex) const;

    /** Get the number of deformers of a given type.
      * \param pType     Type of deformer to count
      * \return          Number of deformers that have been added to this object.
      */
    int GetDeformerCount(KFbxDeformer::EDeformerType pType) const;

    /** Get deformer of a gieven type at given index.
      * \param pIndex     Index of deformer.
      * \param pType      Type of deformer.
      * \return           Pointer to deformer or \c NULL if pIndex is out of range. In this case,
      *                   KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    KFbxDeformer* GetDeformer(int pIndex, KFbxDeformer::EDeformerType pType) const;

    //@}

    /**
      * \name Connected Geometry Weighted Map(s) Management
      */
    //@{

    /** Return the source geometry weighted map connected.
      * \return     Pointer to the source geometry weighted map connected to this object if any.
      */
    KFbxGeometryWeightedMap* GetSourceGeometryWeightedMap();

    /** Get the number of destination geometry weighted map(s) connected.
      * \return     Number of destination geometry weighted map(s) connected to this object.
      */
    int GetDestinationGeometryWeightedMapCount();

    /** Get destination geometry weighted map at a given index.
      * \param pIndex     Index of link.
      * \return           Pointer to the destination geometry weighted map connected to this object if any.
      */
    KFbxGeometryWeightedMap* GetDestinationGeometryWeightedMap(int pIndex);

    //@}

    /**
      * \name Shape Management
      */
    //@{

    /** Add a shape and its associated name.
      * \param pShape         Pointer to the shape object.
      * \param pShapeName     Name given to the shape.
      * \return               Index of added shape, -1 if operation failed.
      *                       If the operation fails, KFbxGeometry::GetLastErrorID() can return one of the following:
      *                            - eNULL_PARAMETER: Pointer to shape is \c NULL.
      *                            - eSHAPE_ALREADY_ADDED: Shape has already been added.
      *                            - eSHAPE_INVALID_NAME: The provided name is empty.
      *                            - eSHAPE_NAME_CLASH: The provided name is already used by another shape.
      * \remarks             The provided name is stripped from surrounding whitespaces before being
      *                      compared with other shape names. It is recommended not to prefix the shape name with its
      *                      enclosing node name because MotionBuilder is known to strip this prefix and not save it back.
      */
    virtual int AddShape(KFbxShape* pShape, char const* pShapeName);

    /** Removes all shapes without destroying them.
      * If shapes aren't explicitly destroyed before calling this function, they will be
      * destroyed along with the SDK manager.
      */
    virtual void ClearShape();

    /** Get the number of shapes.
      * \return     Number of shapes that have been added to this object.
      */
    virtual int GetShapeCount() const;

    /** Get shape at given index.
      * \param pIndex     Index of shape.
      * \return           Pointer to shape or \c NULL if pIndex is out of range. In this case,
      *                   KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    virtual KFbxShape* GetShape(int pIndex);

    /** Get shape at given index.
      * \param pIndex     Index of shape.
      * \return           Pointer to shape or \c NULL if pIndex is out of range. In this case,
      *                   KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    virtual KFbxShape const* GetShape(int pIndex) const;

    /** Get shape name at given index.
      * \param pIndex     Index of shape.
      * \return           Shape name or \c NULL if pIndex is out of range. In this case,
      *                   KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    virtual char const* GetShapeName(int pIndex) const;

    /** Get a shape channel.
      * The shape channel property has a scale from 0 to 100, 100 meaning full shape deformation.
      * The default value is 0.
      * \param pShapeName      Shape Property name.
      * \param pCreateAsNeeded If true, the fcurve is created if not already present.
      * \param pTakeName       Take from which we want the FCurve (if NULL, use the current take).
      * \return                Animation curve or NULL if an error occurred. In this case,
      *                        KFbxGeometry::GetLastErrorID() returns one of the following:
      *                             - eINDEX_OUT_OF_RANGE: Shape index is out of range.
      *                             - eSHAPE_NO_CURVE_FOUND: Shape curve could not be found.
      */
    virtual KFCurve* GetShapeChannel(char const* pShapeName, bool pCreateAsNeeded = false, char const* pTakeName = NULL);

    /** Get a shape channel.
      * The shape channel property has a scale from 0 to 100, 100 meaning full shape deformation.
      * The default value is 0.
      * \param pIndex          Shape index.
      * \param pCreateAsNeeded If true, the fcurve is created if not already present.
      * \param pTakeName       Take from which we want the FCurve (if NULL, use the current take).
      * \return                Animation curve or NULL if an error occurred. In this case,
      *                        KFbxGeometry::GetLastErrorID() returns one of the following:
      *                             - eINDEX_OUT_OF_RANGE: Shape index is out of range.
      *                             - eSHAPE_NO_CURVE_FOUND: Shape curve could not be found.
      */
    virtual KFCurve* GetShapeChannel(int pIndex, bool pCreateAsNeeded = false, char const* pTakeName = NULL);

    //@}

    /** Surface modes
      * This information is only used in nurbs and patches.
      */

    /** \enum ESurfaceMode Types of surfaces.
      * - \e eRAW
      * - \e eLOW_NO_NORMALS
      * - \e eLOW
      * - \e eHIGH_NO_NORMALS
      * - \e eHIGH
      */
    typedef enum
    {
        eRAW,
        eLOW_NO_NORMALS,
        eLOW,
        eHIGH_NO_NORMALS,
        eHIGH
    } ESurfaceMode;

    /**
      * \name Pivot Management
      * The geometry pivot is used to specify additional translation, rotation,
      * and scaling applied to all the control points when the model is
      * exported.
      */
    //@{

    /** Get pivot matrix.
      * \param pXMatrix     Placeholder for the returned matrix.
      * \return             Reference to the passed argument.
      */
    KFbxXMatrix& GetPivot(KFbxXMatrix& pXMatrix) const;

    /** Set pivot matrix.
      * \param pXMatrix     The Transformation matrix.
      */
    void SetPivot(KFbxXMatrix& pXMatrix);

    /** Apply the pivot matrix to all vertices/normals of the geometry.
      */
    void ApplyPivot();

    //@}

    /**
      * \name Default Animation Values
      * These functions provides direct access to default
      * animation values specific to a geometry.
      * These functions only work if the geometry has been associated
      * with a node.
      */
    //@{

    /** Set default deformation for a given shape.
      * The default shape property has a scale from 0 to 100, 100 meaning full shape deformation.
      * The default value is 0.
      * \param pIndex       Shape index.
      * \param pPercent     Deformation percentage on a scale ranging from 0 to 100.
      * \remarks            This function has no effect if pIndex is out of range.
      */
    void SetDefaultShape(int pIndex, double pPercent);
    /** Set default deformation for a given shape.
      * The default shape property has a scale from 0 to 100, 100 meaning full shape deformation.
      * The default value is 0.
      * \param pShapeName   Shape name.
      * \param pPercent     Deformation percentage on a scale ranging from 0 to 100.
      * \remarks            This function has no effect if pShapeName is invalid.
      */
    void SetDefaultShape(char const* pShapeName, double pPercent);

    /** Get default deformation for a given shape.
      * The default shape property has a scale from 0 to 100, 100 meaning full shape deformation.
      * The default value is 0.
      * \param pIndex     Shape index.
      * \return           The deformation value for the given shape, or 0 if pIndex is out of range.
      */
    double GetDefaultShape(int pIndex);
    /** Get default deformation for a given shape.
      * The default shape property has a scale from 0 to 100, 100 meaning full shape deformation.
      * The default value is 0.
      * \param pShapeName     Shape name.
      * \return               The deformation value for the given shape, or 0 if pShapeName is invalid.
      */
    double GetDefaultShape(char const* pShapeName);

    //@}

    /**
      * \name Error Management
      */
    //@{

    /** Retrieve error object.
     *  \return Reference to error object.
     */
    KError& GetError ();

    /** \enum EError Error identifiers.
      * - \e eINDEX_OUT_OF_RANGE
      * - \e eNULL_PARAMETER
      * - \e eMATERIAL_NOT_FOUND
      * - \e eMATERIAL_ALREADY_ADDED
      * - \e eTEXTURE_NOT_FOUND
      * - \e eTEXTURE_ALREADY_ADDED
      * - \e eSHAPE_ALREADY_ADDED
      * - \e eSHAPE_INVALID_NAME
      * - \e eSHAPE_NAME_CLASH
      * - \e eSHAPE_NO_CURVE_FOUND
      * - \e eUNKNOWN_ERROR
      */
    typedef enum
    {
        eINDEX_OUT_OF_RANGE,
        eNULL_PARAMETER,
        eMATERIAL_NOT_FOUND,
        eMATERIAL_ALREADY_ADDED,
        eTEXTURE_NOT_FOUND,
        eTEXTURE_ALREADY_ADDED,
        eSHAPE_ALREADY_ADDED,
        eSHAPE_INVALID_NAME,
        eSHAPE_NAME_CLASH,
        eSHAPE_NO_CURVE_FOUND,
        eUNKNOWN_ERROR,
        eERROR_COUNT
    } EError;

    /** Get last error code.
     *  \return     Last error code.
     */
    EError GetLastErrorID() const;

    /** Get last error string.
     *  \return     Textual description of the last error.
     */
    const char* GetLastErrorString() const;

    //@}

    virtual KFbxObject* Clone(KFbxObject* pContainer, KFbxObject::ECloneType pCloneType) const;

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
    KFbxGeometry(KFbxSdkManager& pManager, char const* pName);
    virtual ~KFbxGeometry();

    //! Assignment operator.
    KFbxGeometry& operator=(KFbxGeometry const& pGeometry);

    virtual void Destruct(bool pRecursive, bool pDependents);

    virtual void SetDocument(KFbxDocument* pDocument);

    /** Overloaded function called in KFbxNode::SetNodeAttribute().
    *   \param pNode Associated node.
    *   \remarks Do not call this function anywhere else.
    */
    virtual void SetNode(KFbxNode& pNode);

    /** Overloaded function called in KFbxNode::SetNodeAttribute().
    *   \remarks Do not call this function anywhere else.
    */
    virtual void UnsetNode();

    /** Add channels specific to a geometry attribute.
    *   \param pTakeNodeName Take node to add specialized channels to.
    */
    bool AddShapeChannel(KString pTakeNodeName, int pShapeIndex);

    /** Remove channels specific to a geometry attribute.
    *   \param pTakeNodeName Take node to remove specialized channels from.
    */
    bool RemoveShapeChannel(KString pTakeNodeName, int pShapeIndex);

    // MotionBuilder 4.01 and earlier versions saved nurb and patch shape channel names
    // following the template "Shape 0x (Shape)" where x is the index of the shape starting
    // at 1. Since then, Jori modified shape channels to turn them into animated properties.
    // As a result, nurb and patch shape channel names are now saved following the template
    // "<shape name> (Shape)". The FBX SDK keeps the old shape channel naming scheme but has
    // been modifed to handle the new one and convert shape channel names to the old shape
    // channel naming scheme.
    void CleanShapeChannels(KString pTakeNodeName);
    void CleanShapeChannel(KString pTakeNodeName, int pShapeIndex);

    // Shape channel name creation for nurb and patch.
    KString CreateShapeChannelName(int pShapeIndex);

    // Shape channel name creation for mesh.
    KString CreateShapeChannelName(KString pShapeName);

    void CreateShapeChannelProperties(KString& pShapeName);

    void ConvertShapeNamesToV5Format(KString pTakeNodeName);
    void ConvertShapeNamesToV5Format(KString pTakeNodeName, int pShapeIndex);
    void RevertShapeNamesToV6Format(KString pTakeNodeName);
    void RevertShapeNamesToV6Format(KString pTakeNodeName, int pShapeIndex);
    void ClearTemporaryShapeNames();

    /** Remove a deformer.
      * \param pIndex Index of deformer to remove.
      * \return Pointer to removed deformer if success, \c NULL otherwise.
      * In the last case, KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    KFbxDeformer* RemoveDeformer(int pIndex);

    /** Remove a shape.
      * \param pIndex Index of shape to remove.
      * \return Pointer to removed shape if success, \c NULL otherwise.
      * In the last case, KFbxGeometry::GetLastErrorID() returns eINDEX_OUT_OF_RANGE.
      */
    // This function implies renaming shape channels. The decision has been made to avoid changing
    // KFCurveNode interface to allow renaming.
    // KFbxShape* RemoveShape(int pIndex);

    void CopyDeformers(KFbxGeometry const* pGeometry);
    void CopyShapes(KFbxGeometry const* pGeometry);

    void CopyPivot(KFbxGeometry const* pSource);

    KArrayTemplate <KFbxShape*> mShapeArray;

    KArrayTemplate <KString*> mShapeNameArray;

    // Used during FBX v5 file store
    KArrayTemplate<KString*> mShapeNameArrayV6;
    KArrayTemplate<KString*> mShapeNameArrayV5;
    KArrayTemplate<KString*> mShapeChannelNameArrayV5;

    KFbxXMatrix* mPivot;


    mutable KError mError;

    friend class KFbxScene;
    friend class KFbxWriterFbx;
    friend class KFbxWriterFbx6;
    friend class KFbxReaderFbx;
    friend class KFbxReaderFbx6;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_GEOMETRY_H_


