/*!  \file kfbxgeometryconverter.h
 */

#ifndef _FBXSDK_GEOMETRY_CONVERTER_H_
#define _FBXSDK_GEOMETRY_CONVERTER_H_

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

#ifndef MB_FBXSDK
#include <kbaselib_nsuse.h>
#endif


#include <fbxfilesdk_nsbegin.h>

class KFbxNode;
class KFbxLayerContainer;
class KFbxGeometryBase;
class KFbxGeometry;
class KFbxMesh;
class KFbxPatch;
class KFbxNurb;
class KFbxCluster;
class KFbxSdkManager;
class KFBXSurfaceEvaluator;
class KFbxWeightedMapping;
class KFbxVector4;
class KFbxNurbsSurface;
class KFbxNurbsCurve;

/** 
  * \brief This class provides functions to triangulate and convert geometry node attributes.
  * \nosubgrouping
  */
class KFBX_DLL KFbxGeometryConverter
{
public:
	KFbxGeometryConverter(KFbxSdkManager* pManager);
	~KFbxGeometryConverter();

	/** 
	  * \name Triangulation
	  */
	//@{

	/** Triangulate a mesh.
	  * \param pMesh     Pointer to the mesh to triangulate.
	  * \return          Pointer to the new triangulated mesh.
	  * \remarks         This method creates a new mesh, leaving the source mesh unchanged.
	  */
	KFbxMesh* TriangulateMesh(KFbxMesh const* pMesh);

	/** Triangulate a patch.
	  * \param pPatch     Pointer to the patch to triangulate.
	  * \return           Pointer to the new triangulated mesh.
	  * \remarks          The links and shapes are also converted to fit the created mesh.
	  */
	KFbxMesh* TriangulatePatch(KFbxPatch const* pPatch);

	/** Triangulate a nurb.
	  * \param pNurb     Pointer to the nurb to triangulate.
	  * \return          Pointer to the new triangulated mesh.
	  * \remarks         The links and shapes are also converted to fit the created mesh.
	  */
	KFbxMesh* TriangulateNurb(KFbxNurb const* pNurb);

	/** Triangulate a mesh, patch or nurb contained in a node in order to preserve 
	  * related animation channels.
	  * \param pNode     Pointer to the node containng the geometry to triangulate.
	  * \return          \c true on success, or \c false if the node attribute is not a mesh, a patch or a nurb.
	  * \remarks         See the remarks for functions TriangulateMesh(), TriangulatePatch() and TriangulateNurb().
	  */
	bool TriangulateInPlace(KFbxNode* pNode);

	/** Add an "alternate" geometry to the node.
	  * \param pNode                        Pointer to the node containing the geometry.
	  * \param pSrcGeom                     Pointer to the source geometry.
	  * \param pAltGeom                     Pointer to the alternate geometry.
	  * \param pSrcToAltWeightedMapping     Pointer to the weighted mapping table (optional).
	  * \param pConvertDeformations         Flag used only if parameter pSrcToAltWeightedMapping is a valid pointer to a weighted mapping table.
	  *                                     Set to \c true to convert deformations using the weighted mapping table.
	  * \return                             \c true on success, or \c false if the node attribute is not a mesh, a patch or a nurb.
	  */
	bool AddAlternateGeometry(
		KFbxNode*			 pNode, 
		KFbxGeometry*		 pSrcGeom, 
		KFbxGeometry*		 pAltGeom,
		KFbxWeightedMapping* pSrcToAltWeightedMapping,
		bool				 pConvertDeformations
	);

	/** Convert shape(s) and link(s) from souce to destination geometry.
	  * \param pNode        Pointer to the node containng the geometry.
	  * \param pSrcGeom     Pointer to the source geometry.
	  * \param pDstGeom     Pointer to the destination geometry.
	  * \return             \c true on success, \c false otherwise.
	  * \remarks            Source and destination geometry must belong to the same node and must be linked by a geometry weighted map.
	  */
	bool ConvertGeometryAnimation(
		KFbxNode*			 pNode, 
		KFbxGeometry*		 pSrcGeom, 
		KFbxGeometry*		 pDstGeom
	);

	/** Compute a "vertex-correspondance" table that helps passing from source to destination geometry.
	  * \param pSrcGeom                     Pointer to the source geometry.
	  * \param pDstGeom                     Pointer to the destination geometry.
	  * \param pSrcToDstWeightedMapping     Pointer to the weighted mapping table.
	  * \param pSwapUV                      Set to \c true to swap UVs.
	  * \return                             \c true on success, \c false if the function fails to compute the correspondance.
	  * \remarks                            Links and shapes are also converted to fit the alternate geometry.
	  */
	bool ComputeGeometryControlPointsWeightedMapping(
		KFbxGeometry*		 pSrcGeom, 
		KFbxGeometry*		 pDstGeom, 
		KFbxWeightedMapping* pSrcToDstWeightedMapping,
		bool				 pSwapUV = false
	);

	//@}

	/** 
	  * \name Geometry Conversion
	  */
	//@{

	/** Convert from patch to nurb.
	  * \param pPatch     Pointer to the patch to convert.
	  * \return           Created nurb or \c NULL if the conversion fails.
	  * \remarks          The patch must be of type eBSPLINE, eBEZIER or eLINEAR.
	  */
	KFbxNurb* ConvertPatchToNurb(KFbxPatch *pPatch);

	/** Convert a patch contained in a node to a nurb. Use this function to preserve the patch's related animation channels.
	  * \param pNode     Pointer to the node containing the patch.
	  * \return          \c true on success, \c false if the node attribute is not a patch.
	  * \remarks         The patch must be of type eBSPLINE, eBEZIER or eLINEAR.
	  */
	bool ConvertPatchToNurbInPlace(KFbxNode* pNode);

	/** Convert a patch to nurb surface.
	  * \param pPatch     Pointer to the patch to convert.
	  * \return           Created nurb surface or \c NULL if conversion fails.
	  * \remarks          The patch must be of type eBSPLINE, eBEZIER or eLINEAR.
	  */
	KFbxNurbsSurface* ConvertPatchToNurbsSurface(KFbxPatch *pPatch);

	/** Convert a patch contained in a node to a nurb surface. Use this function to preserve the patch's related animation channels.
	  * \param pNode     Pointer to the node containing the patch.
	  * \return          \c true on success, \c false if the node attribute is not a patch.
	  * \remarks         The patch must be of type eBSPLINE, eBEZIER or eLINEAR.
	  */
	bool ConvertPatchToNurbsSurfaceInPlace(KFbxNode* pNode);

	/** Convert a KFbxNurb to a KFbxNurbsSurface
	  * \param pNurb     Pointer to the original nurb
	  * \return          A KFbxNurbsSurface that is equivalent to the original nurb.
	  */
	KFbxNurbsSurface* ConvertNurbToNurbsSurface( KFbxNurb* pNurb );

	/** Convert a KFbxNurbsSurface to a KFbxNurb
	  * \param pNurb     Pointer to the original nurbs surface
	  * \return          A KFbxNurb that is equivalent to the original nurbs surface.
	  */
	KFbxNurb* ConvertNurbsSurfaceToNurb( KFbxNurbsSurface* pNurb );

	/** Convert a nurb, contained in a node, to a nurbs surface. Use this function to preserve the nurb's related animation channels.
	  * \param pNode     Pointer to the node containing the nurb.
	  * \return          \c true on success, \c false otherwise
	  */
	bool ConvertNurbToNurbsSurfaceInPlace(KFbxNode* pNode);

	/** Convert a nurb contained in a node to a nurbs surface. Use this function to preserve the nurb's related animation channels.
	  * \param pNode     Pointer to the node containing the nurbs surface.
	  * \return          \c true on success, \c false otherwise
	  */
	bool ConvertNurbsSurfaceToNurbInPlace(KFbxNode* pNode);

	//@}

	/** 
	  * \name Nurb UV and Links Swapping
	  */
	//@{

	/** Flip UV and/or links of a nurb.
	  * \param pNurb             Pointer to the Source nurb.
	  * \param pSwapUV           Set to \c true to swap the UVs.
	  * \param pSwapClusters     Set to \c true to swap the control point indices of clusters.
	  * \return                  A fliped kFbxNurb, or \c NULL if the function fails.
	  */
	KFbxNurb* FlipNurb(KFbxNurb* pNurb, bool pSwapUV, bool pSwapClusters);

	/** Flip UV and/or links of a nurb surface.
	  * \param pNurb             Pointer to the Source nurb surface.
	  * \param pSwapUV           Set to \c true to swap the UVs.
	  * \param pSwapClusters     Set to \c true to swap the control point indices of clusters.
	  * \return                  A fliped kFbxNurbSurface, or \c NULL if the function fails.
	  */
	KFbxNurbsSurface* FlipNurbsSurface(KFbxNurbsSurface* pNurb, bool pSwapUV, bool pSwapClusters);

	//@}

	/** 
	  * \name Normals By Polygon Vertex Emulation
	  */
	//@{

	/** Emulate normals by polygon vertex mode for a mesh.
	  * \param pMesh     Pointer to the mesh object.
	  * \return          \c true on success, \c false if the number of normals in the 
	  *                  mesh and in its associated shapes don't match the number of polygon
	  *                  vertices.
	  * \remarks         Since the FBX file format currently only supports normals by
	  *                  control points, this function duplicates control points to equal the 
	  *                  number of polygon vertices. Links and shapes are also converted.
	  *                  As preconditions:
	  *                       -# polygons must have been created
	  *                       -# the number of normals in the mesh and in its associated shapes must match the 
	  *                          number of polygon vertices.
	  */
	bool EmulateNormalsByPolygonVertex(KFbxMesh* pMesh);

	/** Create edge smoothing information from polygon-vertex mapped normals.
	  * Existing smoothing information is removed and edge data is created if
	  * none exists on the mesh.
	  * \param pMesh     The mesh used to generate edge smoothing.
	  * \return          \c true on success, \c false otherwise.
	  * \remarks         The edge smoothing data is placed on Layer 0 of the mesh.
	  *                  Normals do not need to be on Layer 0, since the first layer with
	  *                  per polygon vertex normals is used.
	  */
	bool ComputeEdgeSmoothingFromNormals( KFbxMesh* pMesh ) const;

    /** Convert edge smoothing to polygon smoothing group.
	  * Existing smoothing information is replaced.
	  * 
	  * \param pMesh     The mesh that contains the smoothing to be converted.
      * \param pIndex    The index of the layer smoothing to be converted.
	  * \return          \c true on success, \c false otherwise.
	  * \remarks         The smoothing group is bitwise.  The each bit of the integer represents
      *                  one smoothing group.  Therefore, there is 32 smoothing groups maximum.
	  */
	bool ComputePolygonSmoothingFromEdgeSmoothing( KFbxMesh* pMesh, int pIndex=0 ) const;
    
    /** Convert polygon smoothing group to edge smoothing.
    * Existing smoothing information is replaced.
    * 
    * \param pMesh     The mesh that contains the smoothing to be converted.
    * \param pIndex    The index of the layer smoothing to be converted
    * \return          \c true on success, \c false otherwise.
    */
    bool ComputeEdgeSmoothingFromPolygonSmoothing( KFbxMesh* pMesh, int pIndex=0 ) const;

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

public:

	/** Add a "triangulated mesh" geometry to the node.
	  * \param pNode Pointer to the node containng the geometry.
	  * \return \c true on success, \c false if the node attribute is not a mesh, 
	  * a patch or a nurb.
	  * \remarks The remarks relative to functions TriangulateMesh(), TriangulatePatch()
	  * , TriangulateNurb() and TriangulateInPlace() are applicable.
	  */
	bool AddTriangulatedMeshGeometry(KFbxNode* pNode, int pUVStepCoeff);
	
protected:

	bool ComputePatchToMeshControlPointsWeightedMapping
	(
		KFbxPatch*			 pSrcPatch, 
		KFbxMesh*			 pDstMesh, 
		KFbxWeightedMapping* pMapping,
		bool				 pSwapUV = false
	);
	bool ComputeNurbToMeshControlPointsWeightedMapping
	(
		KFbxNurbsSurface*	 pSrcNurb, 
		KFbxMesh*			 pDstMesh, 
		KFbxWeightedMapping* pMapping,
		bool				 pRescaleUVs = false,
		bool				 pSwapUV = false
	);

	void InitializeWeightInControlPoints(KFbxGeometryBase* pGeometry);
	void InitializeWeightInNormals(KFbxLayerContainer* pLayerContainer);
	void TriangulateContinuousSurface
	( 
		KFbxMesh* pMesh, 
		KFBXSurfaceEvaluator* pSurface, 
		kUInt pPointCountX, 
		kUInt pPointCountY, 
		bool ClockWise = false
	);
	void CheckForZeroWeightInShape(KFbxGeometry *pGeometry);
	KFbxMesh* CreateMeshFromParametricSurface(KFbxGeometry const* pGeometry);
	KFbxNurb* CreateNurbFromPatch(KFbxPatch* pPatch);
	KFbxNurbsSurface* CreateNurbsSurfaceFromPatch(KFbxPatch* pPatch);

	void ConvertShapes(KFbxGeometry const* pSource, 
					   KFbxGeometry* pDestination, 
					   KFBXSurfaceEvaluator* pEvaluator, 
					   int pUCount, 
					   int pVCount);
	void ConvertShapes(KFbxGeometry const* pSource,
						KFbxGeometry* pDestination, 
						KFbxWeightedMapping* pSourceToDestinationMapping);
	void ConvertClusters(KFbxGeometry const* pSource, 
					  KFbxGeometry* pDestination, 
					  KFbxWeightedMapping* pSourceToDestinationMapping);
	void ConvertClusters(KArrayTemplate<KFbxCluster*> const& pSourceClusters, 
					  int pSourceControlPointsCount,
					  KArrayTemplate<KFbxCluster*>& pDestinationClusters, 
					  int pDestinationControlPointsCount,
					  KFbxWeightedMapping* pSourceToDestinationMapping);
	void BuildClusterToSourceMapping(KArrayTemplate<KFbxCluster*> const& pSourceClusters, 
								  KFbxWeightedMapping* pClusterToSourceMapping);
	void CheckClusterToSourceMapping(KFbxWeightedMapping* pClusterToSourceMapping);
	void ConvertCluster(int pSourceClusterIndex, 
					 KFbxWeightedMapping* pClusterToSourceMapping,
					 KFbxWeightedMapping* pSourceToDestinationMapping,
					 KFbxCluster* pDestinationCluster);
	void DuplicateControlPoints(KArrayTemplate<KFbxVector4>& pControlPoints, 
		                        KArrayTemplate<int>& pPolygonVertices);
	void UpdatePolygon(KFbxMesh *pNewMesh, 
						KFbxMesh  const *pRefMesh,
						int pPolygonIndex, 
						int* pNewIndex,
						int &pVerticeIndexMeshTriangulated, 
						int &pPolygonIndexMeshTriangulated);
	void ClearPolygon(KFbxMesh *pNewMesh, int pNewCountVertices = 0, int pNewCountPolygons =0);

	template <class T1, class T2>
	void ConvertNurbs( T1* pNewNurb, T2* pOldNurb );

	bool CopyAnimationCurves(KFbxNode* pNode, KFbxGeometry* pNewGeometry );

	bool FlipNurbsCurve( KFbxNurbsCurve* pCurve ) const;

	void FlipControlPoints( KFbxGeometryBase* pPoints, int pUCount, int pVCount ) const;

	bool ConvertMaterialReferenceMode( KFbxMesh* pMeshRef ) const;

	void RevertMaterialReferenceModeConversion( KFbxMesh* pMeshRef ) const;

	KFbxSdkManager* mManager;

#endif // #ifndef DOXYGEN_SHOULD_SKIP_THIS 

};

#include <fbxfilesdk_nsend.h>

#endif // #ifndef _FBXSDK_GEOMETRY_CONVERTER_H_


