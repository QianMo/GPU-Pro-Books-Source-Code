/******************************************************************************

 @File         PVRTShadowVol.h

 @Title        PVRTShadowVol

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Declarations of functions relating to shadow volume generation.

******************************************************************************/
#ifndef _PVRTSHADOWVOL_H_
#define _PVRTSHADOWVOL_H_

#include "PVRTContext.h"
#include "PVRTVector.h"

/****************************************************************************
** Defines
****************************************************************************/
#define PVRTSHADOWVOLUME_VISIBLE		0x00000001
#define PVRTSHADOWVOLUME_NEED_CAP_FRONT	0x00000002
#define PVRTSHADOWVOLUME_NEED_CAP_BACK	0x00000004
#define PVRTSHADOWVOLUME_NEED_ZFAIL		0x00000008

/****************************************************************************
** Structures
****************************************************************************/
struct PVRTShadowVolMEdge {
	unsigned short		wV0, wV1;		/*!< Indices of the vertices of the edge */
	int			nVis;			/*!< Bit0 = Visible, Bit1 = Hidden, Bit2 = Reverse Winding */
};

struct PVRTShadowVolMTriangle {
	unsigned short		w[3];				/*!< Source indices of the triangle */
	PVRTShadowVolMEdge	*pE0, *pE1, *pE2;	/*!< Edges of the triangle */
	PVRTVECTOR3	vNormal;			/*!< Triangle normal */
	int			nWinding;			/*!< BitN = Correct winding for edge N */
};

struct PVRTShadowVolShadowMesh {
	PVRTVECTOR3		*pV;	/*!< Unique vertices in object space */
	PVRTShadowVolMEdge		*pE;
	PVRTShadowVolMTriangle	*pT;
	unsigned int	nV;		/*!< Vertex count */
	unsigned int	nE;		/*!< Edge count */
	unsigned int	nT;		/*!< Triangle count */

#ifdef BUILD_DX9
	IDirect3DVertexBuffer9	*pivb;		/*!< Two copies of the vertices */
#endif
#ifdef BUILD_DX10
	ID3D10Buffer	*pivb;		/*!< Two copies of the vertices */
#endif
#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	void			*pivb;		/*!< Two copies of the vertices */
#endif
};

/*
	Renderable shadow-volume information:
*/
struct PVRTShadowVolShadowVol {
#ifdef BUILD_DX9
	IDirect3DIndexBuffer9	*piib;		/*!< Indices to render the volume */
#endif
#ifdef BUILD_DX10
	ID3D10Buffer			*piib;		/*!< Indices to render the volume */
#endif
#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	unsigned short			*piib;		/*!< Indices to render the volume */
#endif
	unsigned int			nIdxCnt;	/*!< Number of indices in piib */

#ifdef _DEBUG
	unsigned int			nIdxCntMax;	/*!< Number of indices which can fit in piib */
#endif
};

/****************************************************************************
** Declarations
****************************************************************************/

/*!***********************************************************************
@Function	PVRTShadowVolMeshCreateMesh
@Modified	psMesh		The shadow volume mesh to populate
@Input		pVertex		A list of vertices
@Input		nNumVertex	The number of vertices
@Input		pFaces		A list of faces
@Input		nNumFaces	The number of faces
@Description	Creates a mesh format suitable for generating shadow volumes
*************************************************************************/
void PVRTShadowVolMeshCreateMesh(
	PVRTShadowVolShadowMesh		* const psMesh,
	const float				* const pVertex,
	const unsigned int		nNumVertex,
	const unsigned short	* const pFaces,
	const unsigned int		nNumFaces);

/*!***********************************************************************
@Function		PVRTShadowVolMeshInitMesh
@Input			psMesh	The shadow volume mesh
@Input			pContext	A struct for API specific data
@Returns 		True on success
@Description	Init the mesh
*************************************************************************/
bool PVRTShadowVolMeshInitMesh(
	PVRTShadowVolShadowMesh		* const psMesh,
	const SPVRTContext		* const pContext);

/*!***********************************************************************
@Function		PVRTShadowVolMeshInitVol
@Modified		psVol	The shadow volume struct
@Input			psMesh	The shadow volume mesh
@Input			pContext	A struct for API specific data
@Returns		True on success
@Description	Init the renderable shadow volume information.
*************************************************************************/
bool PVRTShadowVolMeshInitVol(
	PVRTShadowVolShadowVol			* const psVol,
	const PVRTShadowVolShadowMesh	* const psMesh,
	const SPVRTContext		* const pContext);

/*!***********************************************************************
@Function		PVRTShadowVolMeshDestroyMesh
@Input			psMesh	The shadow volume mesh to destroy
@Description	Destroys all shadow volume mesh data created by PVRTShadowVolMeshCreateMesh
*************************************************************************/
void PVRTShadowVolMeshDestroyMesh(
	PVRTShadowVolShadowMesh		* const psMesh);

/*!***********************************************************************
@Function		PVRTShadowVolMeshReleaseMesh
@Input			psMesh	The shadow volume mesh to release
@Description	Releases all shadow volume mesh data created by PVRTShadowVolMeshInitMesh
*************************************************************************/
void PVRTShadowVolMeshReleaseMesh(
	PVRTShadowVolShadowMesh		* const psMesh);

/*!***********************************************************************
@Function		PVRTShadowVolMeshReleaseVol
@Input			psVol	The shadow volume information to release
@Description	Releases all data create by PVRTShadowVolMeshInitVol
*************************************************************************/
void PVRTShadowVolMeshReleaseVol(
	PVRTShadowVolShadowVol			* const psVol);

/*!***********************************************************************
@Function		PVRTShadowVolSilhouetteProjectedBuild
@Modified		psVol	The shadow volume information
@Input			dwVisFlags	Shadow volume creation flags
@Input			psMesh	The shadow volume mesh
@Input			pvLightModel	The light position/direction
@Input			bPointLight		Is the light a point light
@Description	Using the light set up the shadow volume so it can be extruded.
*************************************************************************/
void PVRTShadowVolSilhouetteProjectedBuild(
	PVRTShadowVolShadowVol			* const psVol,
	const unsigned int				dwVisFlags,
	const PVRTShadowVolShadowMesh	* const psMesh,
	const PVRTVECTOR3		* const pvLightModel,
	const bool				bPointLight);

/*!***********************************************************************
@Function		PVRTShadowVolSilhouetteProjectedBuild
@Modified		psVol	The shadow volume information
@Input			dwVisFlags	Shadow volume creation flags
@Input			psMesh	The shadow volume mesh
@Input			pvLightModel	The light position/direction
@Input			bPointLight		Is the light a point light
@Description	Using the light set up the shadow volume so it can be extruded.
*************************************************************************/
void PVRTShadowVolSilhouetteProjectedBuild(
	PVRTShadowVolShadowVol			* const psVol,
	const unsigned int		dwVisFlags,
	const PVRTShadowVolShadowMesh	* const psMesh,
	const PVRTVec3		* const pvLightModel,
	const bool				bPointLight);

/*!***********************************************************************
@Function		PVRTShadowVolBoundingBoxExtrude
@Modified		pvExtrudedCube	8 Vertices to represent the extruded box
@Input			pBoundingBox	The bounding box to extrude
@Input			pvLightMdl		The light position/direction
@Input			bPointLight		Is the light a point light
@Input			fVolLength		The length the volume has been extruded by
@Description	Extrudes the bounding box of the volume
*************************************************************************/
void PVRTShadowVolBoundingBoxExtrude(
	PVRTVECTOR3				* const pvExtrudedCube,
	const PVRTBOUNDINGBOX	* const pBoundingBox,
	const PVRTVECTOR3		* const pvLightMdl,
	const bool				bPointLight,
	const float				fVolLength);

/*!***********************************************************************
@Function		PVRTShadowVolBoundingBoxIsVisible
@Modified		pdwVisFlags		Visibility flags
@Input			bObVisible		Is the object visible? Unused set to true
@Input			bNeedsZClipping	Does the object require Z clipping? Unused set to true
@Input			pBoundingBox	The volumes bounding box
@Input			pmTrans			The projection matrix
@Input			pvLightMdl		The light position/direction
@Input			bPointLight		Is the light a point light
@Input			fCamZProj		The camera's z projection value
@Input			fVolLength		The length the volume is extruded by
@Description	Determines if the volume is visible and if it needs caps
*************************************************************************/
void PVRTShadowVolBoundingBoxIsVisible(
	unsigned int			* const pdwVisFlags,
	const bool				bObVisible,
	const bool				bNeedsZClipping,
	const PVRTBOUNDINGBOX	* const pBoundingBox,
	const PVRTMATRIX		* const pmTrans,
	const PVRTVECTOR3		* const pvLightMdl,
	const bool				bPointLight,
	const float				fCamZProj,
	const float				fVolLength);

/*!***********************************************************************
@Function		PVRTShadowVolSilhouetteProjectedRender
@Input			psMesh		Shadow volume mesh
@Input			psVol		Renderable shadow volume information
@Input			pContext	A struct for passing in API specific data
@Description	Draws the shadow volume
*************************************************************************/
int PVRTShadowVolSilhouetteProjectedRender(
	const PVRTShadowVolShadowMesh	* const psMesh,
	const PVRTShadowVolShadowVol	* const psVol,
	const SPVRTContext		* const pContext);


#endif /* _PVRTSHADOWVOL_H_ */

/*****************************************************************************
 End of file (PVRTShadowVol.h)
*****************************************************************************/
