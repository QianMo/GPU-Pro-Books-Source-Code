/******************************************************************************

 @File         PVRTShadowVol.cpp

 @Title        PVRTShadowVol

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Declarations of functions relating to shadow volume generation.

******************************************************************************/
#include <stdlib.h>
#include <string.h>

#include "PVRTGlobal.h"
#include "PVRTContext.h"
#include "PVRTFixedPoint.h"
#include "PVRTMatrix.h"
#include "PVRTTrans.h"
#include "PVRTShadowVol.h"

/****************************************************************************
** Build options
****************************************************************************/

/****************************************************************************
** Defines
****************************************************************************/

/****************************************************************************
** Macros
****************************************************************************/

/****************************************************************************
** Structures
****************************************************************************/
struct SVertexShVol {
	float	x, y, z;
	unsigned int	dwExtrude;
};

/****************************************************************************
** Constants
****************************************************************************/
const static unsigned short c_pwLinesHyperCube[64] = {
	// Cube0
	0, 1,  2, 3,  0, 2,  1, 3,
	4, 5,  6, 7,  4, 6,  5, 7,
	0, 4,  1, 5,  2, 6,  3, 7,
	// Cube1
	8, 9,  10, 11,  8, 10,  9, 11,
	12, 13,  14, 15,  12, 14,  13, 15,
	8, 12,  9, 13,  10, 14,  11, 15,
	// Hyper cube jn
	0, 8,  1, 9,  2, 10,  3, 11,
	4, 12,  5, 13,  6, 14,  7, 15
};
const static PVRTVECTOR3 c_pvRect[4] = {
	{ -1, -1, 1 },
	{ -1,  1, 1 },
	{  1, -1, 1 },
	{  1,  1, 1 }
};

/****************************************************************************
** Shared globals
****************************************************************************/

/****************************************************************************
** Globals
****************************************************************************/

/****************************************************************************
** Declarations
****************************************************************************/

/****************************************************************************
** Code
****************************************************************************/
static unsigned short FindOrCreateVertex(PVRTShadowVolShadowMesh * const psMesh, const PVRTVECTOR3 * const pV) {
	unsigned short	wCurr;

	/*
		First check whether we already have a vertex here
	*/
	for(wCurr = 0; wCurr < psMesh->nV; wCurr++) {
		if(memcmp(&psMesh->pV[wCurr], pV, sizeof(*pV)) == 0) {
			/* Don't do anything more if the vertex already exists */
			return wCurr;
		}
	}

	/*
		Add the vertex then!
	*/
	psMesh->pV[psMesh->nV] = *pV;

	return (unsigned short) psMesh->nV++;
}

static PVRTShadowVolMEdge *FindOrCreateEdge(PVRTShadowVolShadowMesh * const psMesh, const PVRTVECTOR3 * const pv0, const PVRTVECTOR3 * const pv1) {
	unsigned int	nCurr;
	unsigned short			wV0, wV1;

	wV0 = FindOrCreateVertex(psMesh, pv0);
	wV1 = FindOrCreateVertex(psMesh, pv1);

	/*
		First check whether we already have a edge here
	*/
	for(nCurr = 0; nCurr < psMesh->nE; nCurr++) {
		if(
			(psMesh->pE[nCurr].wV0 == wV0 && psMesh->pE[nCurr].wV1 == wV1) ||
			(psMesh->pE[nCurr].wV0 == wV1 && psMesh->pE[nCurr].wV1 == wV0))
		{
			/* Don't do anything more if the edge already exists */
			return &psMesh->pE[nCurr];
		}
	}

	/*
		Add the edge then!
	*/
	psMesh->pE[psMesh->nE].wV0	= wV0;
	psMesh->pE[psMesh->nE].wV1	= wV1;
	psMesh->pE[psMesh->nE].nVis	= 0;

	return &psMesh->pE[psMesh->nE++];
}

static void CrossProduct(
	PVRTVECTOR3 * const pvOut,
	const PVRTVECTOR3 * const pv0,
	const PVRTVECTOR3 * const pv1,
	const PVRTVECTOR3 * const pv2)
{
	PVRTVECTOR3 v0, v1;

	v0.x = pv1->x - pv0->x;
	v0.y = pv1->y - pv0->y;
	v0.z = pv1->z - pv0->z;

	v1.x = pv2->x - pv0->x;
	v1.y = pv2->y - pv0->y;
	v1.z = pv2->z - pv0->z;

	PVRTMatrixVec3CrossProduct(*pvOut, v0, v1);
}

static void FindOrCreateTriangle(
	PVRTShadowVolShadowMesh	* const psMesh,
	const PVRTVECTOR3	* const pv0,
	const PVRTVECTOR3	* const pv1,
	const PVRTVECTOR3	* const pv2)
{
	unsigned int	nCurr;
	PVRTShadowVolMEdge	*psE0, *psE1, *psE2;

	psE0 = FindOrCreateEdge(psMesh, pv0, pv1);
	psE1 = FindOrCreateEdge(psMesh, pv1, pv2);
	psE2 = FindOrCreateEdge(psMesh, pv2, pv0);
	_ASSERT(psE0);
	_ASSERT(psE1);
	_ASSERT(psE2);

	if(psE0 == psE1 || psE1 == psE2 || psE2 == psE0) {
		/* Don't add degenerate triangles */
		_RPT0(_CRT_WARN, "FindOrCreateTriangle() Degenerate triangle.\n");
		return;
	}

	/*
		First check whether we already have a triangle here
	*/
	for(nCurr = 0; nCurr < psMesh->nT; nCurr++) {
		if(
			(psMesh->pT[nCurr].pE0 == psE0 || psMesh->pT[nCurr].pE0 == psE1 || psMesh->pT[nCurr].pE0 == psE2) &&
			(psMesh->pT[nCurr].pE1 == psE0 || psMesh->pT[nCurr].pE1 == psE1 || psMesh->pT[nCurr].pE1 == psE2) &&
			(psMesh->pT[nCurr].pE2 == psE0 || psMesh->pT[nCurr].pE2 == psE1 || psMesh->pT[nCurr].pE2 == psE2))
		{
			/* Don't do anything more if the triangle already exists */
			return;
		}
	}

	/*
		Add the triangle then!
	*/
	psMesh->pT[psMesh->nT].pE0 = psE0;
	psMesh->pT[psMesh->nT].pE1 = psE1;
	psMesh->pT[psMesh->nT].pE2 = psE2;

	/*
		Store the triangle indices; these are indices into the shadow mesh, not the source model indices
	*/
	if(psE0->wV0 == psE1->wV0 || psE0->wV0 == psE1->wV1)
		psMesh->pT[psMesh->nT].w[0] = psE0->wV1;
	else
		psMesh->pT[psMesh->nT].w[0] = psE0->wV0;

	if(psE1->wV0 == psE2->wV0 || psE1->wV0 == psE2->wV1)
		psMesh->pT[psMesh->nT].w[1] = psE1->wV1;
	else
		psMesh->pT[psMesh->nT].w[1] = psE1->wV0;

	if(psE2->wV0 == psE0->wV0 || psE2->wV0 == psE0->wV1)
		psMesh->pT[psMesh->nT].w[2] = psE2->wV1;
	else
		psMesh->pT[psMesh->nT].w[2] = psE2->wV0;

	/* Calculate the triangle normal */
	CrossProduct(&psMesh->pT[psMesh->nT].vNormal, pv0, pv1, pv2);

	/* Check which edges have the correct winding order for this triangle */
	psMesh->pT[psMesh->nT].nWinding = 0;
	if(memcmp(&psMesh->pV[psE0->wV0], pv0, sizeof(*pv0)) == 0) psMesh->pT[psMesh->nT].nWinding |= 0x01;
	if(memcmp(&psMesh->pV[psE1->wV0], pv1, sizeof(*pv1)) == 0) psMesh->pT[psMesh->nT].nWinding |= 0x02;
	if(memcmp(&psMesh->pV[psE2->wV0], pv2, sizeof(*pv2)) == 0) psMesh->pT[psMesh->nT].nWinding |= 0x04;

	psMesh->nT++;
}

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
	const unsigned int		nNumFaces)
{
	unsigned int	nCurr;

	/*
		Prep the structure to return
	*/
	memset(psMesh, 0, sizeof(*psMesh));

	/*
		Allocate some working space to find the unique vertices
	*/
	psMesh->pV = (PVRTVECTOR3*)malloc(nNumVertex * sizeof(*psMesh->pV));
	psMesh->pE = (PVRTShadowVolMEdge*)malloc(nNumFaces * sizeof(*psMesh->pE) * 3);
	psMesh->pT = (PVRTShadowVolMTriangle*)malloc(nNumFaces * sizeof(*psMesh->pT));
	_ASSERT(psMesh->pV);
	_ASSERT(psMesh->pE);
	_ASSERT(psMesh->pT);

	for(nCurr = 0; nCurr < nNumFaces; nCurr++) {
		FindOrCreateTriangle(psMesh,
			(PVRTVECTOR3*)&pVertex[3 * pFaces[3 * nCurr + 0]],
			(PVRTVECTOR3*)&pVertex[3 * pFaces[3 * nCurr + 1]],
			(PVRTVECTOR3*)&pVertex[3 * pFaces[3 * nCurr + 2]]);
	}

	_ASSERT(psMesh->nV <= nNumVertex);
	_ASSERT(psMesh->nE < nNumFaces * 3);
	_ASSERT(psMesh->nT == nNumFaces);

	_RPT2(_CRT_WARN, "Unique vertices : %d (from %d)\n", psMesh->nV, nNumVertex);
	_RPT2(_CRT_WARN, "Unique edges    : %d (from %d)\n", psMesh->nE, nNumFaces * 3);
	_RPT2(_CRT_WARN, "Unique triangles: %d (from %d)\n", psMesh->nT, nNumFaces);

	/*
		Create the real unique lists
	*/
	psMesh->pV = (PVRTVECTOR3*)realloc(psMesh->pV, psMesh->nV * sizeof(*psMesh->pV));
	psMesh->pE = (PVRTShadowVolMEdge*)realloc(psMesh->pE, psMesh->nE * sizeof(*psMesh->pE));
	psMesh->pT = (PVRTShadowVolMTriangle*)realloc(psMesh->pT, psMesh->nT * sizeof(*psMesh->pT));
	_ASSERT(psMesh->pV);
	_ASSERT(psMesh->pE);
	_ASSERT(psMesh->pT);

#if defined(_DEBUG) && !defined(_UNICODE) && defined(WIN32)
	/*
		Check we have sensible model data
	*/
	{
		unsigned int nTri, nEdge;
		OutputDebugStringA("ShadowMeshCreate() Sanity check...");

		for(nEdge = 0; nEdge < psMesh->nE; nEdge++) {
			nCurr = 0;

			for(nTri = 0; nTri < psMesh->nT; nTri++) {
				if(psMesh->pT[nTri].pE0 == &psMesh->pE[nEdge])
					nCurr++;

				if(psMesh->pT[nTri].pE1 == &psMesh->pE[nEdge])
					nCurr++;

				if(psMesh->pT[nTri].pE2 == &psMesh->pE[nEdge])
					nCurr++;
			}

			/*
				Every edge should be referenced exactly twice
			*/
			_ASSERTE(nCurr == 2);
		}

		OutputDebugStringA("done.\n");
	}
#endif
}

/*!***********************************************************************
@Function		PVRTShadowVolMeshInitMesh
@Input			psMesh	The shadow volume mesh
@Input			pContext	A struct for API specific data
@Returns 		True on success
@Description	Init the mesh
*************************************************************************/
bool PVRTShadowVolMeshInitMesh(
	PVRTShadowVolShadowMesh		* const psMesh,
	const SPVRTContext		* const pContext)
{
	unsigned int	nCurr;
#if defined(BUILD_DX9) || defined(BUILD_DX10)
	HRESULT			hRes;
#endif
	SVertexShVol	*pvData;

#ifdef BUILD_OGL
	_ASSERT(pContext && pContext->pglExt);

	if(!pContext || !pContext->pglExt)
		return false;
#endif

	_ASSERT(psMesh);
	_ASSERT(psMesh->nV);
	_ASSERT(psMesh->nE);
	_ASSERT(psMesh->nT);

	/*
		Allocate a vertex buffer for the shadow volumes
	*/
	_ASSERT(psMesh->pivb == NULL);
	_RPT3(_CRT_WARN, "ShadowMeshInitMesh() %5d byte VB (%3dv x 2 x size(%d))\n", psMesh->nV * 2 * sizeof(*pvData), psMesh->nV, sizeof(*pvData));

#ifdef BUILD_DX9
	hRes = pContext->pDev->CreateVertexBuffer(psMesh->nV * 2 * sizeof(*pvData), D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &psMesh->pivb, NULL);
	if(FAILED(hRes)) {
		_ASSERT(false);
		return false;
	}

	hRes = psMesh->pivb->Lock(0, 0, (void**)&pvData, 0);
	if(FAILED(hRes)) {
		_ASSERT(false);
		return false;
	}
#endif

#ifdef BUILD_DX10
	pvData = (SVertexShVol*)psMesh->pivb;
#endif

#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	psMesh->pivb = malloc(psMesh->nV * 2 * sizeof(*pvData));
	pvData = (SVertexShVol*)psMesh->pivb;
#endif

	/*
		Fill the vertex buffer with two subtly different copies of the vertices
	*/
	for(nCurr = 0; nCurr < psMesh->nV; ++nCurr) {
		pvData[nCurr].x			= psMesh->pV[nCurr].x;
		pvData[nCurr].y			= psMesh->pV[nCurr].y;
		pvData[nCurr].z			= psMesh->pV[nCurr].z;
		pvData[nCurr].dwExtrude = 0;

		pvData[nCurr + psMesh->nV]				= pvData[nCurr];
		pvData[nCurr + psMesh->nV].dwExtrude	= 0x04030201;		// Order is wzyx
	}

#ifdef BUILD_DX9
	psMesh->pivb->Unlock();
#endif

#ifdef BUILD_DX10
	D3D10_BUFFER_DESC sBufDesc;
	sBufDesc.ByteWidth		= psMesh->nV * 2 * sizeof(*pvData);
	sBufDesc.Usage			= D3D10_USAGE_IMMUTABLE;
	sBufDesc.BindFlags		= D3D10_BIND_VERTEX_BUFFER;
	sBufDesc.CPUAccessFlags	= 0;
	sBufDesc.MiscFlags		= 0;

	D3D10_SUBRESOURCE_DATA sSRData;
	sSRData.pSysMem				= pvData;

	hRes = pContext->pDev->CreateBuffer(&sBufDesc, &sSRData, &psMesh->pivb);
	if(FAILED(hRes))
	{
		_ASSERT(false);
		return false;
	}
#endif

	return true;
}

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
	const SPVRTContext		* const pContext)
{
#ifdef BUILD_DX9
	HRESULT hRes;
#endif

	_ASSERT(psVol);
	_ASSERT(psMesh);
	_ASSERT(psMesh->nV);
	_ASSERT(psMesh->nE);
	_ASSERT(psMesh->nT);

	_RPT1(_CRT_WARN, "ShadowMeshInitVol() %5d byte IB\n", psMesh->nT * 2 * 3 * sizeof(unsigned short));

	/*
		Allocate a index buffer for the shadow volumes
	*/
#ifdef _DEBUG
	psVol->nIdxCntMax = psMesh->nT * 2 * 3;
#endif
#ifdef BUILD_DX9
	hRes = pContext->pDev->CreateIndexBuffer(psMesh->nT * 2 * 3 * sizeof(unsigned short),
		D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &psVol->piib, NULL);
	if(FAILED(hRes)) {
		_ASSERT(false);
		return false;
	}
#endif
#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	psVol->piib = (unsigned short*)malloc(psMesh->nT * 2 * 3 * sizeof(unsigned short));
#endif

	return true;
}

/*!***********************************************************************
@Function		PVRTShadowVolMeshDestroyMesh
@Input			psMesh	The shadow volume mesh to destroy
@Description	Destroys all shadow volume mesh data created by PVRTShadowVolMeshCreateMesh
*************************************************************************/
void PVRTShadowVolMeshDestroyMesh(
	PVRTShadowVolShadowMesh		* const psMesh)
{
	FREE(psMesh->pV);
	FREE(psMesh->pE);
	FREE(psMesh->pT);
}

/*!***********************************************************************
@Function		PVRTShadowVolMeshReleaseMesh
@Input			psMesh	The shadow volume mesh to release
@Description	Releases all shadow volume mesh data created by PVRTShadowVolMeshInitMesh
*************************************************************************/
void PVRTShadowVolMeshReleaseMesh(
	PVRTShadowVolShadowMesh		* const psMesh)
{
#ifdef BUILD_DX9
	RELEASE(psMesh->pivb);
#endif
#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	FREE(psMesh->pivb);
#endif
}

/*!***********************************************************************
@Function		PVRTShadowVolMeshReleaseVol
@Input			psVol	The shadow volume information to release
@Description	Releases all data create by PVRTShadowVolMeshInitVol
*************************************************************************/
void PVRTShadowVolMeshReleaseVol(
	PVRTShadowVolShadowVol			* const psVol)
{
#ifdef BUILD_DX9
	RELEASE(psVol->piib);
#endif
#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	FREE(psVol->piib);
#endif
}

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
	const bool				bPointLight)
{
	PVRTShadowVolSilhouetteProjectedBuild(psVol, dwVisFlags,psMesh, (PVRTVECTOR3*) pvLightModel, bPointLight);
}

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
	const PVRTVECTOR3		* const pvLightModel,
	const bool				bPointLight)
{
	PVRTVECTOR3		v;
	PVRTShadowVolMTriangle	*psTri;
	PVRTShadowVolMEdge		*psEdge;
	unsigned short	*pwIdx;
#if defined(BUILD_DX9) || defined(BUILD_DX10)
	HRESULT			hRes;
#endif
	unsigned int	nCurr;
	float			f;

	/*
		Lock the index buffer; this is where we create the shadow volume
	*/
	_ASSERT(psVol && psVol->piib);
#ifdef BUILD_DX9
	hRes = psVol->piib->Lock(0, 0, (void**)&pwIdx, D3DLOCK_DISCARD);
	_ASSERT(SUCCEEDED(hRes));
#endif
#if defined(BUILD_OGL) || defined(BUILD_OGLES) || defined(BUILD_OGLES2)
	pwIdx = psVol->piib;
#endif

	psVol->nIdxCnt = 0;

	/*
		Run through triangles, testing which face the From point
	*/
	for(nCurr = 0; nCurr < psMesh->nT; nCurr++) {
		psTri = &psMesh->pT[nCurr];

		if(bPointLight) {
			v.x = psMesh->pV[psTri->pE0->wV0].x - pvLightModel->x;
			v.y = psMesh->pV[psTri->pE0->wV0].y - pvLightModel->y;
			v.z = psMesh->pV[psTri->pE0->wV0].z - pvLightModel->z;
			f = PVRTMatrixVec3DotProduct(psTri->vNormal, v);
		} else {
			f = PVRTMatrixVec3DotProduct(psTri->vNormal, *pvLightModel);
		}

		if(f >= 0) {
			/* Triangle is in the light */
			psTri->pE0->nVis |= 0x01;
			psTri->pE1->nVis |= 0x01;
			psTri->pE2->nVis |= 0x01;

			if(dwVisFlags & PVRTSHADOWVOLUME_NEED_CAP_FRONT) {
				// Add the triangle to the volume, unextruded
				pwIdx[psVol->nIdxCnt+0] = psTri->w[0];
				pwIdx[psVol->nIdxCnt+1] = psTri->w[1];
				pwIdx[psVol->nIdxCnt+2] = psTri->w[2];
				psVol->nIdxCnt += 3;
			}
		} else {
			/* Triangle is in shade; set Bit3 if the winding order needs reversed */
			psTri->pE0->nVis |= 0x02 | (psTri->nWinding & 0x01) << 2;
			psTri->pE1->nVis |= 0x02 | (psTri->nWinding & 0x02) << 1;
			psTri->pE2->nVis |= 0x02 | (psTri->nWinding & 0x04);

			if(dwVisFlags & PVRTSHADOWVOLUME_NEED_CAP_BACK) {
				// Add the triangle to the volume, extruded
				pwIdx[psVol->nIdxCnt+0] = (unsigned short) psMesh->nV + psTri->w[0];
				pwIdx[psVol->nIdxCnt+1] = (unsigned short) psMesh->nV + psTri->w[1];
				pwIdx[psVol->nIdxCnt+2] = (unsigned short) psMesh->nV + psTri->w[2];
				psVol->nIdxCnt += 3;
			}
		}
	}

#ifdef _DEBUG
	_ASSERT(psVol->nIdxCnt <= psVol->nIdxCntMax);
	for(nCurr = 0; nCurr < psVol->nIdxCnt; ++nCurr) {
		_ASSERT(pwIdx[nCurr] < psMesh->nV*2);
	}
#endif

	/*
		Run through edges, testing which are silhouette edges
	*/
	for(nCurr = 0; nCurr < psMesh->nE; nCurr++) {
		psEdge = &psMesh->pE[nCurr];

		if((psEdge->nVis & 0x03) == 0x03) {
			/* Silhouette edge found! */
			if(psEdge->nVis & 0x04) {
				pwIdx[psVol->nIdxCnt+0] = psEdge->wV0;
				pwIdx[psVol->nIdxCnt+1] = psEdge->wV1;
				pwIdx[psVol->nIdxCnt+2] = psEdge->wV0 + (unsigned short) psMesh->nV;

				pwIdx[psVol->nIdxCnt+3] = psEdge->wV0 + (unsigned short) psMesh->nV;
				pwIdx[psVol->nIdxCnt+4] = psEdge->wV1;
				pwIdx[psVol->nIdxCnt+5] = psEdge->wV1 + (unsigned short) psMesh->nV;
			} else {
				pwIdx[psVol->nIdxCnt+0] = psEdge->wV1;
				pwIdx[psVol->nIdxCnt+1] = psEdge->wV0;
				pwIdx[psVol->nIdxCnt+2] = psEdge->wV1 + (unsigned short) psMesh->nV;

				pwIdx[psVol->nIdxCnt+3] = psEdge->wV1 + (unsigned short) psMesh->nV;
				pwIdx[psVol->nIdxCnt+4] = psEdge->wV0;
				pwIdx[psVol->nIdxCnt+5] = psEdge->wV0 + (unsigned short) psMesh->nV;
			}

			psVol->nIdxCnt += 6;
		}

		/* Zero for next render */
		psEdge->nVis = 0;
	}

#ifdef _DEBUG
	_ASSERT(psVol->nIdxCnt <= psVol->nIdxCntMax);
	for(nCurr = 0; nCurr < psVol->nIdxCnt; ++nCurr) {
		_ASSERT(pwIdx[nCurr] < psMesh->nV*2);
	}
#endif

#ifdef BUILD_DX9
	psVol->piib->Unlock();
#endif
#ifdef BUILD_DX10
    D3D10_BUFFER_DESC sIdxBufferDesc;
    sIdxBufferDesc.ByteWidth = psVol->nIdxCnt * sizeof(WORD);
    sIdxBufferDesc.Usage = D3D10_USAGE_DEFAULT;
    sIdxBufferDesc.BindFlags = D3D10_BIND_INDEX_BUFFER;
    sIdxBufferDesc.CPUAccessFlags = 0;
    sIdxBufferDesc.MiscFlags = 0;

    D3D10_SUBRESOURCE_DATA sIdxBufferData;
    ZeroMemory(&sIdxBufferData, sizeof(D3D10_SUBRESOURCE_DATA));
    sIdxBufferData.pSysMem = pwIdx;
    hRes = pContext->pDev->CreateBuffer(&sIdxBufferDesc, &sIdxBufferData, &psVol->piib));
	_ASSERT(SUCCEEDED(hRes));
#endif
}

static bool IsBoundingBoxVisibleEx(
	const PVRTVECTOR4	* const pBoundingHyperCube,
	const float			fCamZ)
{
	PVRTVECTOR3	v, vShift[16];
	unsigned int		dwClipFlags;
	int			i, j;
	unsigned short		w0, w1;

	dwClipFlags = 0;		// Assume all are off-screen

	i = 8;
	while(i)
	{
		i--;

		if(pBoundingHyperCube[i].x <  pBoundingHyperCube[i].w)
			dwClipFlags |= 1 << 0;

		if(pBoundingHyperCube[i].x > -pBoundingHyperCube[i].w)
			dwClipFlags |= 1 << 1;

		if(pBoundingHyperCube[i].y <  pBoundingHyperCube[i].w)
			dwClipFlags |= 1 << 2;

		if(pBoundingHyperCube[i].y > -pBoundingHyperCube[i].w)
			dwClipFlags |= 1 << 3;

		if(pBoundingHyperCube[i].z > 0)
			dwClipFlags |= 1 << 4;
	}

	/*
		Volume is hidden if all the vertices are over a screen edge
	*/
	if(dwClipFlags != 0x1F)
		return false;

	/*
		Well, according to the simple bounding box check, it might be
		visible. Let's now test the view frustrum against the bounding
		hyper cube. (Basically the reverse of the previous test!)

		This catches those cases where a diagonal hyper cube passes near a
		screen edge.
	*/

	// Subtract the camera position from the vertices. I.e. move the camera to 0,0,0
	for(i = 0; i < 8; ++i) {
		vShift[i].x = pBoundingHyperCube[i].x;
		vShift[i].y = pBoundingHyperCube[i].y;
		vShift[i].z = pBoundingHyperCube[i].z - fCamZ;
	}

	i = 12;
	while(i) {
		--i;

		w0 = c_pwLinesHyperCube[2 * i + 0];
		w1 = c_pwLinesHyperCube[2 * i + 1];

		PVRTMatrixVec3CrossProduct(v, vShift[w0], vShift[w1]);
		dwClipFlags = 0;

		j = 4;
		while(j) {
			--j;

			if(PVRTMatrixVec3DotProduct(c_pvRect[j], v) < 0)
				++dwClipFlags;
		}

		// dwClipFlagsA will be 0 or 4 if the screen edges are on the outside of
		// this bounding-box-silhouette-edge.
		if(dwClipFlags % 4)
			continue;

		j = 8;
		while(j) {
			--j;

			if((j != w0) & (j != w1) && (PVRTMatrixVec3DotProduct(vShift[j], v) > 0))
				++dwClipFlags;
		}

		// dwClipFlagsA will be 0 or 18 if this is a silhouette edge of the bounding box
		if(dwClipFlags % 12)
			continue;

		return false;
	}

	return true;
}

static bool IsHyperBoundingBoxVisibleEx(
	const PVRTVECTOR4	* const pBoundingHyperCube,
	const float			fCamZ)
{
	const PVRTVECTOR4	*pv0;
	PVRTVECTOR3	v, vShift[16];
	unsigned int		dwClipFlagsA, dwClipFlagsB;
	int			i, j;
	unsigned short		w0, w1;

	pv0 = &pBoundingHyperCube[8];
	dwClipFlagsA = 0;		// Assume all are off-screen
	dwClipFlagsB = 0;

	i = 8;
	while(i)
	{
		i--;

		// Far
		if(pv0[i].x <  pv0[i].w)
			dwClipFlagsA |= 1 << 0;

		if(pv0[i].x > -pv0[i].w)
			dwClipFlagsA |= 1 << 1;

		if(pv0[i].y <  pv0[i].w)
			dwClipFlagsA |= 1 << 2;

		if(pv0[i].y > -pv0[i].w)
			dwClipFlagsA |= 1 << 3;

		if(pv0[i].z >  0)
			dwClipFlagsA |= 1 << 4;

		// Near
		if(pBoundingHyperCube[i].x <  pBoundingHyperCube[i].w)
			dwClipFlagsB |= 1 << 0;

		if(pBoundingHyperCube[i].x > -pBoundingHyperCube[i].w)
			dwClipFlagsB |= 1 << 1;

		if(pBoundingHyperCube[i].y <  pBoundingHyperCube[i].w)
			dwClipFlagsB |= 1 << 2;

		if(pBoundingHyperCube[i].y > -pBoundingHyperCube[i].w)
			dwClipFlagsB |= 1 << 3;

		if(pBoundingHyperCube[i].z > 0)
			dwClipFlagsB |= 1 << 4;
	}

	/*
		Volume is hidden if all the vertices are over a screen edge
	*/
	if((dwClipFlagsA | dwClipFlagsB) != 0x1F)
		return false;

	/*
		Well, according to the simple bounding box check, it might be
		visible. Let's now test the view frustrum against the bounding
		hyper cube. (Basically the reverse of the previous test!)

		This catches those cases where a diagonal hyper cube passes near a
		screen edge.
	*/

	// Subtract the camera position from the vertices. I.e. move the camera to 0,0,0
	for(i = 0; i < 16; ++i) {
		vShift[i].x = pBoundingHyperCube[i].x;
		vShift[i].y = pBoundingHyperCube[i].y;
		vShift[i].z = pBoundingHyperCube[i].z - fCamZ;
	}

	i = 32;
	while(i) {
		--i;

		w0 = c_pwLinesHyperCube[2 * i + 0];
		w1 = c_pwLinesHyperCube[2 * i + 1];

		PVRTMatrixVec3CrossProduct(v, vShift[w0], vShift[w1]);
		dwClipFlagsA = 0;

		j = 4;
		while(j) {
			--j;

			if(PVRTMatrixVec3DotProduct(c_pvRect[j], v) < 0)
				++dwClipFlagsA;
		}

		// dwClipFlagsA will be 0 or 4 if the screen edges are on the outside of
		// this bounding-box-silhouette-edge.
		if(dwClipFlagsA % 4)
			continue;

		j = 16;
		while(j) {
			--j;

			if((j != w0) & (j != w1) && (PVRTMatrixVec3DotProduct(vShift[j], v) > 0))
				++dwClipFlagsA;
		}

		// dwClipFlagsA will be 0 or 18 if this is a silhouette edge of the bounding box
		if(dwClipFlagsA % 18)
			continue;

		return false;
	}

	return true;
}

static bool IsFrontClipInVolume(
	const PVRTVECTOR4	* const pBoundingHyperCube)
{
	const PVRTVECTOR4	*pv0, *pv1;
	unsigned int				dwClipFlags;
	int					i;
	float				fScale, x, y, w;

	/*
		OK. The hyper-bounding-box is in the view frustrum.

		Now decide if we can use Z-pass instead of Z-fail.

		TODO: if we calculate the convex hull of the front-clip intersection
		points, we can use the connecting lines to do a more accurate on-
		screen check (currently it just uses the bounding box of the
		intersection points.)
	*/
	dwClipFlags = 0;

	i = 32;
	while(i) {
		--i;

		pv0 = &pBoundingHyperCube[c_pwLinesHyperCube[2 * i + 0]];
		pv1 = &pBoundingHyperCube[c_pwLinesHyperCube[2 * i + 1]];

		// If both coords are negative, or both coords are positive, it doesn't cross the Z=0 plane
		if(pv0->z * pv1->z > 0)
			continue;

		// TODO: if fScale > 0.5f, do the lerp in the other direction; this is
		// because we want fScale to be close to 0, not 1, to retain accuracy.
		fScale = (0 - pv0->z) / (pv1->z - pv0->z);

		x = fScale * pv1->x + (1.0f - fScale) * pv0->x;
		y = fScale * pv1->y + (1.0f - fScale) * pv0->y;
		w = fScale * pv1->w + (1.0f - fScale) * pv0->w;

		if(x > -w)
			dwClipFlags |= 1 << 0;

		if(x < w)
			dwClipFlags |= 1 << 1;

		if(y > -w)
			dwClipFlags |= 1 << 2;

		if(y < w)
			dwClipFlags |= 1 << 3;
	}

	if(dwClipFlags == 0x0F)
		return true;

	return false;
}

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
	const float				fVolLength)
{
	int i;

	if(bPointLight) {
		i = 8;
		while(i)
		{
			i--;

			pvExtrudedCube[i].x = pBoundingBox->Point[i].x + fVolLength * (pBoundingBox->Point[i].x - pvLightMdl->x);
			pvExtrudedCube[i].y = pBoundingBox->Point[i].y + fVolLength * (pBoundingBox->Point[i].y - pvLightMdl->y);
			pvExtrudedCube[i].z = pBoundingBox->Point[i].z + fVolLength * (pBoundingBox->Point[i].z - pvLightMdl->z);
		}
	} else {
		i = 8;
		while(i)
		{
			i--;

			pvExtrudedCube[i].x = pBoundingBox->Point[i].x + fVolLength * pvLightMdl->x;
			pvExtrudedCube[i].y = pBoundingBox->Point[i].y + fVolLength * pvLightMdl->y;
			pvExtrudedCube[i].z = pBoundingBox->Point[i].z + fVolLength * pvLightMdl->z;
		}
	}
}

/*!***********************************************************************
@Function		PVRTShadowVolBoundingBoxIsVisible
@Modified		pdwVisFlags		Visibility flags
@Input			bObVisible		Unused set to true
@Input			bNeedsZClipping	Unused set to true
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
	const bool				bObVisible,				// Is the object visible?
	const bool				bNeedsZClipping,		// Does the object require Z clipping?
	const PVRTBOUNDINGBOX	* const pBoundingBox,
	const PVRTMATRIX		* const pmTrans,
	const PVRTVECTOR3		* const pvLightMdl,
	const bool				bPointLight,
	const float				fCamZProj,
	const float				fVolLength)
{
	PVRTVECTOR3		pvExtrudedCube[8];
	PVRTVECTOR4		BoundingHyperCubeT[16];
	int				i;
	unsigned int	dwClipFlagsA, dwClipZCnt;
	float			fLightProjZ;

	_ASSERT((bObVisible && bNeedsZClipping) || !bNeedsZClipping);

	/*
		Transform the eight bounding box points into projection space
	*/
	PVRTTransformVec3Array(&BoundingHyperCubeT[0], sizeof(*BoundingHyperCubeT), pBoundingBox->Point,	sizeof(*pBoundingBox->Point),	pmTrans, 8);

	/*
		Get the light Z coordinate in projection space
	*/
#if 0
	fLightProjZ =
		pmTrans->_13 * pvLightMdl->x +
		pmTrans->_23 * pvLightMdl->y +
		pmTrans->_33 * pvLightMdl->z +
		pmTrans->_43;
#else
	fLightProjZ =
		pmTrans->f[ 2] * pvLightMdl->x +
		pmTrans->f[ 6] * pvLightMdl->y +
		pmTrans->f[10] * pvLightMdl->z +
		pmTrans->f[14];
#endif

	/*
		Where is the object relative to the near clip plane and light?
	*/
	dwClipZCnt		= 0;
	dwClipFlagsA	= 0;
	i = 8;
	while(i) {
		--i;

		if(BoundingHyperCubeT[i].z <= 0)
			++dwClipZCnt;

		if(BoundingHyperCubeT[i].z <= fLightProjZ)
			++dwClipFlagsA;
	}

	if(dwClipZCnt == 8 && dwClipFlagsA == 8) {
		// hidden
		*pdwVisFlags = 0;
		return;
	}

	/*
		Shadow the bounding box into pvExtrudedCube.
	*/
	PVRTShadowVolBoundingBoxExtrude(pvExtrudedCube, pBoundingBox, pvLightMdl, bPointLight, fVolLength);

	/*
		Transform to projection space
	*/
	PVRTTransformVec3Array(&BoundingHyperCubeT[8], sizeof(*BoundingHyperCubeT), pvExtrudedCube, sizeof(*pvExtrudedCube), pmTrans, 8);

	/*
		Check whether any part of the hyper bounding box is even visible
	*/
	if(!IsHyperBoundingBoxVisibleEx(BoundingHyperCubeT, fCamZProj)) {
		*pdwVisFlags = 0;
		return;
	}

	/*
		It's visible, so choose a render method
	*/
	if(dwClipZCnt == 8) {
		// 1
		if(IsFrontClipInVolume(BoundingHyperCubeT)) {
			*pdwVisFlags = PVRTSHADOWVOLUME_VISIBLE | PVRTSHADOWVOLUME_NEED_ZFAIL;

			if(IsBoundingBoxVisibleEx(&BoundingHyperCubeT[8], fCamZProj))
			{
				*pdwVisFlags |= PVRTSHADOWVOLUME_NEED_CAP_BACK;
			}
		} else {
			*pdwVisFlags = PVRTSHADOWVOLUME_VISIBLE;
		}
	} else {
		if(!(dwClipZCnt | dwClipFlagsA)) {
			// 3
			*pdwVisFlags = PVRTSHADOWVOLUME_VISIBLE;
		} else {
			// 5
			if(IsFrontClipInVolume(BoundingHyperCubeT)) {
				*pdwVisFlags = PVRTSHADOWVOLUME_VISIBLE | PVRTSHADOWVOLUME_NEED_ZFAIL;

				if(IsBoundingBoxVisibleEx(BoundingHyperCubeT, fCamZProj))
				{
					*pdwVisFlags |= PVRTSHADOWVOLUME_NEED_CAP_FRONT;
				}

				if(IsBoundingBoxVisibleEx(&BoundingHyperCubeT[8], fCamZProj))
				{
					*pdwVisFlags |= PVRTSHADOWVOLUME_NEED_CAP_BACK;
				}
			} else {
				*pdwVisFlags = PVRTSHADOWVOLUME_VISIBLE;
			}
		}
	}
}

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
	const SPVRTContext		* const pContext)
{
#ifdef BUILD_DX9
	HRESULT	hRes;

	_ASSERT(psMesh->pivb);
	pContext->pDev->SetStreamSource(0, psMesh->pivb, 0, sizeof(SVertexShVol));
	pContext->pDev->SetIndices(psVol->piib);

	_ASSERT(psVol->nIdxCnt <= psVol->nIdxCntMax);
	_ASSERT(psVol->nIdxCnt % 3 == 0);
	_ASSERT(psVol->nIdxCnt / 3 <= 0xFFFF);
	hRes = pContext->pDev->DrawIndexedPrimitive(D3DPT_TRIANGLELIST, 0, 0, psMesh->nV * 2, 0, psVol->nIdxCnt / 3);
	_ASSERT(SUCCEEDED(hRes));

	return psVol->nIdxCnt / 3;
#endif

#if defined(BUILD_OGL) || defined(BUILD_OGLES2)
	_ASSERT(psMesh->pivb);

#if defined(BUILD_OGL)
	_ASSERT(pContext && pContext->pglExt);

	pContext->pglExt->glVertexAttribPointerARB(0, 3, GL_FLOAT, GL_FALSE, sizeof(SVertexShVol), &((SVertexShVol*)psMesh->pivb)[0].x);
	pContext->pglExt->glEnableVertexAttribArrayARB(0);

	pContext->pglExt->glVertexAttribPointerARB(1, 4, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(SVertexShVol), &((SVertexShVol*)psMesh->pivb)[0].dwExtrude);
	pContext->pglExt->glEnableVertexAttribArrayARB(1);

#ifdef _DEBUG // To fix error in Linux
	_ASSERT(psVol->nIdxCnt <= psVol->nIdxCntMax);
	_ASSERT(psVol->nIdxCnt % 3 == 0);
	_ASSERT(psVol->nIdxCnt / 3 <= 0xFFFF);
#endif

	glDrawElements(GL_TRIANGLES, psVol->nIdxCnt, GL_UNSIGNED_SHORT, psVol->piib);

	pContext->pglExt->glDisableVertexAttribArrayARB(0);
	pContext->pglExt->glDisableVertexAttribArrayARB(1);

	return psVol->nIdxCnt / 3;

#elif defined(BUILD_OGLES2)
	GLint i32CurrentProgram;
	glGetIntegerv(GL_CURRENT_PROGRAM, &i32CurrentProgram);

	_ASSERT(i32CurrentProgram); //no program currently set

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SVertexShVol), &((SVertexShVol*)psMesh->pivb)[0].x);
	glEnableVertexAttribArray(0);

	glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_FALSE, sizeof(SVertexShVol), &((SVertexShVol*)psMesh->pivb)[0].dwExtrude);
	glEnableVertexAttribArray(1);

#ifdef _DEBUG // To fix error in Linux
	_ASSERT(psVol->nIdxCnt <= psVol->nIdxCntMax);
	_ASSERT(psVol->nIdxCnt % 3 == 0);
	_ASSERT(psVol->nIdxCnt / 3 <= 0xFFFF);
#endif

	glDrawElements(GL_TRIANGLES, psVol->nIdxCnt, GL_UNSIGNED_SHORT, psVol->piib);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	return psVol->nIdxCnt / 3;

#else
	glVertexAttribPointerARB(6, 4, GL_UNSIGNED_BYTE, false, sizeof(SVertexShVol), &((SVertexShVol*)psMesh->pivb)[0].dwExtrude);
	glEnableVertexAttribArrayARB(6);

	#ifdef _DEBUG // To fix error in Linux
		_ASSERT(psVol->nIdxCnt <= psVol->nIdxCntMax);
		_ASSERT(psVol->nIdxCnt % 3 == 0);
		_ASSERT(psVol->nIdxCnt / 3 <= 0xFFFF);
	#endif

	glDrawElements(GL_TRIANGLES, psVol->nIdxCnt, GL_UNSIGNED_SHORT, psVol->piib);

	return psVol->nIdxCnt / 3;
#endif

#endif
}

/*****************************************************************************
 End of file (PVRTShadowVol.cpp)
*****************************************************************************/
