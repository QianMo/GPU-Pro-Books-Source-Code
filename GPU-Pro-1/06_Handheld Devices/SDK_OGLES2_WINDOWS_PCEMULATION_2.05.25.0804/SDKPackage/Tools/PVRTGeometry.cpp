/******************************************************************************

 @File         PVRTGeometry.cpp

 @Title        PVRTGeometry

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independant

 @Description  Code to affect triangle mesh geometry.

******************************************************************************/
#undef PVRTRISORT_ENABLE_VERIFY_RESULTS

/****************************************************************************
** Includes
****************************************************************************/
#include <vector>
#include <math.h>

#include "PVRTGeometry.h"

#ifdef PVRTRISORT_ENABLE_VERIFY_RESULTS
#include "PvrVGPBlockTest.h"
#endif

#include "PVRTGlobal.h"
#include "PVRTContext.h"
/****************************************************************************
** Structures
****************************************************************************/

struct SVtx;

// SEdg: Information about an "edge" - the shared boundary between two triangles
struct SEdg {
	const SVtx	*psVtx[2];		// Identify the edge by the two vertices it joins
	int			nTriNumFree;	// Number of triangle using this edge
};

// STri: Information about a triangle
struct STri {
	const PVRTGEOMETRY_IDX	*pwIdx;			// Vertex indices forming this triangle
	SEdg					*psEdg[3];		// Pointer to the three triangle edges
	bool					bUsed;
};

// SVtx: information about a vertex
struct SVtx {
	STri	**psTri;		// Allocated array of pointers to the triangles sharing this vertex
	int		nTriNumTot;		// Length of the above array
	int		nTriNumFree;	// Number of triangles unused in the above array
	SVtx	**ppMeshPos;	// Position in VtxByMesh list
};

struct SMesh {
	SVtx	**ppVtx;
	int		nVtxNum;
};

// CObject: Collection of information about the vertices & triangles in a mesh
class CObject {
public:
	STri	*m_pTri;		// Array of all the triangles in the mesh
	SEdg	*m_pEdg;		// Array of all the edges in the mesh
	SVtx	*m_pVtx;		// Array of all the vertices in a mesh

	int		m_nTriNumFree;

protected:
	int		m_nVtxTot;		// Total vertices in the object
	int		m_nEdgTot;		// Total edges in the object
	int		m_nTriTot;		// Total triangles in the object

	int		m_nVtxLimit;	// Maximum number of vertices a block can contain
	int		m_nTriLimit;	// Maximum number of triangles a block can contain

public:
	std::vector<SMesh> *m_pvMesh;
	std::vector<SMesh> m_vMeshLg;

protected:
	SVtx	**m_ppVtxByMesh;

public:
	CObject(
		const PVRTGEOMETRY_IDX	* const pwIdx,
		const int				nVtxTot,
		const int				nTriTot,
		const int				nVtxLimit,
		const int				nTriLimit);

	~CObject();

	int GetVertexCount() const;
	int GetTriangleCount() const;

protected:
	SEdg *BuildEdgeList(
		const SVtx * const pVtx0,
		const SVtx * const pVtx1);

	void CreateMeshList();

public:
	void SplitMesh(
		SMesh		* const pMesh,
		const int	nVtxNum,
		SVtx		** const ppVtx);

	void ResizeMesh(
		const int	nVtxNum,
		SVtx		** const ppVtx);
};

// CBlockOption: a possible group of polygons to use
struct CBlockOption {
protected:
	struct SEdgeDelta {
		const SEdg	*pEdg;
		int			nRefCnt;
	};

public:
	int			nVtxNum;			// Number of vertices in the block
	int			nEdgNum;			// Number of edges in the block
	int			nTriNum;			// Number of triangles in the block

	SVtx		**psVtx;			// Pointers to vertices
protected:
	SEdgeDelta	*psEdgeDelta;
	STri		**psTri;			// Pointers to triangles

	int			m_nVtxLimit;		// Maximum number of vertices a block can contain
	int			m_nTriLimit;		// Maximum number of triangles a block can contain

public:
	~CBlockOption();

	void Init(
		const int	nVtxLimit,
		const int	nTriLimit);
	void Copy(const CBlockOption * const pSrc);

	void Clear();

	void Output(
		PVRTGEOMETRY_IDX	* const pwOut,
		int					* const pnVtxCnt,
		int					* const pnTriCnt,
		const CObject		* const pOb) const;

	bool UsingVertex(const SVtx * const pVtx) const;
	bool Contains(const STri * const pTri) const;

	bool IsEmpty() const;
	bool IsFull() const;

	void AddVertex(SVtx * const pVtx);
	void AddVertexCheckDup(SVtx * const pVtx);

	void AddTriangleCheckDup(STri * const pTri);

	void AddEdgeCheckDup(const SEdg * const pEdg);

	void AddTriangle(STri * const pTri);

	void AddOneTriangle(
		STri		* const pTri,
		const CObject	* const pOb);

	int GetClosedEdgeDelta() const;
	bool IsBetterThan(const CBlockOption * const pCmp) const;

	void Add(
		const CBlockOption	* const pSrc,
		const CObject		* const pOb);

	void Add(
		const SMesh		* const pMesh);
};

// CBlock: a model of a HW block [of triangles and vertices]
class CBlock {
protected:
	CBlockOption	m_sOpt, m_sOptBest;

	int				m_nVtxLimit;		// Maximum number of vertices a block can contain
	int				m_nTriLimit;		// Maximum number of triangles a block can contain

	CBlockOption	m_sJob0, m_sJob1;	// Workspace: to find the best triangle to add

public:
	CBlock(
		const int	nBufferVtxLimit,
		const int	nBufferTriLimit);

	void Clear();

	bool FillFrom(
		SMesh	* const pMesh,
		SVtx	* const pVtx,
		CObject	* const pOb);

	int Fill(
		CObject	* const pOb);

	void Output(
		PVRTGEOMETRY_IDX	* const pwOut,
		int					* const pnVtxCnt,
		int					* const pnTriCnt,
		const CObject		* const pOb) const;

protected:
	bool AddBestTrianglesAppraise(
		CBlockOption	* const pJob,
		const CObject		* const pOb,
		const STri		* const pTriAppraise);

	void AddBestTriangles(CObject * const pOb);
};

/****************************************************************************
** Local function prototypes
****************************************************************************/

/****************************************************************************
** Class: CObject
****************************************************************************/
CObject::CObject(
	const PVRTGEOMETRY_IDX	* const pwIdx,
	const int				nVtxTot,
	const int				nTriTot,
	const int				nVtxLimit,
	const int				nTriLimit)
{
	int		i;
	SVtx	*pVtx0, *pVtx1, *pVtx2;

	m_nVtxLimit		= nVtxLimit;
	m_nTriLimit		= nTriLimit;

	m_pvMesh = new std::vector<SMesh>[nVtxLimit-2];
	_ASSERT(m_pvMesh);

	m_ppVtxByMesh = (SVtx**)calloc(nVtxTot, sizeof(*m_ppVtxByMesh));
	_ASSERT(m_ppVtxByMesh);

	m_nVtxTot = nVtxTot;
	m_nEdgTot = 0;
	m_nTriTot = nTriTot;

	m_nTriNumFree = m_nTriTot;

	m_pTri = (STri*)calloc(nTriTot, sizeof(*m_pTri));
	_ASSERT(m_pTri);

	m_pEdg = (SEdg*)calloc(nTriTot*3, sizeof(*m_pEdg));	// Allocate the maximum possible number of edges, though it should be far fewer than this
	_ASSERT(m_pEdg);

	m_pVtx = (SVtx*)calloc(nVtxTot, sizeof(*m_pVtx));
	_ASSERT(m_pVtx);

	// Run through triangles...
	for(i = 0; i < nTriTot; ++i) {
		pVtx0 = &m_pVtx[pwIdx[3*i+0]];
		pVtx1 = &m_pVtx[pwIdx[3*i+1]];
		pVtx2 = &m_pVtx[pwIdx[3*i+2]];

		// Mark each vertex for the number of times it's referenced
		++pVtx0->nTriNumFree;
		++pVtx1->nTriNumFree;
		++pVtx2->nTriNumFree;

		// Build the edge list
		m_pTri[i].psEdg[0] = BuildEdgeList(pVtx0, pVtx1);
		m_pTri[i].psEdg[1] = BuildEdgeList(pVtx1, pVtx2);
		m_pTri[i].psEdg[2] = BuildEdgeList(pVtx2, pVtx0);
	}

	// Run through vertices, creating enough space for pointers to each triangle using this vertex
	for(i = 0; i < nVtxTot; ++i)
		m_pVtx[i].psTri = (STri**)calloc(m_pVtx[i].nTriNumFree, sizeof(*m_pVtx[i].psTri));

	// Run through triangles, marking each vertex used with a pointer to this tri
	for(i = 0; i < nTriTot; ++i) {
		pVtx0 = &m_pVtx[pwIdx[3*i+0]];
		pVtx1 = &m_pVtx[pwIdx[3*i+1]];
		pVtx2 = &m_pVtx[pwIdx[3*i+2]];

		pVtx0->psTri[pVtx0->nTriNumTot++] = &m_pTri[i];
		pVtx1->psTri[pVtx1->nTriNumTot++] = &m_pTri[i];
		pVtx2->psTri[pVtx2->nTriNumTot++] = &m_pTri[i];

		// Give each triangle a pointer to its indices
		m_pTri[i].pwIdx = &pwIdx[3*i];
	}

#ifdef _DEBUG
	for(i = 0; i < nVtxTot; ++i) {
		_ASSERTE(m_pVtx[i].nTriNumFree == m_pVtx[i].nTriNumTot);
	}
#endif

	CreateMeshList();
}

CObject::~CObject()
{
	_ASSERT(m_nTriNumFree == 0);

	while(m_nVtxTot) {
		--m_nVtxTot;
		FREE(m_pVtx[m_nVtxTot].psTri);
		_ASSERTE(m_pVtx[m_nVtxTot].nTriNumFree == 0);
		_ASSERTE(m_pVtx[m_nVtxTot].ppMeshPos);
	}

#ifdef _DEBUG
	while(m_nEdgTot) {
		--m_nEdgTot;
		_ASSERTE(m_pEdg[m_nEdgTot].nTriNumFree == 0);
	}
	while(m_nTriTot) {
		--m_nTriTot;
		_ASSERTE(m_pTri[m_nTriTot].bUsed);
	}
#endif

	FREE(m_pTri);
	FREE(m_pEdg);
	FREE(m_pVtx);

	delete [] m_pvMesh;
	FREE(m_ppVtxByMesh);
}

int CObject::GetVertexCount() const
{
	return m_nVtxTot;
}

int CObject::GetTriangleCount() const
{
	return m_nTriTot;
}

SEdg *CObject::BuildEdgeList(
	const SVtx * const pVtx0,
	const SVtx * const pVtx1)
{
	SEdg		*pEdg;
	const SVtx	*pVtxL, *pVtxH;
	int			i;

	pVtxL = pVtx0 < pVtx1 ? pVtx0 : pVtx1;
	pVtxH = pVtx0 > pVtx1 ? pVtx0 : pVtx1;

	// Do nothing if the edge already exists
	i = m_nEdgTot;
	while(i) {
		--i;

		pEdg = &m_pEdg[i];
		if(pEdg->psVtx[0] == pVtxL && pEdg->psVtx[1] == pVtxH)
		{
			++pEdg->nTriNumFree;
			return pEdg;
		}
	}

	// Add the new edge
	_ASSERT(m_nEdgTot < m_nTriTot*3);
	pEdg				= &m_pEdg[m_nEdgTot++];
	pEdg->psVtx[0]		= pVtxL;
	pEdg->psVtx[1]		= pVtxH;
	pEdg->nTriNumFree	= 1;

	return pEdg;
}

void CObject::CreateMeshList()
{
	SVtx	**ppR, **ppW, *pVtx;
	STri	*pTri;
	int		i, j, k;
	SMesh	sMesh;
	int		nMeshCnt;

	nMeshCnt = 0;

	ppR = m_ppVtxByMesh;
	ppW = m_ppVtxByMesh;

	for(i = 0; i < m_nVtxTot; ++i) {
		pVtx = &m_pVtx[i];

		if(pVtx->ppMeshPos) {
			_ASSERT(pVtx->ppMeshPos < ppW);
			continue;
		}

		++nMeshCnt;
		sMesh.ppVtx = ppW;

		*ppW			= pVtx;
		pVtx->ppMeshPos	= ppW;
		++ppW;

		do {
			// Add all the vertices of all the triangles of *ppR to the list - unless they're already in there
			for(j = 0; j < (*ppR)->nTriNumTot; ++j) {
				pTri = (*ppR)->psTri[j];

				for(k = 0; k < 3; ++k) {
					pVtx = &m_pVtx[pTri->pwIdx[k]];

					if(pVtx->ppMeshPos) {
						_ASSERT(pVtx->ppMeshPos < ppW);
						continue;
					}

					*ppW			= pVtx;
					pVtx->ppMeshPos	= ppW;
					++ppW;
				}
			}

			++ppR;
		} while(ppR != ppW);

		sMesh.nVtxNum = (int)(ppR - sMesh.ppVtx);
//		_RPT2(_CRT_WARN, "CreateMeshList() mesh %d %dvtx\n", nMeshCnt, sMesh.nVtxNum);
		if(sMesh.nVtxNum >= 3)
		{
			if(sMesh.nVtxNum >= m_nVtxLimit)
				m_vMeshLg.push_back(sMesh);
			else
				m_pvMesh[sMesh.nVtxNum-3].push_back(sMesh);
		}
		else
		{
			/*
				Vertex is not used by any triangles; this may be because we're
				optimising a subset of the mesh (e.g. for bone batching).
			*/
			_ASSERT(sMesh.nVtxNum == 1);
		}
	}

	_ASSERT(ppR == &m_ppVtxByMesh[m_nVtxTot]);
	_ASSERT(ppW == &m_ppVtxByMesh[m_nVtxTot]);
//	_RPT1(_CRT_WARN, "CreateMeshList() %d meshes\n", nMeshCnt);

#ifdef _DEBUG
/*	for(i = 0; i < m_nVtxLimit-2; ++i)
		if(m_pvMesh[i].size())
			_RPT2(_CRT_WARN, "%d:%d ", i+3, m_pvMesh[i].size());
	_RPT1(_CRT_WARN, "lg:%d\n", m_vMeshLg.size());*/
#endif
}

void CObject::SplitMesh(
	SMesh		* const pMesh,
	const int	nVtxNum,
	SVtx		** const ppVtx)
{
	SVtx	*pTmp;
	int		i;
	SMesh	sNew;

	_ASSERT(nVtxNum);

	for(i = 0; i < nVtxNum; ++i) {
		pTmp					= pMesh->ppVtx[i];		// Keep a record of the old vtx that's already here

		pMesh->ppVtx[i]			= ppVtx[i];				// Move the new vtx into place
		*ppVtx[i]->ppMeshPos	= pTmp;					// Move the old vtx into place

		pTmp->ppMeshPos			= ppVtx[i]->ppMeshPos;	// Tell the old vtx where it is now
		ppVtx[i]->ppMeshPos		= &pMesh->ppVtx[i];		// Tell the new vtx where it is now

		_ASSERT(pMesh->ppVtx[i]->nTriNumFree);
	}

	sNew.nVtxNum	= nVtxNum;
	sNew.ppVtx		= pMesh->ppVtx;
	m_pvMesh[nVtxNum-3].push_back(sNew);

	pMesh->ppVtx	= &pMesh->ppVtx[nVtxNum];
	pMesh->nVtxNum	-= nVtxNum;
	if(pMesh->nVtxNum < m_nVtxLimit) {
		ResizeMesh(pMesh->nVtxNum, pMesh->ppVtx);
		m_vMeshLg.pop_back();
#ifdef _DEBUG
/*	} else {
		for(i = 0; i < m_nVtxLimit-2; ++i)
			if(m_pvMesh[i].size())
				_RPT2(_CRT_WARN, "%d:%d ", i+3, m_pvMesh[i].size());
		_RPT1(_CRT_WARN, "lg:%d\n", m_vMeshLg.size());*/
#endif
	}
}

void CObject::ResizeMesh(
	const int	nVtxNum,
	SVtx		** const ppVtx)
{
	SVtx	**ppR, **ppW;
	SMesh	sNew;
	int		i;

	ppR = ppVtx;
	ppW = ppVtx;

	for(i = 0; i < nVtxNum; ++i) {
		if((*ppR)->nTriNumFree) {
			(*ppW) = (*ppR);
			++ppW;
		}
		++ppR;
	}

	sNew.nVtxNum = (int)(ppW - ppVtx);
	_ASSERT(sNew.nVtxNum <= nVtxNum);

	// If any mesh still exists, add it to the relevant list
	if(sNew.nVtxNum) {
		_ASSERT(sNew.nVtxNum >= 3);
		_ASSERT(sNew.nVtxNum < m_nVtxLimit);

		sNew.ppVtx = ppVtx;
		m_pvMesh[sNew.nVtxNum-3].push_back(sNew);
	}

#ifdef _DEBUG
/*	for(i = 0; i < m_nVtxLimit-2; ++i)
		if(m_pvMesh[i].size())
			_RPT2(_CRT_WARN, "%d:%d ", i+3, m_pvMesh[i].size());
	_RPT1(_CRT_WARN, "lg:%d\n", m_vMeshLg.size());*/
#endif
}

/****************************************************************************
** Class: CBlockOption
****************************************************************************/
CBlockOption::~CBlockOption()
{
	FREE(psVtx);
	FREE(psTri);
	FREE(psEdgeDelta);
}

void CBlockOption::Init(
	const int	nVtxLimit,
	const int	nTriLimit)
{
	m_nVtxLimit = nVtxLimit;
	m_nTriLimit = nTriLimit;

	psVtx		= (SVtx**)malloc(nVtxLimit * sizeof(*psVtx));
	psTri		= (STri**)malloc(nTriLimit * sizeof(*psTri));
	psEdgeDelta	= (SEdgeDelta*)malloc(3 * nTriLimit * sizeof(*psEdgeDelta));
}

void CBlockOption::Copy(const CBlockOption * const pSrc)
{
	nVtxNum = pSrc->nVtxNum;
	nEdgNum = pSrc->nEdgNum;
	nTriNum = pSrc->nTriNum;

	memcpy(psVtx, pSrc->psVtx, nVtxNum * sizeof(*psVtx));
	memcpy(psEdgeDelta, pSrc->psEdgeDelta, nEdgNum * sizeof(*psEdgeDelta));
	memcpy(psTri, pSrc->psTri, nTriNum * sizeof(*psTri));
}

void CBlockOption::Clear()
{
	nVtxNum = 0;
	nEdgNum = 0;
	nTriNum = 0;
}

void CBlockOption::Output(
	PVRTGEOMETRY_IDX	* const pwOut,
	int					* const pnVtxCnt,
	int					* const pnTriCnt,
	const CObject		* const pOb) const
{
	STri	*pTri;
	int		i, j;

	for(i = 0; i < nTriNum; ++i) {
		pTri = psTri[i];

		_ASSERT(!pTri->bUsed);

		for(j = 0; j < 3; ++j) {
			_ASSERT(pOb->m_pVtx[pTri->pwIdx[j]].nTriNumFree > 0);
			_ASSERT(pTri->psEdg[j]->nTriNumFree > 0);

			--pOb->m_pVtx[pTri->pwIdx[j]].nTriNumFree;
			--pTri->psEdg[j]->nTriNumFree;

			_ASSERT(pOb->m_pVtx[pTri->pwIdx[j]].nTriNumFree >= 0);
			_ASSERT(pTri->psEdg[j]->nTriNumFree >= 0);
		}

		pTri->bUsed = true;

		// Copy indices into output
		memcpy(&pwOut[3*i], pTri->pwIdx, 3 * sizeof(*pTri->pwIdx));
	}

	*pnVtxCnt = nVtxNum;
	*pnTriCnt = nTriNum;
}

bool CBlockOption::UsingVertex(
	const SVtx		* const pVtx) const
{
	int i;

	i = nVtxNum;
	while(i) {
		--i;

		if(psVtx[i] == pVtx)
			return true;
	}

	return false;
}

bool CBlockOption::Contains(const STri * const pTri) const
{
	int i;

	i = nTriNum;
	while(i) {
		--i;

		if(psTri[i] == pTri)
			return true;
	}

	return false;
}

bool CBlockOption::IsEmpty() const
{
	return !(nVtxNum + nEdgNum + nTriNum);
}

bool CBlockOption::IsFull() const
{
	return  (m_nVtxLimit - nVtxNum) < 3 || nTriNum == m_nTriLimit;
}

void CBlockOption::AddVertex(SVtx * const pVtx)
{
	_ASSERT(nVtxNum < m_nVtxLimit);
	psVtx[nVtxNum++] = pVtx;
}

void CBlockOption::AddVertexCheckDup(SVtx * const pVtx)
{
	int i;

	for(i = 0; i < nVtxNum; ++i)
		if(psVtx[i] == pVtx)
			return;

	AddVertex(pVtx);
}

void CBlockOption::AddTriangleCheckDup(STri * const pTri)
{
	int i;

	for(i = 0; i < nTriNum; ++i)
		if(psTri[i] == pTri)
			return;

	_ASSERT(nTriNum < m_nTriLimit);
	psTri[nTriNum++] = pTri;
}

void CBlockOption::AddEdgeCheckDup(const SEdg * const pEdg)
{
	int i;

	for(i = 0; i < nEdgNum; ++i) {
		if(psEdgeDelta[i].pEdg == pEdg) {
			++psEdgeDelta[i].nRefCnt;
			return;
		}
	}

	_ASSERT(nEdgNum < 3*m_nTriLimit);
	psEdgeDelta[nEdgNum].pEdg		= pEdg;
	psEdgeDelta[nEdgNum].nRefCnt	= 1;
	++nEdgNum;
}

// TODO: if this is only used to add fresh triangles, all edges must be added
void CBlockOption::AddTriangle(STri * const pTri)
{
	int i;

	_ASSERT(nTriNum < m_nTriLimit);
	psTri[nTriNum++] = pTri;

	// Keep a count of edges and the number of tris which share them
	for(i = 0; i < 3; ++i)
		AddEdgeCheckDup(pTri->psEdg[i]);
}

// TODO: if this is only called to add a fresh start triangle, all vertices must be added
void CBlockOption::AddOneTriangle(
	STri			* const pTri,
	const CObject	* const pOb)
{
	int		i;

	// Add the triangle to the block
	AddTriangle(pTri);

	// Add the vertices to the block
	for(i = 0; i < 3; ++i)
		AddVertexCheckDup(&pOb->m_pVtx[pTri->pwIdx[i]]);
}

int CBlockOption::GetClosedEdgeDelta() const
{
	int i, nDelta;

	nDelta = 0;
	for(i = 0; i < nEdgNum; ++i) {
		_ASSERT(psEdgeDelta[i].pEdg->nTriNumFree >= psEdgeDelta[i].nRefCnt);

		// Check how many tris will use the edge if these are taken away
		switch(psEdgeDelta[i].pEdg->nTriNumFree - psEdgeDelta[i].nRefCnt) {
			case 0:
				// If the edge was open, and is now closed, that's good
				if(psEdgeDelta[i].pEdg->nTriNumFree == 1)
					++nDelta;
				break;
			case 1:
				// if the edge is now open, that's bad
				--nDelta;
				break;
		}
	}

	return nDelta;
}

bool CBlockOption::IsBetterThan(const CBlockOption * const pCmp) const
{
	float	fWorth0, fWorth1;
	int		nClosed0, nClosed1;

	// Check "worth" - TrisAdded/VtxAdded
	fWorth0 = (float)nTriNum / (float)nVtxNum;
	fWorth1 = (float)pCmp->nTriNum / (float)pCmp->nVtxNum;

	nClosed0 = GetClosedEdgeDelta();
	nClosed1 = pCmp->GetClosedEdgeDelta();

//	if(fWorth0 != fWorth1) {
	if(fabsf(fWorth0 - fWorth1) > 0.1f) {
		return fWorth0 > fWorth1;
	} else if(nClosed0 != nClosed1) {
		return nClosed0 > nClosed1;
	} else {
		return nTriNum > pCmp->nTriNum;
	}
}

void CBlockOption::Add(
	const CBlockOption	* const pSrc,
	const CObject		* const pOb)
{
	int i;

	// Add vertices from job to block
	for(i = 0; i < pSrc->nVtxNum; ++i)
		AddVertexCheckDup(pSrc->psVtx[i]);

	// Add triangles from job to block
	for(i = 0; i < pSrc->nTriNum; ++i)
		AddTriangle(pSrc->psTri[i]);
}

void CBlockOption::Add(
	const SMesh		* const pMesh)
{
	int i, j;
	SVtx	*pVtx;

	for(i = 0; i < pMesh->nVtxNum; ++i) {
		pVtx = pMesh->ppVtx[i];

		AddVertexCheckDup(pVtx);

		for(j = 0; j < pVtx->nTriNumTot; ++j) {
			if(!pVtx->psTri[j]->bUsed)
				AddTriangleCheckDup(pVtx->psTri[j]);
		}
	}
}

/****************************************************************************
** Class: CBlock
****************************************************************************/
CBlock::CBlock(
	const int	nBufferVtxLimit,
	const int	nBufferTriLimit)
{
	m_nVtxLimit = nBufferVtxLimit;
	m_nTriLimit = nBufferTriLimit;

	m_sOpt.Init(m_nVtxLimit, m_nTriLimit);
	m_sOptBest.Init(m_nVtxLimit, m_nTriLimit);

	// Intialise "job" blocks
	m_sJob0.Init(3, m_nTriLimit);
	m_sJob1.Init(3, m_nTriLimit);
}

void CBlock::Clear()
{
	m_sOpt.Clear();
	m_sOptBest.Clear();
}

void CBlock::Output(
	PVRTGEOMETRY_IDX	* const pwOut,
	int					* const pnVtxCnt,
	int					* const pnTriCnt,
	const CObject		* const pOb) const
{
	m_sOptBest.Output(pwOut, pnVtxCnt, pnTriCnt, pOb);
}

bool CBlock::AddBestTrianglesAppraise(
	CBlockOption	* const pJob,
	const CObject	* const pOb,
	const STri		* const pTriAppraise)
{
	SVtx	*pVtx;
	STri	*pTri;
	int		i, j;

	pJob->Clear();

	// Add vertices
	for(i = 0; i < 3; ++i) {
		pVtx = &pOb->m_pVtx[pTriAppraise->pwIdx[i]];
		if(!m_sOpt.UsingVertex(pVtx))
			pJob->AddVertex(pVtx);
	}

	if(pJob->nVtxNum > (m_nVtxLimit-m_sOpt.nVtxNum))
		return false;

	// Add triangles referenced by each vertex
	for(i = 0; i < 3; ++i) {
		pVtx = &pOb->m_pVtx[pTriAppraise->pwIdx[i]];

		_ASSERT(pVtx->nTriNumFree >= 1);
		_ASSERT(pVtx->nTriNumFree <= pVtx->nTriNumTot);
/*		if(pVtx->nTriNumFree == 1)
			continue;*/

		for(j = 0; j < pVtx->nTriNumTot; ++j) {
			if(pJob->nTriNum >= (m_nTriLimit-m_sOpt.nTriNum))
				break;

			pTri = pVtx->psTri[j];

			// Don't count the same triangle twice!
			if(pTri->bUsed || m_sOpt.Contains(pTri) || pJob->Contains(pTri))
				continue;

			// If all the triangle's vertices are or will be in the block, then increase nTri
			if(
					(
						pTri->pwIdx[0] == pTriAppraise->pwIdx[0] ||
						pTri->pwIdx[0] == pTriAppraise->pwIdx[1] ||
						pTri->pwIdx[0] == pTriAppraise->pwIdx[2] ||
						m_sOpt.UsingVertex(&pOb->m_pVtx[pTri->pwIdx[0]])
					) && (
						pTri->pwIdx[1] == pTriAppraise->pwIdx[0] ||
						pTri->pwIdx[1] == pTriAppraise->pwIdx[1] ||
						pTri->pwIdx[1] == pTriAppraise->pwIdx[2] ||
						m_sOpt.UsingVertex(&pOb->m_pVtx[pTri->pwIdx[1]])
					) && (
						pTri->pwIdx[2] == pTriAppraise->pwIdx[0] ||
						pTri->pwIdx[2] == pTriAppraise->pwIdx[1] ||
						pTri->pwIdx[2] == pTriAppraise->pwIdx[2] ||
						m_sOpt.UsingVertex(&pOb->m_pVtx[pTri->pwIdx[2]])
					)
				)
			{
				pJob->AddTriangle(pTri);
			}
		}
	}

	_ASSERT(pJob->nTriNum);
	_ASSERT(pJob->nTriNum <= (m_nTriLimit-m_sOpt.nTriNum));

	return true;
}

void CBlock::AddBestTriangles(CObject * const pOb)
{
	int				i, j;
	const SVtx		*pVtx;
	STri			*pTri;
	CBlockOption	*pJob, *pJobBest;

	pJob = &m_sJob0;

	do {
		pJobBest = 0;

		for(i = 0; i < m_sOpt.nVtxNum; ++i) {
			pVtx = m_sOpt.psVtx[i];

			if(!pVtx->nTriNumFree)
				continue;

			for(j = 0; j < pVtx->nTriNumTot; ++j) {
				pTri = pVtx->psTri[j];

				if(pTri->bUsed || m_sOpt.Contains(pTri))
					continue;

				// Find out how many triangles and vertices this tri adds
				if(!AddBestTrianglesAppraise(pJob, pOb, pTri))
					continue;

				if(!pJobBest || pJob->IsBetterThan(pJobBest)) {
					pJobBest	= pJob;
					pJob		= (pJob == &m_sJob0 ? &m_sJob1 : &m_sJob0);
				}
			}
		}

		if(pJobBest) {
			m_sOpt.Add(pJobBest, pOb);
		}
	} while(pJobBest && m_nTriLimit != m_sOpt.nTriNum);
}

// Return TRUE if Fill() needs to be called again - i.e. blockOption is already filled
bool CBlock::FillFrom(
	SMesh		* const pMesh,
	SVtx		* const pVtx,
	CObject		* const pOb)
{
	// Let's try starting from this vertex then
	_ASSERT(pVtx->nTriNumFree);
	m_sOpt.Clear();
	m_sOpt.AddVertex(pVtx);
	AddBestTriangles(pOb);

	if(m_sOpt.IsFull()) {
		if(m_sOptBest.IsEmpty() || m_sOpt.IsBetterThan(&m_sOptBest))
			m_sOptBest.Copy(&m_sOpt);
		return false;
	}
	else
	{
		_ASSERT(!m_sOpt.IsEmpty());
		pOb->SplitMesh(pMesh, m_sOpt.nVtxNum, m_sOpt.psVtx);		// Split the sub-mesh into its own mesh
		return true;
	}
}

int CBlock::Fill(
	CObject	* const pOb)
{
	SVtx	*pVtx;
	int		i;
	SMesh	*pMesh;

	/*
		Build blocks from the large meshes
	*/
	if(!pOb->m_vMeshLg.empty()) {
		pMesh = &pOb->m_vMeshLg.back();

//		_RPT1(_CRT_WARN, "Fill() using large with %d vtx\n", pMesh->nVtxNum);

		// Find the vertex with the fewest unused triangles
		for(i = 0; i < pMesh->nVtxNum; ++i) {
			pVtx = pMesh->ppVtx[i];

			if(pVtx->nTriNumFree == 1) {
				if(FillFrom(pMesh, pVtx, pOb))
					return Fill(pOb);
			}
		}

		if(m_sOptBest.IsEmpty()) {
			// Just start from any old vertex
			for(i = 0; i < pMesh->nVtxNum; ++i) {
				pVtx = pMesh->ppVtx[i];

				if(pVtx->nTriNumFree) {
					if(FillFrom(pMesh, pVtx, pOb))
						return Fill(pOb);
					break;
				}
			}

			if(m_sOptBest.IsEmpty()) {
				pOb->m_vMeshLg.pop_back();					// Delete the mesh from the list
				return Fill(pOb);
			}
		}

		if(m_sOptBest.IsFull())
			return -1;
	}

	/*
		Match together the small meshes into blocks
	*/
	_ASSERT(m_sOptBest.IsEmpty());
	i = m_nVtxLimit - m_sOptBest.nVtxNum - 3;

//	_RPT0(_CRT_WARN, "Fill() grouping small ");

	// Starting with the largest meshes, lump them into this block
	while(i >= 0 && (m_nVtxLimit - m_sOptBest.nVtxNum) >= 3) {
		if(pOb->m_pvMesh[i].empty()) {
			--i;
			continue;
		}

		pMesh = &pOb->m_pvMesh[i].back();
		m_sOptBest.Add(pMesh);
//		_RPT1(_CRT_WARN, "+%d", pMesh->nVtxNum);
		pOb->m_pvMesh[i].pop_back();
		i = PVRT_MIN(i, m_nVtxLimit - m_sOptBest.nVtxNum - 3);
	}

	// If there's any space left in this block (and clearly there are no blocks
	// just the right size to fit) then take SOME of the largest block available.
	if(!m_sOptBest.IsFull()) {
		m_sOpt.Copy(&m_sOptBest);

		// Note: This loop purposely does not check m_pvMesh[0] - any block
		// which is looking to grab more geometry would have already sucked
		// up those meshes
		for(i = (m_nVtxLimit-3); i; --i) {
			if(!pOb->m_pvMesh[i].empty()) {
				pMesh = &pOb->m_pvMesh[i].back();

				_ASSERT(pMesh->ppVtx[0]->nTriNumFree);
				_ASSERT(!m_sOpt.UsingVertex(pMesh->ppVtx[0]));

				m_sOpt.AddVertex(pMesh->ppVtx[0]);
//				_RPT1(_CRT_WARN, "(+%d)\n", pMesh->nVtxNum);
				AddBestTriangles(pOb);

				m_sOptBest.Copy(&m_sOpt);
				_ASSERT(m_sOptBest.IsFull());
				return i;
			}
		}
	}
//	_RPT0(_CRT_WARN, "\n");
	return -1;
}

/****************************************************************************
** Local functions
****************************************************************************/
static void SortVertices(
	void				* const pVtxData,
	PVRTGEOMETRY_IDX	* const pwIdx,
	const int			nStride,
	const int			nVertNum,
	const int			nIdxNum)
{
	void				*pVtxNew;
	int					*pnVtxDest;
	int					i;
	PVRTGEOMETRY_IDX	wNext;

	pVtxNew		= malloc(nVertNum * nStride);
	_ASSERT(pVtxNew);

	pnVtxDest	= (int*)malloc(nVertNum * sizeof(*pnVtxDest));
	_ASSERT(pnVtxDest);

	wNext = 0;

	// Default all indices to an invalid number
	for(i = 0; i < nVertNum; ++i)
		pnVtxDest[i] = -1;

	// Let's get on with it then.
	for(i = 0; i < nIdxNum; ++i) {
		if(pnVtxDest[pwIdx[i]] == -1) {
			_ASSERT(wNext < nVertNum);
			memcpy((char*)pVtxNew+(wNext*nStride), (char*)pVtxData+(pwIdx[i]*nStride), nStride);
			pnVtxDest[pwIdx[i]] = wNext++;
		}

		pwIdx[i] = pnVtxDest[pwIdx[i]];
	}

	/*
		This assert will fail if sorting a sub-set of the triangles (e.g. if
		the mesh is bone-batched).

		In that situation vertex sorting should be performed only once after
		all the tri sorting is finished, not per tri-sort.
	*/
	_ASSERT(wNext == nVertNum);
	memcpy(pVtxData, pVtxNew, nVertNum * nStride);

	FREE(pnVtxDest);
	FREE(pVtxNew);
}

/****************************************************************************
** Functions
****************************************************************************/
/*!***************************************************************************
 @Function		PVRTGeometrySort
 @Modified		pVtxData		Pointer to array of vertices
 @Modified		pwIdx			Pointer to array of indices
 @Input			nStride			Size of a vertex (in bytes)
 @Input			nVertNum		Number of vertices. Length of pVtxData array
 @Input			nTriNum			Number of triangles. Length of pwIdx array is 3* this
 @Input			nBufferVtxLimit	Number of vertices that can be stored in a buffer
 @Input			nBufferTriLimit	Number of triangles that can be stored in a buffer
 @Input			dwFlags			PVRTGEOMETRY_SORT_* flags
 @Description	Triangle sorter
*****************************************************************************/
void PVRTGeometrySort(
	void				* const pVtxData,
	PVRTGEOMETRY_IDX	* const pwIdx,
	const int			nStride,
	const int			nVertNum,
	const int			nTriNum,
	const int			nBufferVtxLimit,
	const int			nBufferTriLimit,
	const unsigned int	dwFlags)
{
	CObject				sOb(pwIdx, nVertNum, nTriNum, nBufferVtxLimit, nBufferTriLimit);
	CBlock				sBlock(nBufferVtxLimit, nBufferTriLimit);
	PVRTGEOMETRY_IDX	*pwIdxOut;
	int					nTriCnt, nVtxCnt;
	int					nOutTriCnt, nOutVtxCnt, nOutBlockCnt;
	int					nMeshToResize;
#ifdef PVRTRISORT_ENABLE_VERIFY_RESULTS
	int					i;
	int					pnBlockTriCnt[PVRVGPBLOCKTEST_MAX_BLOCKS];
	SVGPModel			sVGPMdlBefore;
	SVGPModel			sVGPMdlAfter;
#endif

	if(dwFlags & PVRTGEOMETRY_SORT_VERTEXCACHE) {
#ifdef PVRTRISORT_ENABLE_VERIFY_RESULTS
		VGP590Test(&sVGPMdlBefore, pwIdx, nTriNum);
		_RPT4(_CRT_WARN, "OptimiseTriListPVR() Before: Tri: %d, Vtx: %d, vtx/tri=%f Blocks=%d\n", nTriNum, sVGPMdlBefore.nVtxCnt, (float)sVGPMdlBefore.nVtxCnt / (float)nTriNum, sVGPMdlBefore.nBlockCnt);
#endif

		pwIdxOut	= (PVRTGEOMETRY_IDX*)malloc(nTriNum * 3 * sizeof(*pwIdxOut));
		_ASSERT(pwIdxOut);

		// Sort geometry into blocks
		nOutTriCnt		= 0;
		nOutVtxCnt		= 0;
		nOutBlockCnt	= 0;
		do {
			// Clear & fill the block
			sBlock.Clear();
			nMeshToResize = sBlock.Fill(&sOb);

			// Copy indices into output
			sBlock.Output(&pwIdxOut[3*nOutTriCnt], &nVtxCnt, &nTriCnt, &sOb);
			sOb.m_nTriNumFree	-= nTriCnt;
			nOutTriCnt			+= nTriCnt;

			if(nMeshToResize >= 0) {
				SMesh	*pMesh;
				_ASSERT(nMeshToResize <= (nBufferVtxLimit-3));
				pMesh = &sOb.m_pvMesh[nMeshToResize].back();
				sOb.ResizeMesh(pMesh->nVtxNum, pMesh->ppVtx);
				sOb.m_pvMesh[nMeshToResize].pop_back();
			}

			_ASSERT(nVtxCnt <= nBufferVtxLimit);
			_ASSERT(nTriCnt <= nBufferTriLimit);

#ifdef PVRTRISORT_ENABLE_VERIFY_RESULTS
			_ASSERT(nOutBlockCnt < PVRVGPBLOCKTEST_MAX_BLOCKS);
			pnBlockTriCnt[nOutBlockCnt] = nTriCnt;
#endif
			nOutVtxCnt += nVtxCnt;
			nOutBlockCnt++;

//			_RPT4(_CRT_WARN, "%d/%d tris (+%d), %d blocks\n", nOutTriCnt, nTriNum, nTriCnt, nOutBlockCnt);

			_ASSERT(nTriCnt == nBufferTriLimit || (nBufferVtxLimit - nVtxCnt) < 3 || nOutTriCnt == nTriNum);
		} while(nOutTriCnt < nTriNum);

		_ASSERT(nOutTriCnt == nTriNum);
		// The following will fail if optimising a subset of the mesh (e.g. a bone batching)
		//_ASSERT(nOutVtxCnt >= nVertNum);

		// Done!
		memcpy(pwIdx, pwIdxOut, nTriNum * 3 * sizeof(*pwIdx));
		FREE(pwIdxOut);

		_RPT3(_CRT_WARN, "OptimiseTriListPVR() In: Tri: %d, Vtx: %d, vtx/tri=%f\n", nTriNum, nVertNum, (float)nVertNum / (float)nTriNum);
		_RPT4(_CRT_WARN, "OptimiseTriListPVR() HW: Tri: %d, Vtx: %d, vtx/tri=%f Blocks=%d\n", nOutTriCnt, nOutVtxCnt, (float)nOutVtxCnt / (float)nOutTriCnt, nOutBlockCnt);

#ifdef PVRTRISORT_ENABLE_VERIFY_RESULTS
		VGP590Test(&sVGPMdlAfter, pwIdx, nTriNum);
		_RPT4(_CRT_WARN, "OptimiseTriListPVR() After : Tri: %d, Vtx: %d, vtx/tri=%f Blocks=%d\n", nTriNum, sVGPMdlAfter.nVtxCnt, (float)sVGPMdlAfter.nVtxCnt / (float)nTriNum, sVGPMdlAfter.nBlockCnt);
		_ASSERTE(sVGPMdlAfter.nVtxCnt <= sVGPMdlBefore.nVtxCnt);
		_ASSERTE(sVGPMdlAfter.nBlockCnt <= sVGPMdlBefore.nBlockCnt);

		for(i = 0; i < nOutBlockCnt; ++i) {
			_ASSERT(pnBlockTriCnt[i] == sVGPMdlAfter.pnBlockTriCnt[i]);
		}
#endif
	}

	if(!(dwFlags & PVRTGEOMETRY_SORT_IGNOREVERTS)) {
		// Re-order the vertices so maybe they're accessed in a more linear
		// manner. Should cut page-breaks on the initial memory read of
		// vertices. Affects both the order of vertices, and the values
		// of indices, but the triangle order is unchanged.
		SortVertices(pVtxData, pwIdx, nStride, nVertNum, nTriNum*3);
	}
}

/*****************************************************************************
 End of file (PVRTGeometry.cpp)
*****************************************************************************/
