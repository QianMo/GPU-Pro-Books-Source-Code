/******************************************************************************

 @File         PVRTBoneBatch.cpp

 @Title        PVRTBoneBatch

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Utility functions which process vertices.

******************************************************************************/
#include "PVRTGlobal.h"
#include "PVRTContext.h"

#include <vector>
#include <list>

#include "PVRTMatrix.h"
#include "PVRTVertex.h"
#include "PVRTBoneBatch.h"

/****************************************************************************
** Defines
****************************************************************************/

/****************************************************************************
** Macros
****************************************************************************/

/****************************************************************************
** Structures
****************************************************************************/
class CBatch
{
protected:
	int	m_nCapacity, m_nCnt, *m_pn;

public:
	CBatch()
	{
		m_pn = NULL;
	}

	CBatch(const CBatch &src)
	{
		m_pn = NULL;
		SetSize(src.m_nCapacity);
		*this = src;
	}

	~CBatch()
	{
		FREE(m_pn);
	}

	void operator= (const CBatch &src)
	{
		_ASSERT(m_nCapacity == src.m_nCapacity);
		m_nCnt = src.m_nCnt;
		memcpy(m_pn, src.m_pn, m_nCnt * sizeof(*m_pn));
	}

	void SetSize(const int nSize)
	{
		FREE(m_pn);

		m_nCapacity	= nSize;
		m_nCnt		= 0;
		m_pn		= (int*)malloc(m_nCapacity * sizeof(*m_pn));
	}

	void Clear()
	{
		m_nCnt = 0;
	}

	bool Add(const int n)
	{
		int i;

		if(n < 0)
			return false;

		// If we already have this item, do nothing
		for(i = 0; i < m_nCnt; ++i)
		{
			if(m_pn[i] == n)
				return true;
		}

		// Add the new item
		if(m_nCnt < m_nCapacity)
		{
			m_pn[m_nCnt] = n;
			++m_nCnt;
			return true;
		}
		else
		{
			return false;
		}
	}

	void Merge(const CBatch &src)
	{
		int i;

		for(i = 0; i < src.m_nCnt; ++i)
			Add(src.m_pn[i]);
	}

	int TestMerge(const CBatch &src)
	{
		int i, nCnt;

		nCnt = 0;
		for(i = 0; i < src.m_nCnt; ++i)
			if(!Contains(src.m_pn[i]))
				++nCnt;

		return m_nCnt+nCnt > m_nCapacity ? -1 : nCnt;
	}

	bool Contains(const CBatch &batch) const
	{
		int i;

		for(i = 0; i < batch.m_nCnt; ++i)
			if(!Contains(batch.m_pn[i]))
				return false;

		return true;
	}

	bool Contains(const int n) const
	{
		int i;

		for(i = 0; i < m_nCnt; ++i)
			if(m_pn[i] == n)
				return true;

		return false;
	}

	void Write(
		int * const pn,
		int * const pnCnt) const
	{
		memcpy(pn, m_pn, m_nCnt * sizeof(*pn));
		*pnCnt = m_nCnt;
	}

	void GetVertexBoneIndices(
		float		* const pfI,
		const float	* const pfW,
		const int	n)
	{
		int i, j;

		for(i = 0; i < n; ++i)
		{
			if(pfW[i] != 0)
			{
				for(j = 0; j < m_nCnt; ++j)
				{
					if(pfI[i] != m_pn[j])
						continue;

					pfI[i] = (float)j;
					break;
				}

				// This batch *must* contain this vertex
				_ASSERT(j != m_nCnt);
			}
			else
			{
				pfI[i] = 0;
			}
		}
	}
};

class CGrowableArray
{
protected:
	char	*m_p;
	int		m_nSize;
	int		m_nCnt;

public:
	CGrowableArray(const int nSize)
	{
		m_p		= NULL;
		m_nSize	= nSize;
		m_nCnt	= 0;
	}

	~CGrowableArray()
	{
		FREE(m_p);
	}

	void Append(const void * const pData, const int nCnt)
	{
		m_p = (char*)realloc(m_p, (m_nCnt + nCnt) * m_nSize);
		_ASSERT(m_p);

		memcpy(&m_p[m_nCnt * m_nSize], pData, nCnt * m_nSize);
		m_nCnt += nCnt;
	}

	char *last()
	{
		return at(m_nCnt-1);
	}

	char *at(const int nIdx)
	{
		return &m_p[nIdx * m_nSize];
	}

	int size() const
	{
		return m_nCnt;
	}

	int Surrender(
		char ** const pData)
	{
		int nCnt;

		*pData = m_p;
		nCnt = m_nCnt;

		m_p		= NULL;
		m_nCnt	= 0;

		return nCnt;
	}
};

/****************************************************************************
** Constants
****************************************************************************/

/****************************************************************************
** Local function definitions
****************************************************************************/
static bool FillBatch(
	CBatch					&batch,
	const unsigned short	* const pwIdx,	// input AND output; index array for triangle list
	const char				* const pVtx,	// Input vertices
	const int				nStride,		// Size of a vertex (in bytes)
	const int				nOffsetWeight,	// Offset in bytes to the vertex bone-weights
	EPVRTDataType			eTypeWeight,	// Data type of the vertex bone-weights
	const int				nOffsetIdx,		// Offset in bytes to the vertex bone-indices
	EPVRTDataType			eTypeIdx,		// Data type of the vertex bone-indices
	const int				nVertexBones);	// Number of bones affecting each vertex

static bool BonesMatch(
	const float * const pfIdx0,
	const float * const pfIdx1);

/*****************************************************************************
** Functions
*****************************************************************************/

/*!***************************************************************************
 @Function		Create
 @Output		pnVtxNumOut		vertex count
 @Output		pVtxOut			Output vertices (program must free() this)
 @Modified		pwIdx			index array for triangle list
 @Input			nVtxNum			vertex count
 @Input			pVtx			vertices
 @Input			nStride			Size of a vertex (in bytes)
 @Input			nOffsetWeight	Offset in bytes to the vertex bone-weights
 @Input			eTypeWeight		Data type of the vertex bone-weights
 @Input			nOffsetIdx		Offset in bytes to the vertex bone-indices
 @Input			eTypeIdx		Data type of the vertex bone-indices
 @Input			nTriNum			Number of triangles
 @Input			nBatchBoneMax	Number of bones a batch can reference
 @Input			nVertexBones	Number of bones affecting each vertex
 @Returns		PVR_SUCCESS if successful
 @Description	Fills the bone batch structure
*****************************************************************************/
EPVRTError CPVRTBoneBatches::Create(
	int					* const pnVtxNumOut,
	char				** const pVtxOut,
	unsigned short		* const pwIdx,
	const int			nVtxNum,
	const char			* const pVtx,
	const int			nStride,
	const int			nOffsetWeight,
	const EPVRTDataType	eTypeWeight,
	const int			nOffsetIdx,
	const EPVRTDataType	eTypeIdx,
	const int			nTriNum,
	const int			nBatchBoneMax,
	const int			nVertexBones)
{
	int							i, j, k, nTriCnt;
	CBatch						batch;
	std::list<CBatch>			lBatch;
	std::list<CBatch>::iterator	iBatch, iBatch2;
	CBatch						**ppBatch;
	unsigned short				*pwIdxNew;
	const char					*pV, *pV2;
	PVRTVECTOR4					vWeight, vIdx;
	PVRTVECTOR4					vWeight2, vIdx2;
	std::vector<int>			*pvDup;
	CGrowableArray				*pVtxBuf;
	unsigned short				wSrcIdx;

	memset(this, 0, sizeof(*this));

	if(nVertexBones <= 0 || nVertexBones > 4)
	{
		_RPT0(_CRT_WARN, "CPVRTBoneBatching() will only handle 1..4 bones per vertex.\n");
		return PVR_FAIL;
	}

	memset(&vWeight, 0, sizeof(vWeight));
	memset(&vWeight2, 0, sizeof(vWeight2));
	memset(&vIdx, 0, sizeof(vIdx));
	memset(&vIdx2, 0, sizeof(vIdx2));

	batch.SetSize(nBatchBoneMax);

	// Allocate some working space
	ppBatch		= (CBatch**)malloc(nTriNum * sizeof(*ppBatch));
	pwIdxNew	= (unsigned short*)malloc(nTriNum * 3 * sizeof(*pwIdxNew));
	pvDup		= new std::vector<int>[nVtxNum];
	pVtxBuf		= new CGrowableArray(nStride);

	// Check what batches are necessary
	for(i = 0; i < nTriNum; ++i)
	{
		// Build the batch
		if(!FillBatch(batch, &pwIdx[i * 3], pVtx, nStride, nOffsetWeight, eTypeWeight, nOffsetIdx, eTypeIdx, nVertexBones))
			return PVR_FAIL;

		// Update the batch list
		for(iBatch = lBatch.begin(); iBatch != lBatch.end(); ++iBatch)
		{
			// Do nothing if an existing batch is a superset of this new batch
			if(iBatch->Contains(batch))
			{
				break;
			}

			// If this new batch is a superset of an existing batch, replace the old with the new
			if(batch.Contains(*iBatch))
			{
				*iBatch = batch;
				break;
			}
		}

		// If no suitable batch exists, create a new one
		if(iBatch == lBatch.end())
		{
			lBatch.push_back(batch);
		}
	}

	//	Group batches into fewer batches. This simple greedy algorithm could be improved.
	if(1) {
		int							nShort, nShortest;
		std::list<CBatch>::iterator	iShortest;

		for(iBatch = lBatch.begin(); iBatch != lBatch.end(); ++iBatch)
		{
			while(true)
			{
				nShortest	= nBatchBoneMax;
				iBatch2		= iBatch;
				++iBatch2;
				for(; iBatch2 != lBatch.end(); ++iBatch2)
				{
					nShort = iBatch->TestMerge(*iBatch2);

					if(nShort >= 0 && nShort < nShortest)
					{
						nShortest	= nShort;
						iShortest	= iBatch2;
					}
				}

				if(nShortest < nBatchBoneMax)
				{
					iBatch->Merge(*iShortest);
					lBatch.erase(iShortest);
				}
				else
				{
					break;
				}
			}
		}
	}

	// Place each triangle in a batch.
	for(i = 0; i < nTriNum; ++i)
	{
		if(!FillBatch(batch, &pwIdx[i * 3], pVtx, nStride, nOffsetWeight, eTypeWeight, nOffsetIdx, eTypeIdx, nVertexBones))
			return PVR_FAIL;

		for(iBatch = lBatch.begin(); iBatch != lBatch.end(); ++iBatch)
		{
			if(iBatch->Contains(batch))
			{
				ppBatch[i] = &*iBatch;
				break;
			}
		}

		_ASSERT(iBatch != lBatch.end());
	}

	// Now that we know how many batches there are, we can allocate the output arrays
	CPVRTBoneBatches::nBatchBoneMax = nBatchBoneMax;
	pnBatches		= new int[lBatch.size() * nBatchBoneMax];
	pnBatchBoneCnt	= new int[lBatch.size()];
	pnBatchOffset	= new int[lBatch.size()];
	memset(pnBatches,		0, lBatch.size() * nBatchBoneMax * sizeof(int));
	memset(pnBatchBoneCnt,	0, lBatch.size() * sizeof(int));
	memset(pnBatchOffset,	0, lBatch.size() * sizeof(int));

	// Create the new triangle index list, the new vertex list, and the batch information.
	nTriCnt = 0;
	nBatchCnt = 0;

	for(iBatch = lBatch.begin(); iBatch != lBatch.end(); ++iBatch)
	{
		// Write pnBatches, pnBatchBoneCnt and pnBatchOffset for this batch.
		iBatch->Write(&pnBatches[nBatchCnt * nBatchBoneMax], &pnBatchBoneCnt[nBatchCnt]);
		pnBatchOffset[nBatchCnt] = nTriCnt;
		++nBatchCnt;

		// Copy any triangle indices for this batch
		for(i = 0; i < nTriNum; ++i)
		{
			if(ppBatch[i] != &*iBatch)
				continue;

			for(j = 0; j < 3; ++j)
			{
				wSrcIdx = pwIdx[3 * i + j];

				// Get desired bone indices for this vertex/tri
				pV = &pVtx[wSrcIdx * nStride];

				PVRTVertexRead(&vWeight, &pV[nOffsetWeight], eTypeWeight, nVertexBones);
				PVRTVertexRead(&vIdx, &pV[nOffsetIdx], eTypeIdx, nVertexBones);

				iBatch->GetVertexBoneIndices(&vIdx.x, &vWeight.x, nVertexBones);
				_ASSERT(vIdx.x == 0 || vIdx.x != vIdx.y);

				// Check the list of copies of this vertex for one with suitable bone indices
				for(k = 0; k < (int)pvDup[wSrcIdx].size(); ++k)
				{
					pV2 = pVtxBuf->at(pvDup[wSrcIdx][k]);

					PVRTVertexRead(&vWeight2, &pV2[nOffsetWeight], eTypeWeight, nVertexBones);
					PVRTVertexRead(&vIdx2, &pV2[nOffsetIdx], eTypeIdx, nVertexBones);

					if(BonesMatch(&vIdx2.x, &vIdx.x))
					{
						pwIdxNew[3 * nTriCnt + j] = pvDup[wSrcIdx][k];
						break;
					}
				}

				if(k != (int)pvDup[wSrcIdx].size())
					continue;

				//	Did not find a suitable duplicate of the vertex, so create one
				pVtxBuf->Append(pV, 1);
				pvDup[wSrcIdx].push_back(pVtxBuf->size() - 1);

				PVRTVertexWrite(&pVtxBuf->last()[nOffsetIdx], eTypeIdx, nVertexBones, &vIdx);

				pwIdxNew[3 * nTriCnt + j] = pVtxBuf->size() - 1;
			}
			++nTriCnt;
		}
	}
	_ASSERTE(nTriCnt == nTriNum);
	_ASSERTE(nBatchCnt == (int)lBatch.size());

	//	Copy indices to output
	memcpy(pwIdx, pwIdxNew, nTriNum * 3 * sizeof(*pwIdxNew));

	//	Move vertices to output
	*pnVtxNumOut = pVtxBuf->Surrender(pVtxOut);

	//	Free working memory
	delete [] pvDup;
	delete pVtxBuf;
	FREE(ppBatch);
	FREE(pwIdxNew);

	if(*pnVtxNumOut > 0x0ffff)
	{
		*pnVtxNumOut = 0;
		FREE(*pVtxOut);
		return PVR_OVERFLOW;
	}
	else
		return PVR_SUCCESS;
}

/****************************************************************************
** Local functions
****************************************************************************/

/*!***********************************************************************
 @Function		FillBatch
 @Modified		batch 			The batch to fill
 @Input			pwIdx			Input index array for triangle list
 @Input			pVtx			Input vertices
 @Input			nStride			Size of a vertex (in bytes)
 @Input			nOffsetWeight	Offset in bytes to the vertex bone-weights
 @Input			eTypeWeight		Data type of the vertex bone-weights
 @Input			nOffsetIdx		Offset in bytes to the vertex bone-indices
 @Input			eTypeIdx		Data type of the vertex bone-indices
 @Input			nVertexBones	Number of bones affecting each vertex
 @Returns		True if successful
 @Description	Creates a bone batch from a triangle.
*************************************************************************/
static bool FillBatch(
	CBatch					&batch,
	const unsigned short	* const pwIdx,
	const char				* const pVtx,
	const int				nStride,
	const int				nOffsetWeight,
	EPVRTDataType			eTypeWeight,
	const int				nOffsetIdx,
	EPVRTDataType			eTypeIdx,
	const int				nVertexBones)
{
	PVRTVECTOR4	vWeight, vIdx;
	const char	*pV;
	int			i;
	bool		bOk;

	bOk = true;
	batch.Clear();
	for(i = 0; i < 3; ++i)
	{
		pV = &pVtx[pwIdx[i] * nStride];

		memset(&vWeight, 0, sizeof(vWeight));
		PVRTVertexRead(&vWeight, &pV[nOffsetWeight], eTypeWeight, nVertexBones);
		PVRTVertexRead(&vIdx, &pV[nOffsetIdx], eTypeIdx, nVertexBones);

		if(nVertexBones >= 1 && vWeight.x != 0)	bOk &= batch.Add((int)vIdx.x);
		if(nVertexBones >= 2 && vWeight.y != 0)	bOk &= batch.Add((int)vIdx.y);
		if(nVertexBones >= 3 && vWeight.z != 0)	bOk &= batch.Add((int)vIdx.z);
		if(nVertexBones >= 4 && vWeight.w != 0)	bOk &= batch.Add((int)vIdx.w);
	}
	return bOk;
}

/*!***********************************************************************
 @Function		BonesMatch
 @Input			pfIdx0 A float 4 array
 @Input			pfIdx1 A float 4 array
 @Returns		True if the two float4 arraus are identical
 @Description	Checks if the two float4 arrays are identical.
*************************************************************************/
static bool BonesMatch(
	const float * const pfIdx0,
	const float * const pfIdx1)
{
	int i;

	for(i = 0; i < 4; ++i)
	{
		if(pfIdx0[i] != pfIdx1[i])
			return false;
	}

	return true;
}

/*****************************************************************************
 End of file (PVRTBoneBatch.cpp)
*****************************************************************************/
