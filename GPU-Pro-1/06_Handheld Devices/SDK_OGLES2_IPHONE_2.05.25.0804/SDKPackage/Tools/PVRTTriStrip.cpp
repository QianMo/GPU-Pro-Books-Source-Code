/******************************************************************************

 @File         PVRTTriStrip.cpp

 @Title        PVRTTriStrip

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Strips a triangle list.

******************************************************************************/
#include <stdlib.h>

#include "PVRTGlobal.h"
#include "PVRTContext.h"
#include "PVRTTriStrip.h"

/****************************************************************************
** Defines
****************************************************************************/
#define RND_TRIS_ORDER

/****************************************************************************
** Structures
****************************************************************************/

/****************************************************************************
** Class: CTri
****************************************************************************/
class CTri;

class CTriState
{
public:
	CTri	*pRev, *pFwd;
	bool	bWindFwd;

	CTriState()
	{
		bWindFwd	= true;		// Initial value irrelevent
		pRev		= NULL;
		pFwd		= NULL;
	}
};

class CTri
{
public:
	CTriState	sNew, sOld;

	CTri	*pAdj[3];
	bool	bInStrip;

	const unsigned short	*pIdx;		// three indices for the tri
	bool					bOutput;

public:
	CTri();
	int FindEdge(const unsigned int pw0, const unsigned int pw1) const;
	void Cement();
	void Undo();
	int EdgeFromAdjTri(const CTri &tri) const;	// Find the index of the adjacent tri
};

class CStrip
{
protected:
	unsigned int	m_nTriCnt;
	CTri			*m_pTri;
	unsigned int	m_nStrips;

	CTri			**m_psStrip;	// Working space for finding strips

public:
	CStrip(
		const unsigned short	* const pwTriList,
		const unsigned int		nTriCnt);
	~CStrip();

protected:
	bool StripGrow(
		CTri				&triFrom,
		const unsigned int	nEdgeFrom,
		const int			nMaxChange);

public:
	void StripFromEdges();
	void StripImprove();

	void Output(
		unsigned short	**ppwStrips,
		unsigned int	**ppnStripLen,
		unsigned int	*pnStripCnt);
};

/****************************************************************************
** Constants
****************************************************************************/

/****************************************************************************
** Code: Class: CTri
****************************************************************************/
CTri::CTri()
{
	pAdj[0]		= NULL;
	pAdj[1]		= NULL;
	pAdj[2]		= NULL;
	bInStrip	= false;
	bOutput		= false;
}

int CTri::FindEdge(const unsigned int pw0, const unsigned int pw1) const
{
	if((pIdx[0] == pw0 && pIdx[1] == pw1))
		return 0;
	if((pIdx[1] == pw0 && pIdx[2] == pw1))
		return 1;
	if((pIdx[2] == pw0 && pIdx[0] == pw1))
		return 2;
	return -1;
}

void CTri::Cement()
{
	sOld = sNew;
}

void CTri::Undo()
{
	sNew = sOld;
}

int CTri::EdgeFromAdjTri(const CTri &tri) const
{
	for(int i = 0; i < 3; ++i)
	{
		if(pAdj[i] == &tri)
		{
			return i;
		}
	}
	_ASSERT(false);
	return -1;
}

/****************************************************************************
** Local code
****************************************************************************/
static int OrphanTri(
	CTri		* const pTri)
{
	_ASSERT(!pTri->bInStrip);
	if(pTri->sNew.bWindFwd || !pTri->sNew.pFwd)
		return 0;

	pTri->sNew.pFwd->sNew.pRev = NULL;
	pTri->sNew.pFwd = NULL;
	return 1;
}

static int TakeTri(
	CTri		* const pTri,
	CTri		* const pRevNew,
	const bool	bFwd)
{
	int	nRet;

	_ASSERT(!pTri->bInStrip);

	if(pTri->sNew.pFwd && pTri->sNew.pRev)
	{
		_ASSERT(pTri->sNew.pFwd->sNew.pRev == pTri);
		pTri->sNew.pFwd->sNew.pRev = NULL;
		_ASSERT(pTri->sNew.pRev->sNew.pFwd == pTri);
		pTri->sNew.pRev->sNew.pFwd = NULL;

		// If in the middle of a Strip, this will generate a new Strip
		nRet = 1;

		// The second tri in the strip may need to be orphaned, or it will have wrong winding order
		nRet += OrphanTri(pTri->sNew.pFwd);
	}
	else if(pTri->sNew.pFwd)
	{
		_ASSERT(pTri->sNew.pFwd->sNew.pRev == pTri);
		pTri->sNew.pFwd->sNew.pRev = NULL;

		// If at the beginning of a Strip, no change
		nRet = 0;

		// The second tri in the strip may need to be orphaned, or it will have wrong winding order
		nRet += OrphanTri(pTri->sNew.pFwd);
	}
	else if(pTri->sNew.pRev)
	{
		_ASSERT(pTri->sNew.pRev->sNew.pFwd == pTri);
		pTri->sNew.pRev->sNew.pFwd = NULL;

		// If at the end of a Strip, no change
		nRet = 0;
	}
	else
	{
		// Otherwise it's a lonesome triangle; one Strip removed!
		nRet = -1;
	}

	pTri->sNew.pFwd		= NULL;
	pTri->sNew.pRev		= pRevNew;
	pTri->bInStrip		= true;
	pTri->sNew.bWindFwd	= bFwd;

	if(pRevNew)
	{
		_ASSERT(!pRevNew->sNew.pFwd);
		pRevNew->sNew.pFwd	= pTri;
	}

	return nRet;
}

static bool TryLinkEdge(
	CTri					&src,
	CTri					&cmp,
	const int				nSrcEdge,
	const unsigned short	idx0,
	const unsigned short	idx1)
{
	int nCmpEdge;

	nCmpEdge = cmp.FindEdge(idx0, idx1);
	if(nCmpEdge != -1 && !cmp.pAdj[nCmpEdge])
	{
		cmp.pAdj[nCmpEdge] = &src;
		src.pAdj[nSrcEdge] = &cmp;
		return true;
	}
	return false;
}

/****************************************************************************
** Code: Class: CStrip
****************************************************************************/

CStrip::CStrip(
	const unsigned short	* const pwTriList,
	const unsigned int		nTriCnt)
{
	unsigned int	i, j;
	bool			b0, b1, b2;

	m_nTriCnt = nTriCnt;

	/*
		Generate adjacency info
	*/
	m_pTri = new CTri[nTriCnt];
	for(i = 0; i < nTriCnt; ++i)
	{
		// Set pointer to indices
		m_pTri[i].pIdx = &pwTriList[3 * i];

		b0 = false;
		b1 = false;
		b2 = false;
		for(j = 0; j < i && !(b0 & b1 & b2); ++j)
		{
			if(!b0)
				b0 = TryLinkEdge(m_pTri[i], m_pTri[j], 0, m_pTri[i].pIdx[1], m_pTri[i].pIdx[0]);

			if(!b1)
				b1 = TryLinkEdge(m_pTri[i], m_pTri[j], 1, m_pTri[i].pIdx[2], m_pTri[i].pIdx[1]);

			if(!b2)
				b2 = TryLinkEdge(m_pTri[i], m_pTri[j], 2, m_pTri[i].pIdx[0], m_pTri[i].pIdx[2]);
		}
	}

	// Initially, every triangle is a strip.
	m_nStrips = m_nTriCnt;

	// Allocate working space for the strippers
	m_psStrip = new CTri*[m_nTriCnt];
}

CStrip::~CStrip()
{
	delete [] m_pTri;
	delete [] m_psStrip;
}

bool CStrip::StripGrow(
	CTri				&triFrom,
	const unsigned int	nEdgeFrom,
	const int			nMaxChange)
{
	unsigned int	i;
	bool			bFwd;
	int				nDiff, nDiffTot, nEdge;
	CTri			*pTri, *pTriPrev, *pTmp;
	unsigned int	nStripLen;

	// Start strip from this tri
	pTri		= &triFrom;
	pTriPrev	= NULL;

	nDiffTot	= 0;
	nStripLen	= 0;

	// Start strip from this edge
	nEdge	= nEdgeFrom;
	bFwd	= true;

	// Extend the strip until we run out, or we find an improvement
	nDiff = 1;
	while(nDiff > nMaxChange)
	{
		// Add nTri to the strip
		_ASSERT(pTri);
		nDiff += TakeTri(pTri, pTriPrev, bFwd);
		_ASSERT(nStripLen < m_nTriCnt);
		m_psStrip[nStripLen++] = pTri;

		// Jump to next tri
		pTriPrev = pTri;
		pTri = pTri->pAdj[nEdge];
		if(!pTri)
			break;	// No more tris, gotta stop

		if(pTri->bInStrip)
			break;	// No more tris, gotta stop

		// Find which edge we came over
		nEdge = pTri->EdgeFromAdjTri(*pTriPrev);

		// Find the edge to leave over
		if(bFwd)
		{
			if(--nEdge < 0)
				nEdge = 2;
		}
		else
		{
			if(++nEdge > 2)
				nEdge = 0;
		}

		// Swap the winding order for the next tri
		bFwd = !bFwd;
	}
	_ASSERT(!pTriPrev->sNew.pFwd);

	/*
		Accept or reject this strip.

		Accepting changes which don't change the number of strips
		adds variety, which can help better strips to develop.
	*/
	if(nDiff <= nMaxChange)
	{
		nDiffTot += nDiff;

		// Great, take the Strip
		for(i = 0; i < nStripLen; ++i)
		{
			pTri = m_psStrip[i];
			_ASSERT(pTri->bInStrip);

			// Cement affected tris
			pTmp = pTri->sOld.pFwd;
			if(pTmp && !pTmp->bInStrip)
			{
				if(pTmp->sOld.pFwd && !pTmp->sOld.pFwd->bInStrip)
					pTmp->sOld.pFwd->Cement();
				pTmp->Cement();
			}

			pTmp = pTri->sOld.pRev;
			if(pTmp && !pTmp->bInStrip)
			{
				pTmp->Cement();
			}

			// Cement this tris
			pTri->bInStrip = false;
			pTri->Cement();
		}
	}
	else
	{
		// Shame, undo the strip
		for(i = 0; i < nStripLen; ++i)
		{
			pTri = m_psStrip[i];
			_ASSERT(pTri->bInStrip);

			// Undo affected tris
			pTmp = pTri->sOld.pFwd;
			if(pTmp && !pTmp->bInStrip)
			{
				if(pTmp->sOld.pFwd && !pTmp->sOld.pFwd->bInStrip)
					pTmp->sOld.pFwd->Undo();
				pTmp->Undo();
			}

			pTmp = pTri->sOld.pRev;
			if(pTmp && !pTmp->bInStrip)
			{
				pTmp->Undo();
			}

			// Undo this tris
			pTri->bInStrip = false;
			pTri->Undo();
		}
	}

#ifdef _DEBUG
	for(int nDbg = 0; nDbg < (int)m_nTriCnt; ++nDbg)
	{
		_ASSERT(m_pTri[nDbg].bInStrip == false);
		_ASSERT(m_pTri[nDbg].bOutput == false);
		_ASSERT(m_pTri[nDbg].sOld.pRev == m_pTri[nDbg].sNew.pRev);
		_ASSERT(m_pTri[nDbg].sOld.pFwd == m_pTri[nDbg].sNew.pFwd);

		if(m_pTri[nDbg].sNew.pRev)
		{
			_ASSERT(m_pTri[nDbg].sNew.pRev->sNew.pFwd == &m_pTri[nDbg]);
		}

		if(m_pTri[nDbg].sNew.pFwd)
		{
			_ASSERT(m_pTri[nDbg].sNew.pFwd->sNew.pRev == &m_pTri[nDbg]);
		}
	}
#endif

	if(nDiffTot)
	{
		m_nStrips += nDiffTot;
		return true;
	}
	return false;
}

void CStrip::StripFromEdges()
{
	unsigned int	i, j, nTest;
	CTri			*pTri, *pTriPrev;
	int				nEdge = 0;

	/*
		Attempt to create grid-oriented strips.
	*/
	for(i = 0; i < m_nTriCnt; ++i)
	{
		pTri = &m_pTri[i];

		// Count the number of empty edges
		nTest = 0;
		for(j = 0; j < 3; ++j)
		{
			if(!pTri->pAdj[j])
			{
				++nTest;
			}
			else
			{
				nEdge = j;
			}
		}

		if(nTest != 2)
			continue;

		while(true)
		{
			// A tri with two empty edges is a corner (there are other corners too, but this works so...)
			while(StripGrow(*pTri, nEdge, -1)) {};

			pTriPrev = pTri;
			pTri = pTri->pAdj[nEdge];
			if(!pTri)
				break;

			// Find the edge we came over
			nEdge = pTri->EdgeFromAdjTri(*pTriPrev);

			// Step around to the next edge
			if(++nEdge > 2)
				nEdge = 0;

			pTriPrev = pTri;
			pTri = pTri->pAdj[nEdge];
			if(!pTri)
				break;

			// Find the edge we came over
			nEdge = pTri->EdgeFromAdjTri(*pTriPrev);

			// Step around to the next edge
			if(--nEdge < 0)
				nEdge = 2;

#if 0
			// If we're not tracking the edge, give up
			nTest = nEdge - 1;
			if(nTest < 0)
				nTest = 2;
			if(pTri->pAdj[nTest])
				break;
			else
				continue;
#endif
		}
	}
}

#ifdef RND_TRIS_ORDER
struct pair
{
	unsigned int i, o;
};

static int compare(const void *arg1, const void *arg2)
{
	return ((pair*)arg1)->i - ((pair*)arg2)->i;
}
#endif

void CStrip::StripImprove()
{
	unsigned int	i, j;
	bool			bChanged;
	int				nRepCnt, nChecks;
	int				nMaxChange;
#ifdef RND_TRIS_ORDER
	pair			*pnOrder;

	/*
		Create a random order to process the tris
	*/
	pnOrder = new pair[m_nTriCnt];
#endif

	nRepCnt = 0;
	nChecks = 2;
	nMaxChange	= 0;

	/*
		Reduce strip count by growing each of the three strips each tri can start.
	*/
	while(nChecks)
	{
		--nChecks;

		bChanged = false;

#ifdef RND_TRIS_ORDER
		/*
			Create a random order to process the tris
		*/
		for(i = 0; i < m_nTriCnt; ++i)
		{
			pnOrder[i].i = rand() * rand();
			pnOrder[i].o = i;
		}
		qsort(pnOrder, m_nTriCnt, sizeof(*pnOrder), compare);
#endif

		/*
			Process the tris
		*/
		for(i = 0; i < m_nTriCnt; ++i)
		{
			for(j = 0; j < 3; ++j)
			{
#ifdef RND_TRIS_ORDER
				bChanged |= StripGrow(m_pTri[pnOrder[i].o], j, nMaxChange);
#else
				bChanged |= StripGrow(m_pTri[i], j, nMaxChange);
#endif
			}
		}
		++nRepCnt;

		// Check the results once or twice
		if(bChanged)
			nChecks = 2;

		nMaxChange = (nMaxChange == 0 ? -1 : 0);
	}

#ifdef RND_TRIS_ORDER
	delete [] pnOrder;
#endif
	_RPT1(_CRT_WARN, "Reps: %d\n", nRepCnt);
}

void CStrip::Output(
	unsigned short	**ppwStrips,
	unsigned int	**ppnStripLen,
	unsigned int	*pnStripCnt)
{
	unsigned short	*pwStrips;
	unsigned int	*pnStripLen;
	unsigned int	i, j, nIdx, nStrip;
	CTri			*pTri;

	/*
		Output Strips
	*/
	pnStripLen = (unsigned int*)malloc(m_nStrips * sizeof(*pnStripLen));
	pwStrips = (unsigned short*)malloc((m_nTriCnt + m_nStrips * 2) * sizeof(*pwStrips));
	nStrip = 0;
	nIdx = 0;
	for(i = 0; i < m_nTriCnt; ++i)
	{
		pTri = &m_pTri[i];

		if(pTri->sNew.pRev)
			continue;
		_ASSERT(!pTri->sNew.pFwd || pTri->sNew.bWindFwd);
		_ASSERT(pTri->bOutput == false);

		if(!pTri->sNew.pFwd)
		{
			pwStrips[nIdx++] = pTri->pIdx[0];
			pwStrips[nIdx++] = pTri->pIdx[1];
			pwStrips[nIdx++] = pTri->pIdx[2];
			pnStripLen[nStrip] = 1;
			pTri->bOutput = true;
		}
		else
		{
			if(pTri->sNew.pFwd == pTri->pAdj[0])
			{
				pwStrips[nIdx++] = pTri->pIdx[2];
				pwStrips[nIdx++] = pTri->pIdx[0];
			}
			else if(pTri->sNew.pFwd == pTri->pAdj[1])
			{
				pwStrips[nIdx++] = pTri->pIdx[0];
				pwStrips[nIdx++] = pTri->pIdx[1];
			}
			else
			{
				_ASSERT(pTri->sNew.pFwd == pTri->pAdj[2]);
				pwStrips[nIdx++] = pTri->pIdx[1];
				pwStrips[nIdx++] = pTri->pIdx[2];
			}

			pnStripLen[nStrip] = 0;
			do
			{
				_ASSERT(pTri->bOutput == false);

				// Increment tris-in-this-strip counter
				++pnStripLen[nStrip];

				// Output the new vertex index
				for(j = 0; j < 3; ++j)
				{
					if(
						(pwStrips[nIdx-2] != pTri->pIdx[j]) &&
						(pwStrips[nIdx-1] != pTri->pIdx[j]))
					{
						break;
					}
				}
				_ASSERT(j != 3);
				pwStrips[nIdx++] = pTri->pIdx[j];

				// Double-check that the previous three indices are the indices of this tris (in some order)
				_ASSERT(
					((pwStrips[nIdx-3] == pTri->pIdx[0]) && (pwStrips[nIdx-2] == pTri->pIdx[1]) && (pwStrips[nIdx-1] == pTri->pIdx[2])) ||
					((pwStrips[nIdx-3] == pTri->pIdx[1]) && (pwStrips[nIdx-2] == pTri->pIdx[2]) && (pwStrips[nIdx-1] == pTri->pIdx[0])) ||
					((pwStrips[nIdx-3] == pTri->pIdx[2]) && (pwStrips[nIdx-2] == pTri->pIdx[0]) && (pwStrips[nIdx-1] == pTri->pIdx[1])) ||
					((pwStrips[nIdx-3] == pTri->pIdx[2]) && (pwStrips[nIdx-2] == pTri->pIdx[1]) && (pwStrips[nIdx-1] == pTri->pIdx[0])) ||
					((pwStrips[nIdx-3] == pTri->pIdx[1]) && (pwStrips[nIdx-2] == pTri->pIdx[0]) && (pwStrips[nIdx-1] == pTri->pIdx[2])) ||
					((pwStrips[nIdx-3] == pTri->pIdx[0]) && (pwStrips[nIdx-2] == pTri->pIdx[2]) && (pwStrips[nIdx-1] == pTri->pIdx[1])));

				// Check that the latest three indices are not degenerate
				_ASSERT(pwStrips[nIdx-1] != pwStrips[nIdx-2]);
				_ASSERT(pwStrips[nIdx-1] != pwStrips[nIdx-3]);
				_ASSERT(pwStrips[nIdx-2] != pwStrips[nIdx-3]);

				pTri->bOutput = true;

				// Check that the next triangle is adjacent to this triangle
				_ASSERT(
					(pTri->sNew.pFwd == pTri->pAdj[0]) ||
					(pTri->sNew.pFwd == pTri->pAdj[1]) ||
					(pTri->sNew.pFwd == pTri->pAdj[2]) ||
					(!pTri->sNew.pFwd));
				// Check that this triangle is adjacent to the next triangle
				_ASSERT(
					(!pTri->sNew.pFwd) ||
					(pTri == pTri->sNew.pFwd->pAdj[0]) ||
					(pTri == pTri->sNew.pFwd->pAdj[1]) ||
					(pTri == pTri->sNew.pFwd->pAdj[2]));

				pTri = pTri->sNew.pFwd;
			} while(pTri);
		}

		++nStrip;
	}
	_ASSERT(nIdx == m_nTriCnt + m_nStrips * 2);
	_ASSERT(nStrip == m_nStrips);

	// Check all triangles have been output
	for(i = 0; i < m_nTriCnt; ++i)
	{
		_ASSERT(m_pTri[i].bOutput == true);
	}

	// Check all triangles are present
	j = 0;
	for(i = 0; i < m_nStrips; ++i)
	{
		j += pnStripLen[i];
	}
	_ASSERT(j == m_nTriCnt);

	// Output data
	*pnStripCnt		= m_nStrips;
	*ppwStrips		= pwStrips;
	*ppnStripLen	= pnStripLen;
}

/****************************************************************************
** Code
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTTriStrip
 @Output			ppwStrips
 @Output			ppnStripLen
 @Output			pnStripCnt
 @Input				pwTriList
 @Input				nTriCnt
 @Description		Reads a triangle list and generates an optimised triangle strip.
*****************************************************************************/
void PVRTTriStrip(
	unsigned short			**ppwStrips,
	unsigned int			**ppnStripLen,
	unsigned int			*pnStripCnt,
	const unsigned short	* const pwTriList,
	const unsigned int		nTriCnt)
{
	unsigned short	*pwStrips;
	unsigned int	*pnStripLen;
	unsigned int	nStripCnt;

	/*
		If the order in which triangles are tested as strip roots is
		randomised, then several attempts can be made. Use the best result.
	*/
	for(int i = 0; i <
#ifdef RND_TRIS_ORDER
		5
#else
		1
#endif
		; ++i)
	{
		CStrip stripper(pwTriList, nTriCnt);

#ifdef RND_TRIS_ORDER
		srand(i);
#endif

		stripper.StripFromEdges();
		stripper.StripImprove();
		stripper.Output(&pwStrips, &pnStripLen, &nStripCnt);

		if(!i || nStripCnt < *pnStripCnt)
		{
			if(i)
			{
				FREE(*ppwStrips);
				FREE(*ppnStripLen);
			}

			*ppwStrips		= pwStrips;
			*ppnStripLen	= pnStripLen;
			*pnStripCnt		= nStripCnt;
		}
		else
		{
			FREE(pwStrips);
			FREE(pnStripLen);
		}
	}
}

/*!***************************************************************************
 @Function			PVRTTriStripList
 @Modified			pwTriList
 @Input				nTriCnt
 @Description		Reads a triangle list and generates an optimised triangle strip.
 					Result is converted back to a triangle list.
*****************************************************************************/
void PVRTTriStripList(unsigned short * const pwTriList, const unsigned int nTriCnt)
{
	unsigned short	*pwStrips;
	unsigned int	*pnStripLength;
	unsigned int	nNumStrips;
	unsigned short	*pwTriPtr, *pwStripPtr;

	/*
		Strip the geometry
	*/
	PVRTTriStrip(&pwStrips, &pnStripLength, &nNumStrips, pwTriList, nTriCnt);

	/*
		Convert back to a triangle list
	*/
	pwStripPtr	= pwStrips;
	pwTriPtr	= pwTriList;
	for(unsigned int i = 0; i < nNumStrips; ++i)
	{
		*pwTriPtr++ = *pwStripPtr++;
		*pwTriPtr++ = *pwStripPtr++;
		*pwTriPtr++ = *pwStripPtr++;

		for(unsigned int j = 1; j < pnStripLength[i]; ++j)
		{
			// Use two indices from previous triangle, flipping tri order alternately.
			if(j & 0x01)
			{
				*pwTriPtr++ = pwStripPtr[-1];
				*pwTriPtr++ = pwStripPtr[-2];
			}
			else
			{
				*pwTriPtr++ = pwStripPtr[-2];
				*pwTriPtr++ = pwStripPtr[-1];
			}

			*pwTriPtr++ = *pwStripPtr++;
		}
	}

	free(pwStrips);
	free(pnStripLength);
}

/*****************************************************************************
 End of file (PVRTTriStrip.cpp)
*****************************************************************************/
