/******************************************************************************

 @File         PVRTBoneBatch.h

 @Title        PVRTBoneBatch

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Utility functions which process vertices.

******************************************************************************/
#ifndef _PVRTBONEBATCH_H_
#define _PVRTBONEBATCH_H_

#include "PVRTVertex.h"

/*!***************************************************************************
 Handles a batch of bones
*****************************************************************************/
/*!***************************************************************************
@Class CPVRTBoneBatches
@Brief A class for processing vertices into bone batches
*****************************************************************************/
class CPVRTBoneBatches
{
public:
	int	*pnBatches;			/*!< Space for nBatchBoneMax bone indices, per batch */
	int	*pnBatchBoneCnt;	/*!< Actual number of bone indices, per batch */
	int	*pnBatchOffset;		/*!< Offset into triangle array, per batch */
	int nBatchBoneMax;		/*!< Stored value as was passed into Create() */
	int	nBatchCnt;			/*!< Number of batches to render */

	/*!***********************************************************************
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
	*************************************************************************/
	EPVRTError Create(
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
		const int			nVertexBones);

	/*!***********************************************************************
	 @Function		Release
	 @Description	Destroy the bone batch structure
	*************************************************************************/
	void Release()
	{
		{ delete pnBatches;			pnBatches = 0; }
		{ delete pnBatchBoneCnt;	pnBatchBoneCnt = 0; }
		{ delete pnBatchOffset;		pnBatchOffset = 0; }
		nBatchCnt = 0;
	}
};


#endif /* _PVRTBONEBATCH_H_ */

/*****************************************************************************
 End of file (PVRTBoneBatch.h)
*****************************************************************************/
