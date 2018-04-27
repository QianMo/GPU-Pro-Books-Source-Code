/******************************************************************************

 @File         PVRTVertex.h

 @Title        PVRTVertex

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Utility functions which process vertices.

******************************************************************************/
#ifndef _PVRTVERTEX_H_
#define _PVRTVERTEX_H_

#include "PVRTError.h"

/****************************************************************************
** Enumerations
****************************************************************************/
enum EPVRTDataType {
	EPODDataNone,
	EPODDataFloat,
	EPODDataInt,
	EPODDataUnsignedShort,
	EPODDataRGBA,
	EPODDataARGB,
	EPODDataD3DCOLOR,
	EPODDataUBYTE4,
	EPODDataDEC3N,
	EPODDataFixed16_16,
	EPODDataUnsignedByte,
	EPODDataShort,
	EPODDataShortNorm,
	EPODDataByte,
	EPODDataByteNorm
};

/*****************************************************************************
** Functions
*****************************************************************************/

/*!***************************************************************************
 @Function			PVRTVertexRead
 @Output			pV
 @Input				pData
 @Input				eType
 @Input				nCnt
 @Description		Read a vector
*****************************************************************************/
void PVRTVertexRead(
	PVRTVECTOR4f		* const pV,
	const void			* const pData,
	const EPVRTDataType	eType,
	const int			nCnt);

/*!***************************************************************************
 @Function			PVRTVertexRead
 @Output			pV
 @Input				pData
 @Input				eType
 @Description		Read an int
*****************************************************************************/
void PVRTVertexRead(
	unsigned int		* const pV,
	const void			* const pData,
	const EPVRTDataType	eType);

/*!***************************************************************************
 @Function			PVRTVertexWrite
 @Output			pOut
 @Input				eType
 @Input				nCnt
 @Input				pV
 @Description		Write a vector
*****************************************************************************/
void PVRTVertexWrite(
	void				* const pOut,
	const EPVRTDataType	eType,
	const int			nCnt,
	const PVRTVECTOR4f	* const pV);

/*!***************************************************************************
 @Function			PVRTVertexWrite
 @Output			pOut
 @Input				eType
 @Input				V
 @Description		Write an int
*****************************************************************************/
void PVRTVertexWrite(
	void				* const pOut,
	const EPVRTDataType	eType,
	const unsigned int	V);

/*!***************************************************************************
 @Function			PVRTVertexTangentBitangent
 @Output			pvTan
 @Output			pvBin
 @Input				pvNor
 @Input				pfPosA
 @Input				pfPosB
 @Input				pfPosC
 @Input				pfTexA
 @Input				pfTexB
 @Input				pfTexC
 @Description		Calculates the tangent and bitangent vectors for
					vertex 'A' of the triangle defined by the 3 supplied
					3D position coordinates (pfPosX) and 2D texture
					coordinates (pfTexX).
*****************************************************************************/
void PVRTVertexTangentBitangent(
	PVRTVECTOR3			* const pvTan,
	PVRTVECTOR3			* const pvBin,
	const PVRTVECTOR3	* const pvNor,
	const float			* const pfPosA,
	const float			* const pfPosB,
	const float			* const pfPosC,
	const float			* const pfTexA,
	const float			* const pfTexB,
	const float			* const pfTexC);

/*!***************************************************************************
 @Function			PVRTVertexGenerateTangentSpace
 @Output			pnVtxNumOut			Output vertex count
 @Output			pVtxOut				Output vertices (program must free() this)
 @Modified			pwIdx				input AND output; index array for triangle list
 @Input				nVtxNum				Input vertex count
 @Input				pVtx				Input vertices
 @Input				nStride				Size of a vertex (in bytes)
 @Input				nOffsetPos			Offset in bytes to the vertex position
 @Input				eTypePos			Data type of the position
 @Input				nOffsetNor			Offset in bytes to the vertex normal
 @Input				eTypeNor			Data type of the normal
 @Input				nOffsetTex			Offset in bytes to the vertex texture coordinate to use
 @Input				eTypeTex			Data type of the texture coordinate
 @Input				nOffsetTan			Offset in bytes to the vertex tangent
 @Input				eTypeTan			Data type of the tangent
 @Input				nOffsetBin			Offset in bytes to the vertex bitangent
 @Input				eTypeBin			Data type of the bitangent
 @Input				nTriNum				Number of triangles
 @Input				fSplitDifference	Split a vertex if the DP3 of tangents/bitangents are below this (range -1..1)
 @Return			PVR_FAIL if there was a problem.
 @Description		Calculates the tangent space for all supplied vertices.
					Writes tangent and bitangent vectors to the output
					vertices, copies all other elements from input vertices.
					Will split vertices if necessary - i.e. if two triangles
					sharing a vertex want to assign it different
					tangent-space matrices. The decision whether to split
					uses fSplitDifference - of the DP3 of two desired
					tangents or two desired bitangents is higher than this,
					the vertex will be split.
*****************************************************************************/
EPVRTError PVRTVertexGenerateTangentSpace(
	int				* const pnVtxNumOut,
	char			** const pVtxOut,
	unsigned short	* const pwIdx,
	const int		nVtxNum,
	const char		* const pVtx,
	const int		nStride,
	const int		nOffsetPos,
	EPVRTDataType	eTypePos,
	const int		nOffsetNor,
	EPVRTDataType	eTypeNor,
	const int		nOffsetTex,
	EPVRTDataType	eTypeTex,
	const int		nOffsetTan,
	EPVRTDataType	eTypeTan,
	const int		nOffsetBin,
	EPVRTDataType	eTypeBin,
	const int		nTriNum,
	const float		fSplitDifference);


#endif /* _PVRTVERTEX_H_ */

/*****************************************************************************
 End of file (PVRTVertex.h)
*****************************************************************************/
