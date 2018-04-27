/******************************************************************************

 @File         PVRTTrans.h

 @Title        PVRTTrans

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Header file of PVRTTrans.cpp Contains structure definitions and
               prototypes of all functions in PVRTTrans.cpp

******************************************************************************/
#ifndef _PVRTTRANS_H_
#define _PVRTTRANS_H_


/****************************************************************************
** Typedefs
****************************************************************************/
typedef struct PVRTBOUNDINGBOX_TAG
{
	PVRTVECTOR3	Point[8];
} PVRTBOUNDINGBOX, *LPPVRTBOUNDINGBOX;

/****************************************************************************
** Functions
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTBoundingBoxCompute
 @Output			pBoundingBox
 @Input				pV
 @Input				nNumberOfVertices
 @Description		Calculate the eight vertices that surround an object.
					This "bounding box" is used later to determine whether
					the object is visible or not.
					This function should only be called once to determine the
					object's bounding box.
*****************************************************************************/
void PVRTBoundingBoxCompute(
	PVRTBOUNDINGBOX		* const pBoundingBox,
	const PVRTVECTOR3	* const pV,
	const int			nNumberOfVertices);

/*!***************************************************************************
 @Function			PVRTBoundingBoxComputeInterleaved
 @Output			pBoundingBox
 @Input				pV
 @Input				nNumberOfVertices
 @Input				i32Offset
 @Input				i32Stride
 @Description		Calculate the eight vertices that surround an object.
					This "bounding box" is used later to determine whether
					the object is visible or not.
					This function should only be called once to determine the
					object's bounding box.
					Takes interleaved data using the first vertex's offset
					and the stride to the next vertex thereafter
*****************************************************************************/
void PVRTBoundingBoxComputeInterleaved(
	PVRTBOUNDINGBOX		* const pBoundingBox,
	const unsigned char	* const pV,
	const int			nNumberOfVertices,
	const int			i32Offset,
	const int			i32Stride);

/*!******************************************************************************
 @Function			PVRTBoundingBoxIsVisible
 @Output			pNeedsZClipping
 @Input				pBoundingBox
 @Input				pMatrix
 @Return			TRUE if the object is visible, FALSE if not.
 @Description		Determine if a bounding box is "visible" or not along the
					Z axis.
					If the function returns TRUE, the object is visible and should
					be displayed (check bNeedsZClipping to know if Z Clipping needs
					to be done).
					If the function returns FALSE, the object is not visible and thus
					does not require to be displayed.
					bNeedsZClipping indicates whether the object needs Z Clipping
					(i.e. the object is partially visible).
					- *pBoundingBox is a pointer to the bounding box structure.
					- *pMatrix is the World, View & Projection matrices combined.
					- *bNeedsZClipping is TRUE if Z clipping is required.
*****************************************************************************/
bool PVRTBoundingBoxIsVisible(
	const PVRTBOUNDINGBOX	* const pBoundingBox,
	const PVRTMATRIX		* const pMatrix,
	bool					* const pNeedsZClipping);

/*!***************************************************************************
 @Function Name		PVRTTransformVec3Array
 @Output			pOut				Destination for transformed vectors
 @Input				nOutStride			Stride between vectors in pOut array
 @Input				pV					Input vector array
 @Input				nInStride			Stride between vectors in pV array
 @Input				pMatrix				Matrix to transform the vectors
 @Input				nNumberOfVertices	Number of vectors to transform
 @Description		Transform all vertices [X Y Z 1] in pV by pMatrix and
 					store them in pOut.
*****************************************************************************/
void PVRTTransformVec3Array(
	PVRTVECTOR4			* const pOut,
	const int			nOutStride,
	const PVRTVECTOR3	* const pV,
	const int			nInStride,
	const PVRTMATRIX	* const pMatrix,
	const int			nNumberOfVertices);

/*!***************************************************************************
 @Function			PVRTTransformArray
 @Output			pTransformedVertex	Destination for transformed vectors
 @Input				pV					Input vector array
 @Input				nNumberOfVertices	Number of vectors to transform
 @Input				pMatrix				Matrix to transform the vectors
 @Input				fW					W coordinate of input vector (e.g. use 1 for position, 0 for normal)
 @Description		Transform all vertices in pVertex by pMatrix and store them in
					pTransformedVertex
					- pTransformedVertex is the pointer that will receive transformed vertices.
					- pVertex is the pointer to untransformed object vertices.
					- nNumberOfVertices is the number of vertices of the object.
					- pMatrix is the matrix used to transform the object.
*****************************************************************************/
void PVRTTransformArray(
	PVRTVECTOR3			* const pTransformedVertex,
	const PVRTVECTOR3	* const pV,
	const int			nNumberOfVertices,
	const PVRTMATRIX	* const pMatrix,
	const VERTTYPE		fW = f2vt(1.0f));

/*!***************************************************************************
 @Function			PVRTTransformArrayBack
 @Output			pTransformedVertex
 @Input				pVertex
 @Input				nNumberOfVertices
 @Input				pMatrix
 @Description		Transform all vertices in pVertex by the inverse of pMatrix
					and store them in pTransformedVertex.
					- pTransformedVertex is the pointer that will receive transformed vertices.
					- pVertex is the pointer to untransformed object vertices.
					- nNumberOfVertices is the number of vertices of the object.
					- pMatrix is the matrix used to transform the object.
*****************************************************************************/
void PVRTTransformArrayBack(
	PVRTVECTOR3			* const pTransformedVertex,
	const PVRTVECTOR3	* const pVertex,
	const int			nNumberOfVertices,
	const PVRTMATRIX	* const pMatrix);

/*!***************************************************************************
 @Function			PVRTTransformBack
 @Output			pOut
 @Input				pV
 @Input				pM
 @Description		Transform vertex pV by the inverse of pMatrix
					and store in pOut.
*****************************************************************************/
void PVRTTransformBack(
	PVRTVECTOR4			* const pOut,
	const PVRTVECTOR4	* const pV,
	const PVRTMATRIX	* const pM);

/*!***************************************************************************
 @Function			PVRTTransform
 @Output			pOut
 @Input				pV
 @Input				pM
 @Description		Transform vertex pV by pMatrix and store in pOut.
*****************************************************************************/
void PVRTTransform(
	PVRTVECTOR4			* const pOut,
	const PVRTVECTOR4	* const pV,
	const PVRTMATRIX	* const pM);


#endif /* _PVRTTRANS_H_ */

/*****************************************************************************
 End of file (PVRTTrans.h)
*****************************************************************************/
