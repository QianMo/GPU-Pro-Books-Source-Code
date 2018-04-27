/******************************************************************************

 @File         PVRTGeometry.h

 @Title        PVRTGeometry

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independant

 @Description  Code to affect triangle mesh geometry.

******************************************************************************/
#ifndef _PVRTGEOMETRY_H_
#define _PVRTGEOMETRY_H_


/****************************************************************************
** Defines
****************************************************************************/
#define PVRTGEOMETRY_IDX	unsigned short

#define PVRTGEOMETRY_SORT_VERTEXCACHE (0x01	/* Sort triangles for optimal vertex cache usage */)
#define PVRTGEOMETRY_SORT_IGNOREVERTS (0x02	/* Do not sort vertices for optimal memory cache usage */)

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
	const unsigned int	dwFlags);


#endif /* _PVRTGEOMETRY_H_ */

/*****************************************************************************
 End of file (PVRTGeometry.h)
*****************************************************************************/
