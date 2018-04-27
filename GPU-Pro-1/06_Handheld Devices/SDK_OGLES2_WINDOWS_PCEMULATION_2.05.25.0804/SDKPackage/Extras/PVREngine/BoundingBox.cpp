/******************************************************************************

 @File         BoundingBox.cpp

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Set of functions used for 3D transformations and projections.

******************************************************************************/
#include "BoundingBox.h"

namespace pvrengine
{

	

	BoundingBox::BoundingBox(const PVRTVec3	* const pV,
		const int			nNumberOfVertices)
	{
		int			i;
		VERTTYPE	MinX, MaxX, MinY, MaxY, MinZ, MaxZ;

		/* Inialise values to first vertex */
		MinX=pV->x;	MaxX=pV->x;
		MinY=pV->y;	MaxY=pV->y;
		MinZ=pV->z;	MaxZ=pV->z;

		/* Loop through all vertices to find extremas */
		for (i=1; i<nNumberOfVertices; i++)
		{
			/* Minimum and Maximum X */
			if (pV[i].x < MinX) MinX = pV[i].x;
			if (pV[i].x > MaxX) MaxX = pV[i].x;

			/* Minimum and Maximum Y */
			if (pV[i].y < MinY) MinY = pV[i].y;
			if (pV[i].y > MaxY) MaxY = pV[i].y;

			/* Minimum and Maximum Z */
			if (pV[i].z < MinZ) MinZ = pV[i].z;
			if (pV[i].z > MaxZ) MaxZ = pV[i].z;
		}

		/* Assign the resulting extremas to the bounding box structure */
		/* Point 0 */
		vPoint[0].x=MinX;
		vPoint[0].y=MinY;
		vPoint[0].z=MinZ;

		/* Point 1 */
		vPoint[1].x=MinX;
		vPoint[1].y=MinY;
		vPoint[1].z=MaxZ;

		/* Point 2 */
		vPoint[2].x=MinX;
		vPoint[2].y=MaxY;
		vPoint[2].z=MinZ;

		/* Point 3 */
		vPoint[3].x=MinX;
		vPoint[3].y=MaxY;
		vPoint[3].z=MaxZ;

		/* Point 4 */
		vPoint[4].x=MaxX;
		vPoint[4].y=MinY;
		vPoint[4].z=MinZ;

		/* Point 5 */
		vPoint[5].x=MaxX;
		vPoint[5].y=MinY;
		vPoint[5].z=MaxZ;

		/* Point 6 */
		vPoint[6].x=MaxX;
		vPoint[6].y=MaxY;
		vPoint[6].z=MinZ;

		/* Point 7 */
		vPoint[7].x=MaxX;
		vPoint[7].y=MaxY;
		vPoint[7].z=MaxZ;
	}

/******************************************************************************/

	BoundingBox::BoundingBox(
		const unsigned char			* const pV,
		const int			nNumberOfVertices,
		const int			i32Offset,
		const int			i32Stride)
	{
		int			i;
		VERTTYPE	MinX, MaxX, MinY, MaxY, MinZ, MaxZ;

		// point to first vertex
		PVRTVec3 *pVertex =(PVRTVec3*)(pV+i32Offset);

		/* Inialise values to first vertex */
		MinX=pVertex->x;	MaxX=pVertex->x;
		MinY=pVertex->y;	MaxY=pVertex->y;
		MinZ=pVertex->z;	MaxZ=pVertex->z;

		/* Loop through all vertices to find extremas */
		for (i=1; i<nNumberOfVertices; i++)
		{
			pVertex = (PVRTVec3*)( (unsigned char*)(pVertex)+i32Stride);

			/* Minimum and Maximum X */
			if (pVertex->x < MinX) MinX = pVertex->x;
			if (pVertex->x > MaxX) MaxX = pVertex->x;

			/* Minimum and Maximum Y */
			if (pVertex->y < MinY) MinY = pVertex->y;
			if (pVertex->y > MaxY) MaxY = pVertex->y;

			/* Minimum and Maximum Z */
			if (pVertex->z < MinZ) MinZ = pVertex->z;
			if (pVertex->z > MaxZ) MaxZ = pVertex->z;
		}

		/* Assign the resulting extremas to the bounding box structure */
		/* Point 0 */
		vPoint[0].x=MinX;
		vPoint[0].y=MinY;
		vPoint[0].z=MinZ;

		/* Point 1 */
		vPoint[1].x=MinX;
		vPoint[1].y=MinY;
		vPoint[1].z=MaxZ;

		/* Point 2 */
		vPoint[2].x=MinX;
		vPoint[2].y=MaxY;
		vPoint[2].z=MinZ;

		/* Point 3 */
		vPoint[3].x=MinX;
		vPoint[3].y=MaxY;
		vPoint[3].z=MaxZ;

		/* Point 4 */
		vPoint[4].x=MaxX;
		vPoint[4].y=MinY;
		vPoint[4].z=MinZ;

		/* Point 5 */
		vPoint[5].x=MaxX;
		vPoint[5].y=MinY;
		vPoint[5].z=MaxZ;

		/* Point 6 */
		vPoint[6].x=MaxX;
		vPoint[6].y=MaxY;
		vPoint[6].z=MinZ;

		/* Point 7 */
		vPoint[7].x=MaxX;
		vPoint[7].y=MaxY;
		vPoint[7].z=MaxZ;
	}


}

/*****************************************************************************
End of file (BoundingBox.cpp)
*****************************************************************************/
