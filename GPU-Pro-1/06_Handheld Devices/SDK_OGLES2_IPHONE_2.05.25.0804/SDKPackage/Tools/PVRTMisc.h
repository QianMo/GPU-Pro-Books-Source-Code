/******************************************************************************

 @File         PVRTMisc.h

 @Title        PVRTMisc

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Miscellaneous functions used in 3D rendering.

******************************************************************************/
#ifndef _PVRTMISC_H_
#define _PVRTMISC_H_

#include "PVRTMatrix.h"
#include "PVRTFixedPoint.h"

/****************************************************************************
** Functions
****************************************************************************/

/*!***************************************************************************
 @Function			PVRTMiscCalculateIntersectionLinePlane
 @Input				pfPlane			Length 4 [A,B,C,D], values for plane
									equation
 @Input				pv0				A point on the line
 @Input				pv1				Another point on the line
 @Output			pvIntersection	The point of intersection
 @Description		Calculates coords of the intersection of a line and an
					infinite plane
*****************************************************************************/
void PVRTMiscCalculateIntersectionLinePlane(
	PVRTVECTOR3			* const pvIntersection,
	const VERTTYPE		pfPlane[4],
	const PVRTVECTOR3	* const pv0,
	const PVRTVECTOR3	* const pv1);

/*!***************************************************************************
 @Function		PVRTMiscCalculateInfinitePlane
 @Input			nStride			Size of each vertex structure containing pfVtx
 @Input			pvPlane			Length 4 [A,B,C,D], values for plane equation
 @Input			pmViewProjInv	The inverse of the View Projection matrix
 @Input			pFrom			Position of the camera
 @Input			fFar			Far clipping distance
 @Output		pfVtx			Position of the first of 3 floats to receive
								the position of vertex 0; up to 5 vertex positions
								will be written (5 is the maximum number of vertices
								required to draw an infinite polygon clipped to screen
								and far clip plane).
 @Returns		Number of vertices in the polygon fan (Can be 0, 3, 4 or 5)
 @Description	Calculates world-space coords of a screen-filling
				representation of an infinite plane The resulting vertices run
				counter-clockwise around the screen, and can be simply drawn using
				non-indexed TRIANGLEFAN
*****************************************************************************/
int PVRTMiscCalculateInfinitePlane(
	VERTTYPE			* const pfVtx,
	const int			nStride,
	const PVRTVECTOR4	* const pvPlane,
	const PVRTMATRIX 	* const pmViewProjInv,
	const PVRTVECTOR3	* const pFrom,
	const VERTTYPE		fFar);

/*!***************************************************************************
 @Function		PVRTCreateSkybox
 @Input			scale			Scale the skybox
 @Input			adjustUV		Adjust or not UVs for PVRT compression
 @Input			textureSize		Texture size in pixels
 @Output		Vertices		Array of vertices
 @Output		UVs				Array of UVs
 @Description	Creates the vertices and texture coordinates for a skybox
*****************************************************************************/
void PVRTCreateSkybox(float scale, bool adjustUV, int textureSize, VERTTYPE** Vertices, VERTTYPE** UVs);

/*!***************************************************************************
 @Function		PVRTDestroySkybox
 @Input			Vertices	Vertices array to destroy
 @Input			UVs			UVs array to destroy
 @Description	Destroy the memory allocated for a skybox
*****************************************************************************/
void PVRTDestroySkybox(VERTTYPE* Vertices, VERTTYPE* UVs);


#endif /* _PVRTMISC_H_ */


/*****************************************************************************
 End of file (PVRTMisc.h)
*****************************************************************************/
