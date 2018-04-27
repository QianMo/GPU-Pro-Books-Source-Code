/******************************************************************************

 @File         BoundingBox.h

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     ANSI compatible

 @Description  Struct to describe a 3D bounding box. Can be generated from POD
               data.

******************************************************************************/
#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

/****************************************************************************
** Includes
****************************************************************************/
#include "../PVRTools.h"


namespace pvrengine
{
	/*!***************************************************************************
	 * @Struct BoundingBox
	 * @Brief Describes a 3D bounding box
 	 * @Description Describes a 3D bounding box
 	 *****************************************************************************/
	struct BoundingBox
	{
	/****************************************************************************
	** Variables
	****************************************************************************/
		// the actual bounding points
		PVRTVec3 vPoint[8];

	/*!***************************************************************************
	@Function			BoundingBox
	@Description		Blank constructor.
	*****************************************************************************/
		BoundingBox(){};

	/*!***************************************************************************
	@Function			BoundingBox
	@Input				vCorners	a pointer to an array of 8 corner points
	@Description		Simple constructor to create BoundingBox directly from
						precalculated points.
	*****************************************************************************/
		BoundingBox(const PVRTVec3* const vCorners)
		{
			for(int i=0;i<8;++i)
			{
				vPoint[i]=vCorners[i];
			}
		}

	/*!***************************************************************************
	@Function			BoundingBox
	@Input				other - another bounding box
	@Description		copy constructor.
	*****************************************************************************/
		BoundingBox(const BoundingBox& other)
		{
			for(int i=0;i<8;++i)
			{
				vPoint[i]=other.vPoint[i];
			}
		}


	/*!***************************************************************************
	@Function			BoundingBox
	@Input				pV array of vertices
	@Input				i32NumberOfVertices size of array pV
	@Description		Constructor that calculates the eight vertices that
						surround an object.
						This function should only be called once to determine the
						object's bounding box.
	*****************************************************************************/
	BoundingBox(const PVRTVec3* const pV,
						const int i32NumberOfVertices);

	/*!***************************************************************************
	@Function			BoundingBox
	@Input				pV array of vertices
	@Input				i32NumberOfVertices size of array pV
	@Input				i32Offset the offset of the array
	@Input				i32Stride the stride of the array
	@Description		Calculate the eight vertices that surround an object.
						This function should only be called once to determine the
						object's bounding box.
						Takes interleaved data using the first vertex's offset
						and the stride to the next vertex thereafter
	*****************************************************************************/
	BoundingBox(const unsigned char	* const pV,
		const int			i32NumberOfVertices,
		const int			i32Offset,
		const int			i32Stride);

	};
}

#endif // BOUNDINGBOX_H

/*****************************************************************************
 End of file (BoundingBox.h)
*****************************************************************************/
