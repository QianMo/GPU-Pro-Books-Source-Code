/******************************************************************************

 @File         BoundingHex.h

 @Title        Bounding Hexahedron

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  More convenient struct for holding irregular bounding volumes
               Stores 6 planes = 6 points + 6 normals to describe a bounding area

******************************************************************************/
#ifndef BOUNDINGHEX_H
#define BOUNDINGHEX_H

/******************************************************************************
Includes
******************************************************************************/

#include "Plane.h"
#include "BoundingBox.h"

namespace pvrengine
{
	/*!***************************************************************************
	* @Struct BoundingHex
	* @Brief Describes a more convenient struct for holding irregular bounding volumes
	* @Description Describes a 3D bounding box
	*****************************************************************************/
	struct BoundingHex
	{
		/****************************************************************************
		** Variables
		****************************************************************************/
		Plane		m_pPlanes[6];

		/*!***************************************************************************
		@Function			BoundingHex
		@Description		Blank constructor. Sets all values to 0.
		*****************************************************************************/
		BoundingHex();

		/*!***************************************************************************
		@Function			BoundingHex
		@Input				sBoundingBox	a bounding box struct
		@Description		Constructor to make a bounding hex from a bounding box.
		*****************************************************************************/
		BoundingHex(const BoundingBox& sBoundingBox);

	};



}
#endif // BOUNDINGHEX

/******************************************************************************
End of file (PVRES.h)
******************************************************************************/
