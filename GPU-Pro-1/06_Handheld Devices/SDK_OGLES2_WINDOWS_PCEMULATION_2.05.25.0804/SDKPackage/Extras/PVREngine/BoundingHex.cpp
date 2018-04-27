/******************************************************************************

 @File         BoundingHex.cpp

 @Title        Bounding Hexahedron

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  More convenient class for holding irregular bounding volumes
               Stores 6 planes = 6 points + 6 normals to describe a bounding area

******************************************************************************/
#include <string.h>
#include "BoundingHex.h"

namespace pvrengine
{
	/******************************************************************************/

	BoundingHex::BoundingHex()
	{
		memset(this,0,sizeof(*this));
	}

	/******************************************************************************/

	BoundingHex::BoundingHex(const BoundingBox& sBoundingBox)
	{
		m_pPlanes[0] = Plane(sBoundingBox.vPoint[0],sBoundingBox.vPoint[1],sBoundingBox.vPoint[2]);
		m_pPlanes[1] = Plane(sBoundingBox.vPoint[1],sBoundingBox.vPoint[3],sBoundingBox.vPoint[5]);
		m_pPlanes[2] = Plane(sBoundingBox.vPoint[5],sBoundingBox.vPoint[4],sBoundingBox.vPoint[7]);
		m_pPlanes[3] = Plane(sBoundingBox.vPoint[4],sBoundingBox.vPoint[0],sBoundingBox.vPoint[6]);
		m_pPlanes[4] = Plane(sBoundingBox.vPoint[0],sBoundingBox.vPoint[4],sBoundingBox.vPoint[1]);
		m_pPlanes[5] = Plane(sBoundingBox.vPoint[3],sBoundingBox.vPoint[7],sBoundingBox.vPoint[2]);
	}
}

/******************************************************************************
End of file (BoundingHex.cpp)
******************************************************************************/
