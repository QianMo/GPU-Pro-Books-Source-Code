/******************************************************************************

 @File         Plane.h

 @Title        Plane

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Struct to hold a plane with some convenient
               constructors/functions/operators. Eqaution of plane is Ax+By+Cz+D=0

******************************************************************************/
#ifndef PLANE_H
#define PLANE_H

/******************************************************************************
Includes
******************************************************************************/

#include "../PVRTools.h"

namespace pvrengine
{
	/*!****************************************************************************
	Struct
	******************************************************************************/
	/*!***************************************************************************
	* @Struct Plane
	* @Brief Struct to hold a plane
	* @Description Struct to hold a plane
 	*****************************************************************************/
	struct Plane
	{
		VERTTYPE	fA,fB,fC,fD;	/*! the four actual coefficients of the plane */

		/*!***************************************************************************
		@Function			Plane
		@Description		blank constructor.
		*****************************************************************************/
		Plane(){}

		/*!***************************************************************************
		@Function			Plane
		@Input				A	coefficient value
		@Input				B	coefficient value
		@Input				C	coefficient value
		@Input				D	coefficient value
		@Description		basic constructor to initialise a Plane object directly
		from equation coefficients.
		*****************************************************************************/
		Plane(const VERTTYPE A, const VERTTYPE B, const VERTTYPE C, const VERTTYPE D)
		{	fA = A; fB = B; fC = C; fD = D;}

		/*!***************************************************************************
		@Function			Plane
		@Input				A	coefficient value
		@Input				B	coefficient value
		@Input				C	coefficient value
		@Description		constructor to initialise a Plane object from 3 points
		*****************************************************************************/
		Plane(const PVRTVec3& A,const PVRTVec3& B,const PVRTVec3& C);

		/*!***************************************************************************
		@Function			normalize
		@Description		normalizes the associated normal to the plane
		essential for the accuracy of some calculations.
		*****************************************************************************/
		void normalize();

		/*!***************************************************************************
		@Function			getNormal
		@Return				A PVRTVec3 containing the normal
		@Description		Retrieves the normal vector associated with this plane object.
		*****************************************************************************/
		PVRTVec3 getNormal() const
		{	return PVRTVec3(fA,fB,fC);	}

		/*!***************************************************************************
		@Function			whichSideIs
		@Input				vPoint	the point in question
		@Return			value indicating the relative position of the
		point to the plane
		@Description		returns a value from which it may be determined which
		side of the plane the passed point is on negative vs positive. if 0 is
		returned then the point is on the plane. It is probably wise to normalize
		the plane first.
		*****************************************************************************/
		VERTTYPE whichSideIs(const PVRTVec3& vPoint) const;

	};

}

#endif // PLANE_H

/******************************************************************************
End of file (Plane.h)
******************************************************************************/
