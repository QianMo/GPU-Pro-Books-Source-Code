/******************************************************************************

 @File         Plane.cpp

 @Title        A Plane

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Struct to hold a plane with some convenient
               constructors/functions/operators

******************************************************************************/
#include "Plane.h"
#include <math.h>


namespace pvrengine
{
	

	/******************************************************************************/

	Plane::Plane(const PVRTVec3& A,const PVRTVec3& B,const PVRTVec3& C)
	{
		fA = VERTTYPEMUL(A.y,(B.z-C.z)) + VERTTYPEMUL(B.y,(C.z-A.z)) + VERTTYPEMUL(C.y,(A.z-B.z));
		fB = VERTTYPEMUL(A.z,(B.x-C.x)) + VERTTYPEMUL(B.z,(C.x-A.x)) + VERTTYPEMUL(C.z,(A.x-B.x));
		fC = VERTTYPEMUL(A.x,(B.y-C.y)) + VERTTYPEMUL(B.x,(C.y-A.y)) + VERTTYPEMUL(C.x,(A.y-B.y));

		// calculate distance from origin aka d
		// d = -(|n|.A) = -nx.Ax-ny.Ay-nz.Az
		fD = - VERTTYPEMUL(A.x,(VERTTYPEMUL(B.y,C.z)-VERTTYPEMUL(B.z,C.y)))
			+ VERTTYPEMUL(A.y,(VERTTYPEMUL(B.x,C.z)-VERTTYPEMUL(B.z,C.x)))
			- VERTTYPEMUL(A.z,(VERTTYPEMUL(B.x,C.y)-VERTTYPEMUL(B.y,C.x)));

		// normalize
		normalize();
	}

	/******************************************************************************/

	void Plane::normalize()
	{
		VERTTYPE fMag = VERTTYPEDIV(1.0f,f2vt(sqrt(vt2f(VERTTYPEMUL(fA,fA)+VERTTYPEMUL(fB,fB)+VERTTYPEMUL(fC,fC)))));

		fA = VERTTYPEMUL(fA,fMag);
		fB = VERTTYPEMUL(fB,fMag);
		fC = VERTTYPEMUL(fC,fMag);
		fD = VERTTYPEMUL(fD,fMag);
	}


	/******************************************************************************/

	VERTTYPE Plane::whichSideIs(const PVRTVec3& vec) const
	{
		return VERTTYPEMUL(fA,vec.x)+ VERTTYPEMUL(fB,vec.y)+VERTTYPEMUL(fC,vec.z)+fD;	// take away???
	}

}

/******************************************************************************
End of file (Plane.cpp)
******************************************************************************/
