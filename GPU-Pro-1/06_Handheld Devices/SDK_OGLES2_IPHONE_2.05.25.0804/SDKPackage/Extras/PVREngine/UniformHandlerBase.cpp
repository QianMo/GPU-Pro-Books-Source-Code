/******************************************************************************

 @File         UniformHandlerBase.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OGLES2 implementation of PVRESemanticHandler

 @Description  Shows how to use the pfx format

******************************************************************************/
#include "PVRTools.h"
#include "UniformHandler.h"
#include "ConsoleLog.h"

namespace pvrengine
{
	

	/******************************************************************************/

	void UniformHandler::setScene(CPVRTModelPOD *psScene)
	{
		m_psScene = psScene;
	}

	/******************************************************************************/

	void UniformHandler::setFrame(const float fFrame)
	{
		m_fFrame = fFrame;
	}

	/******************************************************************************/

	bool UniformHandler::isVisibleSphere(const PVRTVec3& v3Centre, const VERTTYPE fRadius)
	{
		// get in view space
		PVRTVec4 v4TransCentre = m_mWorldView * PVRTVec4(v3Centre,f2vt(1.0f));

		// find clip space coord for centre
		v4TransCentre = m_mProjection * v4TransCentre;

		VERTTYPE fRadX,fRadY;
		// scale radius according to perspective
		if(m_bRotate)
		{
		fRadX = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(0,1)));
		fRadY = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(1,0)));
		}
		else
		{
		fRadX = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(0,0)));
		fRadY = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(1,1)));
		}
		VERTTYPE fRadZ = PVRTABS(VERTTYPEMUL(fRadius,m_mProjection(2,2)));


		// check if inside frustums
		// X
		if(v4TransCentre.x+fRadX<-v4TransCentre.w)
		{	// 'right' side out to 'left' def out
			return false;
		}
		if(v4TransCentre.x-fRadX>v4TransCentre.w)
		{	// 'left' side out to 'right' def out
			return false;
		}

		// Y
		if(v4TransCentre.y+fRadY<-v4TransCentre.w)
		{	// 'up' side out to 'top' def out
			return false;
		}
		if(v4TransCentre.y-fRadY>v4TransCentre.w)
		{	// 'down' side out to 'bottom' def out
			return false;
		}

		// Z
		if(v4TransCentre.z+fRadZ<-v4TransCentre.w)
		{	// 'far' side out to 'back' def out
			return false;
		}
		if(v4TransCentre.z-fRadZ>v4TransCentre.w)
		{	// 'near' side out to 'front' def out
			return false;
		}

		return true;
	}

}

/******************************************************************************
End of file (UniformHandlerAPI.cpp)
******************************************************************************/
