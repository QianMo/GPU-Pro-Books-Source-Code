/******************************************************************************

 @File         LightBase.cpp

 @Title        A Light

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Class to hold a Light with some convenient
               constructors/functions/operators

******************************************************************************/
#include "Light.h"

namespace pvrengine
{
	

	/******************************************************************************/

	LightDirectional::LightDirectional(const PVRTVec3 v3Direction, const PVRTVec3 v3Colour)
	{
		m_v3Colour = v3Colour;
		m_v4Direction = PVRTVec4(v3Direction,f2vt(0.0f));
		m_eLightType = eLight_Directional;
	}

	/******************************************************************************/

	LightDirectional::LightDirectional(const PVRTVec4 v4Direction, const PVRTVec3 v3Colour)
	{
		m_v3Colour = v3Colour;
		m_v4Direction = v4Direction;
		m_eLightType = eLight_Directional;
	}

	/******************************************************************************/

	void LightDirectional::setDirection(const PVRTVec3 v3Direction)
	{
		m_v4Direction.x = v3Direction.x;
		m_v4Direction.y = v3Direction.y;
		m_v4Direction.z = v3Direction.z;
		m_v4Direction.w = f2vt(0.0f);
	}

	/******************************************************************************/

	void LightDirectional::setDirection(const PVRTVec4 v4Direction)
	{
		m_v4Direction = v4Direction;
	}

	/******************************************************************************/

	LightPoint::LightPoint(const PVRTVec3 v3Position, const PVRTVec3 v3Colour)
	{
		m_v3Colour = v3Colour;
		m_v4Position = PVRTVec4(v3Position,f2vt(1.0f));
		m_eLightType = eLight_Point;
	}

	/******************************************************************************/

	LightPoint::LightPoint(const PVRTVec4 v4Position,const PVRTVec3 v3Colour)
	{
		m_v3Colour = v3Colour;
		m_v4Position = v4Position;
		m_eLightType = eLight_Point;
	}

	/******************************************************************************/

	void LightPoint::setPosition(const PVRTVec4 v4Position)
	{
		m_v4Position = v4Position;
	}

	/******************************************************************************/

	void LightPoint::setPosition(const PVRTVec3 v3Position)
	{
		m_v4Position.x = v3Position.x;
		m_v4Position.y = v3Position.y;
		m_v4Position.z = v3Position.z;
		m_v4Position.w = f2vt(1.0f);
	}
}

/******************************************************************************
End of file (Light.cpp)
******************************************************************************/
