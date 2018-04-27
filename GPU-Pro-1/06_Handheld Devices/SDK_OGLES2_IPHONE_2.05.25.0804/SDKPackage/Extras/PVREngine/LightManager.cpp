/******************************************************************************

 @File         LightManager.cpp

 @Title        Introducing the POD 3d file format

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Manages lights

******************************************************************************/
#include "LightManager.h"
#include "Light.h"
#include "Globals.h"


namespace pvrengine
{
	

	/******************************************************************************/

	LightManager::LightManager()
	{
	}

	/******************************************************************************/

	LightManager::~LightManager()
	{
		for(unsigned int i=0;i<m_daLights.getSize();++i)
		{
			PVRDELETE(m_daLights[i]);
		}
	}

	/******************************************************************************/

	LightManager::LightManager(int i32MaxTextures)
	{
		m_daLights.expandToSize(i32MaxTextures);
	}

	/******************************************************************************/

	unsigned int	LightManager::addLight(const CPVRTModelPOD& sScene, const unsigned int i32Index )
	{
		PVRTVec3 vColour(f2vt(1.0f),f2vt(1.0f),f2vt(1.0f));

		if(i32Index<sScene.nNumLight)
		{
			Light *pLight;
			if(sScene.pLight[i32Index].eType==ePODDirectional)
			{// make directional light

				pLight = new LightDirectional(sScene.GetLightDirection(i32Index),
					PVRTVec3((VERTTYPE*)&sScene.pLight[i32Index].pfColour));
			}
			else
			{// make point light
				pLight = new LightPoint(sScene.GetLightPosition(i32Index),
					PVRTVec3((VERTTYPE*)&sScene.pLight[i32Index].pfColour));
			}
			m_daLights.append(pLight);
			return m_daLights.getSize();
		}
		return PVR_INVALID;
	}

	/******************************************************************************/

	unsigned int	LightManager::addDirectionalLight(const PVRTVec3& vec3Direction,
		const PVRTVec3& vec3Colour)
	{
		Light *pLight = new LightDirectional(vec3Direction, vec3Colour);
		m_daLights.append(pLight);
		return m_daLights.getSize();
	}


	/******************************************************************************/

	void	LightManager::shineLights()
	{
		for(unsigned int i=0;i<m_daLights.getSize();i++)
		{
			m_daLights[i]->shineLight(i);
		}
	}


}

/******************************************************************************
End of file (LightManager.cpp)
******************************************************************************/
