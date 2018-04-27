/******************************************************************************

 @File         PVRESettingsBase.cpp

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent/OGLES2

 @Description  API independent settings routines for PVREngine Implements
               functions from PVRESettings.h

******************************************************************************/
#include "PVRESettings.h"
#include "PVRTools.h"
#include "ContextManager.h"

namespace pvrengine
{
	

	/******************************************************************************/

	PVRESettings::PVRESettings()
	{
	}

	/******************************************************************************/

	PVRESettings::~PVRESettings()
	{
	}

	/******************************************************************************/

	bool PVRESettings::InitPrint3D(CPVRTPrint3D& print3d,
		const unsigned int u32Width,
		const unsigned int u32Height,
		const bool bRotate)
	{
		return print3d.SetTextures(ContextManager::inst().getCurrentContext(),u32Width,u32Height,bRotate) == PVR_SUCCESS;
	}

}

/******************************************************************************
End of file (PVRESettingsBase.cpp)
******************************************************************************/
