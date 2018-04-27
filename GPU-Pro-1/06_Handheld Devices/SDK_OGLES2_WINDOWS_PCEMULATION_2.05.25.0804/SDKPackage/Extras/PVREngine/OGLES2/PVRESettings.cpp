/******************************************************************************

 @File         PVRESettings.cpp

 @Title        

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     OGLES2 implementation of PVRESettings

 @Description  Settings class for the PVREngine

******************************************************************************/
#include "PVRESettings.h"
#include "ContextManager.h"

namespace pvrengine
{


	/******************************************************************************/

	void PVRESettings::Init()
	{
		ContextManager::inst().initContext();
		glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
		//glDepthFunc(GL_GEQUAL);
		//glClearDepth(0);
	}

	/******************************************************************************/

	void PVRESettings::setBlend(const bool bBlend)
	{
		if(bBlend)
			glEnable(GL_BLEND);
		else
			glDisable(GL_BLEND);
	}

	/******************************************************************************/

void PVRESettings::setBackColour(const unsigned int cBackColour)
{
	glClearColor((float)(cBackColour>>24&0xff)/255.0f,
		(float)(cBackColour>>16&0xff)/255.0f,
		(float)(cBackColour>>8&0xff)/255.0f,
		(float)(cBackColour&0xff)/255.0f);
}

	/******************************************************************************/

	void PVRESettings::setBackColour(const VERTTYPE fRed, const VERTTYPE fGreen, const VERTTYPE fBlue, const VERTTYPE fAlpha)
{
	glClearColor(fRed, fGreen, fBlue, fAlpha);
}
	/******************************************************************************/

	void PVRESettings::setBackColour(const VERTTYPE fRed, const VERTTYPE fGreen, const VERTTYPE fBlue)
{
	glClearColor(fRed, fGreen, fBlue, 1.0f);
}

	/******************************************************************************/

void PVRESettings::setBackColour(const unsigned int u32Red,
								 const unsigned int u32Green, 
								 const unsigned int u32Blue,
								 const unsigned int u32Alpha)
{
	glClearColor((float(u32Red))/255.0f,(float(u32Green))/255.0f,(float(u32Blue))/255.0f,(float(u32Alpha))/255.0f);
}

	/******************************************************************************/

void PVRESettings::setBackColour(const unsigned int u32Red,
								 const unsigned int u32Green, 
								 const unsigned int u32Blue)
{
	glClearColor((float(u32Red))/255.0f,(float(u32Green))/255.0f,(float(u32Blue))/255.0f,1.0f);
}

	/******************************************************************************/

void PVRESettings::setClearFlags(unsigned int u32ClearFlags)
{
	m_u32ClearFlags = u32ClearFlags;
}

	/******************************************************************************/

unsigned int PVRESettings::getClearFlags()
{
	return m_u32ClearFlags;
}

	/******************************************************************************/

void PVRESettings::Clear()
{
	glClear(m_u32ClearFlags);
}

	/******************************************************************************/

void PVRESettings::setDepthTest(const bool bDepth)
{
	if(bDepth)// Enables depth test using the z-buffer
		glEnable(GL_DEPTH_TEST);
	else
		glDisable(GL_DEPTH_TEST);
}

	/******************************************************************************/

void PVRESettings::setDepthWrite(const bool bDepthWrite)
{
	glDepthMask(bDepthWrite);
}

	/******************************************************************************/

void PVRESettings::setCull(bool bCull)
{
	if(bCull)
		glEnable(GL_CULL_FACE);
	else
		glDisable(GL_CULL_FACE);
}

	/******************************************************************************/

void PVRESettings::setCullMode(unsigned int eMode)
{
	glCullFace(eMode);
}

	/******************************************************************************/

CPVRTString PVRESettings::getAPIName()
{
	return CPVRTString("OpenGL|ES2");	// yes i know
}



}

/******************************************************************************
End of file (PVRESettings.cpp)
******************************************************************************/
