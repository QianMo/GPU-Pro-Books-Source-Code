/******************************************************************************

 @File         Texture.cpp

 @Title        Texture

 @Copyright    Copyright (C)  Imagination Technologies Limited.

 @Platform     Independent

 @Description  Texture container for OGLES2

******************************************************************************/
#include "../PVRTools.h"
#include "Texture.h"
#include "ConsoleLog.h"

namespace pvrengine
{

	

	/******************************************************************************/

	Texture::Texture(const CPVRTString& strFilename)
	{
		PVR_Texture_Header sOldHeader;
		if(PVRTTextureLoadFromPVR(strFilename.c_str(),(GLuint*)&m_u32Handle,&sOldHeader) == PVR_SUCCESS)
		{
			m_strFilename = strFilename;
			m_bIsCubeMap = (sOldHeader.dwpfFlags&PVRTEX_CUBEMAP)!=0;
		}
		else
		{
			ConsoleLog::inst().log("PVREngine::Texture: Could not load texture.\n");
			return;
		}
	}

}

/******************************************************************************
End of file (Texture.cpp)
******************************************************************************/
