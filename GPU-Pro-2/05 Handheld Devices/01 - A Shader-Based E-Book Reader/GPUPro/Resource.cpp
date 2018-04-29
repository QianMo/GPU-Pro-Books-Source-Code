/******************************************************************************

 @File         Resource.cpp

 @Title        EBook demo

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  
 
******************************************************************************/

#include "Resource.h"

#include "PVRTShader.h"

bool LoadShader(const char *vertShader, const char *fragShader,
				const char** const pszAttribs, GLuint uiNumAttribs,
				const char* const* aszDefineArray, GLuint uiDefArraySize,
				GLuint &uiVS, GLuint &uiFS, GLuint &uiID)
{
	CPVRTString errorStr;

	if(PVRTShaderLoadFromFile(0, vertShader, GL_VERTEX_SHADER, 0,
		&uiVS, &errorStr, 0, aszDefineArray, uiDefArraySize) != PVR_SUCCESS)
	{
		return false;
	}

	if(PVRTShaderLoadFromFile(0, fragShader, GL_FRAGMENT_SHADER, 0,
		&uiFS, &errorStr, 0, aszDefineArray, uiDefArraySize) != PVR_SUCCESS)
	{
		return false;
	}

	if(PVRTCreateProgram(
			&uiID, uiVS, uiFS, pszAttribs, uiNumAttribs, &errorStr) != PVR_SUCCESS)
	{
		return false;
	}
	return true;
}

bool LoadTexture(const char * const filename, GLuint * const texName)
{
	if (PVRTTextureLoadFromPVR(filename, texName) != PVR_SUCCESS)
	{
		return false;
	}
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

	return true;
}


int IMod(int numerator, int denominator)
{
	int res = (int)fmod((float)numerator, (float)denominator);
	if (res < 0)
		res += abs(denominator);
	return res;
}

bool LoadPOD(CPVRTModelPOD &scene, const char * const filename)
{
	return scene.ReadFromFile(filename) == PVR_SUCCESS;
}

int FindNodeIndex(CPVRTModelPOD &scene, const char * const name)
{
	for (unsigned int m = scene.nNumMeshNode; m--; )
	{
		SPODNode& Node = scene.pNode[m];
		if (!strcmp(Node.pszName, name))
		{
			return m;
		}
	}
	return -1;
}