/******************************************************************************

 @File         Resource.h

 @Title        EBook demo

 @Copyright    Copyright (C) 2010 by Imagination Technologies Limited.

 @Platform     Independent

 @Description  
 
******************************************************************************/

#ifndef _RESOURCE_H_
#define _RESOURCE_H_

#include "OGLES2Tools.h"

bool LoadShader(const char *vertShader, const char *fragShader,
				const char** const pszAttribs, GLuint uiNumAttribs,
				const char* const* aszDefineArray, GLuint uiDefArraySize,
				GLuint &uiVS, GLuint &uiFS, GLuint &uiID);

bool LoadTexture(const char * const filename, GLuint * const texName);

bool LoadPOD(CPVRTModelPOD &scene, const char * const filename);
int FindNodeIndex(CPVRTModelPOD &scene, const char * const name);

int IMod(int numerator, int denominator);

inline float Inertia(float t, float tau)
{
	if (t <= 0.001f)
		return 0.0f;

	return (float)exp(-tau * t);
}

inline float Exp(float time, float amplitude, float tau)
{
	if (time < -0.001f)
		return 0.0f;

	return amplitude * (1.0f - (float)exp(-tau * time));
}


#endif 
