/**
*	EWA filtering on the GPU
*	Copyright 2010-2011 Pavlos Mavridis, All rights reserved
*/

#ifndef		_OVERLAY_H_
#define		_OVERLAY_H_

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>   

#endif //_WIN32
#include <GL/gl.h>
#include "tga.h"

class Overlay_s{
	Texture		font;
	GLuint		base;
	float		width,height;

	

public:
	Overlay_s(float w,float h);
	void BuildFont(void);
	GLvoid KillFont(GLvoid);
	
	GLvoid Print(GLfloat x, GLfloat y, int set, const char *txt,float alpha=1);
	void Begin(void);
	void End(void);
};

#endif //_OVERLAY_H_
