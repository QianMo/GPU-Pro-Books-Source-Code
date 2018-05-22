/**
*	EWA filtering on the GPU
*	Copyright 2010-2011 Pavlos Mavridis, All rights reserved
*/

#ifndef _TGA_H_
#define _TGA_H_

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN 1
#include <windows.h>   

#endif //_WIN32

#include <GL/gl.h>	
#include <GL/glu.h>	
#include <stdio.h>
#include <stdlib.h>

class Texture{
	GLubyte	*imageData;										
	GLuint	bpp;										
	GLuint	width;										
	GLuint	height;										
	GLuint	texID;
public:
	
	void Bind(){
		glBindTexture(GL_TEXTURE_2D,texID);
	}
	bool LoadTGA(char *filename,GLenum MINF=GL_LINEAR,GLenum MAGF=GL_LINEAR,int mipmap=1);

};	

#endif	//_TGA_H_
