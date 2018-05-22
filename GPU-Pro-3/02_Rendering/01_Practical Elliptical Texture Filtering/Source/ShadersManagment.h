/**
*	EWA filtering on the GPU
*	(original version by Cyril Crassin, adapted by Pavlos Mavridis)
*/

#ifndef SHADERSMANAGMENT_H
#define SHADERSMANAGMENT_H

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// GLEW header for OpenGL extentions loading
#include "GL/glew.h"
#include <string>

std::string loadTextFile(const char *name);
GLuint createShader(const char *fileName, GLuint shaderType, GLuint shaderID=0);
void checkProgramInfos(GLuint programID, GLuint stat);
GLuint createShaderProgram(const char *fileNameVS, const char *fileNameFS, GLuint programID=0);

void setShadersGlobalMacro(const char *macro, int val);
void setShadersGlobalMacro(const char *macro, float val);
void resetShadersGlobalMacros();

#endif
