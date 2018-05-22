#ifndef SRC_LIB_GL_ERROR
#define SRC_LIB_GL_ERROR

#include <iostream>
#include <cassert>
#include <string>

#define WIN32_LEAN_AND_MEAN 
#define VC_EXTRALEAN

#include "OpenGL.h"

void checkOpenGLError(char* file, int line);

std::string checkFramebufferStatus();


#if defined(DEBUG) || defined(_DEBUG)
#ifndef V
#define V(x){ (x); checkOpenGLError(__FILE__, __LINE__); }			
#endif
#else
#ifndef V
#define V(x)           { (x); }
#endif
#endif


#endif
