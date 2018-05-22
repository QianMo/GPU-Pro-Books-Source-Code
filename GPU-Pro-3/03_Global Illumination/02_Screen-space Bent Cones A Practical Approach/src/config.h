#pragma once

#define _USE_MATH_DEFINES

#include <cstddef>

#define BUILD_NVIDIA
#define USE_SAMPLER

namespace vertexAttrib {
    enum type {
        POSITION = 0,
        NORMAL	 = 1,
        VIEWDIR  = 1,
        TEXCOORD = 2,
        DIFFUSE_COLOR = 3,
    };
}

#define GLH_BUFFER_OFFSET(i) ((char *)NULL + (i))

// check for opengl error conditions
// you have to include gl.h and glu.h separately!
#ifdef _DEBUG

#include <iostream>

#define GLCHECK(x)  { x; \
{ \
    GLenum err = glGetError(); \
    if (err!=GL_NO_ERROR) { \
    std::cerr << "GlError in " << __FUNCTION__ << "()@" << __FILE__ << ":" << __LINE__ << " (" << #x << ") : " << gluErrorString(err) << std::endl; \
    __asm { int 3 } \
    } \
} }

#define ASSERTMSG(x, msg) if (!(x)) { \
    std::cerr << "\"" << #x << "\" failed in " << __FUNCTION__ << "()@" << __FILE__ << ":" << __LINE__ << " : " << msg << std::endl; \
    __asm { int 3 } \
}

#define ASSERT(x) if (!(x)) { \
    std::cerr << "\"" << #x << "\" failed in " << __FUNCTION__ << "()@" << __FILE__ << ":" << __LINE__ << " : " << std::endl; \
    __asm { int 3 } \
}

#else

#define GLCHECK(x)  x

#define ASSERTMSG(x, msg) ((void)0)

#define ASSERT(x) ((void)0)

#endif

#ifdef BUILD_NVIDIA
#define SAMPLERUNIT(x) x
#else
#define SAMPLERUNIT(x) GL_TEXTURE0 + x
#endif
