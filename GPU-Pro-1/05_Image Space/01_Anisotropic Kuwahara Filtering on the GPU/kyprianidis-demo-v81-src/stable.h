#ifndef INCLUDED_STABLE_H
#define INCLUDED_STABLE_H

#ifdef __cplusplus

#ifdef WIN32
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#endif

#include <cassert>
#include "GLee.h"
#include <QtGui/QtGui>
#include <QtOpenGL/QtOpenGL>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

extern float perlin_original_noise3( float x, float y, float z );

#endif
#endif
