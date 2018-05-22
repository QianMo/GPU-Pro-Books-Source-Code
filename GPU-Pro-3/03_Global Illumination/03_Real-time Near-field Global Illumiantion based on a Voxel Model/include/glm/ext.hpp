///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-01
// Updated : 2009-05-01
// Licence : This source is under MIT License
// File    : glm/ext.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_ext
#define glm_ext

#include "glm.hpp"
#include "gtc.hpp"
#include "gtx.hpp"
#include "img.hpp"

//const float goldenRatio = 1.618033988749894848f;
//const float pi = 3.141592653589793238f;

#if(defined(GLM_MESSAGE) && (GLM_MESSAGE & (GLM_MESSAGE_EXTS | GLM_MESSAGE_NOTIFICATION)))
#	pragma message("GLM message: Extensions library included")
#endif//GLM_MESSAGE

#define GLM_EXTENSION(extension) namespace glm{using extension;}

#endif //glm_ext
