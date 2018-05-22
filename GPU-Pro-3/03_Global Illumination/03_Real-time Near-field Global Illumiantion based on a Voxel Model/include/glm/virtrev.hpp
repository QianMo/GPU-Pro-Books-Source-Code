///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2008 G-Truc Creation (www.g-truc.net)
// Virtrev SDK copyright matrem (matrem84.free.fr)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-05-29
// Updated : 2008-10-06
// Licence : This source is under MIT License
// File    : glm/virtrev.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Note:
// Virtrev SDK extensions
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_virtrev
#define glm_virtrev

#define GLM_VIRTREV_GLOBAL 1

#if(defined(GLM_DEPENDENCE) && (				\
	(GLM_DEPENDENCE & GLM_DEPENDENCE_GLEW) ||	\
	(GLM_DEPENDENCE & GLM_DEPENDENCE_GLEE) ||	\
	(GLM_DEPENDENCE & GLM_DEPENDENCE_GL)))
#include "./virtrev/gl.hpp"
#endif
#include "./virtrev/address.hpp"
#include "./virtrev/equal_operator.hpp"
#include "./virtrev/xstream.hpp"

#endif//glm_virtrev
