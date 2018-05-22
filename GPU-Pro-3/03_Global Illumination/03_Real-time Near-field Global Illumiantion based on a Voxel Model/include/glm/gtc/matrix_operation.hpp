///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-04-29
// Updated : 2009-04-29
// Licence : This source is under MIT License
// File    : glm/gtc/matrix_operation.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtc_matrix_operation
#define glm_gtc_matrix_operation

// Dependency:
#include "../glm.hpp"

namespace glm{
namespace gtc{
//! GLM_GTC_matrix_operation extension: Matrix operation functions
namespace matrix_operation
{

}//namespace matrix_operation
}//namespace gtc
}//namespace glm

#define GLM_GTC_matrix_operation namespace gtc::matrix_operation
#ifndef GLM_GTC_GLOBAL
namespace glm {using GLM_GTC_matrix_operation;}
#endif//GLM_GTC_GLOBAL

#include "matrix_operation.inl"

#endif//glm_gtc_matrix_operation
