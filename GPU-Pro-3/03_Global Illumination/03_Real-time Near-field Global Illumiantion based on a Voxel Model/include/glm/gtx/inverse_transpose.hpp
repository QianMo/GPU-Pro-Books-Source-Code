///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2006-01-04
// Updated : 2008-10-23
// Licence : This source is under MIT License
// File    : glm/gtx/inverse_transpose.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_inverse_transpose
#define glm_gtx_inverse_transpose

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace gtx{
	//! GLM_GTX_inverse_transpose extension: Inverse transpose matrix functions
	namespace inverse_transpose
	{
		//! Compute the inverse transpose of a matrix.
		//! From GLM_GTX_inverse extension.
		template <typename genType> 
		inline typename genType::value_type inverseTranspose(
			genType const & m);

	}//namespace inverse_transpose
	}//namespace gtx
}//namespace glm

#define GLM_GTX_inverse_transpose namespace gtx::inverse_transpose
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_inverse_transpose;}
#endif//GLM_GTX_GLOBAL

#include "inverse_transpose.inl"

#endif//glm_gtx_inverse_transpose
