///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-04-03
// Updated : 2008-09-17
// Licence : This source is under MIT License
// File    : glm/gtx/mixed_product.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_mixed_product
#define glm_gtx_mixed_product

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace test{
		void main_gtx_matrix_selection();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_mixed_product extension: Mixed product of 3 vectors.
	namespace mixed_product
	{
		//! \brief Mixed product of 3 vectors (from GLM_GTX_mixed_product extension)
		template <typename valType> 
		valType mixedProduct(
			detail::tvec3<valType> const & v1, 
			detail::tvec3<valType> const & v2, 
			detail::tvec3<valType> const & v3);
	}//namespace mixed_product
	}//namespace gtx
}//namespace glm

#define GLM_GTX_mixed_product namespace gtx::mixed_product
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_mixed_product;}
#endif//GLM_GTX_GLOBAL

#include "mixed_product.inl"

#endif//glm_gtx_mixed_product
