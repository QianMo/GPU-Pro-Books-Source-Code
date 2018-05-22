///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-21
// Updated : 2006-11-13
// Licence : This source is under MIT License
// File    : glm/gtx/matrix_cross_product.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_matrix_cross_product
#define glm_gtx_matrix_cross_product

// Dependency:
#include "../glm.hpp"

namespace glm
{
   	namespace test{
		void main_gtx_matrix_cross_product();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_matrix_cross_product: Build cross product matrices
	namespace matrix_cross_product
	{
		//! Build a cross product matrix.
		//! From GLM_GTX_matrix_cross_product extension.
		template <typename T> 
		detail::tmat3x3<T> matrixCross3(
			detail::tvec3<T> const & x);
		
		//! Build a cross product matrix.
		//! From GLM_GTX_matrix_cross_product extension.
		template <typename T> 
		detail::tmat4x4<T> matrixCross4(
			detail::tvec3<T> const & x);

	}//namespace matrix_cross_product
	}//namespace gtx
}//namespace glm

#define GLM_GTX_matrix_cross_product namespace gtx::matrix_cross_product
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_matrix_cross_product;}
#endif//GLM_GTX_GLOBAL

#include "matrix_cross_product.inl"

#endif//glm_gtx_matrix_cross_product
