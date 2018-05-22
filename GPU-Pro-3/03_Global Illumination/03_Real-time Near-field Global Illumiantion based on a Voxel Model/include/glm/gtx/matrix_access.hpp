///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-27
// Updated : 2006-12-06
// Licence : This source is under MIT License
// File    : glm/gtx/matrix_access.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_matrix_access
#define glm_gtx_matrix_access

// Dependency:
#include "../glm.hpp"

namespace glm{
namespace gtx{
//! GLM_GTX_matrix_access extension: Set a column or a row of a matrix
namespace matrix_access
{
	//! Set a specific row to a matrix.
	//! From GLM_GTX_matrix_access extension.
    template <typename genType> 
	genType row(
		const genType& m, 
		int index, 
		typename genType::row_type const & x);

	 //! Set a specific column to a matrix.
	 //! From GLM_GTX_matrix_access extension.
	template <typename genType> 
	genType column(
		const genType& m, 
		int index, 
		typename genType::col_type const & x);

}//namespace matrix_access
}//namespace gtx
}//namespace glm

#define GLM_GTX_matrix_access namespace gtx::matrix_access
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_matrix_access;}
#endif//GLM_GTX_GLOBAL

#include "matrix_access.inl"

#endif//glm_gtx_matrix_access
