///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-03-05
// Updated : 2007-03-05
// Licence : This source is under MIT License
// File    : glm/gtx/matrix_query.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_matrix_query
#define glm_gtx_matrix_query

// Dependency:
#include "../glm.hpp"
//#include <cfloat>
#include <limits>

namespace glm
{
   	namespace test{
		void main_gtx_matrix_query();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_matrix_query: Query to evaluate matrices properties
	namespace matrix_query
	{
		//! Return if a matrix a null matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename T> 
		bool isNull(
			const detail::tmat2x2<T>& m, 
			const T epsilon = std::numeric_limits<T>::epsilon());
		
		//! Return if a matrix a null matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename T> 
		bool isNull(
			const detail::tmat3x3<T>& m, 
			const T epsilon = std::numeric_limits<T>::epsilon());
		
		//! Return if a matrix a null matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename T> 
		bool isNull(
			const detail::tmat4x4<T>& m, 
			const T epsilon = std::numeric_limits<T>::epsilon());
			
		//! Return if a matrix an identity matrix. 
		//! From GLM_GTX_matrix_query extension.
		template<typename genType> 
		bool isIdentity(
			const genType& m, 
			const typename genType::value_type epsilon = std::numeric_limits<typename genType::value_type>::epsilon());

		//! Return if a matrix a normalized matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename T> 
		bool isNormalized(
			const detail::tmat2x2<T>& m, 
			const T epsilon = std::numeric_limits<T>::epsilon());
		
		//! Return if a matrix a normalized matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename T> 
		bool isNormalized(
			const detail::tmat3x3<T>& m, 
			const T epsilon = std::numeric_limits<T>::epsilon());
		
		//! Return if a matrix a normalized matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename T> 
		bool isNormalized(
			const detail::tmat4x4<T>& m, 
			const T epsilon = std::numeric_limits<T>::epsilon());

		//! Return if a matrix an orthonormalized matrix.
		//! From GLM_GTX_matrix_query extension.
		template<typename genType> 
		bool isOrthogonal(
			const genType& m, 
			const typename genType::value_type epsilon = std::numeric_limits<typename genType::value_type>::epsilon());

	}//namespace matrix_query
	}//namespace gtx
}//namespace glm

#define GLM_GTX_matrix_query namespace gtx::matrix_query
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_matrix_query;}
#endif//GLM_GTX_GLOBAL

#include "matrix_query.inl"

#endif//glm_gtx_matrix_query
