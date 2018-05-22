 ///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-03-05
// Updated : 2007-03-05
// Licence : This source is under MIT License
// File    : glm/gtx/vector_query.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_vector_query
#define glm_gtx_vector_query

// Dependency:
#include "../glm.hpp"
#include <cfloat>
#include <limits>

namespace glm
{
	namespace test{
		void main_ext_gtx_vector_query();
	}//namespace test

    namespace gtx{
	//! GLM_GTX_vector_query extension: Query informations of vector types
    namespace vector_query
    {
        //! Check if two vectors are collinears.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType> 
		bool areCollinear(
			const genType & v0, 
			const genType & v1, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());
		
        //! Check if two vectors are opposites.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType> 
		bool areOpposite(
			const genType & v0, 
			const genType & v1, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());
		
        //! Check if two vectors are orthogonals.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType> 
		bool areOrthogonal(
			const genType & v0, 
			const genType & v1, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());

		//! Check if a vector is normalized.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType> 
		bool isNormalized(
			const genType & v, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());
		
		//! Check if a vector is null.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType> 
		bool isNull(
			const genType& v, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());

		//! Check if two vectors are orthonormal.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType>
		bool areOrthonormal(
			const genType & v0, 
			const genType & v1, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());

		//! Check if two vectors are similar.
		//! From GLM_GTX_vector_query extensions.
		template <typename genType> 
		bool areSimilar(
			const genType& v0, 
			const genType& v1, 
			const GLMvalType epsilon = std::numeric_limits<GLMvalType>::epsilon());

    }//namespace vector_query
    }//namespace gtx
}//namespace glm

#define GLM_GTX_vector_query namespace gtx::vector_query
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_vector_query;}
#endif//GLM_GTX_GLOBAL

#include "vector_query.inl"

#endif//glm_gtx_vector_query
