///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-21
// Updated : 2006-11-13
// Licence : This source is under MIT License
// File    : glm/gtx/epsilon.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - GLM_GTX_double
// - GLM_GTX_half
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_epsilon
#define glm_gtx_epsilon

// Dependency:
#include "../glm.hpp"
#include "../gtc/double_float.hpp"
#include "../gtc/half_float.hpp"

namespace glm
{
	namespace test{
		void main_gtx_epsilon();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_epsilon extension: Comparaison functions for a user defined epsilon values.
	namespace epsilon
	{
		//! Returns the component-wise compare of |x - y| < epsilon.
		//! From GLM_GTX_epsilon extension.
		template <typename genTypeT, typename genTypeU> 
		bool equalEpsilon(
			genTypeT const & x, 
			genTypeT const & y, 
			genTypeU const & epsilon);
		
		//! Returns the component-wise compare of |x - y| >= epsilon.
		//! From GLM_GTX_epsilon extension.
		template <typename genTypeT, typename genTypeU>
		bool notEqualEpsilon(
			genTypeT const & x, 
			genTypeT const & y, 
			genTypeU const & epsilon);

	}//namespace epsilon
	}//namespace gtx
}//namespace glm

#define GLM_GTX_epsilon namespace gtx::epsilon
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_epsilon;}
#endif//GLM_GTX_GLOBAL

#include "epsilon.inl"

#endif//glm_gtx_epsilon
