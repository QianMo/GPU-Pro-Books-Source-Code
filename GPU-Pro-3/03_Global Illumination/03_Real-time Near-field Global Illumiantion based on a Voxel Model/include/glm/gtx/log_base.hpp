///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-10-24
// Updated : 2008-10-24
// Licence : This source is under MIT License
// File    : glm/gtx/log_base.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_log_base
#define glm_gtx_log_base

// Dependency:
#include "../glm.hpp"

namespace glm
{
   	namespace test{
		void main_ext_gtx_log_base();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_log_base extension: Logarithm for any base. base can be a vector or a scalar.
	namespace log_base
	{
		//! Logarithm for any base.
		//! From GLM_GTX_log_base.
		template <typename genType> 
		genType log(
			genType const & x, 
			genType const & base);

	}//namespace extend
	}//namespace gtx
}//namespace glm

#define GLM_GTX_log_base namespace gtx::log_base
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_log_base;}
#endif//GLM_GTX_GLOBAL

#include "log_base.inl"

#endif//glm_gtx_log_base
