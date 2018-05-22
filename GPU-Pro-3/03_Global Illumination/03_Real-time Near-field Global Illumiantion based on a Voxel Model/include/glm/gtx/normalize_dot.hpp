//////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
//////////////////////////////////////////////////////////////////////////////////
// Created : 2007-09-28
// Updated : 2008-10-07
// Licence : This source is under MIT License
// File    : glm/gtx/normalize_dot.h
//////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - GLM_GTX_fast_square_root
//////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_normalize_dot
#define glm_gtx_normalize_dot

// Dependency:
#include "../glm.hpp"
#include "../gtx/fast_square_root.hpp"

namespace glm
{
	namespace gtx{
	//! GLM_GTX_normalize_dot extension: Dot product of vectors that need to be normalize with a single square root.
	namespace normalize_dot
	{
		//! Normalize parameters and returns the dot product of x and y.
		//! It's faster that dot(normalize(x), normalize(y)).
		//! From GLM_GTX_normalize_dot extension.
		template <typename genType> 
		typename genType::value_type normalizeDot(
			genType const & x, 
			genType const & y);

		//! Normalize parameters and returns the dot product of x and y.
		//! Faster that dot(fastNormalize(x), fastNormalize(y)).
		//! From GLM_GTX_normalize_dot extension.
		template <typename genType> 
		typename genType::value_type fastNormalizeDot(
			genType const & x, 
			genType const & y);

	}//namespace normalize_dot
	}//namespace gtx
}//namespace glm

#define GLM_GTX_normalize_dot namespace gtx::fast_square_root; using namespace gtx::normalize_dot
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_normalize_dot;}
#endif//GLM_GTX_GLOBAL

#include "normalize_dot.inl"

#endif//glm_gtx_normalize_dot
