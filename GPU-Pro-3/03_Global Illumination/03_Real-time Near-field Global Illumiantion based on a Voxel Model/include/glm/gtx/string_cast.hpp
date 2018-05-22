///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2006 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-04-26
// Updated : 2008-05-24
// Licence : This source is under MIT License
// File    : glm/gtx/string_cast.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - GLM_GTX_double
// - GLM_GTX_half
// - GLM_GTX_integer
// - GLM_GTX_quaternion
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_string_cast
#define glm_gtx_string_cast

// Dependency:
#include "../glm.hpp"
#include "../gtc/double_float.hpp"
#include "../gtc/half_float.hpp"
#include "../gtx/integer.hpp"
#include "../gtx/unsigned_int.hpp"
#include "../gtx/quaternion.hpp"
#include <string>

namespace glm
{
	namespace test{
		void main_gtx_string_cast();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_string_cast extension: Setup strings for GLM type values
	namespace string_cast
	{
		using namespace gtc::double_float; 
		using namespace gtc::half_float; 
		using namespace gtx::integer; 
		using namespace gtx::unsigned_int; 
		using namespace gtx::quaternion; 

		//! Create a string from a GLM type value.
		//! From GLM_GTX_string_cast extension.
		template <typename genType> 
		std::string string(genType const & x);

	}//namespace string_cast
	}//namespace gtx
}//namespace glm

#define GLM_GTX_string_cast namespace gtx::string_cast
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_string_cast;}
#endif//GLM_GTX_GLOBAL

#include "string_cast.inl"

#endif//glm_gtx_string_cast
