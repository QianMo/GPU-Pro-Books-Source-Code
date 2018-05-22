///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-24
// Updated : 2008-10-07
// Licence : This source is under MIT License
// File    : glm/gtx/unsigned_int.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - GLM_GTX_integer
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_unsigned_int
#define glm_gtx_unsigned_int

// Dependency:
#include "../glm.hpp"
#include "../gtx/integer.hpp"

namespace glm
{
	namespace test{
		void main_gtx_unsigned_int();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_unsigned_int extension: Add support for unsigned integer for core functions
	namespace unsigned_int
	{
		//! 32bit signed integer. 
		//! From GLM_GTX_unsigned_int extension.
		typedef signed int					sint;

		//! Returns x raised to the y power.
		//! From GLM_GTX_unsigned_int extension.
		uint pow(uint x, uint y);

		//! Returns the positive square root of x. 
		//! From GLM_GTX_unsigned_int extension.
		uint sqrt(uint x);

		//! Modulus. Returns x - y * floor(x / y) for each component in x using the floating point value y.
		//! From GLM_GTX_unsigned_int extension.
		uint mod(uint x, uint y);

	}//namespace unsigned_int
	}//namespace gtx
}//namespace glm

#define GLM_GTX_unsigned_int namespace gtx::unsigned_int; using namespace gtx::integer
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_unsigned_int;}
#endif//GLM_GTX_GLOBAL

#include "unsigned_int.inl"

#endif//glm_gtx_unsigned_int
