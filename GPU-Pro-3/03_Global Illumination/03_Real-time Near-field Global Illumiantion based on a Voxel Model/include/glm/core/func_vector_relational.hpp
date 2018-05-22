///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-08-03
// Updated : 2008-09-09
// Licence : This source is under MIT License
// File    : glm/core/func_vector_relational.h
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_func_vector_relational
#define glm_core_func_vector_relational

namespace glm
{
	namespace test{
		void main_core_func_vector_relational();
	}//namespace test

	namespace core{
	namespace function{
	//! Define vector relational functions from Section 8.3 of GLSL 1.30.8 specification. Included in glm namespace.
	namespace vector_relational{

	//! Returns the component-wise compare of x < y.
	//! (From GLSL 1.30.08 specification, section 8.6)
	template <typename vecType> 
	typename vecType::bool_type lessThan(vecType const & x, vecType const & y);

	//! Returns the component-wise compare of x <= y.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	typename vecType::bool_type lessThanEqual(vecType const & x, vecType const & y);

	//! Returns the component-wise compare of x > y.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	typename vecType::bool_type greaterThan(vecType const & x, vecType const & y);

	//! Returns the component-wise compare of x >= y.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	typename vecType::bool_type greaterThanEqual(vecType const & x, vecType const & y);

	//! Returns the component-wise compare of x == y.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	typename vecType::bool_type equal(vecType const & x, vecType const & y);

	//! Returns the component-wise compare of x != y.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	typename vecType::bool_type notEqual(vecType const & x, vecType const & y);

	//! Returns true if any component of x is true.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	bool any(vecType const & x);

	//! Returns true if all components of x are true.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	bool all(vecType const & x);

	//! Returns the component-wise logical complement of x.
	//! (From GLSL 1.30.08 specification, section 8.6)
    template <typename vecType> 
	typename vecType::bool_type not_(vecType const & x);

	}//namespace vector_relational
	}//namespace function
	}//namespace core

	using namespace core::function::vector_relational;
}//namespace glm

#include "func_vector_relational.inl"

#endif//glm_core_func_vector_relational
