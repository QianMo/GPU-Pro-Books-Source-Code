///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-30
// Updated : 2008-10-05
// Licence : This source is under MIT License
// File    : glm/gtx/closest_point.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_closest_point
#define glm_gtx_closest_point

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace test{
		void main_gtx_closest_point();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_closest_point extension: Find the point on a straight line which is the closet of a point.
	namespace closest_point{

	//! Find the point on a straight line which is the closet of a point. 
	//! From GLM_GTX_closest_point extension.
	template <typename T> 
	detail::tvec3<T> closestPointOnLine(
		detail::tvec3<T> const & point, 
		detail::tvec3<T> const & a, 
		detail::tvec3<T> const & b);

	}//namespace closest_point
	}//namespace gtx
}//namespace glm

#define GLM_GTX_closest_point namespace gtx::closest_point
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_closest_point;}
#endif//GLM_GTC_GLOBAL

#include "closest_point.inl"

#endif//glm_gtx_closest_point
