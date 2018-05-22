///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-21
// Updated : 2006-11-13
// Licence : This source is under MIT License
// File    : glm/gtx/normal.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_normal
#define glm_gtx_normal

// Dependency:
#include "../glm.hpp"

namespace glm
{
   	namespace test{
		void main_gtx_normal();
	}//namespace test
	
	namespace gtx{
	//! GLM_GTX_normal extension: Compute the normal of a triangle.
    namespace normal
    {
		//! Computes triangle normal from triangle points. 
		//! From GLM_GTX_normal extension.
        template <typename T> 
		detail::tvec3<T> triangleNormal(
			detail::tvec3<T> const & p1, 
			detail::tvec3<T> const & p2, 
			detail::tvec3<T> const & p3);

    }//namespace normal
    }//namespace gtx
}//namespace glm

#define GLM_GTX_normal namespace gtx::normal
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_normal;}
#endif//GLM_GTX_GLOBAL

#include "normal.inl"

#endif//glm_gtx_normal
