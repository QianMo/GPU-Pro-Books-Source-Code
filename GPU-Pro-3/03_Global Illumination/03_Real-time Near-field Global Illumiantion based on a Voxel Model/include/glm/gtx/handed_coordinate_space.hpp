///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2005-12-21
// Updated : 2009-02-19
// Licence : This source is under MIT License
// File    : glm/gtx/handed_coordinate_space.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_handed_coordinate_space
#define glm_gtx_handed_coordinate_space

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace gtx{
	//! GLM_GTX_handed_coordinate_space extension: To know if a triedron is right or left handed
	namespace handed_coordinate_space
	{
		//! Return if a trihedron right handed or not.
		//! From GLM_GTX_handed_coordinate_space extension.
		template <typename T> 
		bool rightHanded(
			detail::tvec3<T> const & tangent, 
			detail::tvec3<T> const & binormal, 
			detail::tvec3<T> const & normal);

		//! Return if a trihedron left handed or not.
		//! From GLM_GTX_handed_coordinate_space extension.
		template <typename T> 
		bool leftHanded(
			detail::tvec3<T> const & tangent, 
			detail::tvec3<T> const & binormal, 
			detail::tvec3<T> const & normal);

	}//namespace handed_coordinate_space
	}//namespace gtx
}//namespace glm

#define GLM_GTX_handed_coordinate_space namespace gtx::handed_coordinate_space
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_handed_coordinate_space;}
#endif//GLM_GTX_GLOBAL

#include "handed_coordinate_space.inl"

#endif//glm_gtx_handed_coordinate_space
