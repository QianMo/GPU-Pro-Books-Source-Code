///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-10-26
// Updated : 2009-10-26
// Licence : This source is under MIT License
// File    : glm/img/multiple.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_img_multiple
#define glm_img_multiple

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace test{
		bool main_img_multiple();
	}//namespace test

	namespace img
	{
		//! GLM_IMG_multiple: Find the closest number of a number multiple of other number.
		namespace multiple
		{
			//! Higher Multiple number of Source.
			//! From GLM_IMG_multiple extension.
			template <typename genType> 
			genType higherMultiple(
				genType const & Source, 
				genType const & Multiple);

			//! Lower Multiple number of Source.
			//! From GLM_IMG_multiple extension.
			template <typename genType> 
			genType lowerMultiple(
				genType const & Source, 
				genType const & Multiple);

		}//namespace multiple
	}//namespace img
}//namespace glm

#define GLM_IMG_multiple namespace img::multiple
#ifndef GLM_IMG_GLOBAL
namespace glm {using GLM_IMG_multiple;}
#endif//GLM_IMG_GLOBAL

#include "multiple.inl"

#endif//glm_img_multiple
