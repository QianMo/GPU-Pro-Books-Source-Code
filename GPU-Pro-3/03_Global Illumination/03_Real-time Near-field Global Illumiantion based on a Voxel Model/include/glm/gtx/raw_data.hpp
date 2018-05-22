///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2008-11-19
// Updated : 2008-11-19
// Licence : This source is under MIT License
// File    : glm/gtx/raw_data.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_raw_data
#define glm_gtx_raw_data

// Dependency:
#include "../glm.hpp"
#include "../gtc/type_precision.hpp"

namespace glm
{
    namespace gtx
    {
		//! GLM_GTX_raw_data extension: Projection of a vector to other one
        namespace raw_data
        {
			//! Type for byte numbers. 
			//! From GLM_GTX_raw_data extension.
			typedef gtc::type_precision::uint8		byte;

			//! Type for word numbers. 
			//! From GLM_GTX_raw_data extension.
			typedef gtc::type_precision::uint16		word;

			//! Type for dword numbers. 
			//! From GLM_GTX_raw_data extension.
			typedef gtc::type_precision::uint32		dword;

			//! Type for qword numbers. 
			//! From GLM_GTX_raw_data extension.
			typedef gtc::type_precision::uint64		qword;
		}
    }
}

#define GLM_GTX_raw_data namespace gtx::raw_data
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_raw_data;}
#endif//GLM_GTX_GLOBAL

#include "raw_data.inl"

#endif//glm_gtx_raw_data
