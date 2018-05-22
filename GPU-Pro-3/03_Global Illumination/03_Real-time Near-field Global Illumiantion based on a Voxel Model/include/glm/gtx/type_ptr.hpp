///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-06
// Updated : 2009-05-06
// Licence : This source is under MIT License
// File    : glm/gtx/type_ptr.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_type_ptr
#define glm_gtx_type_ptr

// Dependency:
#include "../glm.hpp"

namespace glm
{
	namespace test{
		void main_gtx_type_ptr();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_type_ptr extension: Get access to vectors & matrices value type address.
	namespace type_ptr{

		//! Get the const address of the vector content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tvec2<valType> const & vec)
		{
			return &(vec.x);
		}

		//! Get the address of the vector content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tvec2<valType> & vec)
		{
			return &(vec.x);
		}

		//! Get the const address of the vector content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tvec3<valType> const & vec)
		{
			return &(vec.x);
		}

		//! Get the address of the vector content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tvec3<valType>  & vec)
		{
			return &(vec.x);
		}
		
		//! Get the const address of the vector content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tvec4<valType> const & vec)
		{
			return &(vec.x);
		}

		//! Get the address of the vector content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tvec4<valType> & vec)
		{
			return &(vec.x);
		}

		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat2x2<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat2x2<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat3x3<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat3x3<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat4x4<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat4x4<valType> & mat)
		{
			return &(mat[0].x);
		}

		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat2x3<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat2x3<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat3x2<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat3x2<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat2x4<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat2x4<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat4x2<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat4x2<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat3x4<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat3x4<valType> & mat)
		{
			return &(mat[0].x);
		}
		
		//! Get the const address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType const * value_ptr(detail::tmat4x3<valType> const & mat)
		{
			return &(mat[0].x);
		}

		//! Get the address of the matrix content.
		//! From GLM_GTX_type_ptr extension.
		template<typename valType>
		inline valType * value_ptr(detail::tmat4x3<valType> & mat)
		{
			return &(mat[0].x);
		}

	}//namespace type_ptr
	}//namespace gtx
}//namespace glm

#define GLM_GTX_type_ptr namespace gtx::type_ptr
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_type_ptr;}
#endif//GLM_GTX_GLOBAL

#include "type_ptr.inl"

#endif//glm_gtx_type_ptr

