///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-03-14
// Updated : 2007-08-14
// Licence : This source is under MIT License
// File    : gtx_extented_min_max.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
// - GLM_GTX_half_float
// - GLM_GTX_double_float
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_extented_min_max
#define glm_gtx_extented_min_max

// Dependency:
#include "../glm.hpp"
#include "../gtc/half_float.hpp"
#include "../gtc/double_float.hpp"

namespace glm
{
   	namespace test{
		void main_ext_gtx_extented_min_max();
	}//namespace test

	namespace gtx{
	//! GLM_GTX_extented_min_max extension: Min and max functions for 3 to 4 parameters.
	namespace extented_min_max
	{
		template <typename T> T min(const T x, const T y, const T z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> T min(const T x, const T y, const T z, const T w); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> min(const detail::tvec2<T>& x, const T y, const T z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> min(const detail::tvec3<T>& x, const T y, const T z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> min(const detail::tvec4<T>& x, const T y, const T z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> min(const detail::tvec2<T>& x, const T y, const T z, const T w); //!< \brief Return the minimum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> min(const detail::tvec3<T>& x, const T y, const T z, const T w); //!< \brief Return the minimum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> min(const detail::tvec4<T>& x, const T y, const T z, const T w); //!< \brief Return the minimum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> min(const detail::tvec2<T>& x, const detail::tvec2<T>& y, const detail::tvec2<T>& z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> min(const detail::tvec3<T>& x, const detail::tvec3<T>& y, const detail::tvec3<T>& z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> min(const detail::tvec4<T>& x, const detail::tvec4<T>& y, const detail::tvec4<T>& z); //!< \brief Return the minimum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> min(const detail::tvec2<T>& x, const detail::tvec2<T>& y, const detail::tvec2<T>& z, const detail::tvec2<T>& w); //!< \brief Return the minimum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> min(const detail::tvec3<T>& x, const detail::tvec3<T>& y, const detail::tvec3<T>& z, const detail::tvec3<T>& w); //!< \brief Return the minimum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> min(const detail::tvec4<T>& x, const detail::tvec4<T>& y, const detail::tvec4<T>& z, const detail::tvec4<T>& w); //!< \brief Return the minimum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> T max(const T x, const T y, const T z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> T max(const T x, const T y, const T z, const T w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> max(const detail::tvec2<T>& x, const T y, const T z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> max(const detail::tvec3<T>& x, const T y, const T z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> max(const detail::tvec4<T>& x, const T y, const T z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> max(const detail::tvec2<T>& x, const T y, const T z, const T w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> max(const detail::tvec3<T>& x, const T y, const T z, const T w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> max(const detail::tvec4<T>& x, const T y, const T z, const T w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> max(const detail::tvec2<T>& x, const detail::tvec2<T>& y, const detail::tvec2<T>& z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> max(const detail::tvec3<T>& x, const detail::tvec3<T>& y, const detail::tvec3<T>& z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> max(const detail::tvec4<T>& x, const detail::tvec4<T>& y, const detail::tvec4<T>& z); //!< \brief Return the maximum component-wise values of 3 imputs (From GLM_GTX_extented_min_max extension)

		template <typename T> detail::tvec2<T> max(const detail::tvec2<T>& x, const detail::tvec2<T>& y, const detail::tvec2<T>& z, const detail::tvec2<T>& w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec3<T> max(const detail::tvec3<T>& x, const detail::tvec3<T>& y, const detail::tvec3<T>& z, const detail::tvec3<T>& w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)
		template <typename T> detail::tvec4<T> max(const detail::tvec4<T>& x, const detail::tvec4<T>& y, const detail::tvec4<T>& z, const detail::tvec4<T>& w); //!< \brief Return the maximum component-wise values of 4 imputs (From GLM_GTX_extented_min_max extension)

	}//namespace extented_min_max
	}//namespace gtx
}//namespace glm

#define GLM_GTX_extented_min_max namespace gtx::extented_min_max
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_extented_min_max;}
#endif//GLM_GTX_GLOBAL

#include "extented_min_max.inl"

#endif//glm_gtx_extented_min_max
