///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2007-11-21
// Updated : 2007-11-21
// Licence : This source is under MIT License
// File    : glm/gtx/statistics_operation.h
///////////////////////////////////////////////////////////////////////////////////////////////////
// Dependency:
// - GLM core
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_gtx_statistics_operation
#define glm_gtx_statistics_operation

// Dependency:
#include "../glm.hpp"

namespace glm
{
	template <typename T> T statDistanceGTX(const detail::tvec2<T>& v1, const detail::tvec2<T>& v2);
    template <typename T> T statDistanceGTX(const detail::tvec3<T>& v1, const detail::tvec3<T>& v2);
    template <typename T> T statDistanceGTX(const detail::tvec4<T>& v1, const detail::tvec4<T>& v2);

    template <typename T> T statDistanceGTX(const detail::tmat2x2<T>& m1, const detail::tmat2x2<T>& m2);
    template <typename T> T statDistanceGTX(const detail::tmat3x3<T>& m1, const detail::tmat3x3<T>& m2);
    template <typename T> T statDistanceGTX(const detail::tmat4x4<T>& m1, const detail::tmat4x4<T>& m2);

    template <typename T> T expectedValueGTX(const detail::tvec2<T>& v1, const detail::tvec2<T>& v2);
    template <typename T> T expectedValueGTX(const detail::tvec3<T>& v1, const detail::tvec3<T>& v2);
    template <typename T> T expectedValueGTX(const detail::tvec4<T>& v1, const detail::tvec4<T>& v2);

    template <typename T> T expectedValueGTX(const detail::tmat2x2<T>& m1, const detail::tmat2x2<T>& m2);
    template <typename T> T expectedValueGTX(const detail::tmat3x3<T>& m1, const detail::tmat3x3<T>& m2);
    template <typename T> T expectedValueGTX(const detail::tmat4x4<T>& m1, const detail::tmat4x4<T>& m2);

    template <typename T> T varianceGTX(const detail::tvec2<T>& v1, const detail::tvec2<T>& v2);
    template <typename T> T varianceGTX(const detail::tvec3<T>& v1, const detail::tvec3<T>& v2);
    template <typename T> T varianceGTX(const detail::tvec4<T>& v1, const detail::tvec4<T>& v2);

    template <typename T> T varianceGTX(const detail::tmat2x2<T>& m1, const detail::tmat2x2<T>& m2);
    template <typename T> T varianceGTX(const detail::tmat3x3<T>& m1, const detail::tmat3x3<T>& m2);
    template <typename T> T varianceGTX(const detail::tmat4x4<T>& m1, const detail::tmat4x4<T>& m2);

    template <typename T> T standardDevitionGTX(const detail::tvec2<T>& v1, const detail::tvec2<T>& v2);
    template <typename T> T standardDevitionGTX(const detail::tvec3<T>& v1, const detail::tvec3<T>& v2);
    template <typename T> T standardDevitionGTX(const detail::tvec4<T>& v1, const detail::tvec4<T>& v2);

    template <typename T> T standardDevitionGTX(const detail::tmat2x2<T>& m1, const detail::tmat2x2<T>& m2);
    template <typename T> T standardDevitionGTX(const detail::tmat3x3<T>& m1, const detail::tmat3x3<T>& m2);
    template <typename T> T standardDevitionGTX(const detail::tmat4x4<T>& m1, const detail::tmat4x4<T>& m2);

    namespace gtx
    {
		//! GLM_GTX_statistics_operation extension: - Work in progress - Statistics functions
        namespace statistics_operation
        {

        }
    }
}

#define GLM_GTX_statistics_operation namespace gtx::statistics_operation
#ifndef GLM_GTX_GLOBAL
namespace glm {using GLM_GTX_statistics_operation;}
#endif//GLM_GTX_GLOBAL

#include "statistics_operation.inl"

#endif//glm_gtx_statistics_operation
