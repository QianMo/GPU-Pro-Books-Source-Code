///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-06-05
// Updated : 2009-06-05
// Licence : This source is under MIT License
// File    : glm/core/intrinsic_common.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef GLM_DETAIL_INTRINSIC_MATRIX_INCLUDED
#define GLM_DETAIL_INTRINSIC_MATRIX_INCLUDED

#include "../glm.hpp"

#include <xmmintrin.h>
#include <emmintrin.h>

void _mm_add_ps(__m128 in1[4], __m128 in2[4], __m128 out[4]);

void _mm_sub_ps(__m128 in1[4], __m128 in2[4], __m128 out[4]);

__m128 _mm_mul_ps(__m128 m[4], __m128 v);

__m128 _mm_mul_ps(__m128 v, __m128 m[4]);

void _mm_mul_ps(__m128 const in1[4], __m128 const in2[4], __m128 out[4]);

void _mm_transpose_ps(__m128 const in[4], __m128 out[4]);

void _mm_inverse_ps(__m128 const in[4], __m128 out[4]);

void _mm_rotate_ps(__m128 const in[4], float Angle, float const v[3], __m128 out[4]);

#include "intrinsic_matrix.inl"

#endif//GLM_DETAIL_INTRINSIC_MATRIX_INCLUDED
