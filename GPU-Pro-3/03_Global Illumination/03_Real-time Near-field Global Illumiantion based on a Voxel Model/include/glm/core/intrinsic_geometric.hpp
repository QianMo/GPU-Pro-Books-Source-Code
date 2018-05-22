///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-08
// Updated : 2009-05-08
// Licence : This source is under MIT License
// File    : glm/core/intrinsic_geometric.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef glm_core_intrinsic_geometric
#define glm_core_intrinsic_geometric

#include "intrinsic_common.hpp"

//length
__m128 _mm_len_ps(__m128 x);

//distance
__m128 _mm_dst_ps(__m128 p0, __m128 p1);

//dot
__m128 _mm_dot_ps(__m128 v1, __m128 v2);

// SSE1
__m128 _mm_dot_ss(__m128 v1, __m128 v2);

//cross
__m128 _mm_xpd_ps(__m128 v1, __m128 v2);

//normalize
__m128 _mm_nrm_ps(__m128 v);

//faceforward
__m128 _mm_ffd_ps(__m128 N, __m128 I, __m128 Nref);

//reflect
__m128 _mm_rfe_ps(__m128 I, __m128 N);

//refract
__m128 _mm_rfa_ps(__m128 I, __m128 N, __m128 eta);


#include "intrinsic_geometric.inl"

#endif//glm_core_intrinsic_geometric
