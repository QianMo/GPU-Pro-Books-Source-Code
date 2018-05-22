///////////////////////////////////////////////////////////////////////////////////////////////////
// OpenGL Mathematics Copyright (c) 2005 - 2009 G-Truc Creation (www.g-truc.net)
///////////////////////////////////////////////////////////////////////////////////////////////////
// Created : 2009-05-11
// Updated : 2009-05-11
// Licence : This source is under MIT License
// File    : glm/core/intrinsic_common.hpp
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef GLM_DETAIL_INTRINSIC_COMMON_INCLUDED
#define GLM_DETAIL_INTRINSIC_COMMON_INCLUDED

//#include <mmintrin.h>
//#include <emmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>

__m128 _mm_abs_ps(__m128 x);

__m128 _mm_sgn_ps(__m128 x);

//floor
__m128 _mm_flr_ps(__m128 v);

//trunc
__m128 _mm_trc_ps(__m128 v);

//round
__m128 _mm_rnd_ps(__m128 v);

//roundEven
__m128 _mm_rde_ps(__m128 v);

__m128 _mm_ceil_ps(__m128 v);

__m128 _mm_frc_ps(__m128 x);

__m128 _mm_mod_ps(__m128 x, __m128 y);

__m128 _mm_modf_ps(__m128 x, __m128i & i);

//inline __m128 _mm_min_ps(__m128 x, __m128 y)

//inline __m128 _mm_max_ps(__m128 x, __m128 y)

__m128 _mm_clp_ps(__m128 v, __m128 minVal, __m128 maxVal);

__m128 _mm_mix_ps(__m128 v1, __m128 v2, __m128 a);

__m128 _mm_stp_ps(__m128 edge, __m128 x);

__m128 _mm_ssp_ps(__m128 edge0, __m128 edge1, __m128 x);

__m128 _mm_nan_ps(__m128 x);

__m128 _mm_inf_ps(__m128 x);

#include "intrinsic_common.inl"

#endif//GLM_DETAIL_INTRINSIC_COMMON_INCLUDED
