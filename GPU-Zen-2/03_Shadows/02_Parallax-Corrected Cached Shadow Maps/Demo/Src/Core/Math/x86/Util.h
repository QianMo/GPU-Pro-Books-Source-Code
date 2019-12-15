#ifndef __MATH_UTIL_H
#define __MATH_UTIL_H

#ifdef ENABLE_SSE2

static const __m128 c_RGBA_CVT_C0 = {1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f, 1.0f/255.0f};
static const __m128 c_RGBA_CVT_C1 = {2.0f/255.0f, 2.0f/255.0f, 2.0f/255.0f, 2.0f/255.0f};
static const __m128 c_RGBA_CVT_C2 = {1.0f, 1.0f, 1.0f, 1.0f};

finline const Mat4x4 RGBA8UN_to_RGBA32F_4T(const unsigned* c)
{
  __m128i va0 = _mm_loadu_si128((__m128i*)c);
  __m128i lo0 = _mm_unpacklo_epi8(va0, _mm_setzero_si128());
  __m128i hi0 = _mm_unpackhi_epi8(va0, _mm_setzero_si128());
  return Mat4x4(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi8(lo0, _mm_setzero_si128())), c_RGBA_CVT_C0),
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi8(lo0, _mm_setzero_si128())), c_RGBA_CVT_C0),
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi8(hi0, _mm_setzero_si128())), c_RGBA_CVT_C0),
                _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi8(hi0, _mm_setzero_si128())), c_RGBA_CVT_C0));
}

finline const Mat4x4 RGBA8SN_to_RGBA32F_4T(const unsigned* c)
{
  __m128i va0 = _mm_loadu_si128((__m128i*)c);
  __m128i lo0 = _mm_unpacklo_epi8(va0, _mm_setzero_si128());
  __m128i hi0 = _mm_unpackhi_epi8(va0, _mm_setzero_si128());
  return Mat4x4(_mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi8(lo0, _mm_setzero_si128())), c_RGBA_CVT_C1), c_RGBA_CVT_C2),
                _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi8(lo0, _mm_setzero_si128())), c_RGBA_CVT_C1), c_RGBA_CVT_C2),
                _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi8(hi0, _mm_setzero_si128())), c_RGBA_CVT_C1), c_RGBA_CVT_C2),
                _mm_sub_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi8(hi0, _mm_setzero_si128())), c_RGBA_CVT_C1), c_RGBA_CVT_C2));
}

#endif

#endif //#ifndef __MATH_UTIL_H
