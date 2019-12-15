#ifdef ENABLE_SSE4

finline __m128 dp_m128(__m128 a, __m128 b)
{
  return _mm_dp_ps(a, b, 0xff);
}

#else

finline __m128 dp_m128(__m128 a, __m128 b)
{
  __m128 t0 = _mm_mul_ps(a, b);
  __m128 t1 = _mm_add_ps(t0, SWZ_YXWZ(t0));
  return _mm_add_ss(t1, _mm_movehl_ps(t1, t1));
}

#endif

#ifdef ENABLE_SSE2

finline __m128d dp_m128(__m128d a, __m128d b)
{
  __m128d t0 = _mm_mul_pd(a, b);
  return _mm_add_pd(t0, SWZ_YX(t0));
}

#endif
