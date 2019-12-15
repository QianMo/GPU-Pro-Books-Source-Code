finline void Vec4::SinCos(const Vec4& a, Vec4& s, Vec4& c)
{
  // Algorithm from sinf.c http://www.netlib.org/cephes/
  // Cephes Math Library by Stephen L. Moshier
  __m128 t0 = _mm_andnot_ps(c_SignMask.f, a.r);
  __m128i t1 = _mm_cvttps_epi32(_mm_mul_ps(t0, _mm_set1_ps(1.27323954473516f)));
  __m128i t2 = _mm_add_epi32(t1, _mm_and_si128(t1, _mm_set1_epi32(1)));
  __m128 t3 = _mm_cvtepi32_ps(t2);
  __m128 t4 = _mm_sub_ps(_mm_sub_ps(_mm_sub_ps(t0, _mm_mul_ps(_mm_set1_ps(0.78515625f), t3)), _mm_mul_ps(_mm_set1_ps(2.4187564849853515625e-4f), t3)), _mm_mul_ps(_mm_set1_ps(3.77489497744594108e-8f), t3));
  __m128 t5 = _mm_mul_ps(t4, t4);
  __m128 t6 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.5f), t5), _mm_set1_ps(1.0f));
  __m128 t7 = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(2.443315711809948E-005f), t5), _mm_set1_ps(-1.388731625493765E-003f)), t5), _mm_set1_ps(4.166664568298827E-002f)), _mm_mul_ps(t5, t5)), t6);
  __m128 t8 = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(-1.9515295891E-4f), t5), _mm_set1_ps(8.3321608736E-3f)), t5), _mm_set1_ps(-1.6666654611E-1f)), _mm_mul_ps(t4, t5)), t4);
  __m128 t9 = _mm_castsi128_ps(_mm_slli_epi32(t2, 29));
  __m128 t10 = _mm_castsi128_ps(_mm_cmpeq_epi32(_mm_and_si128(t2, _mm_set1_epi32(3)), _mm_set1_epi32(2)));
  s = _mm_xor_ps(_mm_or_ps(_mm_and_ps(t10, t7), _mm_andnot_ps(t10, t8)), _mm_and_ps(_mm_xor_ps(t9, a.r), c_SignMask.f));
  c = _mm_xor_ps(_mm_or_ps(_mm_and_ps(t10, t8), _mm_andnot_ps(t10, t7)), _mm_and_ps(_mm_xor_ps(t9, _mm_castsi128_ps(_mm_slli_epi32(t2, 30))), c_SignMask.f));
}
