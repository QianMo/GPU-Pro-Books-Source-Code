finline __m128i mm_divu3(__m128i n)
{
  // Algorithm from "Hacker's Delight" by Henry S. Warren
  // Chapter 10: Integer Division by Constants
  __m128i q = _mm_add_epi32(_mm_srli_epi32(n, 2), _mm_srli_epi32(n, 4));
  q = _mm_add_epi32(q, _mm_srli_epi32(q, 4));
  q = _mm_add_epi32(q, _mm_srli_epi32(q, 8));
  q = _mm_add_epi32(q, _mm_srli_epi32(q, 16));
  __m128i r = _mm_sub_epi32(n, _mm_add_epi32(q, _mm_add_epi32(q, q)));
  __m128i t = _mm_add_epi32(_mm_add_epi32(r, _mm_set1_epi32(5)), _mm_slli_epi32(r, 2));
  return _mm_add_epi32(q, _mm_srli_epi32(t, 4));
}

static const IntegerMask c_CBRT_c1 = {0x3f000000, 0x3f000000, 0x3f000000, 0x3f000000};
static const IntegerMask c_CBRT_c2 = {0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000};

finline const Vec4 Vec4::ApproxCbrt(const Vec4& a)
{
  // Algorithm from cbrt.c http://www.netlib.org/cephes/ 
  // Cephes Math Library by Stephen L. Moshier
  __m128i r0 = _mm_sub_epi32(_mm_srli_epi32(_mm_castps_si128(_mm_and_ps(a.r, c_CBRT_c2.f)), 23), _mm_set1_epi32(0x7e));
  __m128i signr0 = _mm_cmplt_epi32(r0, _mm_setzero_si128());
  __m128i r2 = _mm_sub_epi32(_mm_xor_si128(signr0, r0), signr0);
  __m128i r3 = mm_divu3(r2);
  __m128i r5 = _mm_sub_epi32(_mm_xor_si128(signr0, r3), signr0);
  __m128i r4 = _mm_sub_epi32(r2, _mm_add_epi32(r3, _mm_add_epi32(r3, r3)));
  __m128 m0 = _mm_castsi128_ps(_mm_cmpeq_epi32(r4, _mm_setzero_si128()));
  __m128 m1 = _mm_castsi128_ps(_mm_cmpeq_epi32(r4, _mm_set1_epi32(1)));
  __m128 m2 = _mm_castsi128_ps(_mm_cmpeq_epi32(r4, _mm_set1_epi32(2)));
  __m128 c0 = _mm_set1_ps(1.0f);
  __m128 c1 = _mm_or_ps(_mm_andnot_ps(_mm_castsi128_ps(signr0), _mm_set1_ps(1.25992104989487316477f)), _mm_and_ps(_mm_castsi128_ps(signr0), _mm_set1_ps(0.793700525984099737374f)));
  __m128 c2 = _mm_or_ps(_mm_andnot_ps(_mm_castsi128_ps(signr0), _mm_set1_ps(1.58740105196819947475f)), _mm_and_ps(_mm_castsi128_ps(signr0), _mm_set1_ps(0.6299605249474365824f)));
  __m128 c3 = _mm_or_ps(_mm_or_ps(_mm_and_ps(c0, m0), _mm_and_ps(c1, m1)), _mm_and_ps(c2, m2));
  __m128 x = _mm_or_ps(_mm_andnot_ps(c_CBRT_c2.f, a.r), c_CBRT_c1.f);
  __m128 t0 = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(-0.13466110473359520655053f), x), _mm_set1_ps(0.54664601366395524503440f));
  __m128 t1 = _mm_sub_ps(_mm_mul_ps(t0, x), _mm_set1_ps(0.95438224771509446525043f));
  __m128 t2 = _mm_add_ps(_mm_mul_ps(t1, x), _mm_set1_ps(1.1399983354717293273738f));
  __m128 x1 = _mm_mul_ps(_mm_add_ps(_mm_mul_ps(t2, x), _mm_set1_ps(0.40238979564544752126924f)), c3);
  __m128i r6 = _mm_add_epi32(_mm_srli_epi32(_mm_castps_si128(_mm_and_ps(x1, c_CBRT_c2.f)), 23), r5);
  return _mm_or_ps(_mm_andnot_ps(c_CBRT_c2.f, x1), _mm_and_ps(_mm_castsi128_ps(_mm_slli_epi32(r6, 23)), c_CBRT_c2.f));
}

finline const Vec4 Vec4::Cbrt(const Vec4& a)
{
  __m128 x0 = ApproxCbrt(a);
  __m128 x1 = _mm_sub_ps(x0, _mm_mul_ps(_mm_sub_ps(x0, _mm_mul_ps(a.r, Rcp(_mm_mul_ps(x0, x0)))), _mm_set1_ps(0.33333333333333333333f)));
  return x1;
}
