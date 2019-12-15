#ifndef __VEC4i
#define __VEC4i

class Vec4i : public MathLibObject
{
public:
  union
  {
    struct { int x, y, z, w; };
    int e[4];
    __m128i r;
  };

  finline Vec4i()                           { }
  finline Vec4i(__m128i a)                  { r = a; }
  finline Vec4i(const Vec4i& a)             { r = a.r; }
  finline Vec4i(int a)                      { r = _mm_set1_epi32(a); }
  finline Vec4i(int a, int b, int c, int d) { r = _mm_set_epi32(d, c, b, a); }
  finline Vec4i(const int* a)               { r = _mm_loadu_si128((__m128i*)a); }

  finline void Store(int* a) const                 { _mm_storeu_si128((__m128i*)a, r); }
  finline operator __m128i() const                 { return r; }
  finline const Vec4i& operator = (const Vec4i& a) { r = a.r; return *this; }

  static finline const Vec4i Zero() { return _mm_setzero_si128(); }

  static finline unsigned AsMask(const Vec4i& a)     { return _mm_movemask_ps(_mm_castsi128_ps(a.r)); }
  static finline const Vec4  Cast(const Vec4i& a)    { return _mm_castsi128_ps(a.r); }
  static finline const Vec4  Convert(const Vec4i& a) { return _mm_cvtepi32_ps(a.r); }
  static finline const Vec4i Cast(const Vec4& a)     { return _mm_castps_si128(a.r); }
  static finline const Vec4i Round(const Vec4& a)    { return _mm_cvtps_epi32(a.r); }
  static finline const Vec4i Truncate(const Vec4& a) { return _mm_cvttps_epi32(a.r); }
  static finline const Vec4i Ceil(const Vec4& a)     { __m128 t = _mm_add_ps(a.r, _mm_set1_ps(0.5f)); return _mm_cvtps_epi32(t); }
  static finline const Vec4i Floor(const Vec4& a)    { __m128 t = _mm_sub_ps(a.r, _mm_set1_ps(0.5f)); return _mm_cvtps_epi32(t); }

  static finline const Vec4i Select(const Vec4i& a, const Vec4i& b, const Vec4i& c) { return _mm_or_si128(_mm_and_si128(a.r, b.r), _mm_andnot_si128(a.r, c.r)); }

  template<Vec4Element x, Vec4Element y, Vec4Element z, Vec4Element w> static finline const Vec4 Shuffle(const Vec4i& a, const Vec4i& b) { return _mm_shuffle_epi32(a.r, b.r, _MM_SHUFFLE(w, z, y, x)); }
  template<Vec4Element x, Vec4Element y, Vec4Element z, Vec4Element w> static finline const Vec4 Swizzle(const Vec4i& a)                 { return _mm_shuffle_epi32(a.r, _MM_SHUFFLE(w, z, y, x)); }

  static finline const Vec4i CmpEqual  (const Vec4i& a, const Vec4i& b) { return _mm_cmpeq_epi32(a.r, b.r); }
  static finline const Vec4i CmpLess   (const Vec4i& a, const Vec4i& b) { return _mm_cmplt_epi32(a.r, b.r); }
  static finline const Vec4i CmpGreater(const Vec4i& a, const Vec4i& b) { return _mm_cmpgt_epi32(a.r, b.r); }

  static finline const Vec4i PackS16(const Vec4i& a, const Vec4i& b) { return _mm_packs_epi32(a.r, b.r); }
  static finline const Vec4i UnpackLoS16(const Vec4i& a) { return _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), a.r), 16); }
  static finline const Vec4i UnpackHiS16(const Vec4i& a) { return _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), a.r), 16); }

  static finline const Vec4i Abs(const Vec4i& a)                 { __m128i s = _mm_srai_epi32(a.r, 31); return _mm_sub_epi32(_mm_xor_si128(a.r, s), s); }
  static finline const Vec4i Min(const Vec4i& a, const Vec4i& b) { __m128i m = _mm_cmplt_epi32(a.r, b.r); return _mm_or_si128(_mm_and_si128(m, a.r), _mm_andnot_si128(m, b.r)); }
  static finline const Vec4i Max(const Vec4i& a, const Vec4i& b) { __m128i m = _mm_cmpgt_epi32(a.r, b.r); return _mm_or_si128(_mm_and_si128(m, a.r), _mm_andnot_si128(m, b.r)); }

  finline int&       operator [] (int i)       { return e[i]; }
  finline const int& operator [] (int i) const { return e[i]; }
};

finline bool operator == (const Vec4i& a, const Vec4i& b) { return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(a.r, b.r)))==15; }
finline bool operator != (const Vec4i& a, const Vec4i& b) { return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(a.r, b.r)))!=15; }
finline bool operator >= (const Vec4i& a, const Vec4i& b) { return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmplt_epi32(a.r, b.r)))==0; }
finline bool operator <= (const Vec4i& a, const Vec4i& b) { return _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpgt_epi32(a.r, b.r)))==0; }

finline const Vec4i operator + (const Vec4i& a, const Vec4i& b) { return _mm_add_epi32(a.r, b.r); }
finline const Vec4i operator - (const Vec4i& a, const Vec4i& b) { return _mm_sub_epi32(a.r, b.r); }

finline const Vec4i operator + (int a, const Vec4i& b) { return _mm_add_epi32(_mm_set1_epi32(a), b.r); }
finline const Vec4i operator - (int a, const Vec4i& b) { return _mm_sub_epi32(_mm_set1_epi32(a), b.r); }

finline const Vec4i operator + (const Vec4i& a, int b) { return _mm_add_epi32(a.r, _mm_set1_epi32(b)); }
finline const Vec4i operator - (const Vec4i& a, int b) { return _mm_sub_epi32(a.r, _mm_set1_epi32(b)); }

finline const Vec4i operator & (const Vec4i& a, const Vec4i& b) { return _mm_and_si128(a.r, b.r); }
finline const Vec4i operator | (const Vec4i& a, const Vec4i& b) { return _mm_or_si128 (a.r, b.r); }
finline const Vec4i operator ^ (const Vec4i& a, const Vec4i& b) { return _mm_xor_si128(a.r, b.r); }

finline const Vec4i operator + (const Vec4i& a) { return a; }
finline const Vec4i operator - (const Vec4i& a) { return _mm_sub_epi32(_mm_setzero_si128(), a.r); }

finline const Vec4i operator >> (const Vec4i& a, int b) { return _mm_srai_epi32(a.r, b); }
finline const Vec4i operator << (const Vec4i& a, int b) { return _mm_slli_epi32(a.r, b); }

#endif //#ifndef __VEC4i
