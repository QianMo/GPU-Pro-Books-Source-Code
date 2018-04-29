/* MSVC SSE math library implementation.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __VEC4
#define __VEC4

static const IntegerMask c_SignMask = {0x80000000, 0x80000000, 0x80000000, 0x80000000};

enum Vec4Element { x, y, z, w };

class Vec4 : public MathLibObject
{
public:
  union
  {
    struct { float x, y, z, w; };
    float e[4];
    __m128 r;
  };

  finline Vec4()                                       { }
  finline Vec4(__m128 a)                               { r = a; }
  finline Vec4(const Vec4& a)                          { r = a.r; }
  finline Vec4(float a)                                { r = _mm_set1_ps(a); }
  finline Vec4(float a, float b, float c, float d)     { x = a; y = b; z = c; w = d; }
  finline Vec4(const float* a)                         { r = _mm_loadu_ps(a); }

  finline operator __m128() const                      { return r; }
  finline const Vec4& operator = (const Vec4& a)       { r = a.r; return *this; }

  static finline const Vec4 Zero()                     { return _mm_setzero_ps(); }
  template<int x, int y, int z, int w> static finline const Vec4 Constant() { static const __m128 r = {(float)x, (float)y, (float)z, (float)w}; return r; }

  static finline float LengthSq(const Vec4& a)         { return a.x*a.x + a.y*a.y + a.z*a.z + a.w*a.w; }
  static finline float Length  (const Vec4& a)         { float f; _mm_store_ss(&f, _mm_sqrt_ss(_mm_set_ss(LengthSq(a)))); return f; }

  static finline const Vec4 Sqrt (const Vec4& a)       { return _mm_sqrt_ps(a.r); }
  static finline const Vec4 Rcp  (const Vec4& a)       { __m128 x = _mm_rcp_ps(a.r); return _mm_sub_ps(_mm_add_ps(x, x), _mm_mul_ps(a.r, _mm_mul_ps(x, x))); }
  static finline const Vec4 Rsqrt(const Vec4& a)       { __m128 x = _mm_rsqrt_ps(a.r); return _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_set1_ps(3.0f), _mm_mul_ps(a.r, _mm_mul_ps(x, x))), x), _mm_set1_ps(0.5f)); }
  static finline const Vec4 Normalize(const Vec4& a)   { return _mm_mul_ps(Rsqrt(_mm_set1_ps(LengthSq(a))).r, a.r); }
  static finline const Vec4 Abs  (const Vec4& a)       { return _mm_andnot_ps(c_SignMask.f, a.r); }

  static finline const Vec4 ApproxSqrt (const Vec4& a) { return _mm_mul_ps(a.r, _mm_rsqrt_ps(a.r)); }
  static finline const Vec4 ApproxRcp  (const Vec4& a) { return _mm_rcp_ps(a.r); }
  static finline const Vec4 ApproxRsqrt(const Vec4& a) { return _mm_rsqrt_ps(a.r); }

  static finline const Vec4  Min(const Vec4& a, const Vec4& b) { return _mm_min_ps(a.r, b.r); }
  static finline const Vec4  Max(const Vec4& a, const Vec4& b) { return _mm_max_ps(a.r, b.r); }
  static finline const float Dot(const Vec4& a, const Vec4& b) { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }

  static finline const Vec4 Lerp(const Vec4& a, const Vec4& b, float c) { return _mm_add_ps(a.r, _mm_mul_ps(_mm_sub_ps(b.r, a.r), _mm_set1_ps(c))); }

  template<Vec4Element x, Vec4Element y, Vec4Element z, Vec4Element w> static finline const Vec4 Shuffle(const Vec4& a, const Vec4& b) { return _mm_shuffle_ps(a.r, b.r, _MM_SHUFFLE(w, z, y, x)); }
  template<Vec4Element x, Vec4Element y, Vec4Element z, Vec4Element w> static finline const Vec4 Swizzle(const Vec4& a)                { return _mm_shuffle_ps(a.r, a.r, _MM_SHUFFLE(w, z, y, x)); }

  template<Vec4Element c> static finline int Truncate(const Vec4& a) { return _mm_cvttss_si32(_mm_shuffle_ps(a.r, a.r, _MM_SHUFFLE(c, c, c, c))); }
  template<Vec4Element c> static finline int Round(const Vec4& a)    { return  _mm_cvtss_si32(_mm_shuffle_ps(a.r, a.r, _MM_SHUFFLE(c, c, c, c))); }
  template<Vec4Element c> static finline int Ceil(const Vec4& a)     { __m128 t = _mm_add_ps(a.r, _mm_set1_ps(0.5f)); return _mm_cvtss_si32(_mm_shuffle_ps(t, t, _MM_SHUFFLE(c, c, c, c))); }
  template<Vec4Element c> static finline int Floor(const Vec4& a)    { __m128 t = _mm_sub_ps(a.r, _mm_set1_ps(0.5f)); return _mm_cvtss_si32(_mm_shuffle_ps(t, t, _MM_SHUFFLE(c, c, c, c))); }

  finline float&       operator [] (int i)       { return e[i]; }
  finline const float& operator [] (int i) const { return e[i]; }
};

finline bool operator == (const Vec4& a, const Vec4& b) { return _mm_movemask_ps(_mm_cmpeq_ps(a.r, b.r))==15; }
finline bool operator != (const Vec4& a, const Vec4& b) { return _mm_movemask_ps(_mm_cmpeq_ps(a.r, b.r))!=15; }
finline bool operator >= (const Vec4& a, const Vec4& b) { return _mm_movemask_ps(_mm_cmpge_ps(a.r, b.r))==15; }
finline bool operator <= (const Vec4& a, const Vec4& b) { return _mm_movemask_ps(_mm_cmple_ps(a.r, b.r))==15; }

finline const Vec4 operator + (const Vec4& a, const Vec4& b) { return _mm_add_ps(a.r, b.r); }
finline const Vec4 operator - (const Vec4& a, const Vec4& b) { return _mm_sub_ps(a.r, b.r); }
finline const Vec4 operator * (const Vec4& a, const Vec4& b) { return _mm_mul_ps(a.r, b.r); }
finline const Vec4 operator / (const Vec4& a, const Vec4& b) { return _mm_div_ps(a.r, b.r); }

finline const Vec4 operator + (float a, const Vec4& b) { return _mm_add_ps(_mm_set1_ps(a), b.r); }
finline const Vec4 operator - (float a, const Vec4& b) { return _mm_sub_ps(_mm_set1_ps(a), b.r); }
finline const Vec4 operator * (float a, const Vec4& b) { return _mm_mul_ps(_mm_set1_ps(a), b.r); }
finline const Vec4 operator / (float a, const Vec4& b) { return _mm_div_ps(_mm_set1_ps(a), b.r); }

finline const Vec4 operator + (const Vec4& a, float b) { return _mm_add_ps(a.r, _mm_set1_ps(b)); }
finline const Vec4 operator - (const Vec4& a, float b) { return _mm_sub_ps(a.r, _mm_set1_ps(b)); }
finline const Vec4 operator * (const Vec4& a, float b) { return _mm_mul_ps(a.r, _mm_set1_ps(b)); }
finline const Vec4 operator / (const Vec4& a, float b) { return _mm_div_ps(a.r, _mm_set1_ps(b)); }

finline const Vec4 operator & (const Vec4& a, const Vec4& b) { return _mm_and_ps(a.r, b.r); }
finline const Vec4 operator | (const Vec4& a, const Vec4& b) { return _mm_or_ps (a.r, b.r); }
finline const Vec4 operator ^ (const Vec4& a, const Vec4& b) { return _mm_xor_ps(a.r, b.r); }

finline const Vec4 operator + (const Vec4& a) { return a; }
finline const Vec4 operator - (const Vec4& a) { return _mm_xor_ps(a.r, c_SignMask.f); }

#endif //#ifndef __VEC4
