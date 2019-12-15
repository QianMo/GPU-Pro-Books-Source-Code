#ifndef __VEC4
#define __VEC4

static const IntegerMask c_SignMask = {0x80000000, 0x80000000, 0x80000000, 0x80000000};

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
  finline Vec4(float a, float b, float c, float d)     { r = _mm_set_ps(d, c, b, a); }
  finline Vec4(const float* a)                         { r = _mm_loadu_ps(a); }
  finline Vec4(const UVec4& a)                         { r = _mm_loadu_ps(&a.x); }

  finline void Store(float* a) const                   { _mm_storeu_ps(a, r); }
  finline void Stream(float* a) const                  { _mm_stream_ps(a, r); }
  static finline void FlushStream()                    { _mm_sfence(); }
  finline operator __m128() const                      { return r; }
  finline operator UVec4() const                       { UVec4 a = { x, y, z, w }; return a; }
  finline const Vec4& operator = (const Vec4& a)       { r = a.r; return *this; }

  static finline const Vec4 Zero()                     { return _mm_setzero_ps(); }
  template<int x, int y, int z, int w> static finline const Vec4 Constant() { static const __m128 r = {(float)x, (float)y, (float)z, (float)w}; return r; }

  static finline float LengthSq(const Vec4& a)         { return _mm_cvtss_f32(dp_m128(a.r, a.r)); }
  static finline float Length  (const Vec4& a)         { return _mm_cvtss_f32(_mm_sqrt_ss(dp_m128(a.r, a.r))); }
  static finline unsigned AsMask(const Vec4& a)        { return _mm_movemask_ps(a.r); }

  static finline const Vec4 Sqrt (const Vec4& a)       { return _mm_sqrt_ps(a.r); }
  static finline const Vec4 Cbrt (const Vec4& a);      // in Vec4Cbrt.inl
  static finline const Vec4 Rcp  (const Vec4& a)       { __m128 x = _mm_rcp_ps(a.r); return _mm_sub_ps(_mm_add_ps(x, x), _mm_mul_ps(a.r, _mm_mul_ps(x, x))); }
  static finline const Vec4 Rsqrt(const Vec4& a)       { __m128 x = _mm_rsqrt_ps(a.r); return _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(_mm_set1_ps(3.0f), _mm_mul_ps(a.r, _mm_mul_ps(x, x))), x), _mm_set1_ps(0.5f)); }
  static finline const Vec4 Normalize(const Vec4& a)   { __m128 t = dp_m128(a.r, a.r); return _mm_mul_ps(Rsqrt(SWZ_XXXX(t)).r, a.r); }
  static finline const Vec4 Abs  (const Vec4& a)       { return _mm_andnot_ps(c_SignMask.f, a.r); }

  static finline const Vec4 ApproxSqrt (const Vec4& a) { return _mm_mul_ps(a.r, _mm_rsqrt_ps(a.r)); }
  static finline const Vec4 ApproxCbrt (const Vec4& a);// in Vec4Cbrt.inl
  static finline const Vec4 ApproxRcp  (const Vec4& a) { return _mm_rcp_ps(a.r); }
  static finline const Vec4 ApproxRsqrt(const Vec4& a) { return _mm_rsqrt_ps(a.r); }

  static finline const Vec4 Min (const Vec4& a, const Vec4& b) { return _mm_min_ps(a.r, b.r); }
  static finline const Vec4 Max (const Vec4& a, const Vec4& b) { return _mm_max_ps(a.r, b.r); }
  static finline const float Dot(const Vec4& a, const Vec4& b) { return _mm_cvtss_f32(dp_m128(a.r, b.r)); }
  static finline const Vec4 CmpEqual        (const Vec4& a, const Vec4& b) { return _mm_cmpeq_ps(a.r, b.r); }
  static finline const Vec4 CmpLess         (const Vec4& a, const Vec4& b) { return _mm_cmplt_ps(a.r, b.r); }
  static finline const Vec4 CmpLessEqual    (const Vec4& a, const Vec4& b) { return _mm_cmple_ps(a.r, b.r); }
  static finline const Vec4 CmpGreater      (const Vec4& a, const Vec4& b) { return _mm_cmpgt_ps(a.r, b.r); }
  static finline const Vec4 CmpGreaterEqual (const Vec4& a, const Vec4& b) { return _mm_cmpge_ps(a.r, b.r); }

  static finline const Vec4 Lerp(const Vec4& a, const Vec4& b, float c) { return _mm_add_ps(a.r, _mm_mul_ps(_mm_sub_ps(b.r, a.r), _mm_set1_ps(c))); }
  static finline const Vec4 Select(const Vec4& a, const Vec4& b, const Vec4& c) { return _mm_or_ps(_mm_and_ps(a.r, b.r), _mm_andnot_ps(a.r, c.r)); }
  static finline void SinCos(const Vec4& a, Vec4& s, Vec4& c); // in Vec4SinCos.inl

  template<Vec4Element x, Vec4Element y, Vec4Element z, Vec4Element w> static finline const Vec4 Shuffle(const Vec4& a, const Vec4& b) { return _mm_shuffle_ps(a.r, b.r, _MM_SHUFFLE(w, z, y, x)); }
  template<Vec4Element x, Vec4Element y, Vec4Element z, Vec4Element w> static finline const Vec4 Swizzle(const Vec4& a)                { return SWZ(a.r, _MM_SHUFFLE(w, z, y, x)); }

  template<Vec4Element c> static finline int Truncate(const Vec4& a) { return _mm_cvttss_si32(SWZ(a.r, _MM_SHUFFLE(c, c, c, c))); }
  template<Vec4Element c> static finline int Round(const Vec4& a)    { return  _mm_cvtss_si32(SWZ(a.r, _MM_SHUFFLE(c, c, c, c))); }
  template<Vec4Element c> static finline int Ceil(const Vec4& a)     { __m128 t = _mm_add_ps(a.r, _mm_set1_ps(0.5f)); return _mm_cvtss_si32(SWZ(t, _MM_SHUFFLE(c, c, c, c))); }
  template<Vec4Element c> static finline int Floor(const Vec4& a)    { __m128 t = _mm_sub_ps(a.r, _mm_set1_ps(0.5f)); return _mm_cvtss_si32(SWZ(t, _MM_SHUFFLE(c, c, c, c))); }

  finline float&       operator [] (int i)       { return e[i]; }
  finline const float& operator [] (int i) const { return e[i]; }

  finline const Vec4& operator += (const Vec4& a) { r = _mm_add_ps(r, a.r); return *this; }
  finline const Vec4& operator -= (const Vec4& a) { r = _mm_sub_ps(r, a.r); return *this; }
  finline const Vec4& operator *= (const Vec4& a) { r = _mm_mul_ps(r, a.r); return *this; }
  finline const Vec4& operator /= (const Vec4& a) { r = _mm_div_ps(r, a.r); return *this; }

  finline const Vec4& operator += (float a) { r = _mm_add_ps(r, _mm_set1_ps(a)); return *this; }
  finline const Vec4& operator -= (float a) { r = _mm_sub_ps(r, _mm_set1_ps(a)); return *this; }
  finline const Vec4& operator *= (float a) { r = _mm_mul_ps(r, _mm_set1_ps(a)); return *this; }
  finline const Vec4& operator /= (float a) { r = _mm_div_ps(r, _mm_set1_ps(a)); return *this; }
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
