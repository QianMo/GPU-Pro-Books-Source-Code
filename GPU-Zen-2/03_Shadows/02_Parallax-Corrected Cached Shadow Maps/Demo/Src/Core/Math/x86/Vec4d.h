#ifndef __VEC4d
#define __VEC4d

static const IntegerMask c_SignMaskD = {0, 0x80000000, 0, 0x80000000};
static const IntegerMask c_NSignMaskD = {0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff};

struct Vec4d_POD : public MathLibObject
{
  union
  {
    struct { double x, y, z, w; };
    __m128d r[2];
    double e[4];
  };
};

class Vec4d : public Vec4d_POD
{
public:
  finline Vec4d()                                       { }
  finline Vec4d(const Vec4d_POD& a)                     { r[0] = a.r[0]; r[1] = a.r[1]; }
  finline Vec4d(__m128d a, __m128d b)                   { r[0] = a; r[1] = b; }
  finline Vec4d(double a)                               { r[0] = r[1] = _mm_set1_pd(a); }
  finline Vec4d(double a, double b, double c, double d) { r[0] = _mm_set_pd(b, a); r[1] = _mm_set_pd(d, c); }
  finline Vec4d(const double* a)                        { r[0] = _mm_loadu_pd(&a[0]); r[1] = _mm_loadu_pd(&a[2]); }

  finline void Store(double* a) const                   { _mm_storeu_pd(&a[0], r[0]); _mm_storeu_pd(&a[2], r[1]); }
  finline const Vec4d& operator = (const Vec4d& a)      { r[0] = a.r[0]; r[1] = a.r[1]; return *this; }

  static finline const Vec4d Zero()                     { return Vec4d(_mm_setzero_pd(), _mm_setzero_pd()); }

  template<int x, int y, int z, int w> static finline const Vec4d Constant() { static const __m128d r0 = { x, y }, r1 = { z, w }; return Vec4d(r0, r1); }

  static finline const Vec4  Convert(const Vec4d& a)    { return _mm_or_ps(_mm_cvtpd_ps(a.r[0]), SWZ_ZWXY(_mm_cvtpd_ps(a.r[1]))); }
  static finline const Vec4d Convert(const Vec4& a)     { return Vec4d(_mm_cvtps_pd(a.r), _mm_cvtps_pd(SWZ_ZWXY(a.r))); }

  static finline double LengthSq(const Vec4d& a)        { return _mm_cvtsd_f64(_mm_add_pd(dp_m128(a.r[0], a.r[0]), dp_m128(a.r[1], a.r[1]))); }
  static finline double Length  (const Vec4d& a)        { return _mm_cvtsd_f64(_mm_sqrt_sd(_mm_setzero_pd(), _mm_add_pd(dp_m128(a.r[0], a.r[0]), dp_m128(a.r[1], a.r[1])))); }

  static finline const Vec4d Sqrt (const Vec4d& a)      { return Vec4d(_mm_sqrt_pd(a.r[0]), _mm_sqrt_pd(a.r[1])); }
  static finline const Vec4d Normalize(const Vec4d& a)  { __m128d t = _mm_sqrt_pd(_mm_add_pd(dp_m128(a.r[0], a.r[0]), dp_m128(a.r[1], a.r[1]))); return Vec4d(_mm_div_pd(a.r[0], t), _mm_div_pd(a.r[1], t)); }
  static finline const Vec4d Abs (const Vec4d& a)       { return Vec4d(_mm_and_pd(c_NSignMaskD.d, a.r[0]), _mm_and_pd(c_NSignMaskD.d, a.r[1])); }

  static finline const Vec4d Min (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_min_pd(a.r[0], b.r[0]), _mm_min_pd(a.r[1], b.r[1])); }
  static finline const Vec4d Max (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_max_pd(a.r[0], b.r[0]), _mm_max_pd(a.r[1], b.r[1])); }
  static finline const double Dot(const Vec4d& a, const Vec4d& b) { return _mm_cvtsd_f64(_mm_add_pd(dp_m128(a.r[0], b.r[0]), dp_m128(a.r[1], b.r[1]))); }

  finline double&       operator [] (int i)       { return e[i]; }
  finline const double& operator [] (int i) const { return e[i]; }

  finline const Vec4d& operator += (const Vec4d& a) { r[0] = _mm_add_pd(r[0], a.r[0]); r[1] = _mm_add_pd(r[1], a.r[1]); return *this; }
  finline const Vec4d& operator -= (const Vec4d& a) { r[0] = _mm_sub_pd(r[0], a.r[0]); r[1] = _mm_sub_pd(r[1], a.r[1]); return *this; }
  finline const Vec4d& operator *= (const Vec4d& a) { r[0] = _mm_mul_pd(r[0], a.r[0]); r[1] = _mm_mul_pd(r[1], a.r[1]); return *this; }
  finline const Vec4d& operator /= (const Vec4d& a) { r[0] = _mm_div_pd(r[0], a.r[0]); r[1] = _mm_div_pd(r[1], a.r[1]); return *this; }

  finline const Vec4d& operator += (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_add_pd(r[0], t); r[1] = _mm_add_pd(r[1], t); return *this; }
  finline const Vec4d& operator -= (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_sub_pd(r[0], t); r[1] = _mm_sub_pd(r[1], t); return *this; }
  finline const Vec4d& operator *= (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_mul_pd(r[0], t); r[1] = _mm_mul_pd(r[1], t); return *this; }
  finline const Vec4d& operator /= (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_div_pd(r[0], t); r[1] = _mm_div_pd(r[1], t); return *this; }
};

finline const Vec4d operator + (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_add_pd(a.r[0], b.r[0]), _mm_add_pd(a.r[1], b.r[1])); }
finline const Vec4d operator - (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_sub_pd(a.r[0], b.r[0]), _mm_sub_pd(a.r[1], b.r[1])); }
finline const Vec4d operator * (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_mul_pd(a.r[0], b.r[0]), _mm_mul_pd(a.r[1], b.r[1])); }
finline const Vec4d operator / (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_div_pd(a.r[0], b.r[0]), _mm_div_pd(a.r[1], b.r[1])); }

finline const Vec4d operator + (double a, const Vec4d& b) { __m128d t = _mm_set1_pd(a); return Vec4d(_mm_add_pd(t, b.r[0]), _mm_add_pd(t, b.r[1])); }
finline const Vec4d operator - (double a, const Vec4d& b) { __m128d t = _mm_set1_pd(a); return Vec4d(_mm_sub_pd(t, b.r[0]), _mm_sub_pd(t, b.r[1])); }
finline const Vec4d operator * (double a, const Vec4d& b) { __m128d t = _mm_set1_pd(a); return Vec4d(_mm_mul_pd(t, b.r[0]), _mm_mul_pd(t, b.r[1])); }
finline const Vec4d operator / (double a, const Vec4d& b) { __m128d t = _mm_set1_pd(a); return Vec4d(_mm_div_pd(t, b.r[0]), _mm_div_pd(t, b.r[1])); }

finline const Vec4d operator + (const Vec4d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec4d(_mm_add_pd(a.r[0], t), _mm_add_pd(a.r[1], t)); }
finline const Vec4d operator - (const Vec4d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec4d(_mm_sub_pd(a.r[0], t), _mm_sub_pd(a.r[1], t)); }
finline const Vec4d operator * (const Vec4d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec4d(_mm_mul_pd(a.r[0], t), _mm_mul_pd(a.r[1], t)); }
finline const Vec4d operator / (const Vec4d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec4d(_mm_div_pd(a.r[0], t), _mm_div_pd(a.r[1], t)); }

finline const Vec4d operator & (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_and_pd(a.r[0], b.r[0]), _mm_and_pd(a.r[1], b.r[1])); }
finline const Vec4d operator | (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_or_pd (a.r[0], b.r[0]), _mm_or_pd (a.r[1], b.r[1])); }
finline const Vec4d operator ^ (const Vec4d& a, const Vec4d& b) { return Vec4d(_mm_xor_pd(a.r[0], b.r[0]), _mm_xor_pd(a.r[1], b.r[1])); }

finline const Vec4d operator + (const Vec4d& a) { return a; }
finline const Vec4d operator - (const Vec4d& a) { return Vec4d(_mm_xor_pd(a.r[0], c_SignMaskD.d), _mm_xor_pd(a.r[1], c_SignMaskD.d)); }

#endif //#ifndef __VEC4d
