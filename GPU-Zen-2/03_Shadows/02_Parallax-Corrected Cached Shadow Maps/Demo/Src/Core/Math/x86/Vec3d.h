#ifndef __VEC3d
#define __VEC3d

static const __m128d c_YOneD = { 0, 1 }; 

class Vec3d : public Vec4d
{
public:
  finline Vec3d()                                   { }
  finline Vec3d(const Vec4d& a)                     { r[0] = a.r[0]; r[1] = _mm_unpacklo_pd(a.r[1], _mm_setzero_pd()); }
  finline Vec3d(__m128d a, __m128d b)               { r[0] = a; r[1] = b; }
  finline Vec3d(double a)                           { r[0] = _mm_set1_pd(a); r[1] = _mm_set_sd(a); }
  finline Vec3d(double a, double b, double c)       { r[0] = _mm_set_pd(b, a); r[1] = _mm_set_pd(0, c); }
  finline Vec3d(const double* a)                    { r[0] = _mm_loadu_pd(&a[0]); r[1] = _mm_load_sd(&a[2]); }

  finline void Store(double* a) const               { _mm_storeu_pd(&a[0], r[0]); _mm_store_sd(&a[2], r[1]); }
  finline const Vec3d& operator = (const Vec3d& a)  { r[0] = a.r[0]; r[1] = a.r[1]; return *this; }

  template<int x, int y, int z> static finline const Vec3d Constant() { static const __m128d r0 = { x, y }, r1 = { z, 0 }; return Vec3d(r0, r1); }

  static finline const Vec4d Point(const Vec3d& a)  { return Vec4d(a.r[0], _mm_or_pd(a.r[1], c_YOneD)); }
  static finline const Vec4d Vector(const Vec3d& a) { return a; }

  static finline const Vec3d Cross(const Vec3d& a, const Vec3d& b)
  {
    __m128d t0 = _mm_mul_pd(_mm_shuffle_pd(a.r[0], b.r[0], _MM_SHUFFLE2(0, 1)), _mm_unpacklo_pd(b.r[1], a.r[1]));
    __m128d t1 = _mm_mul_pd(_mm_shuffle_pd(b.r[0], a.r[0], _MM_SHUFFLE2(0, 1)), _mm_unpacklo_pd(a.r[1], b.r[1]));
    __m128d t2 = _mm_mul_pd(a.r[0], SWZ_YX(b.r[0]));
    return Vec3d(_mm_sub_pd(t0, t1), _mm_sub_pd(t2, SWZ_YY(t2)));
  }

  finline const Vec3d& operator += (const Vec4d& a) { r[0] = _mm_add_pd(r[0], a.r[0]); r[1] = _mm_add_sd(r[1], a.r[1]); return *this; }
  finline const Vec3d& operator -= (const Vec4d& a) { r[0] = _mm_sub_pd(r[0], a.r[0]); r[1] = _mm_sub_sd(r[1], a.r[1]); return *this; }
  finline const Vec3d& operator *= (const Vec4d& a) { r[0] = _mm_mul_pd(r[0], a.r[0]); r[1] = _mm_mul_sd(r[1], a.r[1]); return *this; }
  finline const Vec3d& operator /= (const Vec4d& a) { r[0] = _mm_div_pd(r[0], a.r[0]); r[1] = _mm_div_sd(r[1], a.r[1]); return *this; }

  finline const Vec3d& operator += (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_add_pd(r[0], t); r[1] = _mm_add_sd(r[1], t); return *this; }
  finline const Vec3d& operator -= (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_sub_pd(r[0], t); r[1] = _mm_sub_sd(r[1], t); return *this; }
  finline const Vec3d& operator *= (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_mul_pd(r[0], t); r[1] = _mm_mul_sd(r[1], t); return *this; }
  finline const Vec3d& operator /= (double a) { __m128d t = _mm_set1_pd(a); r[0] = _mm_div_pd(r[0], t); r[1] = _mm_div_sd(r[1], t); return *this; }
};

finline const Vec3d operator + (const Vec3d& a, const Vec4d& b) { return Vec3d(_mm_add_pd(a.r[0], b.r[0]), _mm_add_sd(a.r[1], b.r[1])); }
finline const Vec3d operator - (const Vec3d& a, const Vec4d& b) { return Vec3d(_mm_sub_pd(a.r[0], b.r[0]), _mm_sub_sd(a.r[1], b.r[1])); }
finline const Vec3d operator * (const Vec3d& a, const Vec4d& b) { return Vec3d(_mm_mul_pd(a.r[0], b.r[0]), _mm_mul_sd(a.r[1], b.r[1])); }
finline const Vec3d operator / (const Vec3d& a, const Vec4d& b) { return Vec3d(_mm_div_pd(a.r[0], b.r[0]), _mm_div_sd(a.r[1], b.r[1])); }

finline const Vec3d operator + (double a, const Vec3d& b) { __m128d t = _mm_set1_pd(a); return Vec3d(_mm_add_pd(t, b.r[0]), _mm_add_sd(t, b.r[1])); }
finline const Vec3d operator - (double a, const Vec3d& b) { __m128d t = _mm_set1_pd(a); return Vec3d(_mm_sub_pd(t, b.r[0]), _mm_sub_sd(t, b.r[1])); }
finline const Vec3d operator * (double a, const Vec3d& b) { __m128d t = _mm_set1_pd(a); return Vec3d(_mm_mul_pd(t, b.r[0]), _mm_mul_sd(t, b.r[1])); }
finline const Vec3d operator / (double a, const Vec3d& b) { __m128d t = _mm_set1_pd(a); return Vec3d(_mm_div_pd(t, b.r[0]), _mm_div_sd(t, b.r[1])); }

finline const Vec3d operator + (const Vec3d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec3d(_mm_add_pd(a.r[0], t), _mm_add_sd(a.r[1], t)); }
finline const Vec3d operator - (const Vec3d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec3d(_mm_sub_pd(a.r[0], t), _mm_sub_sd(a.r[1], t)); }
finline const Vec3d operator * (const Vec3d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec3d(_mm_mul_pd(a.r[0], t), _mm_mul_sd(a.r[1], t)); }
finline const Vec3d operator / (const Vec3d& a, double b) { __m128d t = _mm_set1_pd(b); return Vec3d(_mm_div_pd(a.r[0], t), _mm_div_sd(a.r[1], t)); }

#endif //#ifndef __VEC3d
