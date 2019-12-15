#ifndef __VEC2d
#define __VEC2d

class Vec2d : public Vec4d
{
public:
  finline Vec2d()                   { }
  finline Vec2d(const Vec4d& a)     { r[0] = a.r[0]; r[1] = _mm_setzero_pd(); }
  finline Vec2d(__m128d a)          { r[0] = a; r[1] = _mm_setzero_pd(); }
  finline Vec2d(double a)           { r[0] = _mm_set1_pd(a); r[1] = _mm_setzero_pd(); }
  finline Vec2d(double a, double b) { r[0] = _mm_set_pd(b, a); r[1] = _mm_setzero_pd(); }
  finline Vec2d(const double* a)    { r[0] = _mm_loadu_pd(&a[0]); r[1] = _mm_setzero_pd(); }

  finline void Store(double* a) const               { _mm_storeu_pd(a, r[0]); }
  finline const Vec2d& operator = (const Vec2d& a)  { r[0] = a.r[0]; r[1] = _mm_setzero_pd(); return *this; }

  template<int x, int y> static finline const Vec2d Constant() { static const __m128d r0 = { x, y }; return r0; }

  static finline const Vec4d Point(const Vec2d& a)  { return Vec4d(a.r[0], c_YOneD); }
  static finline const Vec4d Vector(const Vec2d& a) { return a; }

  static finline double Cross(const Vec2d& a, const Vec2d& b) { __m128d t = _mm_mul_pd(a.r[0], SWZ_YX(b.r[0])); return _mm_cvtsd_f64(_mm_sub_sd(t, SWZ_YY(t))); }

  finline const Vec2d& operator += (const Vec4d& a) { r[0] = _mm_add_pd(r[0], a.r[0]); return *this; }
  finline const Vec2d& operator -= (const Vec4d& a) { r[0] = _mm_sub_pd(r[0], a.r[0]); return *this; }
  finline const Vec2d& operator *= (const Vec4d& a) { r[0] = _mm_mul_pd(r[0], a.r[0]); return *this; }
  finline const Vec2d& operator /= (const Vec4d& a) { r[0] = _mm_div_pd(r[0], a.r[0]); return *this; }

  finline const Vec2d& operator += (double a) { r[0] = _mm_add_pd(r[0], _mm_set1_pd(a)); return *this; }
  finline const Vec2d& operator -= (double a) { r[0] = _mm_sub_pd(r[0], _mm_set1_pd(a)); return *this; }
  finline const Vec2d& operator *= (double a) { r[0] = _mm_mul_pd(r[0], _mm_set1_pd(a)); return *this; }
  finline const Vec2d& operator /= (double a) { r[0] = _mm_div_pd(r[0], _mm_set1_pd(a)); return *this; }
};

finline const Vec2d operator + (const Vec2d& a, const Vec4d& b) { return _mm_add_pd(a.r[0], b.r[0]); }
finline const Vec2d operator - (const Vec2d& a, const Vec4d& b) { return _mm_sub_pd(a.r[0], b.r[0]); }
finline const Vec2d operator * (const Vec2d& a, const Vec4d& b) { return _mm_mul_pd(a.r[0], b.r[0]); }
finline const Vec2d operator / (const Vec2d& a, const Vec4d& b) { return _mm_div_pd(a.r[0], b.r[0]); }

finline const Vec2d operator + (double a, const Vec2d& b) { return _mm_add_pd(_mm_set1_pd(a), b.r[0]); }
finline const Vec2d operator - (double a, const Vec2d& b) { return _mm_sub_pd(_mm_set1_pd(a), b.r[0]); }
finline const Vec2d operator * (double a, const Vec2d& b) { return _mm_mul_pd(_mm_set1_pd(a), b.r[0]); }
finline const Vec2d operator / (double a, const Vec2d& b) { return _mm_div_pd(_mm_set1_pd(a), b.r[0]); }

finline const Vec2d operator + (const Vec2d& a, double b) { return _mm_add_pd(a.r[0], _mm_set1_pd(b)); }
finline const Vec2d operator - (const Vec2d& a, double b) { return _mm_sub_pd(a.r[0], _mm_set1_pd(b)); }
finline const Vec2d operator * (const Vec2d& a, double b) { return _mm_mul_pd(a.r[0], _mm_set1_pd(b)); }
finline const Vec2d operator / (const Vec2d& a, double b) { return _mm_div_pd(a.r[0], _mm_set1_pd(b)); }

#endif //#ifndef __VEC2d
