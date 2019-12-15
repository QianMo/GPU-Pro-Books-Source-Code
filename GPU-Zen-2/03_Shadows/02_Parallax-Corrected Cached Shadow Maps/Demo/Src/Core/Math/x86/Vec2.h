#ifndef __VEC2
#define __VEC2

class Mat4x4;

class Vec2 : public Vec4
{
public:
  finline Vec2()                                  { }
  finline Vec2(__m128 a)                          { r = _mm_movelh_ps(a, _mm_setzero_ps()); }
  finline Vec2(const Vec4& a)                     { r = _mm_movelh_ps(a.r, _mm_setzero_ps()); }
  finline Vec2(float a)                           { r = SWZ_XXZW(_mm_set_ss(a)); }
  finline Vec2(float a, float b)                  { r = _mm_set_ps(0, 0, b, a); }
  finline Vec2(const float* a)                    { r = _mm_loadl_pi(_mm_setzero_ps(), (__m64*)a); }
  finline Vec2(const UVec2& a)                    { x = a.x; y = a.y; z = w = 0; }

  finline void Store(float* a) const              { _mm_storel_pi((__m64*)a, r); }
  finline operator UVec2() const                  { UVec2 a = { x, y }; return a; }
  finline const Vec2& operator = (const Vec2& a)  { r = a.r; return *this; }

  static finline const Vec4 Point(const Vec2& a)  { return _mm_or_ps(a.r, c_WOne); }
  static finline const Vec4 Vector(const Vec2& a) { return a; }

  template<int x, int y> static finline const Vec2 Constant() { static const __m128 r = {(float)x, (float)y, 0, 0}; Vec2 v; v.r = r; return v; }

  static finline float Cross(const Vec4& a, const Vec4& b) { __m128 t = _mm_mul_ps(a.r, SWZ_YXWZ(b.r)); return _mm_cvtss_f32(_mm_sub_ss(t, SWZ_YYYY(t))); }

  static finline const Vec2 Project(const Vec2&, const Mat4x4&);
};

#endif //#ifndef __VEC2
