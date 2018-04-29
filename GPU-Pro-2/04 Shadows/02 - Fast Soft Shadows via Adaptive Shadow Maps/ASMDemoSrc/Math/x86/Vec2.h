/* MSVC SSE math library implementation.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __VEC2
#define __VEC2

class Mat4x4;

class Vec2 : public Vec4
{
public:
  finline Vec2()                                  { }
  finline Vec2(__m128 a)                          { r = _mm_movelh_ps(a, _mm_setzero_ps()); }
  finline Vec2(const Vec4& a)                     { r = _mm_movelh_ps(a.r, _mm_setzero_ps()); }
  finline Vec2(float a)                           { r = _mm_movelh_ps(_mm_set1_ps(a), _mm_setzero_ps()); }
  finline Vec2(float a, float b)                  { x = a; y = b; z = w = 0; }
  finline Vec2(const float* a)                    { x = a[0]; y = a[1]; z = w = 0; }

  finline const Vec2& operator = (const Vec2& a)  { r = a.r; return *this; }

  static finline const Vec4 Point(const Vec2& a)  { return _mm_or_ps(a.r, c_WOne); }
  static finline const Vec4 Vector(const Vec2& a) { return a; }

  template<int x, int y> static finline const Vec2 Constant() { static const __m128 r = {(float)x, (float)y, 0, 0}; Vec2 v; v.r = r; return v; }

  static finline const float Cross(const Vec4& a, const Vec4& b) { return a.x*b.y - a.y*b.x; }

  static finline const Vec2 Project(const Vec2&, const Mat4x4&);
};

#endif //#ifndef __VEC2
