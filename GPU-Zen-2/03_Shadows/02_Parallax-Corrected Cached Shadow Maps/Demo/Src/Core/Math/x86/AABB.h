#ifndef __AABB2D_H
#define __AABB2D_H

class AABB2D : public Vec4
{
public:
  finline AABB2D()                                 { }
  finline AABB2D(const Vec2& Min, const Vec2& Max) { r = _mm_shuffle_ps(Min.r, Max.r, _MM_SHUFFLE(1,0,1,0)); }
  finline AABB2D(const AABB2D& a)                  { r = a.r; }
  finline AABB2D(const Vec4& a)                    { r = a.r; }

  finline const AABB2D& operator = (const AABB2D& a) { r = a.r; return *this; }

  finline const Vec2 GetMin() const { return r; }
  finline const Vec2 GetMax() const { return SWZ_ZWXY(r); }
  finline const Vec2 Size() const   { return _mm_sub_ps(SWZ_ZWXY(r), r); }
  finline const Vec2 Center() const { return _mm_mul_ps(_mm_set1_ps(0.5f), _mm_add_ps(SWZ_ZWXY(r), r)); }

  static finline bool IsIntersecting(const AABB2D& a, const AABB2D& b)
  {
    __m128 t = SWZ_ZWXY(b.r);
    return ((_mm_movemask_ps(_mm_cmpge_ps(t, a.r))&3) | (_mm_movemask_ps(_mm_cmple_ps(t, a.r))&12))==15;
  }
  static finline float GetOverlapArea(const AABB2D& a, const AABB2D& b)
  {
    __m128 t0 = _mm_min_ps(a.r, b.r);
    __m128 t1 = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(SWZ_ZWXY(t0), _mm_max_ps(a.r, b.r)));
    return _mm_cvtss_f32(_mm_mul_ss(t1, SWZ_YYYY(t1)));
  }
};

#endif //#ifndef __AABB2D_H
