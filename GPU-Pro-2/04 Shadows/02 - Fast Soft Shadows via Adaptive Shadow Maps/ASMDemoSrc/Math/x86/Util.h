/* MSVC SSE math library implementation.

   Pavlo Turchyn <pavlo@nichego-novogo.net> Aug 2010 */

#ifndef __MATH_UTIL
#define __MATH_UTIL

finline unsigned ToD3DColor(const Vec4& c)
{
  __m128 t = _mm_mul_ps(_mm_set1_ps(255), c.r);
  unsigned r = _mm_cvtss_si32(t);
  unsigned g = _mm_cvtss_si32(SWZ_YYYY(t));
  unsigned b = _mm_cvtss_si32(SWZ_ZZZZ(t));
  unsigned a = _mm_cvtss_si32(SWZ_WWWW(t));
  return ((a<<24) | (r<<16) | (g<<8) | b);
}

finline void OBBtoAABB(const Mat4x4& OBB, Vec3* pMin, Vec3* pMax)
{
  __m128 t = _mm_add_ps(_mm_add_ps(_mm_andnot_ps(c_SignMask.f, OBB.r[0]), _mm_andnot_ps(c_SignMask.f, OBB.r[1])), _mm_andnot_ps(c_SignMask.f, OBB.r[2]));
  pMin[0] = _mm_sub_ps(OBB.r[3], t);
  pMax[0] = _mm_add_ps(OBB.r[3], t);
}

#endif //#ifndef __MATH_UTIL
