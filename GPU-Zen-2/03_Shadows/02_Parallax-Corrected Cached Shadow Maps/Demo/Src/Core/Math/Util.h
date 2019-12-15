#ifndef __MATH_UTIL_COMMON
#define __MATH_UTIL_COMMON

#include <algorithm>

finline unsigned PackVectorU32(const Vec4& c)
{
  Vec4 t = 255.0f*c;
  return (Vec4::Round<w>(t)<<24) | (Vec4::Round<z>(t)<<16) | (Vec4::Round<y>(t)<<8) | Vec4::Round<x>(t);
}

static const IntegerMask c_UnpackVectorU32_C0 = {0x007f8000, 0x007f8000, 0x007f8000, 0x007f8000};
static const IntegerMask c_UnpackVectorU32_C1 = {0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000};

finline const Vec4 UnpackVectorU32(unsigned c)
{
  IntegerMask r;
  r.i[0] = c<<15;
  r.i[1] = c<<7;
  r.i[2] = c>>1;
  r.i[3] = c>>9;
  return (((r.f & c_UnpackVectorU32_C0.f) | c_UnpackVectorU32_C1.f) - 1.0f)*1.0039215686f;
}

static const Vec4 c_LumVec(0.27f, 0.67f, 0.06f, 0);

finline float ColorLuminance(const Vec4& c)
{
  return Vec4::Dot(c, c_LumVec);
}

finline int FloatAsInt(float f)
{
  union { float f; int i; } u = { f };
  return u.i;
}

finline const Vec3 GetArbitraryOrthogonalVector(const Vec3& v)
{
  Vec3 t = Vec3::Abs(v);
  Vec4 p = Vec4::Select(Vec4::CmpGreater(Vec4::Swizzle<x,x,x,x>(t), Vec4::Swizzle<y,y,y,y>(t)) &
                        Vec4::CmpGreater(Vec4::Swizzle<x,x,x,x>(t), Vec4::Swizzle<z,z,z,z>(t)),
                        c_YAxis, c_XAxis);
  return Vec3::Normalize(Vec3::Cross(v, p));
}

finline unsigned FloorLog2(unsigned n)
{
  unsigned long i;
  return _BitScanReverse(&i, n)!=0 ? i : 0;
}

template<class T>
finline const T Clamp(const T& x, const T& a, const T& b)
{
  return std::min(std::max(x, a), b);
}

template<>
finline const Vec4 Clamp<Vec4>(const Vec4& x, const Vec4& a, const Vec4& b)
{
  return Vec4::Min(Vec4::Max(x, a), b);
}

template<class T>
finline const T Saturate(const T& x) 
{ 
  return Clamp<T>(x, 0.0f, 1.0f); 
}

template<class T>
finline const T Smoothstep(const T& edge0, const T& edge1, const T& x)
{
  T t = Saturate<T>((x - edge0)/(edge1 - edge0));
  return t*t*(3.0f - 2.0f*t);
}

finline const Vec4 LinearizeColor(const Vec4& c)
{
  const float p = 2.2f;
  return Vec4(powf(c.x, p), powf(c.y, p), powf(c.z, p), c.w);
}

#endif //#ifndef __MATH_UTIL_COMMON
