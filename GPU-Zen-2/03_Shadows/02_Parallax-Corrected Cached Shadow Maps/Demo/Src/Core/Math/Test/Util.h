#include "../Math.h"
#include "../../Util/Random.h"

#define ASSERT(a) \
  if(!(a)) { \
   printf("%s(%d): %s\n", __FILE__, __LINE__, #a); \
   exit(1); }

const char* GetFmtFileName(const char *fmt, ...)
{
  static char buf[256];
  va_list args;
  va_start(args, fmt);
  vsprintf_s(buf, sizeof(buf), fmt, args);
  va_end(args);
  return buf;
}

bool IsEqual(const Mat4x4& a, const Mat4x4& b, float e)
{
  for(int i=0; i<16; ++i)
    if(fabsf(a.e[i] - b.e[i])>e)
      return false;
  return true;
}

Vec3 RandomVec3(RandomLCG& r)
{
  return Vec3(r.GetFloat(), r.GetFloat(), r.GetFloat());
}

Vec4 RandomVec4(RandomLCG& r)
{
  return Vec4(r.GetFloat(), r.GetFloat(), r.GetFloat(), r.GetFloat());
}

#ifdef ENABLE_SSE2

Vec4i RandomVec4i(RandomLCG& r)
{
  return Vec4i(r.GetUInt(), r.GetUInt(), r.GetUInt(), r.GetUInt()) - RandomLCG::c_MaxUINT/2;
}

#endif