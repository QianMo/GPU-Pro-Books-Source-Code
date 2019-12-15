#ifndef __MATH
#define __MATH

#include <math.h>

enum Vec4Element { x, y, z, w };

#ifdef _WIN32

#pragma warning(push)
#pragma warning(disable:4201)

#include <xmmintrin.h>
#include <intrin.h>

#ifndef EXTERNAL_SSE_CONTROL
  #define ENABLE_SSE2
#endif

#ifdef ENABLE_SSE4
  #define ENABLE_SSE2
#endif

#ifdef ENABLE_SSE2
  #include <emmintrin.h>
#endif

#ifdef ENABLE_SSE4
  #include <smmintrin.h>
#endif

#define align16  __declspec(align(16))
#define finline  __forceinline
#define restrict __restrict

align16 union IntegerMask
{
   unsigned i[4];
   __m128 f;
#ifdef ENABLE_SSE2
   __m128d d;
#endif
};

class MathLibObject
{
public:
  inline void* operator new     (size_t, void* p)   { return p; }
  inline void* operator new     (size_t n)          { return _aligned_malloc(n, 16); }
  inline void* operator new[]   (size_t n)          { return _aligned_malloc(n, 16); }
  inline void operator delete   (void* p)           { _aligned_free(p); }
  inline void operator delete[] (void* p)           { _aligned_free(p); }
};

class Math
{
public:
  static void Init()
  {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  }
};

#include "UVec.h"
#include "x86/Swizzle.h"
#include "x86/Dot.inl"
#include "x86/Vec4.h"
#include "x86/Vec3.h"
#include "x86/Vec2.h"
#include "x86/Mat4x4.h"
#include "x86/Quat.h"
#include "x86/AABB.h"

#ifdef ENABLE_SSE2
  #include "x86/Vec4Cbrt.inl"
  #include "x86/Vec4SinCos.inl"
  #include "x86/Vec4i.h"
  #include "x86/Vec4d.h"
  #include "x86/Vec3d.h"
  #include "x86/Vec2d.h"
  #include "x86/Mat4x4d.h"
  #include "Mat4x4d.h"
#endif

#include "x86/Util.h"

#pragma warning(pop)

#endif //#ifdef _WIN32

static const Vec4 c_XAxis(1,0,0,0);
static const Vec4 c_YAxis(0,1,0,0);
static const Vec4 c_ZAxis(0,0,1,0);
static const Vec4 c_WAxis(0,0,0,1);
static const float c_PI = 3.1415926535897932384626433832795f;

#include "Mat4x4.h"
#include "Util.h"

#endif //#ifndef __MATH
