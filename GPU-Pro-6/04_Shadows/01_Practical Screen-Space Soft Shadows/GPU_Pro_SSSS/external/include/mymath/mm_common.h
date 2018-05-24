#ifndef mm_common_h
#define mm_common_h

#include <iostream>
#include <cmath>
#include <algorithm>
#include <assert.h>

//#define MYMATH_USE_SSE2
//#define MYMATH_USE_FMA
//#define MYMATH_FORCE_INLINE

#ifdef MYMATH_USE_SSE2
#include "x86intrin.h"
#endif

#ifdef MYMATH_FORCE_INLINE
#ifdef _MSC_VER //msvc++
#define MYMATH_INLINE __forceinline
#else //g++ and clang
#define MYMATH_INLINE __attribute__((always_inline))
#endif
#else
#define MYMATH_INLINE inline
#endif

//align variables to X bytes
#ifdef MYMATH_USE_SSE2
  #ifdef _MSC_VER //msvc++
    #define MYMATH_ALIGNED(x) __declspec(align(x))
  #elif __GNUC__  //g++
    #define MYMATH_ALIGNED(x) __attribute__((aligned (x)))
  #elif __clang__ //clang
    #define MYMATH_ALIGNED(x) __attribute__((__aligned__(x)))
  #endif
#else
  #define MYMATH_ALIGNED(x)
#endif

#ifdef MYMATH_USE_SSE2
#define MYMATH_SHUFFLE(x, y, z, w) (_MM_SHUFFLE(w, z, y, x))
#endif

//align variables to 16 bytes (GPU friendly)
#define MYMATH_GPU_ALIGNED MYMATH_ALIGNED(16)

//makes sure only explicit cast is available between vecn and float/double etc.
#define MYMATH_STRICT_GLSL 0
#define MYMATH_DOUBLE_PRECISION 0

#ifdef _WIN32
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#endif

namespace mymath
{
  static const float epsilon = 0.00001f;
  static const double depsilon = 0.00000001;

  namespace impl
  {
    typedef int post; //this serves as a flag that an increment is a post increment

    template< class t >
    static bool is_eq( t a, t b )
    {
      return a == b;
    }

    static bool is_eq( float a, float b )
    {
      return std::abs( a - b ) < epsilon;
    }

    static bool is_eq( double a, double b )
    {
      return std::abs( a - b ) < depsilon;
    }
  }

  static const float pi = 3.1415926535897932384626433832795f;
  static const float two_pi = 2.0f * pi;
  static const float pi_div_180 = pi / 180.0f;
  static const float inv_pi_div_180 = 180.0f / pi;

#define MYMATH_INVERSESQRT( t ) \
  MYMATH_INLINE t inversesqrt( const t& a ) \
  { \
    return 1 / std::sqrt( a ); \
  }

#define MYMATH_STEP( t ) \
  MYMATH_INLINE t step( const t& a, const t& b ) \
  { \
    return b < a ? 0 : 1; \
  }

#define MYMATH_MIX( t ) \
  MYMATH_INLINE t mix( const t& a, const t& b, const t& c ) \
  { \
    return a * ( 1 - c ) + b * c; \
  }

#define MYMATH_FRACT( t ) \
  MYMATH_INLINE t fract( const t& a ) \
  { \
    return a - floor( a ); \
  }

#define MYMATH_ATAN( t ) \
  MYMATH_INLINE t atan( const t& a, const t& b ) \
  { \
    return std::atan( b / a ); \
  }

#define MYMATH_CLAMP( t ) \
  MYMATH_INLINE t clamp( const t& a, const t& b, const  t& c ) \
  { \
    return std::min( std::max( a, b ), c ); \
  }

#define MYMATH_SMOOTHSTEP( t ) \
  MYMATH_INLINE t smoothstep( const t& a, const t& b, const t& c ) \
  { \
    float u = ( c - a ) / ( b - a ); \
    u = clamp( u, 0, 1 ); \
    return u * u * ( 3 - 2 * u ); \
  }

#define MYMATH_FMA( t ) \
  MYMATH_INLINE t fma( const t& a, const t& b, const t& c ) \
  { \
    return a * b + c; \
  }

  MYMATH_INLINE float radians( const float& degrees )
  {
    return degrees * pi_div_180;
  }

  MYMATH_INLINE float degrees( const float& radians )
  {
    return radians * inv_pi_div_180;
  }

#define MYMATH_SIGN( t ) \
  MYMATH_INLINE t sign( const t& num ) \
  { \
    if( num > 0 ) \
    { \
      return 1; \
    } \
    else if( num < 0 ) \
    { \
      return -1; \
    } \
    else \
    { \
      return num; \
    } \
  }

#define MYMATH_ASINH( t ) \
  MYMATH_INLINE t asinh( const t& num ) \
  { \
    return std::log( num + std::sqrt( num * num + 1 ) ); \
  }

#define MYMATH_ACOSH( t ) \
  MYMATH_INLINE t acosh( const t& num ) \
  { \
    return std::log( num + std::sqrt( num * num - 1 ) ); \
  }

#define MYMATH_ATANH( t ) \
  MYMATH_INLINE t atanh( const t& num ) \
  { \
    return std::log( ( 1 + num ) / ( 1 - num ) ) / 2; \
  }

#define MYMATH_LOG2( t ) \
  MYMATH_INLINE t log2( const t& num ) \
  { \
    return std::log( num ) / std::log( 2 ); \
  }

#define MYMATH_TRUNC( t ) \
  MYMATH_INLINE t trunc( const t& num ) \
  { \
    return num < 0 ? -floor( -num ) : floor( num ); \
  }

#define MYMATH_ROUND( t ) \
  MYMATH_INLINE t round( const t& num ) \
  { \
    if( num < 0 ) \
    { \
      return t( int( num - 0.5 ) ); \
    } \
    else \
    { \
      return t( int( num + 0.5 ) ); \
    } \
  }

  MYMATH_INLINE bool isnan( const float& num )
  {
#ifdef _WIN32
    return _isnan( num ) != 0;
#else
    return std::isnan( num );
#endif
  }

  MYMATH_INLINE bool isinf( const float& num )
  {
#ifdef _WIN32
    return _fpclass( num ) == _FPCLASS_NINF || _fpclass( num ) == _FPCLASS_PINF;
#else
    return std::isinf( num );
#endif
  }

#define MYMATH_MIN( t ) \
  MYMATH_INLINE t min( const t& a, const t& b ) \
  { \
    return std::min( a, b ); \
  }

#define MYMATH_MAX( t ) \
  MYMATH_INLINE t max( const t& a, const t& b ) \
  { \
    return std::max( a, b ); \
  }

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif
  MYMATH_STEP( float )
  MYMATH_LOG2( float )
#ifdef _WIN32
#pragma warning( pop )
#pragma warning( disable : 4244 )
#endif

  MYMATH_INVERSESQRT( float )
  MYMATH_MIX( float )
  MYMATH_FRACT( float )
  MYMATH_ATAN( float )
  MYMATH_CLAMP( float )
  MYMATH_SMOOTHSTEP( float )
  MYMATH_FMA( float )
  MYMATH_SIGN( float )
  MYMATH_ASINH( float )
  MYMATH_ACOSH( float )
  MYMATH_ATANH( float )
  MYMATH_TRUNC( float )
  MYMATH_ROUND( float )
  MYMATH_MIN( float )
  MYMATH_MAX( float )

#if MYMATH_DOUBLE_PRECISION == 1
  static const double dpi = 3.1415926535897932384626433832795;
  static const double dtwo_pi = 2.0 * dpi;
  static const double dpi_div_180 = dpi / 180.0;
  static const double dinv_pi_div_180 = 180.0 / dpi;

  MYMATH_INLINE double radians( const double& degrees )
  {
    return degrees * dpi_div_180;
  }

  MYMATH_INLINE double degrees( const double& radians )
  {
    return radians * dinv_pi_div_180;
  }

  MYMATH_INLINE bool isnan( const double& num )
  {
#ifdef _WIN32
    return _isnan( num ) != 0;
#else
    return std::isnan( num );
#endif
  }

  MYMATH_INLINE bool isinf( const double& num )
  {
#ifdef _WIN32
    return _fpclass( num ) == _FPCLASS_NINF || _fpclass( num ) == _FPCLASS_PINF;
#else
    return std::isinf( num );
#endif
  }

  MYMATH_INVERSESQRT( double )
  MYMATH_STEP( double )
  MYMATH_MIX( double )
  MYMATH_FRACT( double )
  MYMATH_ATAN( double )
  MYMATH_CLAMP( double )
  MYMATH_SMOOTHSTEP( double )
  MYMATH_FMA( double )
  MYMATH_SIGN( double )
  MYMATH_ASINH( double )
  MYMATH_ACOSH( double )
  MYMATH_ATANH( double )
  MYMATH_LOG2( double )
  MYMATH_TRUNC( double )
  MYMATH_ROUND( double )
  MYMATH_MIN( double )
  MYMATH_MAX( double )
#endif
}

namespace mm = mymath;

#endif


