#ifndef mm_vec_swizzle_out_h
#define mm_vec_swizzle_out_h

#include "mm_common.h"
#include "mm_fvec2_impl.h"
#include "mm_fvec3_impl.h"
#include "mm_fvec4_impl.h"

namespace mymath
{
  namespace impl
  {
    //vec3 swizzlers for vec2
    template< int a >
    vec2i<float>::swizzle < a, a, a, -3 >::operator vec3i<float>() const
    {
      return vec3i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, a, 0 ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle < b, a, a, -3 >::operator vec3i<float>() const
    {
      return vec3i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( b, a, a, 0 ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle < a, b, a, -3 >::operator vec3i<float>() const
    {
      return vec3i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, b, a, 0 ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle < a, a, b, -3 >::operator vec3i<float>() const
    {
      return vec3i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, b, 0 ) ) );
    }

    //vec4 swizzlers for vec2
    template<int a>
    vec2i<float>::swizzle<a, a, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, a, a ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle<b, a, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( b, a, a, a ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle<a, b, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, b, a, a ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle<a, a, b, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, b, a ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle<a, a, a, b>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, a, b ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle<a, b, a, b>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, b, a, b ) ) );
    }

    template<int a, int b>
    vec2i<float>::swizzle<a, a, b, b>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, b, b ) ) );
    }

    //vec4 swizzlers for vec3
    template<int a>
    vec3i<float>::swizzle<a, a, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, a, a ) ) );
    }

    template<int a, int bt>
    vec3i<float>::swizzle<bt, a, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( bt, a, a, a ) ) );
    }

    template<int a, int bt>
    vec3i<float>::swizzle<a, bt, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, bt, a, a ) ) );
    }

    template<int a, int bt>
    vec3i<float>::swizzle<a, a, bt, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, bt, a ) ) );
    }

    template<int a, int bt>
    vec3i<float>::swizzle<a, a, a, bt>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, a, bt ) ) );
    }

    template<int a, int bt>
    vec3i<float>::swizzle<bt, bt, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( bt, bt, a, a ) ) );
    }

    template<int a, int bt>
    vec3i<float>::swizzle<a, bt, bt, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, bt, bt, a ) ) );
    }

    template<int a, int bt, int c>
    vec3i<float>::swizzle<bt, c, a, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( bt, c, a, a ) ) );
    }

    template<int a, int bt, int c>
    vec3i<float>::swizzle<a, bt, c, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, bt, c, a ) ) );
    }

    template<int a, int bt, int c>
    vec3i<float>::swizzle<a, a, bt, c>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( a, a, bt, c ) ) );
    }

    template<int a, int bt, int c>
    vec3i<float>::swizzle<bt, a, c, a>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( bt, a, c, a ) ) );
    }

    template<int a, int bt, int c>
    vec3i<float>::swizzle<bt, a, a, c>::operator vec4i<float>() const
    {
      return vec4i<float>( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( bt, a, a, c ) ) );
    }
  }
}

#endif


