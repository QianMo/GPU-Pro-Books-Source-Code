#ifndef mm_fvec_func_h
#define mm_fvec_func_h

#include "mm_common.h"
#include "mm_sse.h"

namespace mymath
{
  MYMATH_INLINE bool equal( const impl::vec2i<float>& a, const impl::vec2i<float>& b )
  {
    impl::vec2i<float> c = _mm_cmpeq_ps( a.d, b.d );
    return c.x && c.y;
  }
  MYMATH_INLINE bool equal( const impl::vec3i<float>& a, const impl::vec3i<float>& b )
  {
    impl::vec3i<float> c = _mm_cmpeq_ps( a.d, b.d );
    return c.x && c.y && c.z;
  }
  MYMATH_INLINE bool equal( const impl::vec4i<float>& a, const impl::vec4i<float>& b )
  {
    impl::vec4i<float> c = _mm_cmpeq_ps( a.d, b.d );
    return c.x && c.y && c.z && c.w;
  }

  MYMATH_INLINE bool notEqual( const impl::vec2i<float>& a, const impl::vec2i<float>& b )
  {
    return !equal( a, b );
  }
  MYMATH_INLINE bool notEqual( const impl::vec3i<float>& a, const impl::vec3i<float>& b )
  {
    return !equal( a, b );
  }
  MYMATH_INLINE bool notEqual( const impl::vec4i<float>& a, const impl::vec4i<float>& b )
  {
    return !equal( a, b );
  }
}

//mul
MYMATH_INLINE mm::impl::vec2i<float> operator*( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return _mm_mul_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator*( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return _mm_mul_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator*( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return _mm_mul_ps( a.d, b.d );
}

//div
MYMATH_INLINE mm::impl::vec2i<float> operator/( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  assert( !mm::impl::is_eq( b.x, ( float )0 ) && !mm::impl::is_eq( b.y, ( float )0 ) );
  return _mm_div_ps( a.d, b.d );
}

MYMATH_INLINE mm::impl::vec3i<float> operator/( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  assert( !mm::impl::is_eq( b.x, ( float )0 ) && !mm::impl::is_eq( b.y, ( float )0 ) && !mm::impl::is_eq( b.z, ( float )0 ) );
  return _mm_div_ps( a.d, b.d );
}

MYMATH_INLINE mm::impl::vec4i<float> operator/( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  assert( !mm::impl::is_eq( b.x, ( float )0 ) && !mm::impl::is_eq( b.y, ( float )0 ) && !mm::impl::is_eq( b.z, ( float )0 ) && !mm::impl::is_eq( b.w, ( float )0 ) );
  return _mm_div_ps( a.d, b.d );
}

//add
MYMATH_INLINE mm::impl::vec2i<float> operator+( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return _mm_add_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator+( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return _mm_add_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator+( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return _mm_add_ps( a.d, b.d );
}

//sub
MYMATH_INLINE mm::impl::vec2i<float> operator-( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return _mm_sub_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator-( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return _mm_sub_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator-( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return _mm_sub_ps( a.d, b.d );
}

//mod
MYMATH_INLINE mm::impl::vec2i<float> operator%( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return mm::impl::sse_mod_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator%( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return mm::impl::sse_mod_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator%( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return mm::impl::sse_mod_ps( a.d, b.d );
}

//TODO shift left
//TODO shift right

//and
MYMATH_INLINE mm::impl::vec2i<float> operator&( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return _mm_and_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator&( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return _mm_and_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator&( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return _mm_and_ps( a.d, b.d );
}

//xor
MYMATH_INLINE mm::impl::vec2i<float> operator^( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return _mm_xor_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator^( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return _mm_xor_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator^( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return _mm_xor_ps( a.d, b.d );
}

//or
MYMATH_INLINE mm::impl::vec2i<float> operator|( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
{
  return _mm_or_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator|( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
{
  return _mm_or_ps( a.d, b.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator|( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
{
  return _mm_or_ps( a.d, b.d );
}

//negate
MYMATH_INLINE mm::impl::vec2i<float> operator-( const mm::impl::vec2i<float>& a )
{
  return mm::impl::sse_neg_ps( a.d );
}
MYMATH_INLINE mm::impl::vec3i<float> operator-( const mm::impl::vec3i<float>& a )
{
  return mm::impl::sse_neg_ps( a.d );
}
MYMATH_INLINE mm::impl::vec4i<float> operator-( const mm::impl::vec4i<float>& a )
{
  return mm::impl::sse_neg_ps( a.d );
}

//cout
MYMATH_INLINE std::ostream& operator<< ( std::ostream& output, const mm::impl::vec2i<float>& vec )
{
  return output << "( " << vec.x << ", " << vec.y << " )\n";
}
MYMATH_INLINE std::ostream& operator<< ( std::ostream& output, const mm::impl::vec3i<float>& vec )
{
  return output << "( " << vec.x << ", " << vec.y << ", " << vec.z << " )\n";
}
MYMATH_INLINE std::ostream& operator<< ( std::ostream& output, const mm::impl::vec4i<float>& vec )
{
  return output << "( " << vec.x << ", " << vec.y << ", " << vec.z << ", " << vec.w << " )\n";
}

namespace mymath
{
//less
  MYMATH_INLINE bool lessThan( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    mm::impl::vec2i<float> v = _mm_cmplt_ps( a.d, b.d );
    return v.x && v.y;
  }
  MYMATH_INLINE bool lessThan( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    mm::impl::vec3i<float> v = _mm_cmplt_ps( a.d, b.d );
    return v.x && v.y && v.z;
  }
  MYMATH_INLINE bool lessThan( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    mm::impl::vec4i<float> v = _mm_cmplt_ps( a.d, b.d );
    return v.x && v.y && v.z && v.w;
  }

//greater
  MYMATH_INLINE bool greaterThan( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    mm::impl::vec2i<float> v = _mm_cmpgt_ps( a.d, b.d );
    return v.x && v.y;
  }
  MYMATH_INLINE bool greaterThan( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    mm::impl::vec3i<float> v = _mm_cmpgt_ps( a.d, b.d );
    return v.x && v.y && v.z;
  }
  MYMATH_INLINE bool greaterThan( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    mm::impl::vec4i<float> v = _mm_cmpgt_ps( a.d, b.d );
    return v.x && v.y && v.z && v.w;
  }

//less or equal
  MYMATH_INLINE bool lessThanEqual( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    mm::impl::vec2i<float> v = _mm_cmple_ps( a.d, b.d );
    return v.x && v.y;
  }
  MYMATH_INLINE bool lessThanEqual( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    mm::impl::vec3i<float> v = _mm_cmple_ps( a.d, b.d );
    return v.x && v.y && v.z;
  }
  MYMATH_INLINE bool lessThanEqual( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    mm::impl::vec4i<float> v = _mm_cmple_ps( a.d, b.d );
    return v.x && v.y && v.z && v.w;
  }

//greater or equal
  MYMATH_INLINE bool greaterThanEqual( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    mm::impl::vec2i<float> v = _mm_cmpge_ps( a.d, b.d );
    return v.x && v.y;
  }
  MYMATH_INLINE bool greaterThanEqual( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    mm::impl::vec3i<float> v = _mm_cmpge_ps( a.d, b.d );
    return v.x && v.y && v.z;
  }
  MYMATH_INLINE bool greaterThanEqual( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    mm::impl::vec4i<float> v = _mm_cmpge_ps( a.d, b.d );
    return v.x && v.y && v.z && v.w;
  }

//radians
  MYMATH_INLINE mm::impl::vec2i<float> radians( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_rad_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> radians( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_rad_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> radians( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_rad_ps( vec.d );
  }

//degrees
  MYMATH_INLINE mm::impl::vec2i<float> degrees( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_deg_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> degrees( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_deg_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> degrees( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_deg_ps( vec.d );
  }

//asinh
  MYMATH_INLINE mm::impl::vec2i<float> asinh( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_asinh_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> asinh( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_asinh_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> asinh( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_asinh_ps( vec.d );
  }

//acosh
  MYMATH_INLINE mm::impl::vec2i<float> acosh( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_acosh_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> acosh( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_acosh_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> acosh( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_acosh_ps( vec.d );
  }

//atanh
  MYMATH_INLINE mm::impl::vec2i<float> atanh( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_atanh_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> atanh( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_atanh_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> atanh( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_atanh_ps( vec.d );
  }

//exp2
  MYMATH_INLINE mm::impl::vec2i<float> exp2( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_exp2_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> exp2( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_exp2_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> exp2( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_exp2_ps( vec.d );
  }

//log2
  MYMATH_INLINE mm::impl::vec2i<float> log2( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_log2_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> log2( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_log2_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> log2( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_log2_ps( vec.d );
  }

//inversesqrt
  MYMATH_INLINE mm::impl::vec2i<float> inversesqrt( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_inversesqrt_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> inversesqrt( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_inversesqrt_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> inversesqrt( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_inversesqrt_ps( vec.d );
  }

//sign
  MYMATH_INLINE mm::impl::vec2i<float> sign( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_sign_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> sign( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_sign_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> sign( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_sign_ps( vec.d );
  }

//trunc
  MYMATH_INLINE mm::impl::vec2i<float> trunc( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_trunc_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> trunc( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_trunc_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> trunc( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_trunc_ps( vec.d );
  }

//round
  MYMATH_INLINE mm::impl::vec2i<float> round( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_round_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> round( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_round_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> round( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_round_ps( vec.d );
  }

//fract
  MYMATH_INLINE mm::impl::vec2i<float> fract( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_fract_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> fract( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_fract_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> fract( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_fract_ps( vec.d );
  }

//mod
  MYMATH_INLINE mm::impl::vec2i<float> mod( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_mod_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> mod( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_mod_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> mod( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_mod_ps( a.d, b.d );
  }

//mix
  MYMATH_INLINE mm::impl::vec2i<float> mix( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b, const mm::impl::vec2i<float>& c )
  {
    return mm::impl::sse_mix_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> mix( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b, const mm::impl::vec3i<float>& c )
  {
    return mm::impl::sse_mix_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> mix( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b, const mm::impl::vec4i<float>& c )
  {
    return mm::impl::sse_mix_ps( a.d, b.d, c.d );
  }

//step
  MYMATH_INLINE mm::impl::vec2i<float> step( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_step_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> step( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_step_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> step( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_step_ps( a.d, b.d );
  }

//clamp
  MYMATH_INLINE mm::impl::vec2i<float> clamp( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b, const mm::impl::vec2i<float>& c )
  {
    return mm::impl::sse_clamp_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> clamp( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b, const mm::impl::vec3i<float>& c )
  {
    return mm::impl::sse_clamp_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> clamp( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b, const mm::impl::vec4i<float>& c )
  {
    return mm::impl::sse_clamp_ps( a.d, b.d, c.d );
  }

//smoothstep
  MYMATH_INLINE mm::impl::vec2i<float> smoothstep( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b, const mm::impl::vec2i<float>& c )
  {
    return mm::impl::sse_smoothstep_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> smoothstep( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b, const mm::impl::vec3i<float>& c )
  {
    return mm::impl::sse_smoothstep_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> smoothstep( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b, const mm::impl::vec4i<float>& c )
  {
    return mm::impl::sse_smoothstep_ps( a.d, b.d, c.d );
  }

//fma
  MYMATH_INLINE mm::impl::vec2i<float> fma( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b, const mm::impl::vec2i<float>& c )
  {
    return mm::impl::sse_fma_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> fma( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b, const mm::impl::vec3i<float>& c )
  {
    return mm::impl::sse_fma_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> fma( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b, const mm::impl::vec4i<float>& c )
  {
    return mm::impl::sse_fma_ps( a.d, b.d, c.d );
  }

//dot
//WARNING: it's slow to switch to floats
  MYMATH_INLINE float dot( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_dot_ps( a.d, b.d );
  }
  MYMATH_INLINE float dot( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_dot_ps( a.d, b.d );
  }
  MYMATH_INLINE float dot( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_dot_ps( a.d, b.d );
  }

//dot helper
  MYMATH_INLINE mm::impl::vec2i<float> dot_helper( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_dot_ps_helper( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> dot_helper( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_dot_ps_helper( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> dot_helper( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_dot_ps_helper( a.d, b.d );
  }

//length
//WARNING: it's slow to switch to floats
  MYMATH_INLINE float length( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_length_ps( vec.d );
  }
  MYMATH_INLINE float length( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_length_ps( vec.d );
  }
  MYMATH_INLINE float length( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_length_ps( vec.d );
  }

//length helper
  MYMATH_INLINE mm::impl::vec2i<float> length_helper( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_length_ps_helper( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> length_helper( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_length_ps_helper( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> length_helper( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_length_ps_helper( vec.d );
  }

//distance
//WARNING: it's slow to switch to floats
  MYMATH_INLINE float distance( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_distance_ps( a.d, b.d );
  }
  MYMATH_INLINE float distance( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_distance_ps( a.d, b.d );
  }
  MYMATH_INLINE float distance( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_distance_ps( a.d, b.d );
  }

//distance helper
  MYMATH_INLINE mm::impl::vec2i<float> distance_helper( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_distance_ps_helper( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> distance_helper( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_distance_ps_helper( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> distance_helper( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_distance_ps_helper( a.d, b.d );
  }

//normalize
  MYMATH_INLINE mm::impl::vec2i<float> normalize( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_normalize_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> normalize( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_normalize_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> normalize( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_normalize_ps( vec.d );
  }

//reflect
  MYMATH_INLINE mm::impl::vec2i<float> reflect( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return mm::impl::sse_reflect_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> reflect( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_reflect_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> reflect( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return mm::impl::sse_reflect_ps( a.d, b.d );
  }

//refract
  MYMATH_INLINE mm::impl::vec2i<float> refract( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b, const mm::impl::vec2i<float>& c )
  {
    return mm::impl::sse_refract_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> refract( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b, const mm::impl::vec3i<float>& c )
  {
    return mm::impl::sse_refract_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> refract( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b, const mm::impl::vec4i<float>& c )
  {
    return mm::impl::sse_refract_ps( a.d, b.d, c.d );
  }

//faceforward
  MYMATH_INLINE mm::impl::vec2i<float> faceforward( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b, const mm::impl::vec2i<float>& c )
  {
    return mm::impl::sse_faceforward_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> faceforward( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b, const mm::impl::vec3i<float>& c )
  {
    return mm::impl::sse_faceforward_ps( a.d, b.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> faceforward( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b, const mm::impl::vec4i<float>& c )
  {
    return mm::impl::sse_faceforward_ps( a.d, b.d, c.d );
  }

//TODO isnan, isinf

//cross
  MYMATH_INLINE mm::impl::vec3i<float> cross( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return mm::impl::sse_cross_ps( a.d, b.d );
  }

//floor
  MYMATH_INLINE mm::impl::vec2i<float> floor( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_floor_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> floor( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_floor_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> floor( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_floor_ps( vec.d );
  }

//sqrt
  MYMATH_INLINE mm::impl::vec2i<float> sqrt( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_sqrt_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> sqrt( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_sqrt_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> sqrt( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_sqrt_ps( vec.d );
  }

//sin
  MYMATH_INLINE mm::impl::vec2i<float> sin( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_sin_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> sin( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_sin_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> sin( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_sin_ps( vec.d );
  }

//cos
  MYMATH_INLINE mm::impl::vec2i<float> cos( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_cos_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> cos( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_cos_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> cos( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_cos_ps( vec.d );
  }

//tan
  MYMATH_INLINE mm::impl::vec2i<float> tan( const mm::impl::vec2i<float>& vec )
  {
    mm::impl::vec2i<float> s, c;
    mm::impl::sse_sincos_ps( vec.d, &s.d, &c.d );
    return _mm_div_ps( s.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> tan( const mm::impl::vec3i<float>& vec )
  {
    mm::impl::vec3i<float> s, c;
    mm::impl::sse_sincos_ps( vec.d, &s.d, &c.d );
    return _mm_div_ps( s.d, c.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> tan( const mm::impl::vec4i<float>& vec )
  {
    mm::impl::vec4i<float> s, c;
    mm::impl::sse_sincos_ps( vec.d, &s.d, &c.d );
    return _mm_div_ps( s.d, c.d );
  }

//asin
  MYMATH_INLINE mm::impl::vec2i<float> asin( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_asin_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> asin( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_asin_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> asin( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_asin_ps( vec.d );
  }

//acos
  MYMATH_INLINE mm::impl::vec2i<float> acos( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_acos_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> acos( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_acos_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> acos( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_acos_ps( vec.d );
  }

//atan
  MYMATH_INLINE mm::impl::vec2i<float> atan( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_atan_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> atan( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_atan_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> atan( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_atan_ps( vec.d );
  }

//abs
  MYMATH_INLINE mm::impl::vec2i<float> abs( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_abs_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> abs( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_abs_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> abs( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_abs_ps( vec.d );
  }

//pow
  MYMATH_INLINE mm::impl::vec2i<float> pow( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return exp2( log2( abs( a ) ) * b );
  }
  MYMATH_INLINE mm::impl::vec3i<float> pow( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return exp2( log2( abs( a ) ) * b );
  }
  MYMATH_INLINE mm::impl::vec4i<float> pow( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return exp2( log2( abs( a ) ) * b );
  }

//sinh
  MYMATH_INLINE mm::impl::vec2i<float> sinh( const mm::impl::vec2i<float>& vec )
  {
    auto etox = pow( mm::impl::e, vec );
    auto etominusx = pow( mm::impl::e, -vec );
    return _mm_mul_ps( _mm_sub_ps( etox.d, etominusx.d ), mm::impl::half );
  }
  MYMATH_INLINE mm::impl::vec3i<float> sinh( const mm::impl::vec3i<float>& vec )
  {
    auto etox = pow( mm::impl::e, vec );
    auto etominusx = pow( mm::impl::e, -vec );
    return _mm_mul_ps( _mm_sub_ps( etox.d, etominusx.d ), mm::impl::half );
  }
  MYMATH_INLINE mm::impl::vec4i<float> sinh( const mm::impl::vec4i<float>& vec )
  {
    auto etox = pow( mm::impl::e, vec );
    auto etominusx = pow( mm::impl::e, -vec );
    return _mm_mul_ps( _mm_sub_ps( etox.d, etominusx.d ), mm::impl::half );
  }

//cosh
  MYMATH_INLINE mm::impl::vec2i<float> cosh( const mm::impl::vec2i<float>& vec )
  {
    auto etox = pow( mm::impl::e, vec );
    auto etominusx = pow( mm::impl::e, -vec );
    return _mm_mul_ps( _mm_add_ps( etox.d, etominusx.d ), mm::impl::half );
  }
  MYMATH_INLINE mm::impl::vec3i<float> cosh( const mm::impl::vec3i<float>& vec )
  {
    auto etox = pow( mm::impl::e, vec );
    auto etominusx = pow( mm::impl::e, -vec );
    return _mm_mul_ps( _mm_add_ps( etox.d, etominusx.d ), mm::impl::half );
  }
  MYMATH_INLINE mm::impl::vec4i<float> cosh( const mm::impl::vec4i<float>& vec )
  {
    auto etox = pow( mm::impl::e, vec );
    auto etominusx = pow( mm::impl::e, -vec );
    return _mm_mul_ps( _mm_add_ps( etox.d, etominusx.d ), mm::impl::half );
  }

//tanh
  MYMATH_INLINE mm::impl::vec2i<float> tanh( const mm::impl::vec2i<float>& vec )
  {
    return sinh( vec ) / cosh( vec );
  }
  MYMATH_INLINE mm::impl::vec3i<float> tanh( const mm::impl::vec3i<float>& vec )
  {
    return sinh( vec ) / cosh( vec );
  }
  MYMATH_INLINE mm::impl::vec4i<float> tanh( const mm::impl::vec4i<float>& vec )
  {
    return sinh( vec ) / cosh( vec );
  }


//exp
  MYMATH_INLINE mm::impl::vec2i<float> exp( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_exp_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> exp( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_exp_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> exp( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_exp_ps( vec.d );
  }

//log
  MYMATH_INLINE mm::impl::vec2i<float> log( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_log_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> log( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_log_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> log( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_log_ps( vec.d );
  }

//ceil
  MYMATH_INLINE mm::impl::vec2i<float> ceil( const mm::impl::vec2i<float>& vec )
  {
    return mm::impl::sse_ceil_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> ceil( const mm::impl::vec3i<float>& vec )
  {
    return mm::impl::sse_ceil_ps( vec.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> ceil( const mm::impl::vec4i<float>& vec )
  {
    return mm::impl::sse_ceil_ps( vec.d );
  }

//min
  MYMATH_INLINE mm::impl::vec2i<float> min( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return _mm_min_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> min( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return _mm_min_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> min( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return _mm_min_ps( a.d, b.d );
  }

//max
  MYMATH_INLINE mm::impl::vec2i<float> max( const mm::impl::vec2i<float>& a, const mm::impl::vec2i<float>& b )
  {
    return _mm_max_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec3i<float> max( const mm::impl::vec3i<float>& a, const mm::impl::vec3i<float>& b )
  {
    return _mm_max_ps( a.d, b.d );
  }
  MYMATH_INLINE mm::impl::vec4i<float> max( const mm::impl::vec4i<float>& a, const mm::impl::vec4i<float>& b )
  {
    return _mm_max_ps( a.d, b.d );
  }

  namespace impl
  {
    template<int ta, int tb, int tc, int td>
    const vec2i<float>& vec2i<float>::swizzle<ta, tb, tc, td>::operator/=( const vec2i<float>& vec )
    {
      assert( notEqual( vec, vec2i<float>( 0 ) ) );
      vec2i<float>* tmp = (vec2i<float>*)this;
      tmp->d = _mm_div_ps( tmp->d, vec.d );
      return *( vec2i<float>* )this;
    }

    template<int a, int b, int c, int dd>
    const vec3i<float>& vec3i<float>::swizzle<a, b, c, dd>::operator/=( const vec3i<float>& vec )
    {
      assert( notEqual( vec, vec3i<float>( 0 ) ) );
      vec3i<float>* tmp = (vec3i<float>*)this;
      tmp->d = _mm_div_ps( tmp->d, vec.d );
      return *( vec3i<float>* )this;
    }

    template<int a, int b, int c, int dd>
    const vec4i<float>& vec4i<float>::swizzle<a, b, c, dd>::operator/=( const vec4i<float>& vec )
    {
      assert( notEqual( vec, vec4i<float>( 0 ) ) );
      vec4i<float>* tmp = (vec4i<float>*)this;
      tmp->d = _mm_div_ps( tmp->d, vec.d );
      return *( vec4i<float>* )this;
    }

    const vec2i<float>& vec2i<float>::operator/=( const vec2i<float>& vec )
    {
      assert( notEqual( vec, vec2i<float>( 0 ) ) );
      vec2i<float>* tmp = (vec2i<float>*)this;
      tmp->d = _mm_div_ps( tmp->d, vec.d );
      return *this;
    }

    const vec3i<float>& vec3i<float>::operator/=( const vec3i<float>& vec )
    {
      assert( notEqual( vec, vec3i<float>( 0 ) ) );
      vec3i<float>* tmp = (vec3i<float>*)this;
      tmp->d = _mm_div_ps( tmp->d, vec.d );
      return *this;
    }

    const vec4i<float>& vec4i<float>::operator/=( const vec4i<float>& vec )
    {
      assert( notEqual( vec, vec4i<float>( 0 ) ) );
      vec4i<float>* tmp = (vec4i<float>*)this;
      tmp->d = _mm_div_ps( tmp->d, vec.d );
      return *this;
    }
  }
}

#endif
