#ifndef mm_mat_func_h
#define mm_mat_func_h

#include "mm_common.h"

#ifdef MYMATH_USE_SSE2
#include "mm_fvec_func.h"
#endif

#include "mm_mat2_impl.h"
#include "mm_mat3_impl.h"
#include "mm_mat4_impl.h"
#include "mm_vec_func.h"

#define MYMATH_VECMULMAT_FUNC(t) \
  MYMATH_INLINE mm::impl::vec2i<t> operator*( const mm::impl::vec2i<t>& vec, const mm::impl::mat2i<t>& mat ) \
  { \
    mm::impl::vec4i<t> tmp1 = vec.xxyy; \
    mm::impl::vec4i<t> tmp2( mat[0], mat[1] ); \
    tmp1 *= tmp2.xzyw; \
    return tmp1.xy + tmp1.zw; \
  } \
  MYMATH_INLINE mm::impl::vec3i<t> operator*( const mm::impl::vec3i<t>& vec, const mm::impl::mat3i<t>& mat ) \
  { \
    mm::impl::vec4i<t> tmp1 = mat[0].xxxy; \
    tmp1 *= vec.xxxy; \
    mm::impl::vec4i<t> tmp11 = mat[0].yxxz; \
    tmp11 *= vec.yxxz; \
    tmp1 *= mm::impl::vec4i<t>(1,1,1,0); \
    tmp1 += tmp11; \
    mm::impl::vec4i<t> tmp2 = mat[1].xxxy; \
    tmp2 *= vec.xxxy; \
    mm::impl::vec4i<t> tmp21 = mat[1].yxxz; \
    tmp21 *= vec.yxxz; \
    tmp2 *= mm::impl::vec4i<t>(1,1,1,0); \
    tmp2 += tmp21; \
    mm::impl::vec4i<t> tmp3 = mat[2].xxxy; \
    tmp3 *= vec.xxxy; \
    mm::impl::vec4i<t> tmp31 = mat[2].yxxz; \
    tmp31 *= vec.yxxz; \
    tmp3 *= mm::impl::vec4i<t>(1,1,1,0); \
    tmp3 += tmp31; \
    tmp1 += tmp1.wxxx; \
    tmp2 += tmp2.wxxx; \
    tmp3 += tmp3.wxxx; \
    tmp1 *= mm::impl::vec4i<t>(1,0,0,0); \
    tmp2 = tmp2.xxxx; \
    tmp2 *= mm::impl::vec4i<t>(0,1,0,0); \
    tmp3 = tmp3.xxxx; \
    tmp3 *= mm::impl::vec4i<t>(0,0,1,0); \
    return (tmp1 + tmp2 + tmp3).xyz; \
  } \
  MYMATH_INLINE mm::impl::vec4i<t> operator*( const mm::impl::vec4i<t>& vec, const mm::impl::mat4i<t>& mat ) \
  { \
    mm::impl::vec4i<t> tmp1 = vec; \
    tmp1 *= mat[0]; \
    tmp1 += tmp1.yxwx; \
    tmp1 += tmp1.zxxx; \
    tmp1 *= mm::impl::vec4i<t>(1, 0, 0, 0); \
    mm::impl::vec4i<t> tmp2 = vec; \
    tmp2 *= mat[1]; \
    tmp2 += tmp2.xxwx; \
    tmp2 += tmp2.xzxx; \
    tmp2 *= mm::impl::vec4i<t>(0, 1, 0, 0); \
    mm::impl::vec4i<t> tmp3 = vec; \
    tmp3 *= mat[2]; \
    tmp3 += tmp3.xxwx; \
    tmp3 += tmp3.xxyx; \
    tmp3 *= mm::impl::vec4i<t>(0, 0, 1, 0); \
    mm::impl::vec4i<t> tmp4 = vec; \
    tmp4 *= mat[3]; \
    tmp4 += tmp4.xxxz; \
    tmp4 += tmp4.xxxy; \
    tmp4 *= mm::impl::vec4i<t>(0, 0, 0, 1); \
    return tmp1 + tmp2 + tmp3 + tmp4; \
  }

#define MYMATH_MATMULVEC_FUNC(t) \
  MYMATH_INLINE mm::impl::vec2i<t> operator* ( const mm::impl::mat2i<t>& mat, const mm::impl::vec2i<t>& vec ) \
  { \
    return mm::fma( vec.yy, mat[1], vec.xx * mat[0] ); \
  } \
  MYMATH_INLINE mm::impl::vec3i<t> operator* ( const mm::impl::mat3i<t>& mat, const mm::impl::vec3i<t>& vec ) \
  { \
    return mm::fma( vec.zzz, mat[2], mm::fma( vec.yyy, mat[1], vec.xxx * mat[0] ) ); \
  } \
  MYMATH_INLINE mm::impl::vec4i<t> operator*( const mm::impl::mat4i<t>& mat, const mm::impl::vec4i<t>& vec ) \
  { \
    return mm::fma( vec.wwww, mat[3], mm::fma( vec.zzzz, mat[2], mm::fma( vec.yyyy, mat[1], vec.xxxx * mat[0] ) ) ); \
  }

#define MYMATH_VECMULEQUALMAT_FUNC(t) \
  MYMATH_INLINE const mm::impl::vec2i<t>& operator*=( mm::impl::vec2i<t>& vec, const mm::impl::mat2i<t>& mat ) \
  { \
    mm::impl::vec2i<t> res = vec * mat; \
    vec = res; \
    return vec; \
  } \
  MYMATH_INLINE const mm::impl::vec3i<t>& operator*=( mm::impl::vec3i<t>& vec, const mm::impl::mat3i<t>& mat ) \
  { \
    mm::impl::vec3i<t> res = vec * mat; \
    vec = res; \
    return vec; \
  } \
  MYMATH_INLINE const mm::impl::vec4i<t>& operator*=( mm::impl::vec4i<t>& vec, const mm::impl::mat4i<t>& mat ) \
  { \
    mm::impl::vec4i<t> res = vec * mat; \
    vec = res; \
    return vec; \
  }

template< typename t >
MYMATH_INLINE mm::impl::mat2i<t> operator* ( const mm::impl::mat2i<t>& a, const mm::impl::mat2i<t>& b )
{
  mm::impl::vec4i<t> tmp1( a[0], a[0] );
  mm::impl::vec4i<t> tmp2( b[0].xx, b[1].xx );
  mm::impl::vec4i<t> tmp3( a[1], a[1] );
  mm::impl::vec4i<t> tmp4( b[0].yy, b[1].yy );
  mm::impl::vec4i<t> res = tmp1 * tmp2 + tmp3 * tmp4;
  return mm::impl::mat2i<t>( res.xy, res.zw );
}

template< typename t >
MYMATH_INLINE mm::impl::mat3i<t> operator* ( const mm::impl::mat3i<t>& a, const mm::impl::mat3i<t>& b )
{

  mm::impl::vec3i<t> tmp1 = a[0];
  mm::impl::vec3i<t> tmp2 = a[1];
  mm::impl::vec3i<t> tmp3 = a[2];

  return mm::impl::mat3i<t>( tmp1 * b[0].xxx + tmp2 * b[0].yyy + tmp3 * b[0].zzz,
                             tmp1 * b[1].xxx + tmp2 * b[1].yyy + tmp3 * b[1].zzz,
                             tmp1 * b[2].xxx + tmp2 * b[2].yyy + tmp3 * b[2].zzz );
}

template< typename t >
MYMATH_INLINE mm::impl::mat4i<t> operator* ( const mm::impl::mat4i<t>& a, const mm::impl::mat4i<t>& b )
{
  mm::impl::vec4i<t> tmp1 = a[0];
  mm::impl::vec4i<t> tmp2 = a[1];
  mm::impl::vec4i<t> tmp3 = a[2];
  mm::impl::vec4i<t> tmp4 = a[3];

  return mm::impl::mat4i<t>( tmp1 * b[0].xxxx + tmp2 * b[0].yyyy + tmp3 * b[0].zzzz + tmp4 * b[0].wwww,
                             tmp1 * b[1].xxxx + tmp2 * b[1].yyyy + tmp3 * b[1].zzzz + tmp4 * b[1].wwww,
                             tmp1 * b[2].xxxx + tmp2 * b[2].yyyy + tmp3 * b[2].zzzz + tmp4 * b[2].wwww,
                             tmp1 * b[3].xxxx + tmp2 * b[3].yyyy + tmp3 * b[3].zzzz + tmp4 * b[3].wwww );
}

template< typename t >
MYMATH_INLINE mm::impl::mat2i<t> operator*( const mm::impl::mat2i<t>& mat, const t& num )
{
  return mm::impl::mat2i<t>( mat[0] * mm::impl::vec2i<t>( num ), mat[1] * mm::impl::vec2i<t>( num ) );
}

template< typename t >
MYMATH_INLINE mm::impl::mat3i<t> operator*( const mm::impl::mat3i<t>& mat, const t& num )
{
  return mm::impl::mat3i<t>( mat[0] * mm::impl::vec3i<t>( num ), mat[1] * mm::impl::vec3i<t>( num ), mat[2] * mm::impl::vec3i<t>( num ) );
}

template< typename t >
MYMATH_INLINE mm::impl::mat4i<t> operator*( const mm::impl::mat4i<t>& mat, const t& num )
{
  return mm::impl::mat4i<t>( mat[0] * mm::impl::vec4i<t>( num ), mat[1] * mm::impl::vec4i<t>( num ), mat[2] * mm::impl::vec4i<t>( num ), mat[3] * mm::impl::vec4i<t>( num ) );
}

template< typename t >
MYMATH_INLINE mm::impl::mat2i<t> operator*( const t& num, const mm::impl::mat2i<t>& mat )
{
  return mat * num;
}

template< typename t >
MYMATH_INLINE mm::impl::mat3i<t> operator*( const t& num, const mm::impl::mat3i<t>& mat )
{
  return mat * num;
}

template< typename t >
MYMATH_INLINE mm::impl::mat4i<t> operator*( const t& num, const mm::impl::mat4i<t>& mat )
{
  return mat * num;
}

template< typename t >
MYMATH_INLINE std::ostream& operator<< ( std::ostream& output, const mm::impl::mat2i<t>& mat )
{
  return output << "( " << mat[0].x << ", " << mat[1].x << "\n  "
         /*__________*/ << mat[0].y << ", " << mat[1].y << " )\n";
}

template< typename t >
MYMATH_INLINE std::ostream& operator<< ( std::ostream& output, const mm::impl::mat3i<t>& mat )
{
  return output << "( " << mat[0].x << ", " << mat[1].x << ", " << mat[2].x << "\n  "
         /*__________*/ << mat[0].y << ", " << mat[1].y << ", " << mat[2].y << "\n  "
         /*__________*/ << mat[0].z << ", " << mat[1].z << ", " << mat[2].z << " )\n";
}

template< typename t >
MYMATH_INLINE std::ostream& operator<< ( std::ostream& output, const mm::impl::mat4i<t>& mat )
{
  return output << "( " << mat[0].x << ", " << mat[1].x << ", " << mat[2].x << ", " << mat[3].x << "\n  "
         /*__________*/ << mat[0].y << ", " << mat[1].y << ", " << mat[2].y << ", " << mat[3].y << "\n  "
         /*__________*/ << mat[0].z << ", " << mat[1].z << ", " << mat[2].z << ", " << mat[3].z << "\n  "
         /*__________*/ << mat[0].w << ", " << mat[1].w << ", " << mat[2].w << ", " << mat[3].w << " )\n";
}

MYMATH_VECMULMAT_FUNC( float )

MYMATH_MATMULVEC_FUNC( float )

MYMATH_VECMULEQUALMAT_FUNC( float )

#if MYMATH_DOUBLE_PRECISION == 1
MYMATH_VECMULMAT_FUNC( double )
MYMATH_MATMULVEC_FUNC( double )
#endif

namespace mymath
{
  template< typename t >
  MYMATH_INLINE impl::mat2i<t> transpose( const impl::mat2i<t>& mat )
  {
    impl::vec4i<t> tmp1 = mat[0].xxyy;
    tmp1 *= impl::vec4i<t>( 1, 0, 1, 0 );
    impl::vec4i<t> tmp2 = mat[1].xxyy;
    tmp2 *= impl::vec4i<t>( 0, 1, 0, 1 );
    tmp1 += tmp2;
    return impl::mat2i<t>( tmp1.xy, tmp1.zw );
  }

  template< typename t >
  MYMATH_INLINE impl::mat3i<t> transpose( const impl::mat3i<t>& mat )
  {
    impl::vec4i<t> tmp1 = mat[0].xxxx;
    tmp1 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp2 = mat[0].yyyy;
    tmp2 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp3 = mat[0].zzzz;
    tmp3 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp4 = mat[1].xxxx;
    tmp4 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp5 = mat[1].yyyy;
    tmp5 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp6 = mat[1].zzzz;
    tmp6 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp7 = mat[2].xxxx;
    tmp7 *= impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp8 = mat[2].yyyy;
    tmp8 *= impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp9 = mat[2].zzzz;
    tmp9 *= impl::vec4i<t>( 0, 0, 1, 0 );
    return impl::mat3i<t>( tmp1.xyz + tmp4.xyz + tmp7.xyz,
                           tmp2.xyz + tmp5.xyz + tmp8.xyz,
                           tmp3.xyz + tmp6.xyz + tmp9.xyz );
  }

  template< typename t >
  MYMATH_INLINE impl::mat4i<t> transpose( const impl::mat4i<t>& mat )
  {
    impl::vec4i<t> tmp1 = mat[0].xxxx;
    tmp1 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp2 = mat[0].yyyy;
    tmp2 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp3 = mat[0].zzzz;
    tmp3 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp4 = mat[0].wwww;
    tmp4 *= impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp5 = mat[1].xxxx;
    tmp5 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp6 = mat[1].yyyy;
    tmp6 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp7 = mat[1].zzzz;
    tmp7 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp8 = mat[1].wwww;
    tmp8 *= impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp9 = mat[2].xxxx;
    tmp9 *= impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp10 = mat[2].yyyy;
    tmp10 *= impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp11 = mat[2].zzzz;
    tmp11 *= impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp12 = mat[2].wwww;
    tmp12 *= impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp13 = mat[3].xxxx;
    tmp13 *= impl::vec4i<t>( 0, 0, 0, 1 );
    impl::vec4i<t> tmp14 = mat[3].yyyy;
    tmp14 *= impl::vec4i<t>( 0, 0, 0, 1 );
    impl::vec4i<t> tmp15 = mat[3].zzzz;
    tmp15 *= impl::vec4i<t>( 0, 0, 0, 1 );
    impl::vec4i<t> tmp16 = mat[3].wwww;
    tmp16 *= impl::vec4i<t>( 0, 0, 0, 1 );
    return impl::mat4i<t>( tmp1 + tmp5 + tmp9 + tmp13,
                           tmp2 + tmp6 + tmp10 + tmp14,
                           tmp3 + tmp7 + tmp11 + tmp15,
                           tmp4 + tmp8 + tmp12 + tmp16 );
  }

  template< typename t >
  MYMATH_INLINE impl::vec4i<t> determinant_helper( const impl::mat2i<t>& mat )
  {
    impl::vec2i<t> tmp = mat[0] * mat[1].yx;
    return ( tmp - tmp.yx ).xxxx;
  }

  template< typename t >
  MYMATH_INLINE t determinant( const impl::mat2i<t>& mat )
  {
    return determinant_helper( mat ).x;
  }

  template< typename t >
  MYMATH_INLINE impl::vec4i<t> determinant_helper( const impl::mat3i<t>& mat )
  {
    impl::vec3i<t> tmp = cross( mat[0], mat[1] );
    tmp *= mat[2];
    tmp += tmp.yxx * impl::vec3i<t>( 1, 0, 0 );
    tmp += tmp.zxx * impl::vec3i<t>( 1, 0, 0 );
    return tmp.xxxx;
  }

  template< typename t >
  MYMATH_INLINE t determinant( const impl::mat3i<t>& mat )
  {
    return determinant_helper( mat ).x;
  }

  template< typename t >
  MYMATH_INLINE impl::vec4i<t> determinant_helper( const impl::mat4i<t>& mat )
  {
    impl::vec4i<t> deta = mat[0].xxxx * determinant_helper( impl::mat3i<t>( mat[1].yzw, mat[2].yzw, mat[3].yzw ) );
    impl::vec4i<t> detb = mat[1].xxxx * determinant_helper( impl::mat3i<t>( mat[0].yzw, mat[2].yzw, mat[3].yzw ) );
    impl::vec4i<t> detc = mat[2].xxxx * determinant_helper( impl::mat3i<t>( mat[0].yzw, mat[1].yzw, mat[3].yzw ) );
    impl::vec4i<t> detd = mat[3].xxxx * determinant_helper( impl::mat3i<t>( mat[0].yzw, mat[1].yzw, mat[2].yzw ) );
    return ( deta - detb + detc - detd ).xxxx;
  }

  template< typename t >
  MYMATH_INLINE t determinant( const impl::mat4i<t>& mat )
  {
    return determinant_helper( mat ).x;
  }

  template< typename t >
  MYMATH_INLINE impl::mat2i<t> inverse( const impl::mat2i<t>& mat )
  {
    assert( determinant( mat ) != 0 );
    impl::vec4i<t> tmp1 = mat[0].xxxx * impl::vec4i<t>( 0, 0, 0, 1 );
    impl::vec4i<t> tmp2 = -mat[0].yyyy * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp3 = -mat[1].xxxx * impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp4 = mat[1].yyyy * impl::vec4i<t>( 1, 0, 0, 0 );
    tmp1 = tmp1 + tmp2 + tmp3 + tmp4;
    impl::vec4i<t> det = impl::vec4i<t>( 1 ) / determinant_helper( mat );
    return impl::mat2i<t>( tmp1.xy * det.xx, tmp1.zw * det.xx );
  }

  template< typename t >
  MYMATH_INLINE impl::mat3i<t> inverse( const impl::mat3i<t>& mat )
  {
    assert( determinant( mat ) != 0 );

    impl::vec4i<t> tmp1 = determinant_helper( impl::mat2i<t>( mat[1].yz, mat[2].yz ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp2 = -determinant_helper( impl::mat2i<t>( mat[0].yz, mat[2].yz ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp3 = determinant_helper( impl::mat2i<t>( mat[0].yz, mat[1].yz ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    tmp1 = tmp1 + tmp2 + tmp3;

    impl::vec4i<t> tmp4 = -determinant_helper( impl::mat2i<t>( mat[1].xz, mat[2].xz ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp5 = determinant_helper( impl::mat2i<t>( mat[0].xz, mat[2].xz ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp6 = -determinant_helper( impl::mat2i<t>( mat[0].xz, mat[1].xz ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    tmp4 = tmp4 + tmp5 + tmp6;

    impl::vec4i<t> tmp7 = determinant_helper( impl::mat2i<t>( mat[1].xy, mat[2].xy ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp8 = -determinant_helper( impl::mat2i<t>( mat[0].xy, mat[2].xy ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp9 = determinant_helper( impl::mat2i<t>( mat[0].xy, mat[1].xy ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    tmp7 = tmp7 + tmp8 + tmp9;

    impl::vec4i<t> det = impl::vec4i<t>( 1 ) / determinant_helper( mat );

    return impl::mat3i<t>( tmp1.xyz * det.xxx, tmp4.xyz * det.xxx, tmp7.xyz * det.xxx );
  }

  template< typename t >
  MYMATH_INLINE impl::mat4i<t> inverse( const impl::mat4i<t>& mat )
  {
    assert( determinant( mat ) != 0 );

    impl::vec4i<t> tmp1 = determinant_helper( impl::mat3i<t>( mat[1].yzw, mat[2].yzw, mat[3].yzw ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp2 = -determinant_helper( impl::mat3i<t>( mat[0].yzw, mat[2].yzw, mat[3].yzw ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp3 = determinant_helper( impl::mat3i<t>( mat[0].yzw, mat[1].yzw, mat[3].yzw ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp4 = -determinant_helper( impl::mat3i<t>( mat[0].yzw, mat[1].yzw, mat[2].yzw ) ) * impl::vec4i<t>( 0, 0, 0, 1 );
    tmp1 = tmp1 + tmp2 + tmp3 + tmp4;

    impl::vec4i<t> tmp5 = -determinant_helper( impl::mat3i<t>( mat[1].xzw, mat[2].xzw, mat[3].xzw ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp6 = determinant_helper( impl::mat3i<t>( mat[0].xzw, mat[2].xzw, mat[3].xzw ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp7 = -determinant_helper( impl::mat3i<t>( mat[0].xzw, mat[1].xzw, mat[3].xzw ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp8 = determinant_helper( impl::mat3i<t>( mat[0].xzw, mat[1].xzw, mat[2].xzw ) ) * impl::vec4i<t>( 0, 0, 0, 1 );
    tmp5 = tmp5 + tmp6 + tmp7 + tmp8;

    impl::vec4i<t> tmp9 = determinant_helper( impl::mat3i<t>( mat[1].xyw, mat[2].xyw, mat[3].xyw ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp10 = -determinant_helper( impl::mat3i<t>( mat[0].xyw, mat[2].xyw, mat[3].xyw ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp11 = determinant_helper( impl::mat3i<t>( mat[0].xyw, mat[1].xyw, mat[3].xyw ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp12 = -determinant_helper( impl::mat3i<t>( mat[0].xyw, mat[1].xyw, mat[2].xyw ) ) * impl::vec4i<t>( 0, 0, 0, 1 );
    tmp9 = tmp9 + tmp10 + tmp11 + tmp12;

    impl::vec4i<t> tmp13 = -determinant_helper( impl::mat3i<t>( mat[1].xyz, mat[2].xyz, mat[3].xyz ) ) * impl::vec4i<t>( 1, 0, 0, 0 );
    impl::vec4i<t> tmp14 = determinant_helper( impl::mat3i<t>( mat[0].xyz, mat[2].xyz, mat[3].xyz ) ) * impl::vec4i<t>( 0, 1, 0, 0 );
    impl::vec4i<t> tmp15 = -determinant_helper( impl::mat3i<t>( mat[0].xyz, mat[1].xyz, mat[3].xyz ) ) * impl::vec4i<t>( 0, 0, 1, 0 );
    impl::vec4i<t> tmp16 = determinant_helper( impl::mat3i<t>( mat[0].xyz, mat[1].xyz, mat[2].xyz ) ) * impl::vec4i<t>( 0, 0, 0, 1 );
    tmp13 = tmp13 + tmp14 + tmp15 + tmp16;

    impl::vec4i<t> det = impl::vec4i<t>( 1 ) / determinant_helper( mat );

    return impl::mat4i<t>( tmp1 * det, tmp5 * det, tmp9 * det, tmp13 * det );
  }

  template< typename t >
  MYMATH_INLINE impl::mat2i<t> matrixCompMult( const impl::mat2i<t>& a, const impl::mat2i<t>& b )
  {
    return impl::mat2i<t>( a[0] * b[0], a[1] * b[1] );
  }

  template< typename t >
  MYMATH_INLINE impl::mat3i<t> matrixCompMult( const impl::mat3i<t>& a, const impl::mat3i<t>& b )
  {
    return impl::mat3i<t>( a[0] * b[0], a[1] * b[1], a[2] * b[2] );
  }

  template< typename t >
  impl::mat4i<t> matrixCompMult( const impl::mat4i<t>& a, const impl::mat4i<t>& b )
  {
    return impl::mat4i<t>( a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3] );
  }
}

#endif



