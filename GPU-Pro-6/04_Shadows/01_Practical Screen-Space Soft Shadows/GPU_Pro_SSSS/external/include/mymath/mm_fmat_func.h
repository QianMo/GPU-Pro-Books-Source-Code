#ifndef mm_mat_func_h
#define mm_mat_func_h

#include "mm_common.h"

#ifdef MYMATH_USE_SSE2
#include "mm_fvec_func.h"
#endif

#include "mm_mat2_impl.h"
#include "mm_mat3_impl.h"
#include "mm_mat4_impl.h"

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

namespace mymath
{
  template< typename t >
  MYMATH_INLINE impl::mat2i<t> transpose( const impl::mat2i<t>& mat )
  {
    impl::vec4i<t> tmp1 = _mm_shuffle_ps( mat[0].d, mat[1].d, MYMATH_SHUFFLE(0, 1, 0, 1) );
    return impl::mat2i<t>( tmp1.xz, tmp1.yw );
  }

  template< typename t >
  MYMATH_INLINE impl::mat3i<t> transpose( const impl::mat3i<t>& mat )
  {
    impl::vec4i<t> tmp1 = _mm_shuffle_ps( mat[0].d, mat[1].d, MYMATH_SHUFFLE(0, 1, 0, 1) ); //adbe
    impl::vec4i<t> tmp2 = mat[2].xyzz;                                                  //cfii
    impl::vec4i<t> tmp3 = _mm_shuffle_ps( mat[0].d, mat[1].d, MYMATH_SHUFFLE(2, 2, 2, 2) ); //gghh

    return impl::mat3i<t>( _mm_shuffle_ps( tmp1.d, tmp2.d, MYMATH_SHUFFLE( 0, 2, 0, 0 ) ), 
                           _mm_shuffle_ps( tmp1.d, tmp2.d, MYMATH_SHUFFLE( 1, 3, 1, 1 ) ),
                           _mm_shuffle_ps( tmp3.d, tmp2.d, MYMATH_SHUFFLE( 0, 2, 2, 3 ) ) );
  }

  template< typename t >
  MYMATH_INLINE impl::mat4i<t> transpose( const impl::mat4i<t>& mat )
  {
    impl::vec4i<t> tmp1 = _mm_shuffle_ps( mat[0].d, mat[1].d, MYMATH_SHUFFLE(0, 1, 0, 1) );
    impl::vec4i<t> tmp2 = _mm_shuffle_ps( mat[0].d, mat[1].d, MYMATH_SHUFFLE(2, 3, 2, 3) );
    impl::vec4i<t> tmp3 = _mm_shuffle_ps( mat[2].d, mat[3].d, MYMATH_SHUFFLE(0, 1, 0, 1) );
    impl::vec4i<t> tmp4 = _mm_shuffle_ps( mat[2].d, mat[3].d, MYMATH_SHUFFLE(2, 3, 2, 3) );

    return impl::mat4i<t>( _mm_shuffle_ps( tmp1.d, tmp3.d, MYMATH_SHUFFLE(0, 2, 0, 2) ),
                           _mm_shuffle_ps( tmp1.d, tmp3.d, MYMATH_SHUFFLE(1, 3, 1, 3) ),
                           _mm_shuffle_ps( tmp2.d, tmp4.d, MYMATH_SHUFFLE(0, 2, 0, 2) ),
                           _mm_shuffle_ps( tmp2.d, tmp4.d, MYMATH_SHUFFLE(1, 3, 1, 3) ) );
  }
}

#define MYMATH_VECMULMAT_FUNC(t) \
  MYMATH_INLINE mm::impl::vec2i<t> operator*( const mm::impl::vec2i<t>& vec, const mm::impl::mat2i<t>& mat ) \
  { \
    mm::impl::mat2i<t> tmp = mm::transpose( mat ); \
    return tmp * vec; \
  } \
  MYMATH_INLINE mm::impl::vec3i<t> operator*( const mm::impl::vec3i<t>& vec, const mm::impl::mat3i<t>& mat ) \
  { \
    mm::impl::mat3i<t> tmp = mm::transpose( mat ); \
    return tmp * vec; \
  } \
  MYMATH_INLINE mm::impl::vec4i<t> operator*( const mm::impl::vec4i<t>& vec, const mm::impl::mat4i<t>& mat ) \
  { \
    mm::impl::mat4i<t> tmp = mm::transpose( mat ); \
    return tmp * vec; \
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
  mm::impl::vec2i<t> tmp1 = a[0];
  mm::impl::vec2i<t> tmp2 = a[1];

  return mm::impl::mat2i<t>( mm::fma(b[0].xx, tmp1, b[0].yy * tmp2), 
                             mm::fma(b[1].xx, tmp1, b[1].yy * tmp2) );
}

template< typename t >
MYMATH_INLINE mm::impl::mat3i<t> operator* ( const mm::impl::mat3i<t>& a, const mm::impl::mat3i<t>& b )
{

  mm::impl::vec3i<t> tmp1 = a[0];
  mm::impl::vec3i<t> tmp2 = a[1];
  mm::impl::vec3i<t> tmp3 = a[2];

  return mm::impl::mat3i<t>( mm::fma(b[0].zzz, tmp3, fma(b[0].yyy, tmp2, b[0].xxx * tmp1)),
                             mm::fma(b[1].zzz, tmp3, fma(b[1].yyy, tmp2, b[1].xxx * tmp1)),
                             mm::fma(b[2].zzz, tmp3, fma(b[2].yyy, tmp2, b[2].xxx * tmp1)) );
}

template< typename t >
MYMATH_INLINE mm::impl::mat4i<t> operator* ( const mm::impl::mat4i<t>& a, const mm::impl::mat4i<t>& b )
{
  mm::impl::vec4i<t> tmp1 = a[0];
  mm::impl::vec4i<t> tmp2 = a[1];
  mm::impl::vec4i<t> tmp3 = a[2];
  mm::impl::vec4i<t> tmp4 = a[3];

  return mm::impl::mat4i<t>( mm::fma(b[0].wwww, tmp4, fma(b[0].zzzz, tmp3, fma(b[0].yyyy, tmp2, b[0].xxxx * tmp1))),
                             mm::fma(b[1].wwww, tmp4, fma(b[1].zzzz, tmp3, fma(b[1].yyyy, tmp2, b[1].xxxx * tmp1))),
                             mm::fma(b[2].wwww, tmp4, fma(b[2].zzzz, tmp3, fma(b[2].yyyy, tmp2, b[2].xxxx * tmp1))),
                             mm::fma(b[3].wwww, tmp4, fma(b[3].zzzz, tmp3, fma(b[3].yyyy, tmp2, b[3].xxxx * tmp1))) );
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

MYMATH_MATMULVEC_FUNC( float )
MYMATH_VECMULMAT_FUNC( float )

MYMATH_VECMULEQUALMAT_FUNC( float )

#if MYMATH_DOUBLE_PRECISION == 1
MYMATH_VECMULMAT_FUNC( double )
MYMATH_MATMULVEC_FUNC( double )
#endif

namespace mymath
{
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
    impl::vec3i<t> tmp1 = cross( mat[0], mat[1] ); //2 mul, 1 sub, 4 shuffle, 1 mov
    tmp1 *= mat[2]; //1 mul, 1 mov
    return tmp1.xxxx + tmp1.yyyy + tmp1.zzzz; //2 add
  }

  template< typename t >
  MYMATH_INLINE t determinant( const impl::mat3i<t>& mat )
  {
    return determinant_helper( mat ).x;
  }

//helper for _mm_set_ps()
#define MYMATH_SSE_SETTER(x, y, z, w) _mm_set_ps(w, z, y, x)

  template< typename t >
  MYMATH_INLINE impl::vec4i<t> determinant_helper( const impl::mat4i<t>& mat )
  {
    impl::vec4i<t> aa = mat[0].zxxx * mat[1].wyzw; //in, af, aj, an
    impl::vec4i<t> ee = mat[1].zxxx * mat[0].wyzw; //jm, be, bi, bm
    impl::vec4i<t> ii = mat[0].zwyy * mat[1].yyzw; //if, mf, ej, en

    //in - jm, af - be, aj - bi, an - bm
    aa = aa - ee;

    //xx - xx, xx - xx, ej - fi, en - fm
    ii = ii - ii.xxxy;

    //an - bm, aj - bi, en - fm, xx - xx
    impl::vec4i<t> jj = _mm_shuffle_ps(aa.d, ii.d, MYMATH_SHUFFLE(3, 2, 3, 3));
    //-k(en - fm), -k(an - bm), -g(an - bm), -g(aj - bi)
    ee = jj.zxxy * mat[2].zzyy;

    //in - jm, in - jm, en - fm, ej - fi
    jj = _mm_shuffle_ps( aa.d, ii.d, MYMATH_SHUFFLE(0, 0, 3, 2) );
    //g(in - jm), c(in - jm), c(en - fm), c(ej - fi)
    impl::vec4i<t> ww = jj * mat[2].yxxx; 

    //af - be, aj - bi, ej - fi, ej - fi
    jj = _mm_shuffle_ps( aa.d, ii.d, MYMATH_SHUFFLE(1, 2, 2, 2) );
    //o(ej - fi), o(aj - bi), o(af - be), k(af - be)
    jj = jj.zyxx * mat[2].wwwz; 

    jj = mat[3] * MYMATH_SSE_SETTER(-1, 1, -1, 1) * (ww - ee + jj);
    jj = jj + jj.wzyx;   //xw, yz, zy, wx
    jj = jj + jj.zxxz;   //xwzy, yzxw, zyxw, wxzy

    return jj;
  }

  template< typename t >
  MYMATH_INLINE t determinant( const impl::mat4i<t>& mat )
  {
    return determinant_helper( mat ).x;
  }

  //TODO optimize this
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

  //TODO optimize this
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

  //TODO optimize this
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



