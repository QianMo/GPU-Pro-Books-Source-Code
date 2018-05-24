#ifndef mm_util_h
#define mm_util_h

#include "mm_mat_func.h"
#include "mm_vec_func.h"

namespace mymath
{
#define MYMATH_UNPROJECT_FUNC(t) \
  MYMATH_INLINE impl::vec3i<t> unproject( const impl::vec3i<t>& ndc_pos, const impl::mat4i<t>& inv_mvp ) \
  { \
    impl::vec4i<t> obj_space = inv_mvp * impl::vec4i<t>( ndc_pos, 1 ); \
    return obj_space.xyz / obj_space.w; \
  }

#define MYMATH_CREATEROTATION_FUNC(t) \
  MYMATH_INLINE impl::mat4i<t> create_rotation( const t& angle, const impl::vec3i<t>& vec ) \
  { \
    assert( !impl::is_eq( length( vec ), (t)0 ) ); \
    t a = angle; \
    t s = std::sin( a ); \
    t c = std::cos( a ); \
    impl::vec3i<t> v = normalize( vec ); \
    t xx = v.x * v.x; \
    t yy = v.y * v.y; \
    t zz = v.z * v.z; \
    t xy = v.x * v.y; \
    t yz = v.y * v.z; \
    t zx = v.z * v.x; \
    t xs = v.x * s; \
    t ys = v.y * s; \
    t zs = v.z * s; \
    t oc = ( t )1 - c; \
    return impl::mat4i<t>( oc * xx + c, oc * xy + zs, oc * zx - ys, 0, \
                           oc * xy - zs, oc * yy + c, oc * yz + xs, 0, \
                           oc * zx + ys, oc * yz - xs, oc * zz + c, 0, \
                           0, 0, 0, 1 ); \
  }

#define MYMATH_CREATESCALE_FUNC(t) \
  MYMATH_INLINE impl::mat4i<t> create_scale( const impl::vec3i<t>& vec ) \
  { return impl::mat4i<t>( vec.x, 0, 0, 0, \
                           0, vec.y, 0, 0, \
                           0, 0, vec.z, 0, \
                           0, 0, 0, 1); }

#define MYMATH_CREATETRANSLATION_FUNC(t) \
  MYMATH_INLINE impl::mat4i<t> create_translation( const impl::vec3i<t>& vec ) \
  { return impl::mat4i<t>( 1, 0, 0, 0, \
                           0, 1, 0, 0, \
                           0, 0, 1, 0, \
                           vec.x, vec.y, vec.z, 1 ); }

#define MYMATH_GETANGLE_FUNC(v, t) \
  MYMATH_INLINE t get_angle(const impl::v<t>& a, const impl::v<t>& b) \
  { return std::acos( dot( normalize(a), normalize(b) ) ); }

#define MYMATH_RAYINTSPHERE_FUNC(t) \
  MYMATH_INLINE t ray_int_sphere( const impl::vec3i<t>& point, const impl::vec3i<t>& ray, const impl::vec3i<t>& sphere_center, const t& sphere_radius ) \
  { \
    impl::vec3i<t> ray_to_center = sphere_center - point; \
    t a = dot( ray_to_center, ray ); \
    t dist2 = dot( ray_to_center, ray_to_center ); \
    t d_ret = ( sphere_radius * sphere_radius ) - dist2 + ( a * a ); \
    if( d_ret > (t)0 ) \
      { d_ret = a - std::sqrt( d_ret );  } \
    return d_ret; \
  }

#define MYMATH_CLOSEENOUGH_FUNC(t) \
  MYMATH_INLINE bool close_enough( const t& candidate, const t& compare, const t& epsilon ) \
  { return ( std::abs( candidate - compare ) < epsilon ); }

#define MYMATH_PROJECTXY_FUNC(t) \
  MYMATH_INLINE impl::vec2i<t> project_xy( const impl::mat4i<t>& modelview, const impl::mat4i<t>& projection, const impl::vec4i<t>& viewport, const impl::vec3i<t>& point_in ) \
  { \
    impl::vec4i<t> back( point_in, ( t )0 ), forth; \
    impl::vec2i<t> point_out; \
    forth = modelview * back; \
    back = projection * forth; \
    if( ! close_enough( back[3], ( t )0, ( t )0.000001 ) ) \
    { \
      t div = ( t )1 / back[3]; \
      back[0] *= div; \
      back[1] *= div; \
    } \
    point_out = impl::vec2i<t>( viewport[0] + ( ( t )1 + back[0] ) * viewport[2] / ( t )2, viewport[1] + ( ( t )1 + back[1] ) * viewport[3] / ( t )2 ); \
    if( !impl::is_eq( viewport[0], ( t )0 ) ) \
      { point_out[0] -= viewport[0]; } \
    if( !impl::is_eq( viewport[1], ( t )0 ) ) \
      { point_out[1] -= viewport[1]; } \
    return point_out; \
  }

#define MYMATH_FINDNORMAL_FUNC(t) \
  MYMATH_INLINE impl::vec3i<t> find_normal(const impl::vec3i<t>& a, const impl::vec3i<t>& b, const impl::vec3i<t>& c ) \
  { return cross( a - b, b - c ); }

#define MYMATH_CATMULLROM_FUNC(t) \
  MYMATH_INLINE impl::vec3i<t> catmullrom( const impl::vec3i<t>& p0, const impl::vec3i<t>& p1, const impl::vec3i<t>& p2, const impl::vec3i<t>& p3, const t& e ) \
  { \
    t e2 = e * e; \
    t e3 = e2 * e; \
    return impl::vec3i<t>( ( t )0.5 * ( ( ( t )2 * p1[0] ) + ( -p0[0] + p2[0] ) * e + ( ( t )2 * p0[0] - ( t )5 * p1[0] + ( t )4 * p2[0] - p3[0] ) * e2 + ( -p0[0] + ( t )3 * p1[0] - ( t )3 * p2[0] + p3[0] ) * e3 ), \
                           ( t )0.5 * ( ( ( t )2 * p1[1] ) + ( -p0[1] + p2[1] ) * e + ( ( t )2 * p0[1] - ( t )5 * p1[1] + ( t )4 * p2[1] - p3[1] ) * e2 + ( -p0[1] + ( t )3 * p1[1] - ( t )3 * p2[1] + p3[1] ) * e3 ), \
                           ( t )0.5 * ( ( ( t )2 * p1[2] ) + ( -p0[2] + p2[2] ) * e + ( ( t )2 * p0[2] - ( t )5 * p1[2] + ( t )4 * p2[2] - p3[2] ) * e2 + ( -p0[2] + ( t )3 * p1[2] - ( t )3 * p2[2] + p3[2] ) * e3 ) ); \
  }

#define MYMATH_CALCTANGENT_FUNC(t) \
  MYMATH_INLINE impl::vec3i<t> calc_tangent( const impl::vec3i<t> vertices[3], const impl::vec2i<t> texcoords[3], const impl::vec3i<t>& normal ) \
  { \
    impl::vec3i<t> dv2v1 = vertices[1] - vertices[0]; \
    impl::vec3i<t> dv3v1 = vertices[2] - vertices[0]; \
    t dc2c1t = texcoords[1][0] - texcoords[0][0]; \
    t dc2c1b = texcoords[1][1] - texcoords[0][1]; \
    t dc3c1t = texcoords[2][0] - texcoords[0][0]; \
    t dc3c1b = texcoords[2][1] - texcoords[0][1]; \
    t m = (t)1 / ( ( dc2c1t * dc3c1b ) - ( dc3c1t * dc2c1b ) ); \
    dv2v1 *= impl::vec3i<t>(dc3c1b); \
    dv3v1 *= impl::vec3i<t>(dc2c1b); \
    impl::vec3i<t> tangent = dv2v1 - dv3v1; \
    tangent *= impl::vec3i<t>(m); \
    tangent = normalize ( tangent ); \
    impl::vec3i<t> bitangent = cross ( normal, tangent ); \
    tangent = cross ( bitangent, normal ); \
    return normalize ( tangent ); \
  }

#define MYMATH_CLOSESTPOINTONRAY_FUNC(t) \
  MYMATH_INLINE impl::vec3i<t> closest_point_on_ray( const impl::vec3i<t>& ray_origin, const impl::vec3i<t>& unit_ray_dir, const impl::vec3i<t>& point_in_space ) \
  { \
    impl::vec3i<t> v = point_in_space - ray_origin; \
    t e = dot ( unit_ray_dir, v ); \
    return ray_origin + unit_ray_dir * impl::vec3i<t>(e); \
  }

#define MYMATH_PROJECTXYZ_FUNC(t) \
  MYMATH_INLINE impl::vec3i<t> project_xyz( const impl::mat4i<t>& modelview, const impl::mat4i<t>& projection, const impl::vec4i<t>& viewport, const impl::vec3i<t>& point_in ) \
  { \
    impl::vec4i<t> back( point_in, ( t )0 ), forth; \
    impl::vec3i<t> point_out; \
    forth = modelview * back; \
    back = projection * forth; \
    if( !close_enough( back[3], ( t )0, ( t )0.000001 ) ) \
    { \
      t div = ( t )1 / back[3]; \
      back[0] *= div; \
      back[1] *= div; \
      back[2] *= div; \
    } \
    point_out = impl::vec3i<t>( viewport[0] + ( ( t )1 + back[0] ) * viewport[2] / ( t )2, viewport[1] + ( ( t )1 + back[1] ) * viewport[3] / ( t )2, ( t )0 ); \
    if( !impl::is_eq( viewport[0], ( t )0 ) ) \
      { point_out[0] -= viewport[0]; } \
    if( !impl::is_eq( viewport[1], ( t )0 ) ) \
      { point_out[1] -= viewport[1]; } \
    point_out[2] = back[2]; \
    return point_out; \
  }

#define MYMATH_GETPLANEEQ_FUNC(t) \
  MYMATH_INLINE impl::vec4i<t> get_plane_eq( const impl::vec3i<t>& p1, const impl::vec3i<t>& p2, const impl::vec3i<t>& p3 ) \
  { \
    impl::vec3i<t> v1 = p3 - p1; \
    impl::vec3i<t> v2 = p2 - p1; \
    impl::vec3i<t> plane_eq = cross( v1, v2 ); \
    plane_eq = normalize( plane_eq ); \
    return impl::vec4i<t> ( plane_eq[0], plane_eq[1], plane_eq[2], -dot( plane_eq, p3 ) ); \
  }

#define MYMATH_MAKEPLANARSHADOW_FUNC(t) \
  MYMATH_INLINE impl::mat4i<t> make_planar_shadow( const impl::vec4i<t>& plane_eq, const impl::vec3i<t>& light_pos ) \
  { \
    return impl::mat4i<t>( plane_eq[1] * -light_pos[1] + plane_eq[2] * -light_pos[2], -plane_eq[0] * -light_pos[1], -plane_eq[0] * -light_pos[2], ( t )0, \
                           -plane_eq[1] * -light_pos[0], plane_eq[0] * -light_pos[0] + plane_eq[2] * -light_pos[2], -plane_eq[1] * -light_pos[2], ( t )0, \
                           -plane_eq[2] * -light_pos[0], -plane_eq[2] * -light_pos[1], plane_eq[0] * -light_pos[0] + plane_eq[1] * -light_pos[1], ( t )0, \
                           -plane_eq[3] * -light_pos[0], plane_eq[3] * light_pos[1], -plane_eq[3] * -light_pos[2], plane_eq[0] * -light_pos[0] + plane_eq[1] * -light_pos[1] + plane_eq[2] * -light_pos[2] ); \
  }

  MYMATH_UNPROJECT_FUNC( float )
  
  MYMATH_CREATEROTATION_FUNC( float )

  MYMATH_CREATESCALE_FUNC( float )

  MYMATH_CREATETRANSLATION_FUNC( float )

  MYMATH_GETANGLE_FUNC( vec2i, float )
  MYMATH_GETANGLE_FUNC( vec3i, float )
  MYMATH_GETANGLE_FUNC( vec4i, float )

  MYMATH_RAYINTSPHERE_FUNC( float )

#if MYMATH_DOUBLE_PRECISION == 1
  MYMATH_CREATEROTATION_FUNC( double )
  MYMATH_CREATESCALE_FUNC( double )
  MYMATH_CREATETRANSLATION_FUNC( double )
  MYMATH_GETANGLE_FUNC( vec2i, double )
  MYMATH_GETANGLE_FUNC( vec3i, double )
  MYMATH_GETANGLE_FUNC( vec4i, double )
  MYMATH_RAYINTSPHERE_FUNC( double )
#endif

  MYMATH_INLINE bool is_pow_2( const unsigned int& val )
  {
    unsigned int pow2 = 1;

    while( val > pow2 )
    {
      pow2 *= 2;

      if( pow2 == val )
      {
        return true;
      }
    }

    return false;
  }

  MYMATH_CLOSEENOUGH_FUNC( float )
  //MYMATH_PROJECTXY_FUNC( float ) //TODO
  MYMATH_FINDNORMAL_FUNC( float )
  MYMATH_CATMULLROM_FUNC( float )
  MYMATH_CALCTANGENT_FUNC( float )
  MYMATH_CLOSESTPOINTONRAY_FUNC( float )
  //MYMATH_PROJECTXYZ_FUNC( float )
  MYMATH_GETPLANEEQ_FUNC( float )

#if MYMATH_DOUBLE_PRECISION == 1
  MYMATH_CLOSEENOUGH_FUNC( double )
  MYMATH_PROJECTXY_FUNC( double )
  MYMATH_FINDNORMAL_FUNC( double )
  MYMATH_CATMULLROM_FUNC( double )
  MYMATH_CALCTANGENT_FUNC( double )
  MYMATH_CLOSESTPOINTONRAY_FUNC( double )
  MYMATH_PROJECTXYZ_FUNC( double )
  MYMATH_GETPLANEEQ_FUNC( double )
#endif

  template< typename t >
  MYMATH_INLINE impl::mat4i<t> perspective( const t& fovy, const t& aspect, const t& near, const t& far )
  {
    t top = near * std::tan( fovy * ( t )0.5 );
    t bottom = -top;
    t left = bottom * aspect;
    t right = -left;
    impl::mat4i<t> r;
    r[0].x = ( ( t )2 * near ) / ( right - left );
    r[1].y = ( ( t )2 * near ) / ( top - bottom );
    r[2].x = ( right + left ) / ( right - left );
    r[2].y = ( top + bottom ) / ( top - bottom );
    r[2].z = -( far + near ) / ( far - near );
    r[2].w = -( t )1;
    r[3].z = -( ( t )2 * far * near ) / ( far - near );
    r[3].w = ( t )0;
    return r;
  }

  template< typename t >
  MYMATH_INLINE impl::mat4i<t> ortographic( const t& left, const t& right, const t& bottom, const t& top, const t& near, const t& far )
  {
    impl::mat4i<t> r;
    r[0].x = ( t )2 / ( right - left );
    r[1].y = ( t )2 / ( top - bottom );
    r[2].z = -( t )2 / ( far - near );
    r[3].x = -( ( right + left ) / ( right - left ) );
    r[3].y = -( ( top + bottom ) / ( top - bottom ) );
    r[3].z = -( ( far + near ) / ( far - near ) );
    r[3].w = ( t )1;
    return r;
  }

  MYMATH_MAKEPLANARSHADOW_FUNC( float )

#if MYMATH_DOUBLE_PRECISION == 1
  MYMATH_MAKEPLANARSHADOW_FUNC( double )
#endif

}

#endif
