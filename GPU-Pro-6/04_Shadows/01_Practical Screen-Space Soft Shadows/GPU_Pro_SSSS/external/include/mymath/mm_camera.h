#ifndef mm_camera_h
#define mm_camera_h

#include "mm_frame.h"

namespace mymath
{
  template< typename t >
  class camera
  {
    private:
      frame<t>* the_frame;

      impl::vec3i<t> convert_local_to_world( const impl::vec3i<t>& local, const bool& rotation_only ) const
      {
        impl::mat4i<t> rm = get_matrix( true );

        impl::vec3i<t> world = impl::mat3i<t>( rm[0].xyz, rm[1].xyz, rm[2].xyz ) * local;

        if( !rotation_only )
        {
          world += pos;
        }

        return world;
      }

      impl::vec3i<t> convert_world_to_local( const impl::vec3i<t>& world ) const
      {
        impl::vec3i<t> new_world = world - pos;

        impl::mat4i<t> im = inverse( get_matrix( true ) );

        return impl::mat3i<t>( im[0].xyz, im[1].xyz, im[2].xyz ) * new_world;
      }

      impl::vec3i<t> rotate_vector( const impl::vec3i<t>& vec ) const
      {
        impl::mat4i<t> m = get_matrix( true );

        return impl::mat3i<t>( m[0].xyz, m[1].xyz, m[2].xyz ) * vec;
      }

      impl::vec3i<t> transform_point( const impl::vec3i<t>& point ) const
      {
        impl::mat4i<t> m = get_matrix( false );

        return impl::mat3i<t>( m[0].xyz, m[1].xyz, m[2].xyz ) * point + impl::vec3i<t>( m[3].x, m[3].y, m[3].z );
      }
    protected:

    public:
      impl::vec3i<t> pos;
      impl::vec3i<t> view_dir;
      impl::vec3i<t> up_vector;

      impl::vec2i<t> get_rotations( const int& screen_width, const int& screen_height, const int& delta_x, const int& delta_y ) const
      {
        impl::vec2i<t> rot;

        if( delta_x != 0 )
        {
          rot.x = ( ( delta_x - ( screen_width * 0.5 ) ) / ( screen_width * 0.02 ) ) * 0.01 * 360.0f;
        }

        if( delta_y != 0 )
        {
          rot.y = ( ( delta_y - ( screen_height * 0.5 ) ) / ( screen_height * 0.02 ) ) * 0.01 * -360.0f;
        }

        return rot;
      }

      void translate_world( const impl::vec3i<t>& vec )
      {
        pos += vec;
      }

      void translate_local( const impl::vec3i<t>& vec )
      {
        move_right( vec[0] );
        move_up( vec[1] );
        move_forward( vec[2] );
      }

      void move_forward( const t& d )
      {
        pos += view_dir * impl::vec3i<t>( d );
      }

      void move_up( const t& d )
      {
        pos += up_vector * impl::vec3i<t>( d );
      }

      void move_right( const t& d )
      {
        pos += cross( up_vector, view_dir ) * impl::vec3i<t>( -d );
      }

      void rotate_x( const t& angle )
      {
        impl::mat4i<t> dummy;
        impl::mat3i<t> rot_mat;
        impl::vec3i<t> x_vec;
        impl::vec3i<t> rot_vec;

        x_vec = cross( up_vector, view_dir );

        dummy = create_rotation( angle, x_vec );
        rot_mat = impl::mat3i<t>( dummy[0].xyz, dummy[1].xyz, dummy[2].xyz );

        rot_vec = rot_mat * up_vector;
        up_vector = rot_vec;

        rot_vec = rot_mat * view_dir;
        view_dir = rot_vec;
      }

      void rotate_y( const t& angle )
      {
        impl::mat4i<t> rot_mat = create_rotation( angle, up_vector );

        view_dir = impl::mat3i<t>( rot_mat[0].xyz, rot_mat[1].xyz, rot_mat[2].xyz ) * view_dir;
      }

      void rotate_z( const t& angle )
      {
        impl::mat4i<t> rot_mat = create_rotation( angle, view_dir );

        up_vector = impl::mat3i<t>( rot_mat[0].xyz, rot_mat[1].xyz, rot_mat[2].xyz ) * up_vector;
      }

      void rotate_world( const t& angle, const impl::vec3i<t>& vec )
      {
        impl::mat4i<t> rot_mat;
        rot_mat = create_rotation( angle, vec );

        up_vector = impl::mat3i<t>( rot_mat[0].xyz, rot_mat[1].xyz, rot_mat[2].xyz ) * up_vector;
        view_dir = impl::mat3i<t>( rot_mat[0].xyz, rot_mat[1].xyz, rot_mat[2].xyz ) * view_dir;
      }

      void rotate_local( const t& angle, const impl::vec3i<t>& vec )
      {
        rotate_world( angle, convert_local_to_world( vec, true ) );
      }

      impl::mat4i<t> get_matrix( const bool& rotation_only ) const
      {
        impl::vec3i<t> right = cross( up_vector, view_dir );

        if( rotation_only )
        {

          return impl::mat4i<t>( impl::vec4i<t>( right, 0 ),
                                 impl::vec4i<t>( up_vector, 0 ),
                                 impl::vec4i<t>( view_dir, 0 ),
                                 impl::vec4i<t>( 0, 0, 0, 1 ) );
        }
        else
        {
          return impl::mat4i<t>( impl::vec4i<t>( right, 0 ),
                                 impl::vec4i<t>( up_vector, 0 ),
                                 impl::vec4i<t>( view_dir, 0 ),
                                 impl::vec4i<t>( pos, 1 ) );
        }
      }

      impl::mat4i<t> get_camera_matrix( const bool& rotation_only ) const
      {
        impl::vec3i<t> x = cross( up_vector, -view_dir );

        impl::mat4i<t> m( x[0], up_vector[0], -view_dir[0], 0,
                          x[1], up_vector[1], -view_dir[1], 0,
                          x[2], up_vector[2], -view_dir[2], 0,
                          0, 0, 0, 1 );

        if( rotation_only )
        {
          return m;
        }

        impl::mat4i<t> trans = create_translation( -pos );

        return m * trans;
      }

      void normalize_axises()
      {
        impl::vec3i<t> cross_prod = cross( up_vector, view_dir );
        view_dir = cross( cross_prod, up_vector );

        up_vector = normalize( up_vector );
        view_dir = normalize( view_dir );
      }

      const frame<t>* get_frame() const
      {
        assert( the_frame != 0 );
        return the_frame;
      }

      void set_frame( frame<t>* frm )
      {
        assert( frm != 0 );
        the_frame = frm;
      }

      camera() : the_frame( 0 ), pos( impl::vec3i<t>( 0, 0, 0 ) ), view_dir( impl::vec3i<t>( 0, 0, -1 ) ), up_vector( impl::vec3i<t>( 0, 1, 0 ) ) {}

  };
}

#endif
