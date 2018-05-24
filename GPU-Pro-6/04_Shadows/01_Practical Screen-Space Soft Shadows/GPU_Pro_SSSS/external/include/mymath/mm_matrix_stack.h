#ifndef mm_matrix_stack_h
#define mm_matrix_stack_h

#include "mm_camera.h"

namespace mymath
{
  template< typename t >
  class matrix_stack
  {
    private:
      int stack_pointer;
      impl::mat4i<t> stack[64];
    protected:

    public:

      void load_identity()
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] = impl::mat4i<t>( 1 );
      }

      void load_matrix( const impl::mat4i<t>& m )
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] = m;
      }

      void load_matrix( const camera<t>& c )
      {
        load_matrix( c.get_camera_matrix( false ) );
      }

      void mult_matrix( const impl::mat4i<t>& m )
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] *= m;
      }

      void mult_matrix( const camera<t>& c )
      {
        mult_matrix( c.get_camera_matrix( false ) );
      }

      void push_matrix()
      {
        ++stack_pointer;
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] = stack[stack_pointer - 1];
      }

      void push_matrix( const impl::mat4i<t>& m )
      {
        ++stack_pointer;
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] = m;
      }

      void push_matrix( const camera<t>& c )
      {
        push_matrix( c.get_camera_matrix( false ) );
      }

      void pop_matrix()
      {
        --stack_pointer;
        assert( stack_pointer < 64 && stack_pointer >= 0 );
      }

      void scale_matrix( const impl::vec3i<t>& vec )
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] *= create_scale( vec );
      }

      void translate_matrix_v( const impl::vec3i<t>& vec )
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        impl::mat4i<t> translation = create_translation( vec );
        translation[3].w = 0;
        stack[stack_pointer] *= translation;
      }

      void translate_matrix( const impl::vec3i<t>& vec )
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] *= create_translation( vec );
      }

      void rotate_matrix( const t& angle, const impl::vec3i<t>& vec )
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        stack[stack_pointer] *= create_rotation( angle, vec );
      }

      impl::mat4i<t> get_matrix() const
      {
        assert( stack_pointer < 64 && stack_pointer >= 0 );
        return stack[stack_pointer];
      }

      matrix_stack() : stack_pointer( 0 ) {}

  };
}

#endif
