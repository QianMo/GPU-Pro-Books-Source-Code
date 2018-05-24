#ifndef mm_mat2_impl_h
#define mm_mat2_impl_h

#include "mm_vec2_impl.h"
#include "mm_vec_func.h"

namespace mymath
{
  namespace impl
  {
    template< typename t >
    class MYMATH_GPU_ALIGNED mat2i
    {
      private:
        /*
         * matrix layout:
         * m[0].x m[1].x
         * m[0].y m[1].y
         */
        impl::vec2i<t> m[2];

      protected:

      public:
        // 1 column vector per row
        mat2i( const t& m0, const t& m1,
               const t& m2, const t& m3 )
        {
          m[0] = vec2i<t>( m0, m1 );
          m[1] = vec2i<t>( m2, m3 );
        }

        // 1 column per vector
        mat2i( const vec2i<t>& a, const vec2i<t>& b )
        {
          m[0] = a;
          m[1] = b;
        }

        explicit mat2i( const t& num )
        {
          m[0] = vec2i<t>( num, 0 );
          m[1] = vec2i<t>( 0, num );
        }

        mat2i()
        {
          m[0] = vec2i<t>( 1, 0 );
          m[1] = vec2i<t>( 0, 1 );
        }

        vec2i<t>& operator[]( const unsigned int& num )
        {
          assert( num < 2 );
          return m[num];
        }

        const vec2i<t>& operator[]( const unsigned int& num ) const
        {
          assert( num < 2 );
          return m[num];
        }

        const mat2i& operator*= ( const mat2i& mat )
        {       
          vec2i<t> tmp1 = m[0];
          vec2i<t> tmp2 = m[1];
          m[0] = mm::fma(mat[0].xx, tmp1, mat[0].yy * tmp2);
          m[1] = mm::fma(mat[1].xx, tmp1, mat[1].yy * tmp2);

          return *this;
        }

        const mat2i& operator*= ( const t& num )
        {
          m[0] *= num;
          m[1] *= num;
          return *this;
        }

        const mat2i& operator++ () //pre
        {
          ++m[0];
          ++m[1];
          return *this;
        }

        mat2i operator++ ( impl::post )
        {
          mat2i tmp = *this;
          ++( *this );
          return tmp;
        }

        const mat2i& operator-- () //pre
        {
          --m[0];
          --m[1];
          return *this;
        }

        mat2i operator-- ( impl::post )
        {
          mat2i tmp = *this;
          --( *this );
          return tmp;
        }

    };
  }
}

#endif
