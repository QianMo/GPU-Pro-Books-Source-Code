#ifndef mm_mat3_impl_h
#define mm_mat3_impl_h

//class declaration only
namespace mymath
{
  namespace impl
  {
    template< typename t >
    class MYMATH_GPU_ALIGNED mat3i;
  }
}

#include "mm_vec3_impl.h"
#include "mm_vec_func.h"
#include "mm_mat4_impl.h"

#include "mm_quat_func.h"

namespace mymath
{
  namespace impl
  {
    template< typename t >
    class MYMATH_GPU_ALIGNED mat3i
    {
      private:
        /*
         * matrix layout:
         * m[0].x m[1].x m[2].x
         * m[0].y m[1].y m[2].y
         * m[0].z m[1].z m[2].z
         */
        vec3i<t> m[3];

      protected:

      public:
        // 1 column vector per row
        mat3i( const t& m0, const t& m1, const t& m2,
               const t& m3, const t& m4, const t& m5,
               const t& m6, const t& m7, const t& m8 )
        {
          m[0] = vec3i<t>( m0, m1, m2 );
          m[1] = vec3i<t>( m3, m4, m5 );
          m[2] = vec3i<t>( m6, m7, m8 );
        }

        mat3i(const mat4i<t>& mat)
        {
          m[0] = mat[0].xyz;
          m[1] = mat[1].xyz;
          m[2] = mat[2].xyz;
        }

        mat3i(const quati<t>& q)
        {
          const mat3i<t> other = mat3_cast(q);
          for(int i = 0; i < 3; ++i)
          {
            m[i] = other[i];
          }
        }

        // 1 column per vector
        mat3i( const vec3i<t>& a, const vec3i<t>& b, const vec3i<t>& c )
        {
          m[0] = a;
          m[1] = b;
          m[2] = c;
        }

        explicit mat3i( const t& num )
        {
          m[0] = vec3i<t>( num, 0, 0 );
          m[1] = vec3i<t>( 0, num, 0 );
          m[2] = vec3i<t>( 0, 0, num );
        }

        mat3i()
        {
          m[0] = vec3i<t>( 1, 0, 0 );
          m[1] = vec3i<t>( 0, 1, 0 );
          m[2] = vec3i<t>( 0, 0, 1 );
        }

        vec3i<t>& operator[]( const unsigned int& num )
        {
          assert( num < 3 );
          return m[num];
        }

        const vec3i<t>& operator[]( const unsigned int& num ) const
        {
          assert( num < 3 );
          return m[num];
        }

        const mat3i& operator*= ( const mat3i& mat )
        {
          vec3i<t> tmp1 = m[0];
          vec3i<t> tmp2 = m[1];
          vec3i<t> tmp3 = m[2];
          m[0] = mm::fma(mat[0].zzz, tmp3, fma(mat[0].yyy, tmp2, mat[0].xxx * tmp1));
          m[1] = mm::fma(mat[1].zzz, tmp3, fma(mat[1].yyy, tmp2, mat[1].xxx * tmp1));
          m[2] = mm::fma(mat[2].zzz, tmp3, fma(mat[2].yyy, tmp2, mat[2].xxx * tmp1));

          return *this;
        }

        const mat3i& operator*= ( const t& num )
        {
          m[0] *= num;
          m[1] *= num;
          m[2] *= num;
          return *this;
        }

        const mat3i& operator++ () //pre
        {
          ++m[0];
          ++m[1];
          ++m[2];
          return *this;
        }

        mat3i operator++ ( impl::post )
        {
          mat3i tmp = ( *this );
          ++( *this );
          return tmp;
        }

        const mat3i& operator-- () //pre
        {
          --m[0];
          --m[1];
          --m[2];
          return *this;
        }

        mat3i operator-- ( impl::post )
        {
          mat3i tmp = ( *this );
          --( *this );
          return tmp;
        }
    };
  }
}

#endif
