#ifndef mm_fvec2_impl_h
#define mm_fvec2_impl_h

#include "mm_common.h"
#include "mm_sse.h"

namespace mymath
{
  namespace impl
  {
    template< typename ty >
    class MYMATH_GPU_ALIGNED vec2i;
    template< typename ty >
    class MYMATH_GPU_ALIGNED vec3i;
    template< typename ty >
    class MYMATH_GPU_ALIGNED vec4i;

    template<>
    class vec2i<float>
    {
      private:
        template< int at, int bt, int ct, int dt >
        class swizzle
        {
          private:
            __m128 v;
          public:
            //For cases like swizzle = vec2 and swizzle = swizzle
            const vec2i& operator=( const vec2i& other )
            {
              v = _mm_shuffle_ps( other.d, other.d, MYMATH_SHUFFLE( at, bt, 0, 0 ) );
              return *( vec2i* )this;
            }

            //For cases like swizzle *= vec2 and swizzle *= swizzle
            const vec2i& operator*=( const vec2i& other )
            {
              v = _mm_mul_ps( v, _mm_shuffle_ps( other.d, other.d, MYMATH_SHUFFLE( at, bt, 0, 0 ) ) );
              return *( vec2i* )this;
            }

            const vec2i& operator/=( const vec2i& other ); //needs notEqual, defined elsewhere

            const vec2i& operator+=( const vec2i& other )
            {
              v = _mm_add_ps( v, _mm_shuffle_ps( other.d, other.d, MYMATH_SHUFFLE( at, bt, 0, 0 ) ) );
              return *( vec2i* )this;
            }

            const vec2i& operator-=( const vec2i& other )
            {
              v = _mm_sub_ps( v, _mm_shuffle_ps( other.d, other.d, MYMATH_SHUFFLE( at, bt, 0, 0 ) ) );
              return *( vec2i* )this;
            }

            operator vec2i() const
            {
              return vec2i( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( at, bt, 0, 0 ) ) );
            }
        };

        template<int at>
        class swizzle < at, at, -2, -3 >
        {
          private:
            __m128 v;
          public:
            operator vec2i() const
            {
              return vec2i( _mm_shuffle_ps( v, v, MYMATH_SHUFFLE( at, at, 0, 0 ) ) );
            }
        };

        //vec2 to vec3 and vec4 declarations
        template< int a >
        class swizzle < a, a, a, -3 >
        {
          private:
            __m128 v;
          public:
            operator vec3i<float>() const;
        };

        template<int a, int b>
        class swizzle < b, a, a, -3 >
        {
          private:
            __m128 v;
          public:
            operator vec3i<float>() const;
        };

        template<int a, int b>
        class swizzle < a, b, a, -3 >
        {
          private:
            __m128 v;
          public:
            operator vec3i<float>() const;
        };

        template<int a, int b>
        class swizzle < a, a, b, -3 >
        {
          private:
            __m128 v;
          public:
            operator vec3i<float>() const;
        };

        //vec4 swizzlers for vec2
        template<int a>
        class swizzle<a, a, a, a>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

        template<int a, int b>
        class swizzle<b, a, a, a>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

        template<int a, int b>
        class swizzle<a, b, a, a>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

        template<int a, int b>
        class swizzle<a, a, b, a>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

        template<int a, int b>
        class swizzle<a, a, a, b>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

        template<int a, int b>
        class swizzle<a, b, a, b>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

        template<int a, int b>
        class swizzle<a, a, b, b>
        {
          private:
            __m128 v;
          public:
            operator vec4i<float>() const;
        };

      protected:

      public:
#ifdef __GNUC__  //g++
#pragma GCC diagnostic ignored "-pedantic"
#endif
        union
        {
          struct
          {
            float x, y;
            float _dummy[2];
          };

          struct
          {
            float r, g;
            float _dummy2[2];
          };

          struct
          {
            float s, t;
            float _dummy3[2];
          };

#include "includes/vec2_swizzle_declarations.h"

          struct
          {
            float v[2];
            float _dummy4[2];
          };

          __m128 d;
        };
#ifdef __GNUC__  //g++
#pragma GCC diagnostic pop
#endif

        vec2i( const float& at, const float& bt ) : x( at ), y( bt ) {}
#if MYMATH_STRICT_GLSL == 1
        explicit
#endif
        vec2i( const float& num ) { d = _mm_set1_ps(num); }
        vec2i( const __m128& num ) : d( num ) {}
        //vec2i() { d = impl::zero; }
        vec2i(){}

        float& operator[]( const unsigned int& num ) //read-write
        {
          assert( num < 2 && this );
          return v[num];
        }

        const float& operator[]( const unsigned int& num ) const //read only, constant ref
        {
          assert( num < 2 && this );
          return v[num];
        }

        const vec2i& operator*= ( const vec2i& vec )
        {
          d = _mm_mul_ps( d, vec.d );
          return *this;
        }

        const vec2i& operator/= ( const vec2i& vec ); //needs notEqual defined elsewhere

        const vec2i& operator+= ( const vec2i& vec )
        {
          d = _mm_add_ps( d, vec.d );
          return *this;
        }

        const vec2i& operator-= ( const vec2i& vec )
        {
          d = _mm_sub_ps( d, vec.d );
          return *this;
        }

        const vec2i& operator%= ( const vec2i& vec )
        {
          d = sse_mod_ps( d, vec.d );
          return *this;
        }

        //TODO
        /*const vec2i& operator<<= ( const vec2i& vec )
        {
          x = ( int )x << ( int )vec.x;
          y = ( int )y << ( int )vec.y;
          return *this;
        }

        const vec2i& operator>>= ( const vec2i& vec )
        {
          x = ( int )x >> ( int )vec.x;
          y = ( int )y >> ( int )vec.y;
          return *this;
        }*/

        const vec2i& operator&= ( const vec2i& vec )
        {
          d = _mm_and_ps( d, vec.d );
          return *this;
        }

        const vec2i& operator^= ( const vec2i& vec )
        {
          d = _mm_xor_ps( d, vec.d );
          return *this;
        }

        const vec2i& operator|= ( const vec2i& vec )
        {
          d = _mm_or_ps( d, vec.d );
          return *this;
        }

        const vec2i operator++ () //pre
        {
          d = _mm_add_ps( d, impl::one );
          return *this;
        }

        const vec2i operator++ ( impl::post )
        {
          vec2i tmp = *this;
          ++( *this );
          return tmp;
        }

        const vec2i operator-- () //pre
        {
          d = _mm_sub_ps( d, impl::one );
          return *this;
        }

        const vec2i operator-- ( impl::post )
        {
          vec2i tmp = *this;
          --( *this );
          return tmp;
        }

        const unsigned int length() const
        {
          return 2;
        }
    };
  }
}

#endif
