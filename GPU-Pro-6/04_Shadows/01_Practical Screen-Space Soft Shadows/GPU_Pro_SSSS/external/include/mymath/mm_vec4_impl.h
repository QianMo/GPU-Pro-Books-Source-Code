#ifndef mm_vec4_impl_h
#define mm_vec4_impl_h

#include "mm_common.h"
#include "mm_vec2_impl.h"
#include "mm_vec3_impl.h"

namespace mymath
{
  namespace impl
  {
    template< typename ty >
    class vec4i
    {
      private:
        template< int at, int bt, int ct, int dt >
        class swizzle
        {
          private:
            ty v[4];
          public:
            //For cases like swizzle = vec2 and swizzle = swizzle
            const vec4i& operator=( const vec4i& other )
            {
              v[at] = other.x;
              v[bt] = other.y;
              v[ct] = other.z;
              v[dt] = other.w;
              return *( vec4i* )this;
            }

            //For cases like swizzle *= vec2 and swizzle *= swizzle
            const vec4i& operator*=( const vec4i& other )
            {
              v[at] *= other.x;
              v[bt] *= other.y;
              v[ct] *= other.z;
              v[dt] *= other.w;
              return *( vec4i* )this;
            }

            const vec4i& operator/=( const vec4i& other ); //needs notEqual defined elsewhere

            const vec4i& operator+=( const vec4i& other )
            {
              v[at] += other.x;
              v[bt] += other.y;
              v[ct] += other.z;
              v[dt] += other.w;
              return *( vec4i* )this;
            }

            const vec4i& operator-=( const vec4i& other )
            {
              v[at] -= other.x;
              v[bt] -= other.y;
              v[ct] -= other.z;
              v[dt] -= other.w;
              return *( vec4i* )this;
            }

            operator vec4i() const
            {
              return vec4i( v[at], v[bt], v[ct], v[dt] );
            }
        };

        template<int at>
        class swizzle<at, at, at, at>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[at] );
            }
        };

        template<int at, int bt>
        class swizzle<bt, at, at, at>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[bt], v[at], v[at], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle<at, bt, at, at>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[at], v[bt], v[at], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle<at, at, bt, at>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[at], v[at], v[bt], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle<at, at, at, bt>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[at], v[at], v[at], v[bt] );
            }
        };

        template<int at, int bt, int ct>
        class swizzle<bt, ct, at, at>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[bt], v[ct], v[at], v[at] );
            }
        };

        template<int at, int bt, int ct>
        class swizzle<bt, at, ct, at>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[bt], v[at], v[ct], v[at] );
            }
        };

        template<int at, int bt, int ct>
        class swizzle<bt, at, at, ct>
        {
          private:
            ty v[4];
          public:
            operator vec4i() const
            {
              return vec4i( v[bt], v[at], v[at], v[ct] );
            }
        };

        //vec3 swizzlers
        template<int at>
        class swizzle < at, at, at, -3 >
        {
          private:
            ty v[4];
          public:
            operator vec3i<ty>() const
            {
              return vec3i<ty>( v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < bt, at, at, -3 >
        {
          private:
            ty v[4];
          public:
            operator vec3i<ty>() const
            {
              return vec3i<ty>( v[bt], v[at], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < at, bt, at, -3 >
        {
          private:
            ty v[4];
          public:
            operator vec3i<ty>() const
            {
              return vec3i<ty>( v[at], v[bt], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < at, at, bt, -3 >
        {
          private:
            ty v[4];
          public:
            operator vec3i<ty>() const
            {
              return vec3i<ty>( v[at], v[at], v[bt] );
            }
        };

        template<int at, int bt, int ct> //3 component
        class swizzle < at, bt, ct, -3 >
        {
          private:
            ty v[4];
          public:
            //For cases like swizzle = vec2 and swizzle = swizzle
            const vec3i<ty>& operator=( const vec3i<ty>& other )
            {
              v[at] = other.x;
              v[bt] = other.y;
              v[ct] = other.z;
              return *( vec3i<ty>* )this;
            }

            //For cases like swizzle *= vec2 and swizzle *= swizzle
            const vec3i<ty>& operator*=( const vec3i<ty>& other )
            {
              v[at] *= other.x;
              v[bt] *= other.y;
              v[ct] *= other.z;
              return *( vec3i<ty>* )this;
            }

            const vec3i<ty>& operator/=( const vec3i<ty>& other )
            {
              assert( !impl::is_eq( other.x, ( ty )0 ) && !impl::is_eq( other.y, ( ty )0 ) && !impl::is_eq( other.z, ( ty )0 ) );
              vec3i<ty> tmp( ( ty )1 / other.x, ( ty )1 / other.y, ( ty )1 / other.z );
              v[at] *= tmp.x;
              v[bt] *= tmp.y;
              v[ct] *= tmp.z;
              return *( vec3i<ty>* )this;
            }

            const vec3i<ty>& operator+=( const vec3i<ty>& other )
            {
              v[at] += other.x;
              v[bt] += other.y;
              v[ct] += other.z;
              return *( vec3i<ty>* )this;
            }

            const vec3i<ty>& operator-=( const vec3i<ty>& other )
            {
              v[at] -= other.x;
              v[bt] -= other.y;
              v[ct] -= other.z;
              return *( vec3i<ty>* )this;
            }

            operator vec3i<ty>() const
            {
              return vec3i<ty>( v[at], v[bt], v[ct] );
            }
        };

        //vec2 swizzlers
        template<int at>
        class swizzle < at, at, -2, -3 >
        {
          private:
            ty v[4];
          public:
            operator vec2i<ty>() const
            {
              return vec2i<ty>( v[at], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < at, bt, -2, -3 >
        {
          private:
            ty v[4];
          public:
            //For cases like swizzle = vec2 and swizzle = swizzle
            const vec2i<ty>& operator=( const vec2i<ty>& other )
            {
              v[at] = other.x;
              v[bt] = other.y;
              return *( vec2i<ty>* )this;
            }

            //For cases like swizzle *= vec2 and swizzle *= swizzle
            const vec2i<ty>& operator*=( const vec2i<ty>& other )
            {
              v[at] *= other.x;
              v[bt] *= other.y;
              return *( vec2i<ty>* )this;
            }

            const vec2i<ty>& operator/=( const vec2i<ty>& other )
            {
              assert( other.x != ( ty )0 && other.y != ( ty )0 );
              vec2i<ty> tmp( ( ty )1 / other.x, ( ty )1 / other.y );
              v[at] *= tmp.x;
              v[bt] *= tmp.y;
              return *( vec2i<ty>* )this;
            }

            const vec2i<ty>& operator+=( const vec2i<ty>& other )
            {
              v[at] += other.x;
              v[bt] += other.y;
              return *( vec2i<ty>* )this;
            }

            const vec2i<ty>& operator-=( const vec2i<ty>& other )
            {
              v[at] -= other.x;
              v[bt] -= other.y;
              return *( vec2i<ty>* )this;
            }

            operator vec2i<ty>() const
            {
              return vec2i<ty>( v[at], v[bt] );
            }
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
            ty x, y, z, w;
          };

          struct
          {
            ty r, g, b, a;
          };

          struct
          {
            ty s, t, q, p;
          };

#include "includes/vec4_swizzle_declarations.h"

          ty v[4];
        };
#ifdef __GNUC__  //g++
#pragma GCC diagnostic pop
#endif

        vec4i( const ty& at, const ty& bt, const ty& ct, const ty& dt ) : x( at ), y( bt ), z( ct ), w( dt ) {}
        vec4i( const vec3i<ty>& vec, const ty& num ) : x( vec.x ), y( vec.y ), z( vec.z ), w( num ) {}
        vec4i( const ty& num, const vec3i<ty>& vec ) : x( num ), y( vec.x ), z( vec.y ), w( vec.z ) {}
        vec4i( const vec2i<ty>& at, const vec2i<ty>& bt ) : x( at.x ), y( at.y ), z( bt.x ), w( bt.y ) {}
        vec4i( const vec2i<ty>& vec, const ty& num1, const ty& num2 ) : x( vec.x ), y( vec.y ), z( num1 ), w( num2 ) {}
        vec4i( const ty& num1, const vec2i<ty>& vec, const ty& num2 ) : x( num1 ), y( vec.x ), z( vec.y ), w( num2 ) {}
        vec4i( const ty& num1, const ty& num2, const vec2i<ty>& vec ) : x( num1 ), y( num2 ), z( vec.x ), w( vec.y ) {}
#if MYMATH_STRICT_GLSL == 1
        explicit
#endif
        vec4i( const ty& num ) : x( num ), y( num ), z( num ), w( num ) {}
        vec4i() : x( 0 ), y( 0 ), z( 0 ), w( 0 ) {}

        ty& operator[]( const unsigned int& num )
        {
          assert( num < 4 && this );
          return v[num];
        }

        const ty& operator[]( const unsigned int& num ) const
        {
          assert( num < 4 && this );
          return v[num];
        }

        const vec4i& operator*= ( const vec4i& vec )
        {
          x *= vec.x;
          y *= vec.y;
          z *= vec.z;
          w *= vec.w;
          return *this;
        }

        const vec4i& operator/= ( const vec4i& vec ); //needs notEqual defined elsewhere

        const vec4i& operator+= ( const vec4i& vec )
        {
          x += vec.x;
          y += vec.y;
          z += vec.z;
          w += vec.w;
          return *this;
        }

        const vec4i& operator-= ( const vec4i& vec )
        {
          x -= vec.x;
          y -= vec.y;
          z -= vec.z;
          w -= vec.w;
          return *this;
        }

        const vec4i& operator%= ( const vec4i& vec )
        {
          x = ( int )x % ( int )vec.x;
          y = ( int )y % ( int )vec.y;
          z = ( int )z % ( int )vec.z;
          w = ( int )w % ( int )vec.w;
          return *this;
        }

        const vec4i& operator<<= ( const vec4i& vec )
        {
          x = ( int )x << ( int )vec.x;
          y = ( int )y << ( int )vec.y;
          z = ( int )z << ( int )vec.z;
          w = ( int )w << ( int )vec.w;
          return *this;
        }

        const vec4i& operator>>= ( const vec4i& vec )
        {
          x = ( int )x >> ( int )vec.x;
          y = ( int )y >> ( int )vec.y;
          z = ( int )z >> ( int )vec.z;
          w = ( int )w >> ( int )vec.w;
          return *this;
        }

        const vec4i& operator&= ( const vec4i& vec )
        {
          x = ( int )x & ( int )vec.x;
          y = ( int )y & ( int )vec.y;
          z = ( int )z & ( int )vec.z;
          w = ( int )w & ( int )vec.w;
          return *this;
        }

        const vec4i& operator^= ( const vec4i& vec )
        {
          x = ( int )x ^( int )vec.x;
          y = ( int )y ^( int )vec.y;
          z = ( int )z ^( int )vec.z;
          w = ( int )w ^( int )vec.w;
          return *this;
        }

        const vec4i& operator|= ( const vec4i& vec )
        {
          x = ( int )x | ( int )vec.x;
          y = ( int )y | ( int )vec.y;
          z = ( int )z | ( int )vec.z;
          w = ( int )w | ( int )vec.w;
          return *this;
        }

        const vec4i operator++ () //pre
        {
          ++x;
          ++y;
          ++z;
          ++w;
          return *this;
        }

        const vec4i operator++ ( impl::post )
        {
          vec4i tmp = *this;
          ++( *this );
          return tmp;
        }

        const vec4i operator-- () //pre
        {
          --x;
          --y;
          --z;
          --w;
          return *this;
        }

        const vec4i operator-- ( impl::post )
        {
          vec4i tmp = *this;
          --( *this );
          return tmp;
        }

        const unsigned int length() const
        {
          return 4;
        }
    };
  }
}

#endif
