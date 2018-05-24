#ifndef mm_vec3_impl_h
#define mm_vec3_impl_h

#include "mm_common.h"
#include "mm_vec2_impl.h"

namespace mymath
{
  namespace impl
  {
    template< typename ty >
    class vec3i
    {
      private:
        template< int at, int bt, int ct, int dt >
        class swizzle
        {
          private:
            ty v[3];
          public:
            //For cases like swizzle = vec2 and swizzle = swizzle
            const vec3i& operator=( const vec3i& other )
            {
              v[at] = other.x;
              v[bt] = other.y;
              v[ct] = other.z;
              return *( vec3i* )this;
            }

            //For cases like swizzle *= vec2 and swizzle *= swizzle
            const vec3i& operator*=( const vec3i& other )
            {
              v[at] *= other.x;
              v[bt] *= other.y;
              v[ct] *= other.z;
              return *( vec3i* )this;
            }

            const vec3i& operator/=( const vec3i& other ); //needs notEqual, defined elsewhere

            const vec3i& operator+=( const vec3i& other )
            {
              v[at] += other.x;
              v[bt] += other.y;
              v[ct] += other.z;
              return *( vec3i* )this;
            }

            const vec3i& operator-=( const vec3i& other )
            {
              v[at] -= other.x;
              v[bt] -= other.y;
              v[ct] -= other.z;
              return *( vec3i* )this;
            }

            operator vec3i() const
            {
              return vec3i( v[at], v[bt], v[ct] );
            }
        };

        template<int at>
        class swizzle < at, at, at, -3 >
        {
          private:
            ty v[3];
          public:
            operator vec3i() const
            {
              return vec3i( v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < bt, at, at, -3 >
        {
          private:
            ty v[3];
          public:
            operator vec3i() const
            {
              return vec3i( v[bt], v[at], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < at, bt, at, -3 >
        {
          private:
            ty v[3];
          public:
            operator vec3i() const
            {
              return vec3i( v[at], v[bt], v[at] );
            }
        };

        template<int at, int bt>
        class swizzle < at, at, bt, -3 >
        {
          private:
            ty v[3];
          public:
            operator vec3i() const
            {
              return vec3i( v[at], v[at], v[bt] );
            }
        };

        //vec2 swizzlers
        template<int at>
        class swizzle < at, at, -2, -3 >
        {
          private:
            ty v[3];
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
            ty v[3];
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
              assert( !impl::is_eq( other.x, ( ty )0 ) && !impl::is_eq( other.y, ( ty )0 ) );
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
            ty x, y, z;
          };

          struct
          {
            ty r, g, b;
          };

          struct
          {
            ty s, t, q;
          };

#include "includes/vec3_swizzle_declarations.h"

          ty v[3];
        };
#ifdef __GNUC__  //g++
#pragma GCC diagnostic pop
#endif

        vec3i( const ty& at, const ty& bt, const ty& ct ) : x( at ), y( bt ), z( ct ) {}
        vec3i( const vec2i<ty>& vec, const ty& num ) : x( vec.x ), y( vec.y ), z( num ) {}
        vec3i( const ty& num, const vec2i<ty>& vec ) : x( num ), y( vec.x ), z( vec.y ) {}
#if MYMATH_STRICT_GLSL == 1
        explicit
#endif
        vec3i( const ty& num ) : x( num ), y( num ), z( num ) {}
        vec3i() : x( 0 ), y( 0 ), z( 0 ) {}

        ty& operator[]( const unsigned int& num )
        {
          assert( num < 3 && this );
          return v[num];
        }

        const ty& operator[]( const unsigned int& num ) const
        {
          assert( num < 3 && this );
          return v[num];
        }

        const vec3i& operator*= ( const vec3i& vec )
        {
          x *= vec.x;
          y *= vec.y;
          z *= vec.z;
          return *this;
        }

        const vec3i& operator/= ( const vec3i& vec ); //needs notEqual defined elsewhere

        const vec3i& operator+= ( const vec3i& vec )
        {
          x += vec.x;
          y += vec.y;
          z += vec.z;
          return *this;
        }

        const vec3i& operator-= ( const vec3i& vec )
        {
          x -= vec.x;
          y -= vec.y;
          z -= vec.z;
          return *this;
        }

        const vec3i& operator%= ( const vec3i& vec )
        {
          x = ( int )x % ( int )vec.x;
          y = ( int )y % ( int )vec.y;
          z = ( int )z % ( int )vec.z;
          return *this;
        }

        const vec3i& operator<<= ( const vec3i& vec )
        {
          x = ( int )x << ( int )vec.x;
          y = ( int )y << ( int )vec.y;
          z = ( int )z << ( int )vec.z;
          return *this;
        }

        const vec3i& operator>>= ( const vec3i& vec )
        {
          x = ( int )x >> ( int )vec.x;
          y = ( int )y >> ( int )vec.y;
          z = ( int )z >> ( int )vec.z;
          return *this;
        }

        const vec3i& operator&= ( const vec3i& vec )
        {
          x = ( int )x & ( int )vec.x;
          y = ( int )y & ( int )vec.y;
          z = ( int )z & ( int )vec.z;
          return *this;
        }

        const vec3i& operator^= ( const vec3i& vec )
        {
          x = ( int )x ^( int )vec.x;
          y = ( int )y ^( int )vec.y;
          z = ( int )z ^( int )vec.z;
          return *this;
        }

        const vec3i& operator|= ( const vec3i& vec )
        {
          x = ( int )x | ( int )vec.x;
          y = ( int )y | ( int )vec.y;
          z = ( int )z | ( int )vec.z;
          return *this;
        }

        const vec3i operator++ () //pre
        {
          ++x;
          ++y;
          ++z;
          return *this;
        }

        const vec3i operator++ ( impl::post )
        {
          vec3i tmp = *this;
          ++( *this );
          return tmp;
        }

        const vec3i operator-- () //pre
        {
          --x;
          --y;
          --z;
          return *this;
        }

        const vec3i operator-- ( impl::post )
        {
          vec3i tmp = *this;
          --( *this );
          return tmp;
        }

        const unsigned int length() const
        {
          return 3;
        }
    };
  }
}

#endif
