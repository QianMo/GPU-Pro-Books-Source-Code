#ifndef mm_vec2_impl_h
#define mm_vec2_impl_h

#include "mm_common.h"

namespace mymath
{
  namespace impl
  {
    template< typename ty >
    class vec2i
    {
      private:
        template< int at, int bt, int ct, int dt >
        class swizzle
        {
          private:
            ty v[2];
          public:
            //For cases like swizzle = vec2 and swizzle = swizzle
            const vec2i& operator=( const vec2i& other )
            {
              v[at] = other.x;
              v[bt] = other.y;
              return *( vec2i* )this;
            }

            //For cases like swizzle *= vec2 and swizzle *= swizzle
            const vec2i& operator*=( const vec2i& other )
            {
              v[at] *= other.x;
              v[bt] *= other.y;
              return *( vec2i* )this;
            }

            const vec2i& operator/=( const vec2i& other ); //needs notEqual, defined elsewhere

            const vec2i& operator+=( const vec2i& other )
            {
              v[at] += other.x;
              v[bt] += other.y;
              return *( vec2i* )this;
            }

            const vec2i& operator-=( const vec2i& other )
            {
              v[at] -= other.x;
              v[bt] -= other.y;
              return *( vec2i* )this;
            }

            operator vec2i() const
            {
              return vec2i( v[at], v[bt] );
            }
        };

        template<int at>
        class swizzle < at, at, -2, -3 >
        {
          private:
            ty v[2];
          public:
            operator vec2i() const
            {
              return vec2i( v[at] );
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
            ty x, y;
          };

          struct
          {
            ty r, g;
          };

          struct
          {
            ty s, t;
          };

#include "includes/vec2_swizzle_declarations.h"

          ty v[2];
        };
#ifdef __GNUC__  //g++
#pragma GCC diagnostic pop
#endif

        vec2i( const ty& at, const ty& bt ) : x( at ), y( bt ) {}
#if MYMATH_STRICT_GLSL == 1
        explicit
#endif
        vec2i( const ty& num ) : x( num ), y( num ) {}
        vec2i() : x( 0 ), y( 0 ) {}

        ty& operator[]( const unsigned int& num ) //read-write
        {
          assert( num < 2 && this );
          return v[num];
        }

        const ty& operator[]( const unsigned int& num ) const //read only, constant ref
        {
          assert( num < 2 && this );
          return v[num];
        }

        const vec2i& operator*= ( const vec2i& vec )
        {
          x *= vec.x;
          y *= vec.y;
          return *this;
        }

        const vec2i& operator/= ( const vec2i& vec ); //needs notEqual defined elsewhere

        const vec2i& operator+= ( const vec2i& vec )
        {
          x += vec.x;
          y += vec.y;
          return *this;
        }

        const vec2i& operator-= ( const vec2i& vec )
        {
          x -= vec.x;
          y -= vec.y;
          return *this;
        }

        const vec2i& operator%= ( const vec2i& vec )
        {
          x = ( int )x % ( int )vec.x;
          y = ( int )y % ( int )vec.y;
          return *this;
        }

        const vec2i& operator<<= ( const vec2i& vec )
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
        }

        const vec2i& operator&= ( const vec2i& vec )
        {
          x = ( int )x & ( int )vec.x;
          y = ( int )y & ( int )vec.y;
          return *this;
        }

        const vec2i& operator^= ( const vec2i& vec )
        {
          x = ( int )x ^( int )vec.x;
          y = ( int )y ^( int )vec.y;
          return *this;
        }

        const vec2i& operator|= ( const vec2i& vec )
        {
          x = ( int )x | ( int )vec.x;
          y = ( int )y | ( int )vec.y;
          return *this;
        }

        const vec2i operator++ () //pre
        {
          ++x;
          ++y;
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
          --x;
          --y;
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
