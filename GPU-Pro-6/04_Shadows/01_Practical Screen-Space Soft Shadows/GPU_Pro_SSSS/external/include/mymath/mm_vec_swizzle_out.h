#ifndef mm_vec_swizzle_out_h
#define mm_vec_swizzle_out_h

#include "mm_common.h"
#include "mm_vec2_impl.h"
#include "mm_vec3_impl.h"
#include "mm_vec4_impl.h"

namespace mymath
{
  namespace impl
  {
    //vec3 swizzlers for vec2
    template<typename ty>
    template<int a>
    class vec2i<ty>::swizzle < a, a, a, -3 >
    {
      private:
        ty v[2];
      public:
        operator vec3i<ty>() const
        {
          return vec3i<ty>( v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle < b, a, a, -3 >
    {
      private:
        ty v[2];
      public:
        operator vec3i<ty>() const
        {
          return vec3i<ty>( v[b], v[a], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle < a, b, a, -3 >
    {
      private:
        ty v[2];
      public:
        operator vec3i<ty>() const
        {
          return vec3i<ty>( v[a], v[b], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle < a, a, b, -3 >
    {
      private:
        ty v[2];
      public:
        operator vec3i<ty>() const
        {
          return vec3i<ty>( v[a], v[a], v[b] );
        }
    };

    //vec4 swizzlers for vec2
    template<typename ty>
    template<int a>
    class vec2i<ty>::swizzle<a, a, a, a>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle<b, a, a, a>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[b], v[a], v[a], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle<a, b, a, a>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[b], v[a], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle<a, a, b, a>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[a], v[b], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle<a, a, a, b>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[a], v[a], v[b] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle<a, b, a, b>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[b], v[a], v[b] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec2i<ty>::swizzle<a, a, b, b>
    {
      private:
        ty v[2];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[a], v[b], v[b] );
        }
    };

    //vec4 swizzlers for vec3
    template<typename ty>
    template<int a>
    class vec3i<ty>::swizzle<a, a, a, a>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec3i<ty>::swizzle<b, a, a, a>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[b], v[a], v[a], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec3i<ty>::swizzle<a, b, a, a>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[b], v[a], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b>
    class vec3i<ty>::swizzle<a, a, b, a>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[a], v[b], v[a] );
        }
    };

    template<typename ty>
    template<int a, int tb>
    class vec3i<ty>::swizzle<a, a, a, tb>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[a], v[a], v[a], v[tb] );
        }
    };

    template<typename ty>
    template<int a, int tb, int c>
    class vec3i<ty>::swizzle<tb, c, a, a>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[tb], v[c], v[a], v[a] );
        }
    };

    template<typename ty>
    template<int a, int b, int c>
    class vec3i<ty>::swizzle<b, a, c, a>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[b], v[a], v[c], v[a] );
        }
    };

    template<typename ty>
    template<int a, int tb, int c>
    class vec3i<ty>::swizzle<tb, a, a, c>
    {
      private:
        ty v[3];
      public:
        operator vec4i<ty>() const
        {
          return vec4i<ty>( v[tb], v[a], v[a], v[c] );
        }
    };
  }
}

#endif
