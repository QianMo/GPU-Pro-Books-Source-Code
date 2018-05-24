#ifndef mm_quat_func_h
#define mm_quat_func_h

#include "mm_quat_impl.h"

template<typename ty>
mymath::impl::quati<ty> operator*(const mymath::impl::quati<ty>& p, const mymath::impl::quati<ty>& q)
{
  const mymath::impl::vec3i<ty> pv = p.vector();
  const ty ps = p.scalar();
  const mymath::impl::vec3i<ty> qv = q.vector();
  const ty qs = q.scalar();

  return mymath::impl::quati<ty>(mymath::impl::vec4i<ty>(ps * qv + qs * pv + mymath::cross(pv, qv),
      ps * qs - mymath::dot(pv, qv)));
}

template<typename ty>
mymath::impl::quati<ty> operator*(const mymath::impl::quati<ty>& p, const ty& num)
{
  return mymath::impl::quati<ty>(p.value * num);
}

template<typename ty>
mymath::impl::quati<ty> operator/(const mymath::impl::quati<ty>& q, const ty& num)
{
  return mymath::impl::quati<ty>(q.value / num);
}

template<typename ty>
mymath::impl::quati<ty> operator+(const mymath::impl::quati<ty>& q1, const mymath::impl::quati<ty>& q2)
{
  return mymath::impl::quati<ty>(q1.value + q2.value);
}

namespace mymath
{
  template<typename ty>
  impl::quati<ty> conjugate(const impl::quati<ty>& q)
  {
    return impl::vec4i<ty>(-1 * q.vector(), q.scalar());
  }

  template<typename ty>
  impl::quati<ty> inverse(const impl::quati<ty>& q)
  {
    return impl::quati<ty>(conjugate(q) / dot(q.value, q.value));

    //return conjugate(q);
  }

  template<typename ty>
  impl::quati<ty> cross(const impl::quati<ty>& q1, const impl::quati<ty>& q2)
  {
    return cross(q1.vector(), q2.vector());
  }

  template<typename ty>
  ty dot(const impl::quati<ty>& q1, const impl::quati<ty>& q2)
  {
    return dot(q1.vector(), q2.vector());
  }

  template<typename ty>
  impl::quati<ty> normalize(const impl::quati<ty>& q)
  {
    return normalize(q.value);
  }

  template<typename ty>
  ty length(const impl::quati<ty>& q)
  {
    return length(q.vector());
  }

  template<typename ty>
  ty norm(const impl::quati<ty>& q)
  {
    return length(q.value);
  }

  template<typename ty>
  impl::quati<ty> pow(const impl::quati<ty>& q, const ty& power)
  {
    const impl::vec3i<ty> vec_normal = normalize(q.vector());
    const ty norm_raised_to_pow = std::pow(norm(q), power);
    const ty theta = std::acos(q.scalar()/norm(q));

    return impl::quati<ty>(vec_normal*norm_raised_to_pow*std::sin(power*theta),
        norm_raised_to_pow*std::cos(power*theta));
  }

  template<typename ty>
  impl::mat3i<ty> mat3_cast(const impl::quati<ty>& q)
  {
    impl::mat3i<ty> m;

    const ty& x = q.value.x;
    const ty& y = q.value.y;
    const ty& z = q.value.z;
    const ty& w = q.value.w;

    m[0].x = 1 - 2 * (y * y + z * z);	m[1].x = 2 * (x * y - z * w);		m[2].x = 2 * (x * z + y * w);
    m[0].y = 2 * (x * y + z * w);		m[1].y = 1 - 2 * (x * x + z * z);	m[2].y = 2 * (y * z - x * w);
    m[0].z = 2 * (x * z - y * w);		m[1].z = 2 * (y * z + x * w);		m[2].z = 1 - 2 * (x * x + y * y);

    return m;
  }

  template<typename ty>
  impl::mat4i<ty> mat4_cast(const impl::quati<ty>& q)
  {
    return impl::mat4i<ty>(mat3_cast<ty>(q));
  }

  ///This function does a linear interpolation
  ///
  template<typename ty>
  impl::quati<ty> mix(const impl::quati<ty>& q1, const impl::quati<ty>& q2, const ty& t)
  {
    return impl::quati<ty>(normalize(q1*(1-t)+q2*t));
  }

  template<typename ty>
  impl::quati<ty> slerp(const impl::quati<ty>& q1, const impl::quati<ty>& q2, const ty& t)
  {
    return q1*pow((inverse(q1)*q2), t);
  }

  template<typename ty>
  impl::quati<ty> quat_cast(const impl::mat3i<ty>& m)
  {
    ty trace = m[0][0] + m[1][1] + m[2][2];

    impl::quati<ty> result;

    //If the trace of the matrix is greater than zero, then the result is
    if (trace > 0)
    {
      /* X = ( m21 - m12 ) * S
       * Y = ( m02 - m20 ) * S
       * Z = ( m10 - m01 ) * S
       * our matrices are column-major so the indexing is reversed
       */

      ty s = 0.5 / std::sqrt(trace + 1);
      result.value.w = 0.25 / s;
      result.value.x = (m[1][2] - m[2][1]) * s;
      result.value.y = (m[2][0] - m[0][2]) * s;
      result.value.z = (m[0][1] - m[1][0]) * s;
    }
    else
    {
      if (m[0][0] > m[1][1] && m[0][0] > m[2][2])
      {
        ty s = 2 * std::sqrt(1 + m[0][0] - m[1][1] - m[2][2]);

        result.value.w = (m[1][2] - m[2][1]) / s;
        result.value.x = 0.25 * s;
        result.value.y = (m[1][0] + m[0][1]) / s;
        result.value.z = (m[2][0] + m[0][2]) / s;
      }
      else if (m[1][1] > m[2][2])
      {
        ty s = 2 * std::sqrt(1 + m[1][1] - m[0][0] - m[2][2]);

        result.value.w = (m[2][0] - m[0][2]) / s;
        result.value.x = (m[1][0] + m[0][1]) / s;
        result.value.y = 0.25 * s;
        result.value.z = (m[2][1] + m[1][2]) / s;
      }
      else
      {
        ty s = 2 * std::sqrt(1 + m[2][2] - m[0][0] - m[1][1]);

        result.value.w = (m[0][1] - m[1][0]) / s;
        result.value.x = (m[2][0] + m[0][2]) / s;
        result.value.y = (m[2][1] + m[1][2]) / s;
        result.value.z = 0.25 * s;
      }
    }

    return result;
  }

  template<typename ty>
  impl::quati<ty> quat_cast(const impl::mat4i<ty>& m)
  {
    return quat_cast(impl::mat3i < ty > (m));
  }

  //angle in radians!
  template<typename ty>
  impl::quati<ty> rotate(const impl::quati<ty>& q, const ty& angle, const impl::vec3i<ty>& axis)
  {
    return q * impl::quati<ty>(angle, axis);
  }

  template<typename ty>
  impl::vec3i<ty> rotate_vector(const impl::quati<ty>& q, const impl::vec3i<ty>& v)
  {
    impl::quati<ty> v_quat = impl::quati < ty > (impl::vec4i < ty > (v, 0));

    v_quat = q * v_quat * inverse(q);
    return v_quat.vector();
  }
}

#endif
