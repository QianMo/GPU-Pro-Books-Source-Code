#ifndef VECTOR4_H
#define VECTOR4_H

// Vector4
//   
// 4D vector.
class Vector4
{
public:
  Vector4()
  {
    x = y = z = w = 0.0f;
  }

  Vector4(float x, float y, float z, float w)
  {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }

  Vector4(const Vector3 &rhs)
  {
    x = rhs.x;
    y = rhs.y;
    z = rhs.z;
    w = 1.0f;
  }

  bool operator== (const Vector4 &vec) const
  {
    if(!IS_EQUAL(x, vec.x))
      return false;
    if(!IS_EQUAL(y, vec.y))
      return false;
    if(!IS_EQUAL(z, vec.z))
      return false;
    if(!IS_EQUAL(w, vec.w))
      return false;
    return true;
  }

  bool operator!= (const Vector4 &vec) const
  {
    return !((*this) == vec);
  }

  Vector4 operator+ (const Vector4 &vec) const
  {
    return Vector4(x + vec.x, y + vec.y, z + vec.z, w + vec.w);
  }

  Vector4 operator- (const Vector4 &vec) const
  {
    return Vector4(x - vec.x, y - vec.y, z - vec.z, w - vec.w);
  }

  Vector4 operator- () const
  {
    return Vector4(-x, -y, -z, -w);
  }

  Vector4 operator* (const float scalar) const
  {
    return Vector4(x * scalar, y * scalar, z * scalar, w * scalar);
  }

  Vector4 operator/ (const float scalar) const
  {
    const float inv = 1.0f / scalar;
    return Vector4(x * inv, y * inv, z * inv, w * inv);
  }

  void operator+= (const Vector4 &rhs)
  {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
    w += rhs.w;
  }

  void operator-= (const Vector4 &rhs)
  {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
    w -= rhs.w;
  }

  void operator*= (float rhs)
  {
    x *= rhs;
    y *= rhs;
    z *= rhs;
    w *= rhs;
  }

  void operator/= (float rhs)
  {
    const float inv = 1.0f / rhs;
    x *= inv;
    y *= inv;
    z *= inv;
    w *= inv;
  }

  float operator[](int index) const
  {
    assert((index >= 0) && (index <= 3));
    return (&x)[index];
  }

  float& operator[](int index)
  {
    assert((index >= 0) && (index <= 3));
    return (&x)[index];
  }

  operator float* () const
  {
    return (float*) this;
  }

  operator const float* () const
  {
    return (const float*) this;
  }

  void Set(float x, float y, float z, float w)
  {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }

  void Set(const Vector3 &vec)
  {
    x = vec.x;
    y = vec.y;
    z = vec.z;
    w = 1.0f;
  }

  void SetZero()
  {
    x = y = z = w = 0.0f;
  }

  float x, y, z, w;

};

#endif