#ifndef VECTOR3_H
#define VECTOR3_H

// Vector3
//   
// 3D vector.
class Vector3
{
public:
  Vector3()
  {
    x = y = z = 0.0f;
  }

  Vector3(float x, float y, float z)
  {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  bool operator== (const Vector3 &vec) const
  {
    if(!IS_EQUAL(x, vec.x))
      return false;
    if(!IS_EQUAL(y, vec.y))
      return false;
    if(!IS_EQUAL(z, vec.z))
      return false;
    return true;
  }

  bool operator!= (const Vector3 &vec) const
  {
    return !((*this) == vec);
  }

  Vector3 operator+ (const Vector3 &vec) const
  {
    return Vector3(x + vec.x, y + vec.y, z + vec.z);
  }

  Vector3 operator- (const Vector3 &vec) const
  {
    return Vector3(x - vec.x, y - vec.y, z - vec.z);
  }

  Vector3 operator- () const
  {
    return Vector3(-x, -y, -z);
  }

  Vector3 operator* (float scalar) const
  {
    return Vector3(x * scalar, y * scalar, z * scalar);
  }

  Vector3 operator/ (float scalar) const
  {
    const float inv = 1.0f / scalar;
    return Vector3(x * inv, y * inv, z * inv);
  }

  void operator+= (const Vector3 &rhs)
  {
    x += rhs.x;
    y += rhs.y;
    z += rhs.z;
  }

  void operator-= (const Vector3 &rhs)
  {
    x -= rhs.x;
    y -= rhs.y;
    z -= rhs.z;
  }

  void operator*= (const float rhs)
  {
    x *= rhs;
    y *= rhs;
    z *= rhs;
  }

  void operator/= (const float rhs)
  {
    const float inv = 1.0f / rhs;
    x *= inv;
    y *= inv;
    z *= inv;
  }

  float operator[](int index) const
  {
    assert((index >= 0) && (index <= 2));
    return (&x)[index];
  }

  float& operator[](int index)
  {
    assert((index >= 0) && (index <= 2));
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

  void Set(float x, float y, float z)
  {
    this->x = x;
    this->y = y;
    this->z = z;
  }

  void SetZero()
  {
    x = y = z = 0.0f;
  }

  void SetMin()
  {
    x = y = z = -FLT_MAX;
  }

  void SetMax()
  {
    x = y = z = FLT_MAX;
  }

  float DotProduct(const Vector3 &vec) const
  {
    return ((vec.x * x) + (vec.y * y) + (vec.z * z));
  }

  Vector3 CrossProduct(const Vector3 &vec) const
  {
    return Vector3((y * vec.z) - (z * vec.y),
                   (z * vec.x) - (x * vec.z),
                   (x * vec.y) - (y * vec.x));
  }

  float GetLength() const
  {
    return sqrt((x * x) + (y * y) + (z * z));
  }

  float GetSquaredLength() const
  {
    return ((x * x) + (y * y) + (z * z));
  }

  float Distance(const Vector3 &vec) const
  {
    return (*this - vec).GetLength();
  }

  void Normalize()
  {
    float length = sqrt((x * x) + (y * y) + (z * z));
    if(length == 0.0f)
      return;
    const float inv = 1.0f / length;
    x *= inv;
    y *= inv;
    z *= inv;
  }

  Vector3 GetNormalized() const
  {
    Vector3 result(*this);
    result.Normalize();
    return result;
  }

  void Floor()
  {
    x = floor(x);
    y = floor(y);
    z = floor(z);
  }

  Vector3 GetFloored()
  {
    Vector3 result(*this);
    result.Floor();
    return result;
  }

  void Clamp(const Vector3 &mins, const Vector3 &maxes)
  {
    CLAMP(x, mins.x, maxes.x);
    CLAMP(y, mins.y, maxes.y);
    CLAMP(z, mins.z, maxes.z);
  }

  Vector3 GetClamped(const Vector3 &mins, const Vector3 &maxes)
  {
    Vector3 result(*this);
    result.Clamp(mins, maxes);
    return result;
  }

  Vector3 Lerp(const Vector3 &vector, float factor) const
  {
    return (*this) * (1.0f - factor) + (vector * factor);
  }

  float x, y, z;

};

#endif