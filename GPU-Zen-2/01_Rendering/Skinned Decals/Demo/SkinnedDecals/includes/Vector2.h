#ifndef VECTOR2_H
#define VECTOR2_H

// Vector2
//
// 2D vector.
class Vector2
{
public:
  Vector2()
  {
    x = y = 0.0f;
  }

  Vector2(float x, float y)
  {
    this->x = x;
    this->y = y;
  }

  bool operator== (const Vector2 &vec) const
  {
    if(!IS_EQUAL(x, vec.x))
      return false;
    if(!IS_EQUAL(y, vec.y))
      return false;
    return true;
  }

  bool operator!= (const Vector2 &vec) const
  {	
    return !((*this) == vec);
  }

  Vector2 operator+ (const Vector2 &vec) const
  {
    return Vector2(x + vec.x, y + vec.y);
  }

  Vector2 operator- (const Vector2 &vec) const
  {
    return Vector2(x - vec.x, y - vec.y);
  }

  Vector2 operator- () const
  {
    return Vector2(-x, -y);
  }

  Vector2 operator* (float scalar) const
  {
    return Vector2(x * scalar, y * scalar);
  }

  Vector2 operator/ (float scalar) const
  {	
    const float inv = 1.0f / scalar;
    return Vector2(x * inv, y * inv);	
  }

  void operator+= (const Vector2 &rhs)
  {	
    x += rhs.x;	
    y += rhs.y;
  }

  void operator-= (const Vector2 &rhs)
  {	
    x -= rhs.x;	
    y -= rhs.y;
  }

  void operator*= (float rhs)
  {	
    x *= rhs;	
    y *= rhs;	
  }

  void operator/= (float rhs)
  {	
    const float inv = 1.0f / rhs;
    x *= inv; 
    y *= inv;	
  }

  float operator[](int index) const
  {
    assert((index >= 0) && (index <= 1));
    return (&x)[index];
  }

  float& operator[](int index) 
  {
    assert((index >= 0) && (index <= 1));
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

  void Set(float x, float y)
  {
    this->x = x;
    this->y = y; 
  }

  void SetZero()
  {  
    x = y = 0.0f; 
  }

  void SetMin()
  {
    x = y = -FLT_MAX;
  }

  void SetMax()
  {
    x = y = FLT_MAX;
  }

  Vector2 Lerp(const Vector2 &vec, float factor) const
  {
    return ((*this) + (vec - (*this)) * factor);
  }

  float x, y;
  
};

#endif
