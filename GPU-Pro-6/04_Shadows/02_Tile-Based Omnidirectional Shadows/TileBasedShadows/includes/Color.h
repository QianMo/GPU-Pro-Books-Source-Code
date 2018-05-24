#ifndef COLOR_H
#define COLOR_H

// Color
//
// RGBA-color.
class Color
{
public:
  Color()
  {
    r = g = b = a = 0.0f;
  }

  Color(float r, float g, float b, float a=1.0f)
  {
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
  }
  
  bool operator== (const Color &color) const
  {
    if(!IS_EQUAL(r, color.r))
      return false;
    if(!IS_EQUAL(g, color.g))
      return false;
    if(!IS_EQUAL(b, color.b))
      return false;
    if(!IS_EQUAL(a, color.a))
      return false;
    return true;
  }

  bool operator!= (const Color &color) const
  {	
    return !((*this) == color);
  } 

  Color operator+ (const Color &color) const
  {	
    return Color(r+color.r, g+color.g, b+color.b, a+color.a);	
  }

  Color operator* (float scalar) const
  {	
    return Color(r*scalar, g*scalar, b*scalar, a*scalar); 	
  }
  
  operator float* () const 
  {
    return (float*)this;
  }
  operator const float* () const 
  {
    return (const float*)this;
  }

  void Set(float r, float g, float b, float a=1.0f)
  {
    this->r = r;
    this->g = g;
    this->b = b;
    this->a = a;
  }

  float r, g, b, a;
  
};

#endif