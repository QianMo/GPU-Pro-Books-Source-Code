#ifndef COLOR_H
#define COLOR_H

// Color
//
// RGBA32 color.
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
    return Color(r + color.r, g + color.g, b + color.b, a + color.a);	
  }

  Color operator* (float scalar) const
  {	
    return Color(r * scalar, g * scalar, b * scalar, a * scalar); 	
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

  Color Lerp(const Color &color, float factor)
  {
    return (*this) * (1.0f - factor) + (color * factor);
  }

  void ConvertSrgbToLinear()
  {
    r = (r <= 0.04045f) ? (r / 12.92f) : powf((r + 0.055f) / 1.055f, 2.4f);
    g = (g <= 0.04045f) ? (g / 12.92f) : powf((g + 0.055f) / 1.055f, 2.4f);
    b = (b <= 0.04045f) ? (b / 12.92f) : powf((b + 0.055f) / 1.055f, 2.4f);
  }

  void ConvertLinearToSrgb()
  { 
    r = (r <= 0.0031308f) ? (r * 12.92f) : ((powf(fabs(r), 1.0f / 2.4f) * 1.055f) - 0.055f);
    g = (g <= 0.0031308f) ? (g * 12.92f) : ((powf(fabs(g), 1.0f / 2.4f) * 1.055f) - 0.055f);
    b = (b <= 0.0031308f) ? (b * 12.92f) : ((powf(fabs(b), 1.0f / 2.4f) * 1.055f) - 0.055f);
  }

  UINT GetAsRgba8() const
  {
    Color clampedColor = *this;
    CLAMP(clampedColor.r, 0.0f, 1.0f);
    CLAMP(clampedColor.g, 0.0f, 1.0f);
    CLAMP(clampedColor.b, 0.0f, 1.0f);
    CLAMP(clampedColor.a, 0.0f, 1.0f);
    return (UINT(clampedColor.a * 255.0f) << 24 | UINT(clampedColor.b * 255.0f) << 16 | UINT(clampedColor.g * 255.0f) << 8 | UINT(clampedColor.r * 255.0f));
  }

  float r, g, b, a;
  
};

#endif