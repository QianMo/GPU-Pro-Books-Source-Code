#ifndef __COLOR__H__
#define __COLOR__H__

#define COLOR_CONV (0.003921568627450980392156862745098f) // = 1/255

class Color
{
public:
	union
	{
		struct
		{
			float r;
			float g;
			float b;
			float a;
		};

		float c[4];
	};

	inline Color()																																															{}
	inline Color(const float f) : r(f), g(f), b(f), a(1.0f)																																					{}
	inline Color(const float _r, const float _g, const float _b) : r(_r), g(_g), b(_b), a(1.0f)																												{}
	inline Color(const float _r, const float _g, const float _b, const float _a) : r(_r), g(_g), b(_b), a(_a)																								{}
	inline Color(const float* pfComponent) : r(pfComponent[0]), g(pfComponent[1]), b(pfComponent[2]), a(pfComponent[3])																						{}

	inline operator float* ()			{return (float*)(c);}

	inline Color operator + (const Color& c) const					{return Color(r + c.r, g + c.g, b + c.b, a + c.a);}
	inline Color operator - (const Color& c) const					{return Color(r - c.r, g - c.g, b - c.b, a - c.a);}
	inline Color operator - () const									{return Color(-r, -g, -b, -a);}
	inline Color operator * (const Color& c) const					{return Color(r * c.r, g * c.g, b * c.b, a * c.a);}
	inline Color operator * (const float f) const						{return Color(r * f, g * f, b * f, a * f);}
	inline Color operator / (const Color &c) const					{return Color(r / c.r, g / c.g, b / c.b, a / c.a);}
	inline Color operator / (const float f) const						{return Color(r / f, g / f, b / f, a / f);}
	inline friend Color operator * (const float f, const Color& c)	{return Color(c.r * f, c.g * f, c.b * f, c.a * f);}

	inline Color operator = (const Color& c)	{r = c.r; g = c.g; b = c.b; a = c.a; return *this;}
	inline Color operator += (const Color& c)	{r += c.r; g += c.g; b += c.b; a += c.a; return *this;}
	inline Color operator -= (const Color& c)	{r -= c.r; g -= c.g; b -= c.b; a -= c.a; return *this;}
	inline Color operator *= (const Color& c)	{r *= c.r; g *= c.g; b *= c.b; a *= c.a; return *this;}
	inline Color operator *= (const float f)		{r *= f; g *= f; b *= f; a *= f; return *this;}
	inline Color operator /= (const Color& c)	{r /= c.r; g /= c.g; b /= c.b; a /= c.a; return *this;}
	inline Color operator /= (const float f)		{r /= f; g /= f; b /= f; a /= f; return *this;}

	inline bool operator == (const Color& c) const {if(r != c.r) return false; if(g != c.g) return false; if(b != c.b) return false; return a == c.a;}
	inline bool operator != (const Color& c) const {if(r != c.r) return true; if(g != c.g) return true; if(b != c.b) return true; return a != c.a;}
};

inline Color	ColorNegate(const Color& c)											{return Color(1.0f - c.r, 1.0f - c.g, 1.0f - c.b, 1.0f - c.a);}
inline float	ColorBrightness(const Color& c)										{return c.r * 0.3f + c.g * 0.6f + c.b * 0.1f;}
inline Color	ColorInterpolate(const Color& c1, const Color& c2, const float p)	{return c1 + p * (c2 - c1);}

#endif