#ifndef COLOR_H
#define COLOR_H
/*
Szymon Rusinkiewicz
Princeton University

Color.h
Random class for encapsulating colors...
*/

#include "Vec.h"
#include <cmath>
#include <algorithm>
using std::fmod;
using std::floor;
using std::min;
using std::max;
#ifndef M_PI
# define M_PI 3.14159265358979323846
#endif


class Color : public Vec<3,float> {
public:
	Color()
		{}
	Color(const Vec<3,float> &v_) : Vec<3,float>(v_)
		{}
	Color(const Vec<3,double> &v_) : Vec<3,float>((float)v_[0], (float)v_[1], (float)v_[2])
		{}
	Color(float r, float g, float b) : Vec<3,float>(r,g,b)
		{}
	Color(double r, double g, double b) : Vec<3,float>((float)r, (float)g, (float)b)
		{}
	explicit Color(const float *rgb) : Vec<3,float>(rgb[0], rgb[1], rgb[2])
		{}
	explicit Color(const double *rgb) : Vec<3,float>((float)rgb[0], (float)rgb[1], (float)rgb[2])
		{}

	// Implicit conversion from float would be bad, so we have an
	// explicit constructor and an assignment statement.
	explicit Color(float c) : Vec<3,float>(c,c,c)
		{}
	explicit Color(double c) : Vec<3,float>((float)c, (float)c, (float)c)
		{}
	Color &operator = (float c)
		{ return *this = Color(c); }
	Color &operator = (double c)
		{ return *this = Color(c); }

	// Assigning from ints divides by 255
	Color(int r, int g, int b)
	{
		const float mult = 1.0f / 255.0f;
		*this = Color(mult*r, mult*g, mult*b);
	}
	explicit Color(const int *rgb)
		{ *this = Color(rgb[0], rgb[1], rgb[2]); }
	explicit Color(const unsigned char *rgb)
		{ *this = Color(rgb[0], rgb[1], rgb[2]); }
	explicit Color(int c)
		{ *this = Color(c,c,c); }
	Color &operator = (int c)
		{ return *this = Color(c); }

	static Color black()
		{ return Color(0.0f, 0.0f, 0.0f); }
	static Color white()
		{ return Color(1.0f, 1.0f, 1.0f); }
	static Color red()
		{ return Color(1.0f, 0.0f, 0.0f); }
	static Color green()
		{ return Color(0.0f, 1.0f, 0.0f); }
	static Color blue()
		{ return Color(0.0f, 0.0f, 1.0f); }
	static Color yellow()
		{ return Color(1.0f, 1.0f, 0.0f); }
	static Color cyan()
		{ return Color(0.0f, 1.0f, 1.0f); }
	static Color magenta()
		{ return Color(1.0f, 0.0f, 1.0f); }
	static Color hsv(float h, float s, float v)
	{
		// From FvD
		if (s <= 0.0f)
			return Color(v,v,v);
		h = fmod(h, float(2.0f * M_PI));
		if (h < 0.0)
			h += (float)(2.0 * M_PI);
		h /= (float)(M_PI / 3.0);
		int i = int(floor(h));
		float f = h - i;
		float p = v * (1.0f - s);
		float q = v * (1.0f - (s*f));
		float t = v * (1.0f - (s*(1.0f-f)));
		switch(i) {
			case 0: return Color(v, t, p);
			case 1: return Color(q, v, p);
			case 2: return Color(p, v, t);
			case 3: return Color(p, q, v);
			case 4: return Color(t, p, v);
			default: return Color(v, p, q);
		}
	}
};

#endif
