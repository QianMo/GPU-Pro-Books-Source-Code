#pragma once

#include <windows.h>

#define COLOR_CONVERT 0.003921568627450980392156862745098f
class CoreColor;

// Inverts the whole color
CoreColor CoreColorInvert(CoreColor &pColor);
// Inverts only r, g, b
CoreColor CoreColorInvertExceptAlpha(CoreColor &pColor);
// Adds 2 Colors
CoreColor CoreColorAdd(CoreColor &c1, CoreColor &c2);
// Subs 2 Colors
CoreColor CoreColorSub(CoreColor &c1, CoreColor &c2);
// Muls 2 Colors
CoreColor CoreColorMul(CoreColor &c1, CoreColor &c2);
// Divs 2 Colors
CoreColor CoreColorDiv(CoreColor &c1, CoreColor &c2);


class CoreColor
{
public:
	union
	{
		struct
		{
			 float r, 
				   g, 
				   b, 
				   a;
		};
		float arr[4];
	};
	// various Constructors
	inline CoreColor()	{};
	CoreColor(float r, float g, float b);
	CoreColor(float r, float g, float b, float a);
	CoreColor(DWORD Color);
	CoreColor(BYTE r, BYTE g, BYTE b);
	CoreColor(BYTE r, BYTE g, BYTE b, BYTE a);

	// various Operators
	inline operator DWORD () const
	{
		return ((a >= 1.0f ? 255 : a <= 0.0f ? 0 : (DWORD)(a * 255.0f)) << 24) |
			   ((r >= 1.0f ? 255 : r <= 0.0f ? 0 : (DWORD)(r * 255.0f)) << 16) |
			   ((g >= 1.0f ? 255 : g <= 0.0f ? 0 : (DWORD)(g * 255.0f)) << 8)  |
			   (b >= 1.0f ? 255 : b <= 0.0f ? 0 : (DWORD)(b * 255.0f));
	}

	inline DWORD ToBGRA()
	{
		return ((a >= 1.0f ? 255 : a <= 0.0f ? 0 : (DWORD)(a * 255.0f)) << 24) |
			   ((r >= 1.0f ? 255 : r <= 0.0f ? 0 : (DWORD)(r * 255.0f))) |
			   ((g >= 1.0f ? 255 : g <= 0.0f ? 0 : (DWORD)(g * 255.0f)) << 8)  |
			   ((b >= 1.0f ? 255 : b <= 0.0f ? 0 : (DWORD)(b * 255.0f)) << 16);
	}

	// Adds a Color to this one and writes it to cOut
	inline CoreColor operator + (CoreColor &c2)		{ return CoreColorAdd(*this, c2); }
	// Adds a Color to this one
	inline CoreColor operator += (CoreColor &c2)		{ *this = CoreColorAdd(*this, c2); return *this;}
	// Subs a Color from this one and writes it to cOut
	inline CoreColor operator - (CoreColor &c2)		{ return CoreColorSub(*this, c2); }
	// Subs a Color from this one
	inline CoreColor operator -= (CoreColor &c2)		{ *this = CoreColorSub(*this, c2); return *this;}
	// Muls a Color with this one and writes it to cOut
	inline CoreColor operator * (CoreColor &c2)		{ return CoreColorMul(*this, c2); }
	// Muls a Color with this one
	inline CoreColor operator *= (CoreColor &c2)		{ *this = CoreColorMul(*this, c2); return *this;}
	// Divs a Color through this one and writes it to cOut
	inline CoreColor operator / (CoreColor &c2)		{ return CoreColorDiv(*this, c2); }
	// Divs a Color through this one
	inline CoreColor operator /= (CoreColor &c2)		{ *this = CoreColorDiv(*this, c2); return *this;}

	// Inverts the Color
	inline CoreColor Invert()							{  return CoreColorInvert(*this); }
	// Inverts this Color
	inline CoreColor InvertThis()						{  *this = CoreColorInvert(*this); }

	
};