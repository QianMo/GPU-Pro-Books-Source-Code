#include "Core.h"

CoreColor::CoreColor(float r, float g, float b)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = 1.0f;
}

CoreColor::CoreColor(float r, float g, float b, float a)
{
	this->r = r;
	this->g = g;
	this->b = b;
	this->a = a;
}

CoreColor::CoreColor(DWORD Color)
{
	this->r = COLOR_CONVERT * (float)(BYTE)(Color >> 16);
	this->g = COLOR_CONVERT * (float)(BYTE)(Color >> 8);
	this->b = COLOR_CONVERT * (float)(BYTE)(Color);
	this->a = COLOR_CONVERT * (float)(BYTE)(Color >> 24);
}

CoreColor::CoreColor(BYTE r, BYTE g, BYTE b)
{
	this->r = COLOR_CONVERT * (float)r;
	this->g = COLOR_CONVERT * (float)g;
	this->b = COLOR_CONVERT * (float)b;
	this->a = 1.0f;
}

CoreColor::CoreColor(BYTE r, BYTE g, BYTE b, BYTE a)
{
	this->r = COLOR_CONVERT * (float)r;
	this->g = COLOR_CONVERT * (float)g;
	this->b = COLOR_CONVERT * (float)b;
	this->a = COLOR_CONVERT * (float)a;
}

// Inverts the whole color
CoreColor CoreColorInvert(CoreColor &pColor)
{
	return CoreColor(1.0f - pColor.r, 1.0f - pColor.g, 1.0f - pColor.b, 1.0f - pColor.a); 	
}

// Inverts only r, g, b
CoreColor CoreColorInvertExceptAlpha(CoreColor &pColor)
{
	return CoreColor(1.0f - pColor.r, 1.0f - pColor.g, 1.0f - pColor.b, pColor.a); 	
}

// Adds 2 Colors
CoreColor CoreColorAdd(CoreColor &c1, CoreColor &c2)
{
	return CoreColor(c1.r + c2.r,
					c1.g + c2.g,
					c1.b + c2.b,
					c1.a + c2.a);
}

// Subs 2 Colors
CoreColor CoreColorSub(CoreColor &c1, CoreColor &c2)
{
	return CoreColor(c1.r - c2.r,
					c1.g - c2.g,
					c1.b - c2.b,
					c1.a - c2.a);
}

// Muls 2 Colors
CoreColor CoreColorMul(CoreColor &c1, CoreColor &c2)
{
	return CoreColor(c1.r * c2.r,
					c1.g * c2.g,
					c1.b * c2.b,
					c1.a * c2.a);
}

// Divs 2 Colors
CoreColor CoreColorDiv(CoreColor &c1, CoreColor &c2)
{
	return CoreColor(c1.r / c2.r,
					c1.g / c2.g,
					c1.b / c2.b,
					c1.a / c2.a);
}