#include "RGBColor.h"
#include "RGBAColor.h"

RGBColor::RGBColor( const RGBAColor& copy)
{
	d[0] = copy.Red();
	d[1] = copy.Green();
	d[2] = copy.Blue();
}