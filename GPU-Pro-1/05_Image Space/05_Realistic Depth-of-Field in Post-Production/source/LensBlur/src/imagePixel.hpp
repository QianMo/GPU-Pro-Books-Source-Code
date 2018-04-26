/* ******************************************************************************
* Description: This class represents an image pixel. Contains the pixel color (RBG), z
*              and calculated luminance value, and other attributes (like zDepth or group)
*              which helps in dof calculation.
*
*  Version 1.0.0
*  Date: Nov 22, 2008
*  Author: David Illes, Peter Horvath
*          www.postpipe.hu
*
* GPUPro
***************************************************************************** */

#ifndef _IMAGEPIXEL_
#define _IMAGEPIXEL_

#include <math.h>

//===============================================================
class ImagePixel {
//===============================================================
public:
	float r, g, b;		// color
	float lum;			// luminance

	// to speed up
	int zValue;
	int focusDistance;
	float zDepth;

	int group;

	// to recalculation
	//bool hasRecalcPixel;
	//bool hasBGRecalcPixel;

//-----------------------------------------------------------------
// Summary: Constructs default pixel.
//-----------------------------------------------------------------
ImagePixel() { 
//-----------------------------------------------------------------
	reset(); 
	//hasRecalcPixel = false;
	//hasBGRecalcPixel = false;
}

//-----------------------------------------------------------------
// Summary: Constructs a pixel with color. Does not calcualte luminance.
// Arguments: rr - R color component
//            gg - G color component
//            bb - B color component
//-----------------------------------------------------------------
ImagePixel(float rr, float gg, float bb) {
//-----------------------------------------------------------------
		r = rr; g = gg; b = bb;
		//hasRecalcPixel = false;
		//hasBGRecalcPixel = false;
		//lum = luminance();
}

//-----------------------------------------------------------------
// Summary: Constructs a pixel.
// Arguments: rr - R color component
//            gg - G color component
//            bb - B color component
//            l - luminance
//-----------------------------------------------------------------
ImagePixel(float rr, float gg, float bb, float l) {
//-----------------------------------------------------------------
		r = rr; g = gg; b = bb; 
		lum = l;
		//hasRecalcPixel = false;
		//hasBGRecalcPixel = false;
}

//-----------------------------------------------------------------
// Summary: Resets color and luminance values.
//-----------------------------------------------------------------
inline void reset() {
//-----------------------------------------------------------------
		r = 0; g = 0; b = 0;
		lum = 0;
}

//-----------------------------------------------------------------
// Summary: Sets color and luminance values from an array.
// Arguments: rgblum - 4 dimensional array of R,G,B and luminance
//-----------------------------------------------------------------
inline void set(float* rgblum) {
//-----------------------------------------------------------------
		r = rgblum[0];
		g = rgblum[1];
		b = rgblum[2];
		lum = rgblum[3];
}

//-----------------------------------------------------------------
// Summary: Increase the color and luminance values.
// Arguments: rgblum - 4 dimensional array of R,G,B and luminance
//-----------------------------------------------------------------
inline void append(float* rgblum) {
//-----------------------------------------------------------------
		r += rgblum[0];
		g += rgblum[1];
		b += rgblum[2];
		lum += rgblum[3];
}

//-----------------------------------------------------------------
// Summary: Sets pixel color.
// Arguments: rr - R color component
//            gg - G color component
//            bb - B color component
//-----------------------------------------------------------------
inline void setRGB(float rr, float gg, float bb) {
//-----------------------------------------------------------------
		r = rr; g = gg; b = bb;
}

//-----------------------------------------------------------------
// Summary: Sets pixel luminance value.
// Arguments: l - luminance
//-----------------------------------------------------------------
inline void setLum(float l) {
//-----------------------------------------------------------------
		lum = l;
}

//-----------------------------------------------------------------
// Summary: Checks if the two pixel's color are the same or not.
// Arguments: p - other pixel
// Returns: true - when the two color are the same
//          false - otherwise
//-----------------------------------------------------------------
inline bool operator==(const ImagePixel& p) const {
//-----------------------------------------------------------------
	return(r == p.r && g == p.g && b == p.b);
}

//-----------------------------------------------------------------
// Summary: Increases the color value with an other pixel's color.
// Arguments: p - other pixel
// Returns: pixel instance with the new color value
//-----------------------------------------------------------------
inline ImagePixel operator+(const ImagePixel& p) const {
//-----------------------------------------------------------------
		return ImagePixel(r + p.r, g + p.g, g + p.g);
}

//-----------------------------------------------------------------
// Summary: Descreases the color value with an other pixel's color.
// Arguments: p - other pixel
// Returns: pixel instance with the new color value
//-----------------------------------------------------------------
inline ImagePixel operator-(const ImagePixel& p) const {
//-----------------------------------------------------------------
		return ImagePixel(r - p.r, g - p.g, b - p.b);
}

//-----------------------------------------------------------------
// Summary: Multiplies the color value.
// Arguments: f - multiplicator
// Returns: pixel instance with the new color value
//-----------------------------------------------------------------
inline ImagePixel operator*(float f) const {
//-----------------------------------------------------------------
		return ImagePixel(r * f, g * f, b * f);
}

//-----------------------------------------------------------------
// Summary: Divides the color value.
// Arguments: f - quotient
// Returns: pixel instance with the new color value
//-----------------------------------------------------------------
inline ImagePixel operator/(float f) const {
//-----------------------------------------------------------------
		return ImagePixel(r / f, g / f, b / f);
}

//-----------------------------------------------------------------
// Summary: Increases the color value.
// Arguments: f - additional value
//-----------------------------------------------------------------
inline void operator+=(float f) { 
//-----------------------------------------------------------------
		r += f; g += f; b += f; 
}

//-----------------------------------------------------------------
// Summary: Sets the color value from an other pixel.
// Arguments: p - other pixel
//-----------------------------------------------------------------
inline void operator=(const ImagePixel& p) { 
//-----------------------------------------------------------------
		r = p.r; g = p.g; b = p.b; 
}

//-----------------------------------------------------------------
// Summary: Increases the color value with the color of an other pixel.
// Arguments: p - other pixel
//-----------------------------------------------------------------
inline void operator+=(const ImagePixel& p) { 
//-----------------------------------------------------------------
		r += p.r; g += p.g; b += p.b;
}

//-----------------------------------------------------------------
// Summary: Multiplies the color value.
// Arguments: f - multiplicator
//-----------------------------------------------------------------
inline void operator*=(float f) { 
//-----------------------------------------------------------------
		r *= f; g *= f; b *= f; 
}

//-----------------------------------------------------------------
// Summary: Multiplies the color value with the color of an other pixel.
// Arguments: p - other pixel
// Returns: sum of the multiplied values.
//-----------------------------------------------------------------
inline float operator*(const ImagePixel& p) const {
//-----------------------------------------------------------------
		return r * p.r + g * p.g + b * p.b; 
}

//-----------------------------------------------------------------
// Summary: Inverts color components.
// Returns: pixel instance with the new color value
//-----------------------------------------------------------------
inline ImagePixel operator-(void) const { 
//-----------------------------------------------------------------
		return ImagePixel(-r, -g, -b);
}

//-----------------------------------------------------------------
// Returns: the average of the color components.
//-----------------------------------------------------------------
inline float avg() {
//-----------------------------------------------------------------
	return (r + g + b) / 3.0f;
}

//-----------------------------------------------------------------
// Summary: Calculates luminance value from the r,g,b color components.
// Returns: luminance value.
//-----------------------------------------------------------------
inline float luminance() {
//-----------------------------------------------------------------
	lum = 0.3f*r + 0.59f*g + 0.11f*b;
	return lum;
}

//-----------------------------------------------------------------
// Returns: luminance value.
//-----------------------------------------------------------------
inline float getLumValue() {
//-----------------------------------------------------------------
	return lum;
}

//-----------------------------------------------------------------
// Summary: Multiplies the color with the multiplicator and luminance.
// Arguments: f - multiplicator
// Returns: pixel instance with the new color value
//-----------------------------------------------------------------
inline ImagePixel operator<<(float f) const {
//-----------------------------------------------------------------
	float m = f*lum;
	return ImagePixel(r * m, g * m, b * m, m);
}

//-----------------------------------------------------------------
// Summary: Increases color with the product of kernel value, 
//          luminance and other pixel's color. 
//          Accumulation of the color by a neighboring pixel through
//          the kernel mask.
// Arguments: pixel - neighbor pixel
//            kernelValue - kernel value
//-----------------------------------------------------------------
inline void add(ImagePixel& pixel, float kernelValue) { 
//-----------------------------------------------------------------
	float multiplicator = kernelValue * pixel.getLumValue();
	r += pixel.r * multiplicator;
	g += pixel.g * multiplicator;
	b += pixel.b * multiplicator; 
	lum += multiplicator;
}

};

#endif
