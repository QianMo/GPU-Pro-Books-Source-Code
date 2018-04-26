
#ifndef RGBCOLOR_H
#define RGBCOLOR_H

#include <stdio.h>
#include "Utils/TextParsing.h"

/* A Color is a RGB float. 
**
** R, G, and B are all in the range [0..1]
**
** This class allows you to add, subtract, and multiply colors, 
** It also allows you to get the separate components (e.g., myColor.Red() ),
**    use constructions like "myColor += yourColor;"
*/
class RGBAColor;

class RGBColor 
{
    float d[3];
public:
	// Constructors & destructors
	inline RGBColor();                          // Default constructor
	inline RGBColor( char *buffer );            // reads in 3 floats from the white-space delimted string
    inline RGBColor(float r, float g, float b); // color from three floats
	inline RGBColor( float *data );				// color from an array of 3 floats
	inline RGBColor(const RGBColor& copy);      // Copy constructor
	RGBColor( const RGBAColor& copy);
	inline RGBColor( FILE * ) { printf("RGBColor() error!\n"); } // a file constructor.  currently bogus.      

	// Mathematical operations
    inline RGBColor operator*(const RGBColor& c) const  { return RGBColor(d[0]*c.d[0], d[1]*c.d[1], d[2]*c.d[2]); }
	inline RGBColor operator*(float s) const            { return RGBColor(d[0]*s, d[1]*s, d[2]*s); }
	inline RGBColor& operator*=(float s) 	            { d[0]*=s; d[1]*=s; d[2]*=s; return *this; }
    inline RGBColor operator+(const RGBColor& c) const  { return RGBColor(d[0]+c.d[0], d[1]+c.d[1], d[2]+c.d[2]); }
	inline RGBColor& operator+=(const RGBColor& c) 	    { d[0]+=c.d[0]; d[1]+=c.d[1]; d[2]+=c.d[2]; return *this; }
    inline RGBColor operator-(const RGBColor& c) const 	{ return RGBColor(d[0]-c.d[0], d[1]-c.d[1], d[2]-c.d[2]); }
	inline RGBColor& operator-=(const RGBColor& c) 	    { d[0]-=c.d[0]; d[1]-=c.d[1]; d[2]-=c.d[2]; return *this; }

	// Accessor functions
	inline float Red() const                            { return d[0]; }
    inline float Green() const                          { return d[1]; }
    inline float Blue() const                           { return d[2]; }
	inline float Alpha() const                          { return 1; }

	inline void Clamp( float minVal=0, float maxVal=1 );

	// Returns a grayscale ('luminance') value roughly corresponding to the color
    inline float Luminance() const { return (float)(0.3*d[0] + 0.6*d[1] + 0.1*d[2]); }

	// Returns the maximum (or minimum component)
    inline float MaxComponent() const { float temp = (d[1] > d[0]? d[1] : d[0]); return (d[2] > temp? d[2] : temp); }
	inline float MinComponent() const { float temp = (d[1] > d[0]? d[0] : d[1]); return (d[2] > temp? temp : d[2]); }

	// Static methods.  Useful for defining commonly used colors
	static RGBColor Black( void )  { return RGBColor(0,0,0); }
	static RGBColor White( void )  { return RGBColor(1,1,1); }

	static bool IsSpectralColor( void ) { return false; }
	static bool IsRGBColor( void )      { return true; }
};




inline RGBColor::RGBColor() 
{ 
	d[0] = d[1] = d[2] = 0; 
}

inline RGBColor::RGBColor(float r, float g, float b) 
{ 
	d[0] = r; 
	d[1] = g; 
	d[2] = b; 
}
	
inline RGBColor::RGBColor( float *data )
{
	d[0] = data[0];
	d[1] = data[1];
	d[2] = data[2];
}

inline RGBColor::RGBColor(const RGBColor& copy) 
{ 
	d[0] = copy.d[0]; 
	d[1] = copy.d[1]; 
	d[2] = copy.d[2]; 
}
    
// Read a color from a string
inline RGBColor::RGBColor( char *buffer )
{
	// Default values, in case something goes wrong.
	d[0] = d[1] = d[2] = 0;

	char *ptr;
	ptr = StripLeadingNumber( buffer, &d[0] );
	ptr = StripLeadingNumber( ptr, &d[1] );
	ptr = StripLeadingNumber( ptr, &d[2] );
}

inline void RGBColor::Clamp( float minVal, float maxVal ) 
{
	d[0] = d[0] < maxVal ? d[0] : maxVal;
	d[0] = d[0] > minVal ? d[0] : minVal;
	d[1] = d[1] < maxVal ? d[1] : maxVal;
	d[1] = d[1] > minVal ? d[1] : minVal;
	d[2] = d[2] < maxVal ? d[2] : maxVal;
	d[2] = d[2] > minVal ? d[2] : minVal;
}


#endif

