
#ifndef RGBACOLOR_H
#define RGBACOLOR_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Utils/TextParsing.h"
#include "RGBColor.h"

/* A Color is a RGBA float. 
**
** R, G, B, and A are all in the range [0..1]
**
** This class allows you to add, subtract, and multiply colors, 
** It also allows you to get the separate components (e.g., myColor.Red() ),
**    use constructions like "myColor += yourColor;"
*/
class RGBAColor 
{
    float d[4];
public:
	// Constructors & destructors
	inline RGBAColor();                                       // Default constructor
	inline RGBAColor( char *buffer );                         // reads in 4 floats from the white-space delimted string
    inline RGBAColor(float r, float g, float b, float a = 1); // color from three or four floats
	inline RGBAColor( float *data );                          // color from an array of 4 floats
	inline RGBAColor(const RGBAColor& copy);                  // Copy constructor
	inline RGBAColor(const RGBColor& copy);                   // Copy constructor
	inline RGBAColor( FILE * ) { printf("RGBAColor() error!\n"); } // a file constructor.  currently bogus.    

	// Mathematical operations
    inline RGBAColor operator*(const RGBAColor& c) const  { return RGBAColor(d[0]*c.d[0], d[1]*c.d[1], d[2]*c.d[2], d[3]*c.d[3]); }
	inline RGBAColor operator*(float s) const             { return RGBAColor(d[0]*s, d[1]*s, d[2]*s, d[3]*s); }
	inline RGBAColor& operator*=(float s) 	              { d[0]*=s; d[1]*=s; d[2]*=s; d[3]*=s; return *this; }
    inline RGBAColor operator+(const RGBAColor& c) const  { return RGBAColor(d[0]+c.d[0], d[1]+c.d[1], d[2]+c.d[2], d[3]+c.d[3]); }
	inline RGBAColor& operator+=(const RGBAColor& c) 	    { d[0]+=c.d[0]; d[1]+=c.d[1]; d[2]+=c.d[2]; d[3]+=c.d[3]; return *this; }
    inline RGBAColor operator-(const RGBAColor& c) const 	{ return RGBAColor(d[0]-c.d[0], d[1]-c.d[1], d[2]-c.d[2], d[3]-c.d[3]); }
	inline RGBAColor& operator-=(const RGBAColor& c) 	    { d[0]-=c.d[0]; d[1]-=c.d[1]; d[2]-=c.d[2]; d[3]-=c.d[3]; return *this; }

	// Accessor functions
	inline float Red() const                            { return d[0]; }
    inline float Green() const                          { return d[1]; }
    inline float Blue() const                           { return d[2]; }
	inline float Alpha() const                          { return d[3]; }
	inline float *GetDataPtr()							{ return d; }

	inline void Clamp( float minVal=0, float maxVal=1 );

	// Returns a grayscale ('luminance') value roughly corresponding to the color
    inline float Luminance() const { return (float)(0.3*d[0] + 0.6*d[1] + 0.1*d[2]); }

	// Returns the maximum (or minimum component)
    inline float MaxComponent() const { float temp = (d[1] > d[0]? d[1] : d[0]); return (d[2] > temp? d[2] : temp); }
	inline float MinComponent() const { float temp = (d[1] > d[0]? d[0] : d[1]); return (d[2] > temp? temp : d[2]); }

	// Static methods.  Useful for defining commonly used colors
	static RGBAColor Black( void )  { return RGBAColor(0,0,0,1); }
	static RGBAColor White( void )  { return RGBAColor(1,1,1,1); }

	static bool IsSpectralColor( void ) { return false; }
	static bool IsRGBColor( void )      { return true; }
};




inline RGBAColor::RGBAColor() 
{ 
	d[0] = d[1] = d[2] = 0; d[3] = 1;
}

inline RGBAColor::RGBAColor(float r, float g, float b, float a) 
{ 
	d[0] = r; 
	d[1] = g; 
	d[2] = b; 
	d[3] = a;
}

inline RGBAColor::RGBAColor( float *data )
{
	d[0] = data[0];
	d[1] = data[1];
	d[2] = data[2];
	d[3] = data[3];
}
	
inline RGBAColor::RGBAColor(const RGBAColor& copy) 
{ 
	d[0] = copy.d[0]; 
	d[1] = copy.d[1]; 
	d[2] = copy.d[2]; 
	d[3] = copy.d[3];
}

inline RGBAColor::RGBAColor(const RGBColor& copy) 
{ 
	d[0] = copy.Red(); 
	d[1] = copy.Green(); 
	d[2] = copy.Blue(); 
	d[3] = 1;
}
    
// Read a color from a string
inline RGBAColor::RGBAColor( char *buffer )
{
	char buf[256];
	// Default values, in case something goes wrong.
	d[0] = d[1] = d[2] = 0; d[3] = 1;

	char *ptr;
	ptr = StripLeadingNumber( buffer, &d[0] );
	ptr = StripLeadingNumber( ptr, &d[1] );
	ptr = StripLeadingNumber( ptr, &d[2] );
	ptr = StripLeadingTokenToBuffer( ptr, buf );
	d[3] = strlen(buf)>0 ? (float)atof( buf ) : 1.0f;
}

inline void RGBAColor::Clamp( float minVal, float maxVal ) 
{
	d[0] = d[0] < maxVal ? d[0] : maxVal;
	d[0] = d[0] > minVal ? d[0] : minVal;
	d[1] = d[1] < maxVal ? d[1] : maxVal;
	d[1] = d[1] > minVal ? d[1] : minVal;
	d[2] = d[2] < maxVal ? d[2] : maxVal;
	d[2] = d[2] > minVal ? d[2] : minVal;
	d[3] = d[3] < maxVal ? d[3] : maxVal;
	d[3] = d[3] > minVal ? d[3] : minVal;
}


#endif

