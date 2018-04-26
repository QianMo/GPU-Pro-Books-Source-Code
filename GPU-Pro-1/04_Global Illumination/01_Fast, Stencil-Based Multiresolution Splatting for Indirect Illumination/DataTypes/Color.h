/******************************************************************/
/* Color.h                                                        */
/* -------------                                                  */
/*                                                                */
/* The file defines the color class used by the ray tracer.  In   */
/*    order to allow maximum flexibility, the actual color class  */
/*    currently used (RGBColor) is defined in a different file,   */
/*    and "Color" is typedef'd to be RGBColor, to allow easily    */
/*    changing to another color space (e.g., spectral colors).    */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/


#ifndef COLOR_H
#define COLOR_H

// Here's the file that actually defines the RGB color class
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "RGBAColor.h"
#include "SpectralColor.h"

// For a ray tracer, one might want to switch to a Spectral Color
//    at some point, but in OpenGL, RGBA is the only way to go!
typedef RGBAColor Color;




// A utility class to load colors from a file.  To use this, simply call:
//                       Color fromFile = LoadColor::Load( token, filePtr, strPtr );
// Where "token" is a null-terminated string (char *) to the command in the file
//                 that specified a color was to be loaded (e.g., "color" or "rgb" or "spectral")
// And   "filePtr" is a FILE * pointer to the currently open file.
// And   "strPtr"  is a null-terminated string (char *) to the rest of the 
//                 current line (this string may contain the data for RGB colors)
class LoadColor
{
public:
	LoadColor() {}
	~LoadColor() {}

	// Return the X, Y, or Z stimuli for the specified wavelength.
	static Color Load( char *token, FILE *filePtr, char *strPtr )
	{
		if (!strcmp(token,"rgb"))
		{
			// This reads in an RGB color to a standard 3-component color using the rest of
			//     the current line of the file for data.  If we're using Color=RGBColor, then
			//     the Color() refers to a copy constructor.  A bit inefficient, but oh well.
			//     If Color!=RGBColor, it should define a constructor that takes an RGBColor 
			//     and converts it to an appropriate internal format.
			return Color( RGBColor( strPtr ) ); 
		}
		else if (!strcmp(token,"rgba"))
		{
			// This reads in an RGBA color to a standard 4-component color using the rest of
			//     the current line of the file for data.  If we're using Color=RGBAColor, then
			//     the Color() refers to a copy constructor.  A bit inefficient, but oh well.
			//     If Color!=RGBAColor, it should define a constructor that takes an RGBColor 
			//     and converts it to an appropriate internal format.
			return Color( RGBAColor( strPtr ) ); 
		}
		else if (!strcmp(token,"spectral"))
		{
			// A call to "return Color( SpectralColor( f ) );" may work here if you define 
			//     a RGBColor( const SpectralColor &specColor ) constructor.  Until then, this
			//     checks if the current type is a spectral color.  If not, it exits with a message
			if (!Color::IsSpectralColor())
			{
				printf("Error: Scene specifies spectral color, but currently using RGB colors!\n");
				exit(1);
			}
			return Color( filePtr );
		}
		else // some other non-specfic color type, e.g., "albedo" or "color" etc.
		{
			// In this case, we don't have enough information to decide which type of constructor
			//     to use (the file uses a generic term), so we'll just have to guess based upon
			//     the current type specified by the above typedef!  This may cause silent and
			//     very wrong errors!
			if (Color::IsSpectralColor())
				return Color( filePtr );
			else
				return Color( strPtr ); 
		}
	}
};


#endif

