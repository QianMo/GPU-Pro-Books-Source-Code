
#ifndef SPECTRALCOLOR_H
#define SPECTRALCOLOR_H

#include "MathDefs.h"
#include "Utils/TextParsing.h"
#include "XYZSpectrum.h"
#include <stdio.h>
#include <string.h>

#define SPECTRALCOLORBINS   20
#define MINWAVELENGTH		380
#define MAXWAVELENGTH		780
#define WAVELENGTH_INCR		((MAXWAVELENGTH-MINWAVELENGTH)/((float)SPECTRALCOLORBINS))


/* A SpectralColor contains a larger number of bins to represent better (than
**    a RGBColor) a whole wavelength of light
**
** SpectralColor bins have no limitation on scale.  They are physically based units!
**
** This class allows you to add, subtract, and multiply colors, 
** It also allows you to get the separate components (e.g., myColor.Red() ),
**    use constructions like "myColor += yourColor;"
*/
class SpectralColor 
{
    float d[SPECTRALCOLORBINS];
public:
	// Constructors & destructors
	inline SpectralColor();                            // Default constructor
	inline SpectralColor( float constValue );          // spreads a constant intensity overall bins
	inline SpectralColor( FILE *f );                   // Reads a spectral color from a file
    inline SpectralColor( float r, float g, float b ); // color from three floats
	inline SpectralColor( const RGBColor &rgbColor );  // color from a 3-component RGBColor
	inline SpectralColor( const float *copyBins );     // copies data from a temporary bin array 
	inline SpectralColor( const SpectralColor& copy ); // Copy constructor   

	// Mathematical operations
    inline SpectralColor operator*(const SpectralColor& c) const;
	inline SpectralColor operator*(float s) const;
	inline SpectralColor& operator*=(float s);
    inline SpectralColor operator+(const SpectralColor& c) const;
	inline SpectralColor& operator+=(const SpectralColor& c);
    inline SpectralColor operator-(const SpectralColor& c) const;
	inline SpectralColor& operator-=(const SpectralColor& c);

	// Accessor functions
	inline float Red( void ) const;                           
    inline float Green( void ) const;                     
    inline float Blue( void ) const;                       
	inline RGBColor ToXYZ() const;
	inline float GetBinValue( int i ) const { return d[i]; }

	inline void Clamp( float minVal=0, float maxVal=1 );

	// Returns a grayscale ('luminance') value roughly corresponding to the color
    inline float Luminance() const;

	// Returns the maximum (or minimum component)
    inline float MaxComponent() const; 
	inline float MinComponent() const;

	// Static methods.  Useful for defining commonly used colors
	static SpectralColor Black( void )  { return SpectralColor(0.0f); }
	static SpectralColor White( void )  { return SpectralColor(1.0f); }

	static bool IsSpectralColor( void ) { return true; }
	static bool IsRGBColor( void )      { return false; }
};

inline RGBColor SpectralColor::ToXYZ( void ) const
{
	RGBColor normalization=RGBColor::Black();
	RGBColor result=RGBColor::Black();
	for (int i=0;i<SPECTRALCOLORBINS;i++)
	{
		float currWavelen = MINWAVELENGTH + i*WAVELENGTH_INCR;
		RGBColor xyzResponse = XYZSpectrum::GetXYZStimulus( currWavelen );
		normalization += xyzResponse;
		result += xyzResponse * d[i];
	}
	return RGBColor( result.Red()/normalization.Red(), 
		result.Green()/normalization.Green(), 
		result.Blue()/normalization.Blue() );
}

inline float SpectralColor::Luminance( void ) const
{
	float normalization=0;
	float result=0;
	for (int i=0;i<SPECTRALCOLORBINS;i++)
	{
		float currWavelen = MINWAVELENGTH + i*WAVELENGTH_INCR;
		RGBColor xyzResponse = XYZSpectrum::GetXYZStimulus( currWavelen );
		normalization += xyzResponse.Green();
		result += xyzResponse.Green() * d[i];
	}
	return result / normalization;
}

inline float SpectralColor::Red( void ) const
{
	return ToXYZ().Red();
}

inline float SpectralColor::Green( void ) const
{
	return ToXYZ().Green();
}

inline float SpectralColor::Blue( void ) const
{
	return ToXYZ().Blue();
}

inline SpectralColor::SpectralColor() 
{ 
	for (int i=0;i<SPECTRALCOLORBINS;i++)
		d[i] = 0;
}

inline SpectralColor::SpectralColor( float constValue ) 
{ 
	for (int i=0;i<SPECTRALCOLORBINS;i++)
		d[i] = constValue;
}

inline SpectralColor::SpectralColor( float r, float g, float b ) 
{ 
	RGBColor tmp( r, g, b ), xyzColor;
	xyzColor = XYZSpectrum::ConvertRGBToXYZ( tmp );
	for (int i=0;i<SPECTRALCOLORBINS;i++)
	{
		float currWavelen = MINWAVELENGTH + i*WAVELENGTH_INCR;
		RGBColor xyzResponse = XYZSpectrum::GetXYZStimulus( currWavelen );
		d[i] = xyzColor.Red()*xyzResponse.Red() + 
			xyzColor.Green()*xyzResponse.Green() +
			xyzColor.Blue()*xyzResponse.Blue();
	}
}

inline SpectralColor::SpectralColor( const RGBColor &rgbColor ) 
{ 
	RGBColor xyzColor;
	xyzColor = XYZSpectrum::ConvertRGBToXYZ( rgbColor );
	for (int i=0;i<SPECTRALCOLORBINS;i++)
	{
		float currWavelen = MINWAVELENGTH + i*WAVELENGTH_INCR;
		RGBColor xyzResponse = XYZSpectrum::GetXYZStimulus( currWavelen );
		d[i] = xyzColor.Red()*xyzResponse.Red() + 
			xyzColor.Green()*xyzResponse.Green() +
			xyzColor.Blue()*xyzResponse.Blue();
	}
}

inline SpectralColor::SpectralColor( const float *copyBins ) 
{ 
	memcpy( d, copyBins, SPECTRALCOLORBINS*sizeof(float) );
	//for (int i=0;i<SPECTRALCOLORBINS;i++)
	//	d[i] = copyBins[i];
}

inline SpectralColor::SpectralColor( const SpectralColor& copy ) 
{ 
	memcpy( d, copy.d, SPECTRALCOLORBINS*sizeof(float) );
	//for (int i=0;i<SPECTRALCOLORBINS;i++)
	//	d[i] = copy.d[i];
}
    
// Read a color from a string
inline SpectralColor::SpectralColor( FILE *f )
{
	// Set some defaults... (zero out color)
	for (int i=0;i<SPECTRALCOLORBINS;i++)
		d[i] = 0;
	
	// Search the scene file.
	char buf[ MAXLINELENGTH ], token[256], *ptr;
	while( fgets(buf, MAXLINELENGTH, f) != NULL )  
	{
		// Is this line a comment?
		ptr = StripLeadingWhiteSpace( buf );
		if (ptr[0] == '#') continue;

		// Nope.  So find out what the command is...
		//   The "command" is either "end" or a wavelength for the bin.
		ptr = StripLeadingTokenToBuffer( ptr, token );
		MakeLower( token );
	
		// Take different measures, depending on the command.
		if (!strcmp(token,"end")) break;

		// Convert to a numeric wavelength
		float binWavelen = (float)atof( token );

		// Check if it's valid (invalid numeric gives binWavelen = 0)
		if (binWavelen < MINWAVELENGTH || binWavelen >= MAXWAVELENGTH)
		{
			printf("Warning: SpectralColor read '%s' as a bin value.\n", token);
			printf("   (This is either non-numeric or out of the valid range [%d..%d)!)\n",
				MINWAVELENGTH, MAXWAVELENGTH);
			continue;	
		}

		// Read the intensity of this wavelength
		float waveValue;
		ptr = StripLeadingNumber( ptr, &waveValue );
		
		// Find which bin it corresponds to
		int binNum = (int)((binWavelen-MINWAVELENGTH)/WAVELENGTH_INCR);
		if (binNum >= SPECTRALCOLORBINS) binNum = SPECTRALCOLORBINS-1;
		if (binNum < 0) binNum = 0;

		// Add the value to that bin
		d[binNum] = waveValue;
	}
}


inline void SpectralColor::Clamp( float minVal, float maxVal )
{
	for (int i=0; i<SPECTRALCOLORBINS;i++)
	{
		d[i] = (d[i] < maxVal ? d[0] : maxVal);
		d[i] = (d[i] > minVal ? d[0] : minVal);
	}
}

inline float SpectralColor::MaxComponent() const 
{ 
	float maxVal = d[0];
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		maxVal = (d[i] > maxVal ? d[i] : maxVal);
	return maxVal;
}


inline float SpectralColor::MinComponent() const 
{ 
	float minVal = d[0];
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		minVal = (d[i] < minVal ? d[i] : minVal);
	return minVal;
}


inline SpectralColor SpectralColor::operator*(const SpectralColor& c) const     
{ 
	float tmpBins[SPECTRALCOLORBINS];
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		tmpBins[i] = d[i]*c.d[i];
	return SpectralColor( tmpBins );
}
inline SpectralColor SpectralColor::operator*(float s) const
{ 
	float tmpBins[SPECTRALCOLORBINS];
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		tmpBins[i] = d[i]*s;
	return SpectralColor( tmpBins );
}
inline SpectralColor& SpectralColor::operator*=(float s) 	                     
{ 
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		d[i] *= s;
	return *this; 
}
inline SpectralColor SpectralColor::operator+(const SpectralColor& c) const     
{ 
	float tmpBins[SPECTRALCOLORBINS];
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		tmpBins[i] = d[i]+c.d[i];
	return SpectralColor( tmpBins );
}
inline SpectralColor& SpectralColor::operator+=(const SpectralColor& c) 	    
{ 
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		d[i] += c.d[i];
	return *this; 
}
inline SpectralColor SpectralColor::operator-(const SpectralColor& c) const 	
{ 
	float tmpBins[SPECTRALCOLORBINS];
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		tmpBins[i] = d[i]-c.d[i];
	return SpectralColor( tmpBins );
}
inline SpectralColor& SpectralColor::operator-=(const SpectralColor& c) 	    
{ 
	for (int i=0; i<SPECTRALCOLORBINS;i++)
		d[i] -= c.d[i];
	return *this; 
}



#endif

