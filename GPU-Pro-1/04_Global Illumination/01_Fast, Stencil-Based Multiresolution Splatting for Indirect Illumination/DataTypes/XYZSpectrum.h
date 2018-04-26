/******************************************************************/
/* XYZSpectrum.h                                                  */
/* -------------                                                  */
/*                                                                */
/* This file defines the spectral matching curves for the XYZ     */
/*    color-space.  The curves are sampled from 375 to 780 nm, in */
/*    increments of 5 nm per bin (82 total bins).                 */
/*                                                                */
/* Note the class only has static methods.                        */
/*                                                                */
/* You may use this file in one of two ways:                      */
/* 1) Without thinking about the underlying bin sampling (useful  */
/*    if you decide to later change it).  You can do this by      */
/*    accessing the methods:                                      */
/*      RGBColor XYZSpectrum::GetXYZStimulus(float wavelen)       */
/* and  RGBColor XYZSpectrum::ConvertXYZToRGB(RGBColor &xyzColor) */
/*                                                                */
/* 2) With an understanding that there are 82 bins sampled in     */
/*    increments of 5 nm.  5 nm should be enough for nearly all   */
/*    applications.  This approach allows you direct access to    */
/*    the underlying data.  In addition, the normalization        */
/*    factors for the x, y, and z stimuli have already been       */
/*    computed.  (This can be a speed advantage, but ties you     */
/*    into using 82 bins, which can be hard to change later, and  */
/*    [probably] is overkill for most applications).              */
/*                                                                */
/* Also note that the use of the RGBColor class is on purpose!    */
/*    Functions taking (or returning) them explicitly handle      */
/*    3-component colors, and should not change, even if the      */
/*    rest of your code uses an arbitrarily different color       */
/*    representation.  Ideally, this code will allow you to write */
/*    a function converting from an arbitrary spectrum to RGB.    */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/



#ifndef XYZSPECTRUM_H
#define XYZSPECTRUM_H

#include "DataTypes/RGBColor.h"

class XYZSpectrum {
private:
	static const int spectrumStart      = 375;
	static const int spectrumEnd        = 780;
	static const int numBins            = 82;
	static const int wavelengthSampling = 5;

	static const float xStimulus[numBins];
	static const float yStimulus[numBins];
	static const float zStimulus[numBins];

public:
	// These shouldn't be necessary.  You should be able to call all class methods directly
	XYZSpectrum() {}
	~XYZSpectrum() {}

	/////////////////////////////////////////////////////////////////////////////////////
	// The following two functions do not require you to think about the internal data //
	//     representation.  The first gives you a 3-component color XYZ value from a   //
	//     specified wavelength.   The second converts a XYZ color to a RGB color.     //
    /////////////////////////////////////////////////////////////////////////////////////

	// Return the X, Y, or Z stimuli for the specified wavelength.
	static RGBColor GetXYZStimulus( float atWavelength )
	{
		if ( atWavelength < spectrumStart || atWavelength > spectrumEnd ) 
			return RGBColor::Black();
	
		int bin = (int)((atWavelength - spectrumStart) / wavelengthSampling);
		float binWeight = ((atWavelength - spectrumStart) / wavelengthSampling) - bin;
		return RGBColor( xStimulus[bin], yStimulus[bin], zStimulus[bin] ) * (1-binWeight) +
			   RGBColor( xStimulus[bin+1], yStimulus[bin+1], zStimulus[bin+1] ) * binWeight ;
	}

	// Convert an color in XYZ format to RGB.  NOTE: This is NOT clamped to [0..1]!
	static RGBColor ConvertXYZToRGB( const RGBColor &xyzColor )
	{
		return RGBColor( 3.240479f * xyzColor.Red() - 1.537250f * xyzColor.Green() - 0.498535f * xyzColor.Blue(),
						-0.969256f * xyzColor.Red() + 1.875991f * xyzColor.Green() + 0.041556f * xyzColor.Blue(),
						 0.055648f * xyzColor.Red() - 0.204043f * xyzColor.Green() + 1.057311f * xyzColor.Blue() );
	}

	// Convert an color in XYZ format to RGB.  NOTE: This is NOT clamped to [0..1]!
	static RGBColor ConvertRGBToXYZ( const RGBColor &rgbColor )
	{
		return RGBColor( 0.412453f * rgbColor.Red() + 0.357580f * rgbColor.Green() + 0.180423f * rgbColor.Blue(),
						 0.212671f * rgbColor.Red() + 0.715160f * rgbColor.Green() + 0.072169f * rgbColor.Blue(),
						 0.019334f * rgbColor.Red() + 0.119193f * rgbColor.Green() + 0.950227f * rgbColor.Blue() );
	}



	/////////////////////////////////////////////////////////////////////////////////////
	// The following functions are usable if you are aware of the data representation! //
	//           (82 bins in each spectra.  Sectra sampled at 5 nm resolution.)        //
	//             (Start wavelengh 375 nm.  Last sampled wavelength 780 nm.)          //
    /////////////////////////////////////////////////////////////////////////////////////

	// Returns the wavelength for a given bin number.
	static int BinWavelength( int binNum ) { return spectrumStart + wavelengthSampling*binNum; }

	// Return the X, Y, or Z stimuli for the given bin number.
	//    Bin #i has a wavelength of 375 + i*5 nanometers, for 0 <= i < 82
	inline static float XStimulus( int binNum ) { return xStimulus[binNum]; }
	inline static float YStimulus( int binNum ) { return yStimulus[binNum]; }
	inline static float ZStimulus( int binNum ) { return zStimulus[binNum]; }

	// Normalization factors.  These are the sums of the 82 entries in each
	//    of the X, Y, and Z stimuli tables (respectively).
	// Please note:  These normalization factors WILL BE WRONG if you use
	//    different sampling rates (the tables assume 5 nm sampling)
	inline static float XNormalizationFactor( void ) { return 21.3714f; }
	inline static float YNormalizationFactor( void ) { return 21.3711f; }
	inline static float ZNormalizationFactor( void ) { return 21.3895f; }


};

#endif

