//-------------------------------------------------------------------------------------------------
// File: Noise.h
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
class CNoise
{
public: // Construction / Destruction

	CNoise();
	virtual ~CNoise();

public: // Public Member Functions

	double Noise( double x, double y, double z );
	double Noise( double x, double y, double z, double fScale, double fOffset, double fRotation );
	double RepeatingNoise( double x, double y, double z, double fScale, double fOffset, double fRotation );
	double AccumulatingNoise( double x, double y, double z, double fScale, int nNumOctaves, double fOffset, double fStartRotation, double fRotation );
 
protected: // Protected Member Functions

	double Fade( double t ) { return t * t * t * ( t * ( t * 6 - 15 ) + 10 ); }
	double Lerp( double t, double a, double b ) { return a + t * ( b - a ); }
	double Grad( int nHash, double x, double y, double z )
	{
		int h = nHash & 15;
		double u = ( h < 8 ) ? x : y;
		double v = ( h < 4 ) ? y : ( ( h == 12 || h == 14 ) ? x : z );
		return ( ( ( h & 1 ) == 0 ) ? u : -u ) + ( ( ( h & 2 ) == 0 ) ? v : -v );

	} // end double grad(int hash, double x, double y, double z)

protected: // Protected Member Variables

   int m_nPermutationArray[ 512 ];

}; // end class CNoise