//-------------------------------------------------------------------------------------------------
// File: Noise.cpp
// Author: Ben Mistal
// Copyright 2010-2012 Mistal Research, Inc.
//-------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <math.h>
#include "Helpers.h"
#include "Noise.h"

// Adapted from Perlin noise
CNoise::CNoise()
{
	int nPermutationArray[] =
	{
		151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 
		140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 
		247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 
		57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
		74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
		60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 
		65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 
		200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
		52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
		207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 
		119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
		129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 
		218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 
		81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 
		184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 
		222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180
	};
   
	for (int i = 0; i < 256; i++)
	{
		m_nPermutationArray[ i ] = nPermutationArray[ i ];
		m_nPermutationArray[ 256 + i ] = m_nPermutationArray[ i ];

	} // end for (int i = 0; i < 256; i++)
   
} // end CNoise::CNoise()

CNoise::~CNoise()
{
} // end CNoise::~CNoise()

double CNoise::Noise( double x, double y, double z )
{
	int X = ( int )floor( x ) & 255;
	int Y = ( int )floor( y ) & 255;
	int Z = ( int )floor( z ) & 255;
	x -= floor( x );
	y -= floor( y );
	z -= floor( z );
	double u = Fade( x );
	double v = Fade( y );
	double w = Fade( z );

	int A = m_nPermutationArray[ X ] + Y;
	int AA = m_nPermutationArray[ A ] + Z;
	int AB = m_nPermutationArray[ A + 1 ] + Z;      
	int B = m_nPermutationArray[ X + 1 ] + Y;
	int BA = m_nPermutationArray[ B ] + Z;
	int BB = m_nPermutationArray[ B + 1 ] + Z;      

	double fNoise =  Lerp( w,	Lerp( v,	Lerp( u,	Grad( m_nPermutationArray[ AA ], x, y, z ),
														Grad( m_nPermutationArray[ BA ], x - 1, y, z ) ),
								  			Lerp( u,	Grad( m_nPermutationArray[ AB ], x  , y - 1, z ),
														Grad( m_nPermutationArray[ BB ], x - 1, y - 1, z ) ) ),
								Lerp( v,	Lerp( u,	Grad( m_nPermutationArray[ AA + 1 ], x, y, z - 1 ),
														Grad( m_nPermutationArray[ BA + 1 ], x - 1, y  , z - 1 ) ),
											Lerp( u,	Grad( m_nPermutationArray[ AB + 1 ], x, y - 1, z - 1 ),
														Grad( m_nPermutationArray[ BB + 1 ], x - 1, y - 1, z - 1 ) ) ) ) ;

	return fNoise;

} // end double CNoise::Noise( double x, double y, double z )

double CNoise::Noise( double x, double y, double z, double fScale, double fOffset, double fRotation )
{
	double fSin = sin( fRotation );
	double fCos = cos( fRotation );
	double fX = ( ( ( fCos * x ) + ( fSin * y ) ) * fScale ) + fOffset;
	double fY = ( ( ( fCos * y ) - ( fSin * x ) ) * fScale ) + fOffset;
	double fZ = ( z * fScale ) + fOffset;

	return Noise( fX, fY, fZ );

} // end double CNoise::Noise( double x, double y, double z, double fScale, double fOffset, double fRotation )

double CNoise::RepeatingNoise( double x, double y, double z, double fScale, double fOffset, double fRotation )
{
	double n = Noise( x, y, z, fScale, fOffset, fRotation );
	double nx = Noise( ( x - 1.0 ), y,z, fScale, fOffset, fRotation );
	double ny1 = Noise( x, ( y - 1.0 ), z, fScale, fOffset, fRotation );
	double ny2 = Noise( ( x - 1.0 ), ( y - 1.0 ), z, fScale, fOffset, fRotation );
	double nz1 = Noise( x, y, ( z - 1.0 ), fScale, fOffset, fRotation );
	double nz2 = Noise( ( x - 1.0 ), y, ( z - 1.0 ), fScale, fOffset, fRotation );
	double nz3 = Noise( x, ( y - 1.0 ), ( z - 1.0 ), fScale, fOffset, fRotation );
	double nz4 = Noise( ( x - 1.0 ), ( y - 1.0 ), ( z - 1.0 ), fScale, fOffset, fRotation );

	double ny = Lerp( x, ny1, ny2 );
	double nza = Lerp( x, nz1, nz2 );
	double nzb = Lerp( x, nz3, nz4 );
	double nz = Lerp( y, nza, nzb );

	return Lerp( z, Lerp( y, Lerp( x, n, nx ), ny ), nz );

} // end double CNoise::RepeatingNoise( double x, double y, double z, double fScale, double fOffset, double fRotation )

double CNoise::AccumulatingNoise( double x, double y, double z, double fScale, int nNumOctaves, double fOffset, double fStartRotation, double fRotation )
{
	double fNoise = 0.0;

	for( int nOctaveIndex = 0; nOctaveIndex < nNumOctaves; nOctaveIndex++ )
	{
		fNoise += RepeatingNoise( x, y, z, pow( 2.0, ( ( double )nOctaveIndex ) + fScale ), fOffset * nOctaveIndex, fStartRotation + fRotation * double( nOctaveIndex + 1 ) ) / pow( 2.0, nOctaveIndex );

	} // end for( int nOctaveIndex = 0; nOctaveIndex < nNumOctaves; nOctaveIndex++ )

	return fNoise;

} // end double CNoise::AccumulatingNoise( double x, double y, double z,  double fScale, int nNumOctaves, double fOffset, double fStartRotation, double fRotation )