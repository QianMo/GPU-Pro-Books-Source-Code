/******************************************************************/
/* Random.h                                                       */
/* -----------------------                                        */
/*                                                                */
/* The file defines a random number generator class.  This may    */
/*   not be the best random number generator around, but it works */
/*   reasonably well, and gives a portable random class for use   */
/*   on multiple platforms (which may or may not have rand() or   */
/*   drand48(), or they may be broken).                           */
/*                                                                */
/* Chris Wyman (10/26/2006)                                       */
/******************************************************************/
#include <math.h>
#include "Utils/Random.h"
//#include "Definitions.h"

Random::Random( unsigned long seed ) :
   maxshort( 65536L ), multiplier( 1194211693L ), adder( 12345L )
{
	randSeed = ( seed ? seed : (unsigned long)time(0) );
}

double Random::dRandom( void )
{
	randSeed = multiplier * randSeed + adder;
	return ((randSeed >> 16) % maxshort) / (double)maxshort;
}

float  Random::fRandom( void )
{
	randSeed = multiplier * randSeed + adder;
	return ((randSeed >> 16) % maxshort) / (float)maxshort;
}

unsigned short Random::sRandom ( void )
{
	randSeed = multiplier * randSeed + adder;
	return (unsigned short)((randSeed >> 16) % maxshort);
}

unsigned char Random::cRandom ( void )
{
	randSeed = multiplier * randSeed + adder;
	return (unsigned char)((randSeed >> 16) % 256L );
}

bool Random::bRandom ( void )
{
	randSeed = multiplier * randSeed + adder;
	return ((randSeed >> 16) & 0x00000001) > 0;  // check if the last bit is on or off
}

Vector Random::RandomHemisphereVector( void )
{
	float f1 = fRandom(), f2 = fRandom();
	return Vector( sqrt(f1)*cos(2.0f*M_PI*f2), 
				   sqrt(f1)*sin(2.0f*M_PI*f2), 
				   sqrt(1.0f - f1) );
}

Vector Random::RandomSphereVector( void )
{
	float f1 = 2.0f * fRandom() - 1.0f;
	float f2 = 2.0f * (float)M_PI * fRandom();
	float r = sqrt( 1.0f - f1*f1 );
	return Vector( r*cos( f2 ), r*sin( f2 ), f1 );
}

Vector Random::RandomHemisphereVector( const Vector &inDir )
{
	float f1 = fRandom(), f2 = fRandom();

	// find an orthonormal basis
	Vector up = inDir;
	up.Normalize();
	Vector u = ( up.X() > 0.99 || up.X() < -0.99 ? Vector(0,1,0) : Vector(1,0,0) );
	Vector v = up.Cross( u );
	u = v.Cross( up );

	return (sqrt(f1)*cos(2.0f*M_PI*f2))*u + (sqrt(f1)*sin(2.0f*M_PI*f2))*v + (sqrt(1-f1))*up;

}

Vector Random::RandomCosineWeightedHemisphereVector( const Vector &inDir )
{
	float f1 = fRandom(), f2 = fRandom();
	float phi = M_PI*f1;
	float theta = 2.0f*M_PI*f2;

	// find an orthonormal basis
	Vector up = inDir;
	up.Normalize();
	Vector u = ( up.X() > 0.99 || up.X() < -0.99 ? Vector(0,1,0) : Vector(1,0,0) );
	Vector v = up.Cross( u );
	u = v.Cross( up );

	return sin(phi)*cos(theta)*u + sin(phi)*sin(theta)*v + cos(phi)*up;
}

Point Random::RandomPointOnTriangle( const Point &p0, const Point &p1, const Point &p2 )
{
	float f1 = fRandom(), f2 = fRandom();
	if (f1 + f2 > 1) { f1 = 1-f1; f2 = 1-f2; }
	return (1-f1-f2)*p0 + f1*p1 + f2*p2;
}

Point Random::RandomPointOnTriangle( const Point &p0, const Vector &edge01, const Vector &edge02 )
{
	float f1 = fRandom(), f2 = fRandom();
	if (f1 + f2 > 1) { f1 = 1-f1; f2 = 1-f2; }
	return p0 + f1*edge01 + f2*edge02;
}

Point Random::RandomPointOnSphere( void )
{
	float f1 = 2.0f * fRandom() - 1.0f;
	float f2 = 2.0f * (float)M_PI * fRandom();
	float r = sqrt( 1.0f - f1*f1 );
	return Point( r*cos( f2 ), r*sin( f2 ), f1 );
}

Point Random::RandomPointOnDisk( void )
{
	float r = sqrt( fRandom() );
	float angle = 2.0f * (float)M_PI * fRandom();
	return Point( r*cos( angle ), r*sin( angle ), 0 );
}


