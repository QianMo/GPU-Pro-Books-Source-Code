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

#ifndef RANDOM_H
#define RANDOM_H

#include <time.h>
#include "DataTypes/Vector.h"
#include "DataTypes/Point.h"

class Random
{
private:
	const unsigned long maxshort;
	const unsigned long multiplier;
	const unsigned long adder;

	unsigned long randSeed;

public:
	Random( unsigned long seed=0 );
	~Random() {}

	double         dRandom( void );          /* in [0..1]     */
	float          fRandom( void );          /* in [0..1]     */
	unsigned short sRandom( void );          /* in [0..65535] */
	bool           bRandom( void );          /* true or false */
	unsigned char  cRandom( void );          /* in [0..255]   */
	
	/* Give a random vector on the unit sphere, +z unit hemisphere, or arbitrary unit hemisphere */
	Vector RandomSphereVector( void );                    
	Vector RandomHemisphereVector( void );               
	Vector RandomHemisphereVector( const Vector &inDir ); 
	Vector RandomCosineWeightedHemisphereVector( const Vector &inDir ); 

	/* Give a random point on a triangle described by 3 points (or 1 point and two edges) */
	Point RandomPointOnTriangle( const Point &p0, const Point &p1, const Point &p2 );
	Point RandomPointOnTriangle( const Point &p0, const Vector &edge01, const Vector &edge02 );

	/* Give a random point on the unit sphere */
	Point RandomPointOnSphere( void );

	/* Give a random point on the unit xy-disk (i.e., z=0) */
	Point RandomPointOnDisk( void );
};


#endif

