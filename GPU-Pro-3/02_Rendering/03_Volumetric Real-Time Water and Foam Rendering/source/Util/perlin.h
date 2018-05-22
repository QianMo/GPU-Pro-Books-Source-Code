#ifndef PERLIN_H
#define PERLIN_H

//#define STRICT				// Enable strict type checking
#define WIN32_LEAN_AND_MEAN // Don't include stuff that are rarely used
#include <windows.h>


#define SIZE 256
#define MASK 0xFF

class Perlin
{
public:
	Perlin();

	void Initialize(UINT nSeed);

	float Noise1(float x);
	float Noise2(float x, float y);
	float Noise3(float x, float y, float z);

protected:
	// Permutation table
	BYTE p[SIZE];

	// Gradients
	float gx[SIZE];
	float gy[SIZE];
	float gz[SIZE];
};

#endif