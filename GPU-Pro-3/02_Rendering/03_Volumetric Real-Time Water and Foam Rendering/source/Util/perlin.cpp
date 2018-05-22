// TODO:
//
// By doubling the length of the arrays and duplicating the first half onto the 
// second half there is no need for the bitwise and when computing the 
// permuation.
//
// Is it possible to make a generalized noise function that takes a number for 
// the dimension and a vector with equal amount of values for the coordinate?
// I propose that a one dimensional gradient array can be used. The permutation
// array can be used to pseudo randomly pick n gradients from this array and 
// combine into a gradient vector.

#include "perlin.h"
#include <stdlib.h>
#include <math.h>

//=============================================================================
Perlin::Perlin()
{
}

//=============================================================================
void Perlin::Initialize(UINT nSeed)
{
	int i, j, nSwap;

	srand(nSeed);

	// Initialize the permutation table
	for(i = 0; i < SIZE; i++)
		p[i] = (BYTE)i;

	for(i = 0; i < SIZE; i++)
	{
		j = rand() & MASK;

		nSwap = p[i];
		p[i]  = p[j];
		p[j]  = (BYTE)nSwap;
	}

	// Generate the gradient lookup tables
	for(i = 0; i < SIZE; i++)
	{
		// Ken Perlin proposes that the gradients are taken from the unit 
		// circle/sphere for 2D/3D, but there are no noticable difference 
		// between that and what I'm doing here. For the sake of generality 
		// I will not do that.

		gx[i] = float(rand())/(RAND_MAX/2) - 1.0f; 
		gy[i] = float(rand())/(RAND_MAX/2) - 1.0f;
		gz[i] = float(rand())/(RAND_MAX/2) - 1.0f;
	}
}

//=============================================================================
float Perlin::Noise1(float x)
{
	// Compute what gradients to use
	int qx0 = (int)floorf(x);
	int qx1 = qx0 + 1;
	float tx0 = x - (float)qx0;
	float tx1 = tx0 - 1;

	// Make sure we don't come outside the lookup table
	qx0 = qx0 & MASK;
	qx1 = qx1 & MASK;

	// Compute the dotproduct between the vectors and the gradients
	float v0 = gx[qx0]*tx0;
	float v1 = gx[qx1]*tx1;

	// Modulate with the weight function
	float wx = (3 - 2*tx0)*tx0*tx0;
	float v = v0 - wx*(v0 - v1);

	return v;
}

//=============================================================================
float Perlin::Noise2(float x, float y)
{
	// Compute what gradients to use
	int qx0 = (int)floorf(x);
	int qx1 = qx0 + 1;
	float tx0 = x - (float)qx0;
	float tx1 = tx0 - 1;

	int qy0 = (int)floorf(y);
	int qy1 = qy0 + 1;
	float ty0 = y - (float)qy0;
	float ty1 = ty0 - 1;

	// Make sure we don't come outside the lookup table
	qx0 = qx0 & MASK;
	qx1 = qx1 & MASK;

	qy0 = qy0 & MASK;
	qy1 = qy1 & MASK;

	// Permutate values to get pseudo randomly chosen gradients
	int q00 = p[(qy0 + p[qx0]) & MASK];
	int q01 = p[(qy0 + p[qx1]) & MASK];

	int q10 = p[(qy1 + p[qx0]) & MASK];
	int q11 = p[(qy1 + p[qx1]) & MASK];

	// Compute the dotproduct between the vectors and the gradients
	float v00 = gx[q00]*tx0 + gy[q00]*ty0;
	float v01 = gx[q01]*tx1 + gy[q01]*ty0;

	float v10 = gx[q10]*tx0 + gy[q10]*ty1;
	float v11 = gx[q11]*tx1 + gy[q11]*ty1;

	// Modulate with the weight function
	float wx = (3 - 2*tx0)*tx0*tx0;
	float v0 = v00 - wx*(v00 - v01);
	float v1 = v10 - wx*(v10 - v11);

	float wy = (3 - 2*ty0)*ty0*ty0;
	float v = v0 - wy*(v0 - v1);

	return v;
}

//=============================================================================
float Perlin::Noise3(float x, float y, float z)
{
	// Compute what gradients to use
	int qx0 = (int)floorf(x);
	int qx1 = qx0 + 1;
	float tx0 = x - (float)qx0;
	float tx1 = tx0 - 1;

	int qy0 = (int)floorf(y);
	int qy1 = qy0 + 1;
	float ty0 = y - (float)qy0;
	float ty1 = ty0 - 1;

	int qz0 = (int)floorf(z);
	int qz1 = qz0 + 1;
	float tz0 = z - (float)qz0;
	float tz1 = tz0 - 1;

	// Make sure we don't come outside the lookup table
	qx0 = qx0 & MASK;
	qx1 = qx1 & MASK;

	qy0 = qy0 & MASK;
	qy1 = qy1 & MASK;

	qz0 = qz0 & MASK;
	qz1 = qz1 & MASK;

	// Permutate values to get pseudo randomly chosen gradients
	int q000 = p[(qz0 + p[(qy0 + p[qx0]) & MASK]) & MASK];
	int q001 = p[(qz0 + p[(qy0 + p[qx1]) & MASK]) & MASK];

	int q010 = p[(qz0 + p[(qy1 + p[qx0]) & MASK]) & MASK];
	int q011 = p[(qz0 + p[(qy1 + p[qx1]) & MASK]) & MASK];

	int q100 = p[(qz1 + p[(qy0 + p[qx0]) & MASK]) & MASK];
	int q101 = p[(qz1 + p[(qy0 + p[qx1]) & MASK]) & MASK];

	int q110 = p[(qz1 + p[(qy1 + p[qx0]) & MASK]) & MASK];
	int q111 = p[(qz1 + p[(qy1 + p[qx1]) & MASK]) & MASK];

	// Compute the dotproduct between the vectors and the gradients
	float v000 = gx[q000]*tx0 + gy[q000]*ty0 + gz[q000]*tz0;
	float v001 = gx[q001]*tx1 + gy[q001]*ty0 + gz[q001]*tz0;  

	float v010 = gx[q010]*tx0 + gy[q010]*ty1 + gz[q010]*tz0;
	float v011 = gx[q011]*tx1 + gy[q011]*ty1 + gz[q011]*tz0;

	float v100 = gx[q100]*tx0 + gy[q100]*ty0 + gz[q100]*tz1;
	float v101 = gx[q101]*tx1 + gy[q101]*ty0 + gz[q101]*tz1;  

	float v110 = gx[q110]*tx0 + gy[q110]*ty1 + gz[q110]*tz1;
	float v111 = gx[q111]*tx1 + gy[q111]*ty1 + gz[q111]*tz1;

	// Modulate with the weight function
	float wx = (3 - 2*tx0)*tx0*tx0;
	float v00 = v000 - wx*(v000 - v001);
	float v01 = v010 - wx*(v010 - v011);
	float v10 = v100 - wx*(v100 - v101);
	float v11 = v110 - wx*(v110 - v111);

	float wy = (3 - 2*ty0)*ty0*ty0;
	float v0 = v00 - wy*(v00 - v01);
	float v1 = v10 - wy*(v10 - v11);

	float wz = (3 - 2*tz0)*tz0*tz0;
	float v = v0 - wz*(v0 - v1);

	return v;
}
