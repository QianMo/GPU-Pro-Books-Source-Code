#ifndef _STRUCT_GPU_H_
#define _STRUCT_GPU_H_

#include <vector_types.h>

// blocks of parameters needed to evaluate the model and extract an isosurface

struct CONVOLUTION_SEGMENT 
{
	CONVOLUTION_SEGMENT() {}

	CONVOLUTION_SEGMENT(	float nx1, float ny1, float nz1,
					float nx2, float ny2, float nz2,
					float nS) : x1(nx1), y1(ny1), z1(nz1), x2(nx2), y2(ny2), z2(nz2),s(nS) {}
	
   float x1, y1, z1;
   float x2, y2, z2;
   float s;
   float padding; // this is to be that sure segments are aligned (due to a known bug in CUDA compiler)

};

struct BOUNDING_BOX 
{
   float minX, minY, minZ;
   float maxX, maxY, maxZ;
};

struct BLENDING_PARAMS
{
   float a0, a1, a2;
};

struct SUBMODEL_PARAMS
{
   BLENDING_PARAMS   blendingParams;
   BOUNDING_BOX      implicitBox;

};

struct POLYGONIZATION_PARAMS
{
   // box around the volume where the polygonization is performed
   BOUNDING_BOX   volumeBox;

   // size of a single cell within the volume
   float3         cellSize;

   // helpers for indexing
   uint3 gridSizeShift, gridSize, gridSizeMask;

	float	threshold;

   int   numCells;
};

extern const int segmentsNumMax;

#endif //_STRUCT_GPU_H_
