#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include "defines.h"

// copy all model params to GPU memory
extern "C" bool
copyParamsToDevice(  const CONVOLUTION_SEGMENT* segmentsOnHost, int segmentsNum, float convIsoValue, 
                     BOUNDING_BOX volumeBound, SUBMODEL_PARAMS submodelParams
                     );

// initiate GPU kernel evaluating scalar field within the volume and writing it out
extern "C" void
launch_writeVolume(	dim3 grid, dim3 threads, float *volume,
							POLYGONIZATION_PARAMS params, bool isBlendingOn
                     );

extern "C" void launch_preprocessCells(	dim3 grid, dim3 threads, 
														uint* cellVerts, uint *cellsNonEmpty,// float *volume, 
														POLYGONIZATION_PARAMS params);

extern "C" void 
launch_compactCells(	dim3 grid, dim3 threads, uint *compactedCells, uint *cellsNonEmpty, 
							uint *cellsNonEmptyScan, uint numCells);

extern "C" void
launch_generateTriangles(	dim3 grid, dim3 threads, float4 *positions, float4 *normals, 
                           uint *compactedCells, uint *cellVBOffsets, POLYGONIZATION_PARAMS params,
									uint nonEmptyCellNumber, uint maxVerts, bool isBlendingOn);

extern "C"
void allocateTextures(	uint **d_edgeTable, uint **d_triTable,  uint **d_numVertsTable );

extern "C"
void allocateLUT(float **d_atanLUT, float *srcLUT, int sizeLUT);

extern "C"
void bindVolumeTexture(float *d_volume);


#endif _FUNCTIONS_H_