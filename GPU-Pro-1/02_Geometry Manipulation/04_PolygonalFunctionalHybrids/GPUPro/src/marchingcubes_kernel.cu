// This file is derived from the NVIDIA CUDA SDK example 'marchingCubes'.
// all CUDA kernels for functions evaluation and isosurface extraction are defined here

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#ifndef _MARCHING_CUBES_KERNEL_CU_
#define _MARCHING_CUBES_KERNEL_CU_

#include <stdio.h>
#include <string.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include "cutil_math.h"

#include "defines.h"
#include "MC/tables.h"

#include "FRep/structGPU.h"

// textures containing look-up tables
texture<uint, 1, cudaReadModeElementType> edgeTex;
texture<uint, 1, cudaReadModeElementType> triTex;
texture<uint, 1, cudaReadModeElementType> numVertsTex;

// volume data
texture<float, 1, cudaReadModeElementType> volumeTex;

// map arrays to 1D textures, hoping texture cache will help when reading
extern "C"
void allocateTextures(	uint **d_edgeTable, uint **d_triTable,  uint **d_numVertsTable)
{
    cutilSafeCall(cudaMalloc((void**) d_edgeTable, 256*sizeof(uint)));
    cutilSafeCall(cudaMemcpy((void *)*d_edgeTable, (void *)edgeTable, 256*sizeof(uint), cudaMemcpyHostToDevice) );
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cutilSafeCall(cudaBindTexture(0, edgeTex, *d_edgeTable, channelDesc) );

    cutilSafeCall(cudaMalloc((void**) d_triTable, 256*16*sizeof(uint)));
    cutilSafeCall(cudaMemcpy((void *)*d_triTable, (void *)triTable, 256*16*sizeof(uint), cudaMemcpyHostToDevice) );
    cutilSafeCall(cudaBindTexture(0, triTex, *d_triTable, channelDesc) );

    cutilSafeCall(cudaMalloc((void**) d_numVertsTable, 256*sizeof(uint)));
    cutilSafeCall(cudaMemcpy((void *)*d_numVertsTable, (void *)numVertsTable, 256*sizeof(uint), cudaMemcpyHostToDevice) );
    cutilSafeCall(cudaBindTexture(0, numVertsTex, *d_numVertsTable, channelDesc) );
}

// bind volume to a 1D texture	
extern "C"
void bindVolumeTexture(float *d_volume)
{  
   cutilSafeCall(cudaBindTexture(0, volumeTex, d_volume, cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat)));
}

// will need a few "passes" if more segments is used
static const int segmentsNumMax = 64;
__constant__ CONVOLUTION_SEGMENT    d_segmentsOnDevice[segmentsNumMax];
__constant__ POLYGONIZATION_PARAMS  d_polygonizationParams;

// polygonization parameters
__constant__ BOUNDING_BOX d_volumeBound;
__constant__ BOUNDING_BOX d_implicitBox;
__constant__ int d_segmentsNum;
__constant__ float d_convIsoValue;

__constant__ BLENDING_PARAMS d_blendParams;

// copy all model params to GPU memory
extern "C"
bool copyParamsToDevice(   const CONVOLUTION_SEGMENT* segmentsOnHost, int segmentsNum, float convIsoValue, 
                           BOUNDING_BOX volumeBound, SUBMODEL_PARAMS submodelParams
                          )
{
   if (segmentsNum > segmentsNumMax) {
      // too many segments
      // could split them and render separately summing the field (but not in this demo)
      return false;
   }

   CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_segmentsOnDevice, segmentsOnHost, sizeof(*segmentsOnHost) * segmentsNum) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_segmentsNum, &segmentsNum, sizeof(segmentsNum)) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_convIsoValue, &convIsoValue, sizeof(convIsoValue)) );

   CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_volumeBound, &volumeBound, sizeof(volumeBound)) );
   CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_implicitBox, &submodelParams.implicitBox, sizeof(submodelParams.implicitBox)) );
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(d_blendParams, &submodelParams.blendingParams, sizeof(submodelParams.blendingParams)) );

   return true;
}

// evaluates the field producede by a set of line segments (Cauchy kernel)
__device__
float convolution(float x, float y, float z)
{
	float f = 0.f;
	
	for(int n=0; n < d_segmentsNum; n++) {

		const CONVOLUTION_SEGMENT& segment = d_segmentsOnDevice[n];

		const float S = segment.s;
		const float S2 = S * S;
		float3 segmentVector;
		segmentVector.x = segment.x2 - segment.x1;
		segmentVector.y = segment.y2 - segment.y1;
		segmentVector.z = segment.z2 - segment.z1;

		float l = sqrtf( segmentVector.x *  segmentVector.x + segmentVector.y *  segmentVector.y + segmentVector.z *  segmentVector.z);

		float ax = (segmentVector.x) / l;
		float ay = (segmentVector.y) / l;
		float az = (segmentVector.z) / l;

		float dx = x - segment.x1;
		float dy = y - segment.y1;
		float dz = z - segment.z1;

		float xx = dx*ax + dy*ay + dz*az;
		float p = sqrtf(1.f + S2 * ( dx*dx + dy*dy + dz*dz - xx*xx));
		float q = sqrtf(1.f + S2 * ( dx*dx + dy*dy + dz*dz + l*l - 2.f*l*xx ));

		f +=  xx / (2.f*p*p*(p*p + S2*xx*xx)) + (l - xx) / (2.f*p*p*q*q) 
			   + (atan(S*xx/p) + atan(S*(l - xx)/p)) / (2.f * S * p * p * p);
   }	

    return f - d_convIsoValue;
}

// convolution surface blended with a box
__device__
float modelFunction(float x, float y, float z)
{
	float convVal = convolution(x, y, z);

	float dvX = x - d_implicitBox.minX;
	float dvY = y - d_implicitBox.minY;
	float dvZ = z - d_implicitBox.minZ;

	float dlvX = d_implicitBox.minX + d_implicitBox.maxX - x;
	float dlvY = d_implicitBox.minY + d_implicitBox.maxY - y;
	float dlvZ = d_implicitBox.minZ + d_implicitBox.maxZ - z;	

	float xt = dvX * dlvX;
	float yt = dvY * dlvY;
	float zt = dvZ * dlvZ;

	float f2 = xt + yt - sqrt(xt*xt + yt*yt);
	f2 = f2 + zt - sqrt(f2*f2 + zt*zt);

	double d = d_blendParams.a0 / (1 + (convVal / d_blendParams.a1) * (convVal / d_blendParams.a1) + (f2 / d_blendParams.a2) * (f2 / d_blendParams.a2));
	convVal = f2 + convVal + sqrt(f2 * f2 + convVal * convVal) + d;


	return convVal;
}

// sample volume data mapped to a 1D texture at a point
__device__
float sampleVolume(uint3 p, uint3 gridSize)
{
    p.x = min(p.x, gridSize.x - 1);
    p.y = min(p.y, gridSize.y - 1);
    p.z = min(p.z, gridSize.z - 1);
    uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;

    float val = tex1Dfetch(volumeTex, i);
   
    return val;
}

// calculates linear address of a point on a specified grid
__device__
uint getVolumeAddress(uint3 p, uint3 gridSize)
{
	p.x = min(p.x, gridSize.x - 1);
   p.y = min(p.y, gridSize.y - 1);
   p.z = min(p.z, gridSize.z - 1);
   uint i = (p.z*gridSize.x*gridSize.y) + (p.y*gridSize.x) + p.x;
	return i;
}

// compute position in 3d grid from 1d index (only works for power of 2 sizes)
__device__
uint3 calcGridPos(uint cellIndex, uint3 gridSizeShift, uint3 gridSizeMask)
{
    uint3 gridPos;
    gridPos.x = cellIndex & gridSizeMask.x;
    gridPos.y = (cellIndex >> gridSizeShift.y) & gridSizeMask.y;
    gridPos.z = (cellIndex >> gridSizeShift.z) & gridSizeMask.z;
    return gridPos;
}

// writes out function value using the functor specified by the template
template <typename FUNCTOR>
__global__ void
writeVolume(	float *volume, POLYGONIZATION_PARAMS params, FUNCTOR functor)
{
   uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
   // linear address
   uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

   // 3 indices in the discretized set of cells
   uint3 gridPos = calcGridPos(i, params.gridSizeShift, params.gridSizeMask);

   uint3 p;
   // need to clamp the values, as the number of sheduled threads might be greater the
   // number of remaining cells
   p.x = min(gridPos.x, params.gridSize.x - 1);
   p.y = min(gridPos.y, params.gridSize.y - 1);
   p.z = min(gridPos.z, params.gridSize.z - 1);

   // now calculate the "clamped index" in the 1D texture
   uint index = (p.z*params.gridSize.x*params.gridSize.y) + (p.y*params.gridSize.x) + p.x;

   float3 pointInVolume;

   pointInVolume.x = params.volumeBox.minX + (gridPos.x * params.cellSize.x);
   pointInVolume.y = params.volumeBox.minY + (gridPos.y * params.cellSize.y);
   pointInVolume.z = params.volumeBox.minZ + (gridPos.z * params.cellSize.z);

   float val = functor.evaluate(pointInVolume.x, pointInVolume.y, pointInVolume.z);

   volume[index] = val;
}

// get MC case index depending on function values
__device__ uint getMCIndex(const float* field, float threshold) 
{
   uint indexMC;
   indexMC =  uint(field[0] < threshold); 
   indexMC |= uint(field[1] < threshold) << 1; 
   indexMC |= uint(field[2] < threshold) << 2; 
   indexMC |= uint(field[3] < threshold) << 3; 
   indexMC |= uint(field[4] < threshold) << 4; 
   indexMC |= uint(field[5] < threshold) << 5; 
   indexMC |= uint(field[6] < threshold) << 6; 
   indexMC |= uint(field[7] < threshold) << 7;

   return indexMC;
}

// find out whether this cell outputs any triangles ("non-empty cell") and find the number of triangles
__global__ void
preprocessCell(   uint* cellVerts, uint *cellsNonEmpty,              
				      POLYGONIZATION_PARAMS params
				  )
{
   uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
   // absolute index of the current cell
   uint cellIndex = __mul24(blockId, blockDim.x) + threadIdx.x;
   // calculate 3D position of the cell
   uint3 gridPos = calcGridPos(cellIndex, params.gridSizeShift, params.gridSizeMask);

	uint3 gridSize = params.gridSize;

   // might be better to clamp the value as in writeVolume to avoid branch divergence
   if (cellIndex >= params.numCells) {
      return;
   }
   float field [8];

   // retrieve function values at 8 corners of a cube
   field[0]= sampleVolume(gridPos, gridSize);
   field[1]= sampleVolume(gridPos + make_uint3(1,0,0), gridSize);
   field[2]= sampleVolume(gridPos + make_uint3(1,1,0), gridSize);
   field[3]= sampleVolume(gridPos + make_uint3(0,1,0), gridSize);
   field[4]= sampleVolume(gridPos + make_uint3(0,0,1), gridSize);
   field[5]= sampleVolume(gridPos + make_uint3(1,0,1), gridSize);
   field[6]= sampleVolume(gridPos + make_uint3(1,1,1), gridSize);
   field[7]= sampleVolume(gridPos + make_uint3(0,1,1), gridSize);

   // find out case index in the MC table
   uint indexMC =  getMCIndex(field, params.threshold); 

   // read number of vertices produced by this case
   uint numVerts = tex1Dfetch(numVertsTex, indexMC);

   //if (cellIndex < numCells) {
   if (cellIndex < params.numCells) {
      // save the number of vertices for later usage
      cellVerts[cellIndex] = numVerts;
      // flag indicating whether this cell outputs any triangles
      cellsNonEmpty[cellIndex] = (numVerts > 0);
   }
}

// these structures are only used to vary the evaluated functions using template instantiation:

// convolution only
struct FUNCTOR_CONVOLUTION
{
   __device__ float evaluate(float x, float y, float z)
   {      
      return convolution(x, y, z);      
   }
};

// convolutions blended with the box
struct FUNCTOR_BLENDED_CONVOLUTION
{
   __device__ float evaluate(float x, float y, float z)
   {
      return modelFunction(x, y, z);
   }
};

// initiate GPU kernel evaluating scalar field within the volume and writing it out
extern "C" void
launch_writeVolume(	dim3 grid, dim3 threads, float *volume,
                     POLYGONIZATION_PARAMS params, bool isBlendingOn)
{
   // this is to speed up computations if blending operation is not required
   // compiler generates 2 functions
   if (!isBlendingOn) {

      FUNCTOR_CONVOLUTION convFunctor;

      writeVolume<<<grid, threads>>>( volume, params, convFunctor);

   } else {

      FUNCTOR_BLENDED_CONVOLUTION convFunctor;

      writeVolume<<<grid, threads>>>(	volume, params, convFunctor);
   }
}

// output the number of vertices that need to be generated for 
// current cell and output flag indicating whether current 
// cell contains any triangles at all
extern "C" void
launch_preprocessCells( dim3 grid, dim3 threads, uint* cellVerts, uint *cellsNonEmpty,// float *volume,
					  POLYGONIZATION_PARAMS params
					  )
{
   // find out the number of vertices being output by each cell and find out if the cell is empty
   preprocessCell<<<grid, threads>>>(	cellVerts, cellsNonEmpty, params);

   cutilCheckMsg("preprocessCells failed");
}
              

// apply compact operation to remove empty cells from the list
__global__ void
compactCells(uint *compactedCells, uint *cellsNonEmpty, uint *cellsNonEmptyScan, uint numCells)
{
   uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
   uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

   if (cellsNonEmpty[i] && (i < numCells)) {
      // if save current cell index, only if it contains some triangles
      compactedCells[ cellsNonEmptyScan[i] ] = i;
   }
}

// initiate compact operation to get rid of empty cells
extern "C" void 
launch_compactCells(dim3 grid, dim3 threads, uint *compactedCells, uint *cellsNonEmpty, uint *cellsNonEmptyScan, uint numCells)
{
   compactCells<<<grid, threads>>>( compactedCells, cellsNonEmpty, cellsNonEmptyScan, numCells);
   cutilCheckMsg("compactCells failed");
}

// compute interpolated vertex along an edge
__device__
float3 interpolatePosition(float threshold, float3 cellVertex1,float3 cellVertex2, float funcValue1, float funcValue2)
{
   float t = (threshold - funcValue1) / (funcValue2 - funcValue1);
   return lerp(cellVertex1, cellVertex2, t);
} 

// use the information from non-empty cells to produce actual triangles
// of the extracted isosurface
template <typename FUNCTOR>
__global__ void
generateTriangles(	float4 *positions, float4 *normals, uint *compactedCells, uint *cellVBOffsets,
                     POLYGONIZATION_PARAMS params, uint nonEmptyCellNumber, uint maxVerts, FUNCTOR functor)
{

   uint blockId = __mul24(blockIdx.y, gridDim.x) + blockIdx.x;
   uint i = __mul24(blockId, blockDim.x) + threadIdx.x;

   if (i > nonEmptyCellNumber - 1) {
      i = nonEmptyCellNumber - 1;
   }

   uint cellIndex = compactedCells[i];

   // compute position in 3D grid
   uint3 gridPos = calcGridPos(cellIndex, params.gridSizeShift, params.gridSizeMask);

   float3 p;

   float3 cellSize = params.cellSize;

   p.x = params.volumeBox.minX + (gridPos.x * cellSize.x);
   p.y = params.volumeBox.minY + (gridPos.y * cellSize.y);
   p.z = params.volumeBox.minZ + (gridPos.z * cellSize.z);

   // calculate cell vertex positions
   float3 cellVertices[8];
   cellVertices[0] = p;
   cellVertices[1] = p + make_float3(cellSize.x, 0, 0);
   cellVertices[2] = p + make_float3(cellSize.x, cellSize.y, 0);
   cellVertices[3] = p + make_float3(0, cellSize.y, 0);
   cellVertices[4] = p + make_float3(0, 0, cellSize.z);
   cellVertices[5] = p + make_float3(cellSize.x, 0, cellSize.z);
   cellVertices[6] = p + make_float3(cellSize.x, cellSize.y, cellSize.z);
   cellVertices[7] = p + make_float3(0, cellSize.y, cellSize.z);

   uint3 gridSize = params.gridSize;

   // sampling filed value at all vertices of the cell again
   float field[8];

   field[0] = sampleVolume(gridPos, gridSize);
   field[1] = sampleVolume(gridPos + make_uint3(1, 0, 0), gridSize);

   field[2] = sampleVolume(gridPos + make_uint3(1, 1, 0), gridSize);
   field[3] = sampleVolume(gridPos + make_uint3(0, 1, 0), gridSize);

   field[4] = sampleVolume(gridPos + make_uint3(0, 0, 1), gridSize);
   field[5] = sampleVolume(gridPos + make_uint3(1, 0, 1), gridSize);

   field[6] = sampleVolume(gridPos + make_uint3(1, 1, 1), gridSize);
   field[7] = sampleVolume(gridPos + make_uint3(0, 1, 1), gridSize);


	// find the vertices where the surface intersects the cell/cube (each of the vertices
   // is situated somewhere along an edge of a cell, each cell has 12 edges) :

   float threshold = params.threshold;
    // use shared memory to avoid using local
	__shared__ float3 vertlist[12*NTHREADS];

   // each thread block writes to exactly 1 memory address within a bank
   // thus avoiding bank conflicts when using shared memory
	vertlist[threadIdx.x] = interpolatePosition(threshold, cellVertices[0], cellVertices[1], field[0], field[1]);
    vertlist[NTHREADS+threadIdx.x] = interpolatePosition(threshold, cellVertices[1], cellVertices[2], field[1], field[2]);
    vertlist[(NTHREADS*2)+threadIdx.x] = interpolatePosition(threshold, cellVertices[2], cellVertices[3], field[2], field[3]);
    vertlist[(NTHREADS*3)+threadIdx.x] = interpolatePosition(threshold, cellVertices[3], cellVertices[0], field[3], field[0]);
	vertlist[(NTHREADS*4)+threadIdx.x] = interpolatePosition(threshold, cellVertices[4], cellVertices[5], field[4], field[5]);
    vertlist[(NTHREADS*5)+threadIdx.x] = interpolatePosition(threshold, cellVertices[5], cellVertices[6], field[5], field[6]);
    vertlist[(NTHREADS*6)+threadIdx.x] = interpolatePosition(threshold, cellVertices[6], cellVertices[7], field[6], field[7]);
    vertlist[(NTHREADS*7)+threadIdx.x] = interpolatePosition(threshold, cellVertices[7], cellVertices[4], field[7], field[4]);
	vertlist[(NTHREADS*8)+threadIdx.x] = interpolatePosition(threshold, cellVertices[0], cellVertices[4], field[0], field[4]);
    vertlist[(NTHREADS*9)+threadIdx.x] = interpolatePosition(threshold, cellVertices[1], cellVertices[5], field[1], field[5]);
    vertlist[(NTHREADS*10)+threadIdx.x] = interpolatePosition(threshold, cellVertices[2], cellVertices[6], field[2], field[6]);
    vertlist[(NTHREADS*11)+threadIdx.x] = interpolatePosition(threshold, cellVertices[3], cellVertices[7], field[3], field[7]);
    __syncthreads();

   // get the number of triangles that need to be output
   // in this MC case
   uint indexMC = getMCIndex(field, params.threshold);    
   uint numVerts = tex1Dfetch(numVertsTex, indexMC);

   for(int i=0; i < numVerts; i++) {
      // find the offset of this vertex in the vertex buffer
      uint vertexOffset = cellVBOffsets[cellIndex] + i;

      if (vertexOffset >= maxVerts) {       
         continue;
      }

      // will get the vertex from the appropriate edge of the cell
      uint edge = tex1Dfetch(triTex, (indexMC << 4) + i);

      // write out vertex position to VB
      float3 p = vertlist[(edge*NTHREADS)+threadIdx.x];
      positions[vertexOffset] = make_float4(p, 1.0f);

      // the same for normal, only it is evaluated on the fly
      float val = functor.evaluate( p.x, p.y, p.z );
      const float d = 0.01f;

      float dx = functor.evaluate(p.x + d,	p.y,		p.z)     - val;
      float dy = functor.evaluate(p.x,		   p.y + d,	p.z)     - val;
      float dz = functor.evaluate(p.x,		   p.y,		p.z + d) - val;

      normals[vertexOffset] = make_float4(-dx, -dy, -dz, 0.0f);
   }
}

// start a kernel generating triangles and normals from the early generated dataset
extern "C" void
launch_generateTriangles(  dim3 grid, dim3 threads, float4 *positions, float4 *normals, uint *compactedCells, uint *cellVBOffsets,
						         POLYGONIZATION_PARAMS params,	uint nonEmptyCellNumber, uint maxVerts, bool isBlendingOn)
{
   if (!isBlendingOn) {
   
      FUNCTOR_CONVOLUTION convFunctor;

      generateTriangles<<<grid, NTHREADS>>>( positions, normals, 
                                             compactedCells, cellVBOffsets, params,
														   nonEmptyCellNumber, maxVerts, convFunctor);
   } else {
      FUNCTOR_BLENDED_CONVOLUTION blendedConvFunctor;
      generateTriangles<<<grid, NTHREADS>>>( positions, normals,
                                             compactedCells, cellVBOffsets, params,
														   nonEmptyCellNumber, maxVerts, blendedConvFunctor);
   }    
}

#endif
