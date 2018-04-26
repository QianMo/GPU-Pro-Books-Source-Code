// this one mixes CUDA code and has some GL functionality to map CUDA objects
// but does not do any rendering, only mesh extraction

#include <stdlib.h>
#include <assert.h>

#include "../gl/glUtils.h"

#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <vector_functions.h>
#include "cudpp/cudpp.h"

#include "../struct.h"

#include "../functions.h"

#include "evaluator.h"

// allocates memory on the GPU and copies some data there
bool FREP_EVALUATOR::init()
{
	m_maxVerts = GRID_SIZE_MAX * GRID_SIZE_MAX * 100;
	m_maxCells = GRID_SIZE_MAX * GRID_SIZE_MAX * GRID_SIZE_MAX;

	printf("max verts = %d\n", m_maxVerts);   

	int size = GRID_SIZE_MAX * GRID_SIZE_MAX * GRID_SIZE_MAX * sizeof(uint);

	cutilSafeCall(cudaMalloc((void**) &d_volume, size));

	bindVolumeTexture(d_volume);

	createVBO(&m_posistionVBO, m_maxVerts*sizeof(float)*4);
	cutilSafeCall(cudaGLRegisterBufferObject(m_posistionVBO));

	createVBO(&m_normalsVBO, m_maxVerts*sizeof(float)*4);
	cutilSafeCall(cudaGLRegisterBufferObject(m_normalsVBO));

	// allocate textures for MC tables
	allocateTextures(	&d_edgeTable, &d_triTable, &d_numVertsTable );

	// allocate device memory
	unsigned int memSize = sizeof(uint) * m_maxCells;
	cutilSafeCall(cudaMalloc((void**) &d_cellVerts, memSize));
	cutilSafeCall(cudaMalloc((void**) &d_cellVBOffsets, memSize));
	cutilSafeCall(cudaMalloc((void**) &d_cellsNonEmpty, memSize));
	cutilSafeCall(cudaMalloc((void**) &d_cellsNonEmptyScan, memSize));
	cutilSafeCall(cudaMalloc((void**) &d_compactedCells, memSize)); 

	return true;
}

// free GPU resources etc
void FREP_EVALUATOR::clean()
{
	cutilSafeCall(cudaGLUnregisterBufferObject(m_posistionVBO));
	cutilSafeCall(cudaGLUnregisterBufferObject(m_normalsVBO));

	deleteVBO(&m_posistionVBO);
	deleteVBO(&m_normalsVBO);   

	cutilSafeCall(cudaFree(d_volume));

	cutilSafeCall(cudaFree(d_edgeTable));
	cutilSafeCall(cudaFree(d_triTable));
	cutilSafeCall(cudaFree(d_numVertsTable));

	cutilSafeCall(cudaFree(d_cellVerts));
	cutilSafeCall(cudaFree(d_cellVBOffsets));
	cutilSafeCall(cudaFree(d_cellsNonEmpty));
	cutilSafeCall(cudaFree(d_cellsNonEmptyScan));
	cutilSafeCall(cudaFree(d_compactedCells));

	cudppDestroyPlan(m_scanPlanExclusive);   
}

// core function, performing function evaluation and isosurface extraction
void FREP_EVALUATOR::evaluate( FREP_DESCRIPTION& model, int resolutionMode, int currentFrame )
{
	int volumeThreads = 128;   

   // update polygonization structures according to current resolution
	calcPolygonizationParams( resolutionMode, &model.polygonizationParams );

	int numCells = model.polygonizationParams.numCells;

   // update prefix sum structures as well
	updatePlan(numCells, &m_scanPlanExclusive);

	dim3 volumeGrid( numCells / volumeThreads, 1, 1);
	// get around maximum grid size of 65535 in each dimension
	if (volumeGrid.x > 65535) {
		volumeGrid.y = volumeGrid.x / 32768;
		volumeGrid.x = 32768;
	}

	currentFrame = (currentFrame ) % (int)model.sampledModel.size();

	const MODEL_PARAMETERS& currentParams = model.sampledModel[currentFrame];

	BOUNDING_BOX volumeBound = model.polygonizationParams.volumeBox;

   // upload all necessary model parameters to GPU
	bool result = copyParamsToDevice(&currentParams.segments[0], (int)currentParams.segments.size(), model.convolutionIsoValue, model.polygonizationParams.volumeBox, currentParams.submodelParams);

	if (result == false) {      
		assert(0);
		return;
	}

   // perform all function evaluations and save the results 
   // OUTPUT: "d_volume" - function values across the volume
	launch_writeVolume(	volumeGrid, volumeThreads, d_volume, model.polygonizationParams, model.isBlendingOn);

	int cellThreads = 128;   
	dim3 grid(numCells / cellThreads, 1, 1);
	// get around maximum grid size of 65535 in each dimension
	if (grid.x > 65535) {
		grid.y = grid.x / 32768;
		grid.x = 32768;
	}

	// output number of vertices that need to be generated for current cell and output flag indicating whether 
   // current cell contains any triangles at all 
   // OUTPUT: "d_cellVerts" - number of vertices, "d_cellsNonEmpty" - empty/non-empty flags
   launch_preprocessCells( grid, cellThreads, d_cellVerts, d_cellsNonEmpty, model.polygonizationParams );

	// scan non-empty cells 
   // OUTPUT: "d_cellsNonEmptyScan" - each cell contains the number of non-empty cells preceding it
	cudppScan(m_scanPlanExclusive, d_cellsNonEmptyScan, d_cellsNonEmpty, numCells);      

	// read back values to calculate total number of non-empty cells
	// It is exclusive scan, thus the total number is the last value of
	// the scan result plus the last value in the input array   
	uint lastCellIsEmpty, nonEmptyCellNumber;

	cutilSafeCall(cudaMemcpy((void *) &nonEmptyCellNumber, 
		(void *) (d_cellsNonEmpty + numCells-1), 
		sizeof(uint), cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaMemcpy((void *) &lastCellIsEmpty, 
		(void *) (d_cellsNonEmptyScan + numCells-1), 
		sizeof(uint), cudaMemcpyDeviceToHost));

	nonEmptyCellNumber += lastCellIsEmpty;   

	if (nonEmptyCellNumber==0) {
		// all cells are empty
		m_vertexNumber = 0;
		return;
	}

	// filter out cells which do not contain any triangles
   // OUTPUT: "d_compactedCells" - array of indices of all non-empty cells
	launch_compactCells(grid, cellThreads, d_compactedCells, d_cellsNonEmpty, d_cellsNonEmptyScan, numCells);
	cutilCheckMsg("compactCells failed");

	// now d_compactedCells contains cell indices of only non-empty cells

	// scan cell vertex offset array
   // OUTPUT: "d_cellVBOffsets" - each element of the array contains the offset in the vertex buffer 
   // of the first vertex it produces
	cudppScan(m_scanPlanExclusive, d_cellVBOffsets, d_cellVerts, numCells);

	uint allVertexNumber, lastCellVertexNumber;

	// last element of scanned array is the number of vertices
	// in all previous cells
	cutilSafeCall( cudaMemcpy( (void *) &allVertexNumber, 
		(void *) (d_cellVBOffsets + numCells-1), 
		sizeof(uint), cudaMemcpyDeviceToHost));
	// get number of vertices in the last cell (exclusive scan did not consider it)
	cutilSafeCall( cudaMemcpy( (void *) &lastCellVertexNumber,
		(void *) (d_cellVerts + numCells-1), 
		sizeof(uint), cudaMemcpyDeviceToHost));

   // the numbe of vertices that will be extracted for this isosurface
	m_vertexNumber = allVertexNumber + lastCellVertexNumber;

	// now map vertex and normal buffers to fill them from the kernel
	float4 *d_pos = 0, *d_normal = 0;
	cutilSafeCall(cudaGLMapBufferObject((void**)&d_pos, m_posistionVBO));
	cutilSafeCall(cudaGLMapBufferObject((void**)&d_normal, m_normalsVBO));

	dim3 grid2((int) ceil(nonEmptyCellNumber / (float) NTHREADS), 1, 1);

	while(grid2.x > 65535) {
		grid2.x /= 2;
		grid2.y *= 2;
	}

	// finally generate vertices and normals of the isosurface
   // OUTPUT: "d_pos" - vertices of the isosurface, "d_normal" - normals of the isosurface
	launch_generateTriangles(  grid2, NTHREADS, d_pos, d_normal, d_compactedCells, d_cellVBOffsets,
		                        model.polygonizationParams, nonEmptyCellNumber, m_maxVerts, model.isBlendingOn);

	cutilSafeCall(cudaGLUnmapBufferObject(m_normalsVBO));
	cutilSafeCall(cudaGLUnmapBufferObject(m_posistionVBO));
}

// update polygonization structures according to current resolution
bool FREP_EVALUATOR::calcPolygonizationParams(int resolutionIndex, POLYGONIZATION_PARAMS* params)
{   
	// 2^4 = 16 is the minimum
	int resolutionLog = resolutionIndex + 4;   

	int newResolution = 1 << (resolutionLog);

	if (!params || newResolution > GRID_SIZE_MAX) {
		assert(0);
		return false;
	}

	params->gridSize = make_uint3(newResolution, newResolution, newResolution);
	params->gridSizeMask = make_uint3(newResolution - 1, newResolution - 1, newResolution - 1);
	params->gridSizeShift = make_uint3(0, resolutionLog, resolutionLog + resolutionLog);

	params->numCells = newResolution * newResolution * newResolution;

	BOUNDING_BOX& volumeBox = params->volumeBox;

	float3 volumeBound;

	volumeBound.x =  volumeBox.maxX - volumeBox.minX;
	volumeBound.y =  volumeBox.maxY - volumeBox.minY;
	volumeBound.z =  volumeBox.maxZ - volumeBox.minZ; 

	params->cellSize = make_float3(  volumeBound.x / float(params->gridSize.x-1), 
		volumeBound.y / float(params->gridSize.y-1), 
		volumeBound.z / float(params->gridSize.z-1) );

	params->threshold = 0.f;  

	return true;
}

// update prefix sum structures according to current resolution
bool FREP_EVALUATOR::updatePlan(int currentMaxCells, CUDPPHandle* handle)
{
	if (!handle) {
		assert(0);
		return false;
	}

	// initialize CUDPP scan
	CUDPPConfiguration config;
	config.algorithm = CUDPP_SCAN;
	config.datatype = CUDPP_UINT;
	config.op = CUDPP_ADD;
	config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;

	cudppPlan(handle, config, currentMaxCells, 1, 0);

	return true;
}

const GEOMETRY_DESC& FREP_EVALUATOR::getGeometryDesc()
{
   static GEOMETRY_DESC geometryDesc;

   geometryDesc.positions  =  m_posistionVBO;
   geometryDesc.normals    =  m_normalsVBO;
   geometryDesc.numElemets =  m_vertexNumber;
   

   return geometryDesc;

}
