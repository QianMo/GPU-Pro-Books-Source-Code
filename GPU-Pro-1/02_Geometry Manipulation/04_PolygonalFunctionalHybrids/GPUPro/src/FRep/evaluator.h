#ifndef __FREP_EVALUATOR_H__
#define __FREP_EVALUATOR_H__

#include <cudpp/cudpp.h>

#include "../struct.h"

// performs evaluation on the GPU, extracts isosurface
class FREP_EVALUATOR {
public:

	static const unsigned int GRID_SIZE_MAX = 128;

	enum GEOMETRY_TYPE {
		VERTEX_DATA,
		NORMAL_DATA
	};

	FREP_EVALUATOR() :m_maxVerts(0), m_maxCells(0), m_vertexNumber(0), m_threshold(0.f),
		               d_volume(NULL), d_cellVerts(NULL), d_cellVBOffsets(NULL), 
		               d_cellsNonEmpty(NULL), d_cellsNonEmptyScan(NULL), d_compactedCells(NULL),   
		               d_numVertsTable(NULL), d_edgeTable(NULL), d_triTable(NULL) {}

	// allocates memory on the GPU and copies some data there
	bool	init ();
	// free GPU resources etc
	void	clean();

	// core function, performing function evaluation and isosurface extraction
	void evaluate(FREP_DESCRIPTION& model, int resolutionMode, int currentFrame);

	//unsigned int getGeometryData(GEOMETRY_TYPE type) { return type == VERTEX_DATA ? m_posistionVBO : m_normalsVBO; }

   const GEOMETRY_DESC& getGeometryDesc();

	//int getVertexNumber() const { return m_vertexNumber; }

private:

	// update polygonization structures according to current resolution
   bool calcPolygonizationParams(int resolutionIndex, POLYGONIZATION_PARAMS* params);

   // update prefix sum structures according to current resolution
	bool updatePlan(int currentMaxCells, CUDPPHandle* handle);

	// upper bound for worst case scenario
	uint  m_maxVerts;
	uint  m_maxCells;

	uint	m_vertexNumber;
	float	m_threshold;

	CUDPPHandle m_scanPlanExclusive;

	// handles of buffers where the extracted mesh will be output
	unsigned int m_posistionVBO, m_normalsVBO;

	// handles of some variables on the GPU side

	float	*d_volume; // volume array where sampled field values are stored
	uint	*d_cellVerts; // number of vertices contained in each cell

	// arrays used for the processing of cells and mesh extraction:
	uint	*d_cellVBOffsets, *d_cellsNonEmpty, *d_cellsNonEmptyScan, *d_compactedCells;

	// MC tables
	uint	*d_numVertsTable, *d_edgeTable, *d_triTable;
};

#endif // __FREP_EVALUATOR_H__