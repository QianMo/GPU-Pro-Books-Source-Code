#pragma once
#include "ShortVector.h"

class BIHBuilder
{
	unsigned int cRayTraceMesh;

	unsigned short* indexArray;
	unsigned int faceStart;
	unsigned int faceCount;
	BYTE* vertexArray;
	unsigned int vertexByteSize;
	unsigned int positionByteOffset;
	unsigned int normalByteOffset;

	struct KDNode
	{
		signed short cuts[4];
	};
	KDNode* kdNodeTableData;

	D3DXVECTOR3* kdTriangleTableData;

	struct Face
	{
		D3DXVECTOR3 fullprecision[3];
		ShortVector min;
		ShortVector max;
		ShortVector v[3];
		D3DXVECTOR3 normals;
		void updateExtents()
		{
			min = ShortVector::ONE;
			min <= v[0] <= v[1] <= v[2];
			max = ShortVector::ZERO;
			max >= v[0] >= v[1] >= v[2];
		}
	};
	Face* faces;
public:
	D3DXVECTOR3 bbMin;
	D3DXVECTOR3 bbMax;
	D3DXVECTOR3 bbExt;
	D3DXVECTOR3 lengthPerUnitizedBit;
	D3DXMATRIX unitizer;
	D3DXMATRIX deunitizer;


	class FaceIndex
	{
	public:
		static Face* faces;
		static unsigned int sortAxis;
		unsigned short index;
		bool operator<(const FaceIndex& b)
		{
			unsigned short x = index;
			unsigned short y = b.index;
			if (faces[x].min[sortAxis] < faces[y].min[sortAxis])
				return true;
			else if (faces[x].min[sortAxis] > faces[y].min[sortAxis])
				return false;
			else if (faces[x].max[sortAxis] < faces[y].max[sortAxis])
				return true;
			else
				return false;
		}
		unsigned short operator=(unsigned short s){index = s; return s;}
	};

	FaceIndex* sortedArrays[3];
	unsigned short currentDepth;

	void buildTree();

public:
	BIHBuilder(
		unsigned short* indexArray,
		unsigned int faceStart,
		unsigned int faceCount,
		BYTE* vertexArray,
		unsigned int vertexByteSize,
		unsigned int positionByteOffset,
		unsigned int normalByteOffset,
		void* nodeTableMemory,
		void* triangleMemory
		);

	~BIHBuilder(void);

	void buildLevel(unsigned short cellNode, unsigned int cellFaceStart, unsigned int cellFaceCount, const ShortVector& cellMin, const ShortVector& cellMax);
	void writeTriangles();
};

