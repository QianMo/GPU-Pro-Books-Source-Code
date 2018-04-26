#include "DXUT.h"
#include "BIHBuilder.h"
#include <algorithm>

BIHBuilder::BIHBuilder(
		unsigned short* indexArray,
		unsigned int faceStart,
		unsigned int faceCount,
		BYTE* vertexArray,
		unsigned int vertexByteSize,
		unsigned int positionByteOffset,
		unsigned int normalByteOffset,
		void* nodeTableMemory,
		void* triangleMemory
		)
{
	this->indexArray = indexArray;
	this->faceStart = faceStart;
	this->faceCount = faceCount;
	this->vertexArray = vertexArray;
	this->vertexByteSize = vertexByteSize;
	this->positionByteOffset = positionByteOffset;
	this->normalByteOffset = normalByteOffset;

	kdNodeTableData = (KDNode*)nodeTableMemory;
	kdTriangleTableData = (D3DXVECTOR3*)triangleMemory;

	buildTree();
}

void BIHBuilder::buildTree()
{
	faces = new Face[faceCount];
	bbMin = D3DXVECTOR3(FLT_MAX, FLT_MAX, FLT_MAX);
	bbMax = D3DXVECTOR3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	for(unsigned int i=0; i<faceCount; i++)
	{
		unsigned int ia = indexArray[i*3];
		unsigned int ib = indexArray[i*3+1];
		unsigned int ic = indexArray[i*3+2];
		D3DXVec3Minimize(&bbMin, &bbMin, (D3DXVECTOR3*)(vertexArray + vertexByteSize * ia + positionByteOffset));
		D3DXVec3Minimize(&bbMin, &bbMin, (D3DXVECTOR3*)(vertexArray + vertexByteSize * ib + positionByteOffset));
		D3DXVec3Minimize(&bbMin, &bbMin, (D3DXVECTOR3*)(vertexArray + vertexByteSize * ic + positionByteOffset));
		D3DXVec3Maximize(&bbMax, &bbMax, (D3DXVECTOR3*)(vertexArray + vertexByteSize * ia + positionByteOffset));
		D3DXVec3Maximize(&bbMax, &bbMax, (D3DXVECTOR3*)(vertexArray + vertexByteSize * ib + positionByteOffset));
		D3DXVec3Maximize(&bbMax, &bbMax, (D3DXVECTOR3*)(vertexArray + vertexByteSize * ic + positionByteOffset));
	}

	bbExt = bbMax - bbMin;
	lengthPerUnitizedBit = bbExt / SHRT_MAX;

	D3DXMATRIX translationMatrix, scaleMatrix;
	D3DXMatrixTranslation(&translationMatrix, -bbMin.x, -bbMin.y, -bbMin.z);
	D3DXMatrixScaling(&scaleMatrix, 1.0 / bbExt.x, 1.0 / bbExt.y, 1.0 / bbExt.z);
	D3DXMatrixMultiply(&unitizer, &translationMatrix, &scaleMatrix);
	D3DXMatrixInverse(&deunitizer, NULL, &unitizer);

	for(unsigned int i=0; i<faceCount; i++)
	{
		unsigned int ia = indexArray[i*3];
		unsigned int ib = indexArray[i*3+1];
		unsigned int ic = indexArray[i*3+2];
		D3DXVECTOR3& pa = *(D3DXVECTOR3*)(vertexArray + vertexByteSize * ia + positionByteOffset);
		D3DXVECTOR3& pb = *(D3DXVECTOR3*)(vertexArray + vertexByteSize * ib + positionByteOffset);
		D3DXVECTOR3& pc = *(D3DXVECTOR3*)(vertexArray + vertexByteSize * ic + positionByteOffset);
		D3DXVECTOR3 upa;
		D3DXVec3TransformCoord(&upa, &pa, &unitizer);
		faces[i].v[0] = upa;
		faces[i].fullprecision[0] = upa;
		D3DXVECTOR3 upb;
		D3DXVec3TransformCoord(&upb, &pb, &unitizer);
		faces[i].v[1] = upb;
		faces[i].fullprecision[1] = upb;
		D3DXVECTOR3 upc;
		D3DXVec3TransformCoord(&upc, &pc, &unitizer);
		faces[i].v[2] = upc;
		faces[i].fullprecision[2] = upc;
		faces[i].updateExtents();

		D3DXVECTOR3 ttt;
		D3DXVec3Cross( &ttt, &(upa - upb), &(upa - upc));
		if(D3DXVec3Length(&ttt) < 0.0000001)
			bool breki = true;

		D3DXVECTOR3& na = *(D3DXVECTOR3*)(vertexArray + vertexByteSize * ia + normalByteOffset);
		D3DXVECTOR3& nb = *(D3DXVECTOR3*)(vertexArray + vertexByteSize * ib + normalByteOffset);
		D3DXVECTOR3& nc = *(D3DXVECTOR3*)(vertexArray + vertexByteSize * ic + normalByteOffset);

		unsigned char ina[4], inb[4], inc[4];
		for(int cf=0; cf<3; cf++)
		{
			ina[cf] = na[cf] * 127 + 128;
			inb[cf] = nb[cf] * 127 + 128;
			inc[cf] = nc[cf] * 127 + 128;
		}
		ina[3] = inb[3] = inc[3] = 0;

		faces[i].normals[0] = *(float*)ina;
		faces[i].normals[1] = *(float*)inb;
		faces[i].normals[2] = *(float*)inc;
	}

	sortedArrays[0] = new FaceIndex[faceCount];
	sortedArrays[1] = new FaceIndex[faceCount];
	sortedArrays[2] = new FaceIndex[faceCount];

	for (unsigned int i = 0; i < faceCount; i++)
	{
		sortedArrays[0][i] = sortedArrays[1][i] = sortedArrays[2][i] = i;
	}

	currentDepth = 0;
	FaceIndex::faces = faces;
	buildLevel(0, 0, faceCount, ShortVector.ZERO, ShortVector.ONE);
	writeTriangles();
}

void BIHBuilder::buildLevel(unsigned short cellNode, unsigned int cellFaceStart, unsigned int cellFaceCount, const ShortVector& cellMin, const ShortVector& cellMax)
{
	currentDepth++;
	ShortVector cellSize = cellMax - cellMin;
	D3DXVECTOR3 cellExt = cellSize.deunitize(lengthPerUnitizedBit);

	for(int iAxis=0; iAxis<3; iAxis++)
	{
		FaceIndex::sortAxis = iAxis;
		std::sort(sortedArrays[iAxis] + cellFaceStart, sortedArrays[iAxis] + cellFaceStart + cellFaceCount);
	}

	float bestCost = FLT_MAX;
	unsigned int bestLeftCellFaceCount = 0;
	unsigned short bestAxis = 0;

	ShortVector bestLeftCellMin = ShortVector.ONE;
	ShortVector bestLeftCellMax = ShortVector.ZERO;
	ShortVector bestRightCellMin = ShortVector.ONE;
	ShortVector bestRightCellMax = ShortVector.ZERO;

	for(int iAxis=0; iAxis<3; iAxis++)
	{
		float baseCost = cellExt[(iAxis+1)%3] * cellExt[(iAxis+2)%3] * 4.0f * cellFaceCount;
		float widthCost = (cellExt[(iAxis+1)%3]	+ cellExt[(iAxis+2)%3]) * 2.0f;

		ShortVector leftCellMin = faces[sortedArrays[iAxis][cellFaceStart].index].min; // left cell limits if cut at iFace
		ShortVector leftCellMax = faces[sortedArrays[iAxis][cellFaceStart].index].max;
		ShortVector rightCellMin = cellMax; // right cell limit accumulator for faces after best
		ShortVector rightCellMax = cellMin;
		for (unsigned int iFace = 1; iFace < cellFaceCount; iFace++)
		{
			unsigned int faceId = sortedArrays[iAxis][cellFaceStart + iFace].index;
			unsigned short firstMinInRightCell = faces[faceId].min[iAxis];
			float cost = baseCost + widthCost *
				(
				(cellMax[iAxis] - firstMinInRightCell) * (cellFaceCount - iFace)
				+ (leftCellMax[iAxis] - cellMin[iAxis]) * iFace
				) * lengthPerUnitizedBit[iAxis];
			if (cost < bestCost)
			{
				bestCost = cost;
				bestLeftCellMin = leftCellMin;
				bestLeftCellMax = leftCellMax;
				// clear right cell
				rightCellMin = cellMax;
				rightCellMax = cellMin;
				
				bestLeftCellFaceCount = iFace;
				bestAxis = iAxis;
			}
			// add face to right cell
			rightCellMin <= faces[faceId].min;
			rightCellMax >= faces[faceId].max;
			// add face to left cell
			leftCellMin <= faces[faceId].min;
			leftCellMax >= faces[faceId].max;
		}
		if(bestAxis == iAxis)
		{
			bestRightCellMin = rightCellMin;
			bestRightCellMax = rightCellMax;
		}
	}

	if(currentDepth < 13 && 
		bestCost < (cellFaceCount / 16) * 2.0f * cellFaceCount * (cellExt.x * cellExt.y + cellExt.x * cellExt.z + cellExt.y * cellExt.z) )
	{
		// make node
		kdNodeTableData[cellNode].cuts[0] = bestLeftCellMin[bestAxis];
		if(kdNodeTableData[cellNode].cuts[0] == 0)
			kdNodeTableData[cellNode].cuts[0] = 1;
		kdNodeTableData[cellNode].cuts[1] = bestLeftCellMax[bestAxis];
		kdNodeTableData[cellNode].cuts[2] = bestRightCellMin[bestAxis];
		if(kdNodeTableData[cellNode].cuts[2] == 0)
			kdNodeTableData[cellNode].cuts[2] = 1;
		kdNodeTableData[cellNode].cuts[3] = bestRightCellMax[bestAxis];
		kdNodeTableData[cellNode].cuts[bestAxis] *= -1;
		// copy arrays
		for(unsigned int iCopy=cellFaceStart; iCopy < cellFaceStart + cellFaceCount; iCopy++)
		{
			FaceIndex f = sortedArrays[bestAxis][iCopy];
			for(int iAxis=0; iAxis<3; iAxis++)
				sortedArrays[iAxis][iCopy] = f;
		}

		buildLevel(cellNode * 2 + 1, cellFaceStart, bestLeftCellFaceCount, bestLeftCellMin, bestLeftCellMax);
		buildLevel(cellNode * 2 + 2, cellFaceStart + bestLeftCellFaceCount, cellFaceCount - bestLeftCellFaceCount, bestRightCellMin, bestRightCellMax);
	}
	else
	{
		// make leaf
		//kdNodeTableData[cellNode].cuts[0] = (cellFaceStart % 8192) * 4;
		//kdNodeTableData[cellNode].cuts[1] = cellFaceCount * 4;
		//kdNodeTableData[cellNode].cuts[2] = (cellFaceStart / 8192) * 4;
		kdNodeTableData[cellNode].cuts[0] = cellFaceStart * 4;
		kdNodeTableData[cellNode].cuts[1] = (cellFaceStart + cellFaceCount) * 4;
		kdNodeTableData[cellNode].cuts[2] = 0;
		kdNodeTableData[cellNode].cuts[3] = -10000;
	}
	currentDepth--;
}

void BIHBuilder::writeTriangles()
{
	for(unsigned int iFace = 0; iFace< faceCount; iFace++)
	{
//		if(iFace >= 60000)
//			break;
		unsigned int faceId = sortedArrays[0][iFace].index;
//		unsigned int faceId = iFace;
		Face& face = faces[faceId];
		D3DXVECTOR3 a = face.v[0].asFloatVector();
		D3DXVECTOR3 b = face.v[1].asFloatVector();
		D3DXVECTOR3 c = face.v[2].asFloatVector();
		//D3DXVECTOR3 a = face.fullprecision[0];
		//D3DXVECTOR3 b = face.fullprecision[1];
		//D3DXVECTOR3 c = face.fullprecision[2];

		D3DXVECTOR3 planeNormal;
		D3DXVec3Cross(&planeNormal, &(b-a), &(c-a));
		D3DXVec3Normalize(&planeNormal, &planeNormal);
		float planeOffset = D3DXVec3Dot(&planeNormal, &a);

		float bestDistance = 0;
		D3DXVECTOR3 bestRefencePoint = D3DXVECTOR3( 0.5f, 0.5f, 0.5f);
		
		static const D3DXVECTOR3 referencePoints[8] = {
			D3DXVECTOR3( 0.f, 0.f, 0.f), 
			D3DXVECTOR3( 0.f, 0.f, 1.f),
			D3DXVECTOR3( 0.f, 1.f, 0.f),
			D3DXVECTOR3( 1.f, 0.f, 0.f), 
			D3DXVECTOR3( 0.f, 1.f, 1.f),
			D3DXVECTOR3( 1.f, 0.f, 1.f),
			D3DXVECTOR3( 1.f, 1.f, 0.f),
			D3DXVECTOR3( 1.f, 1.f, 1.f) };

		for(int iRef=0; iRef<8; iRef++)
		{
			float distance = abs(D3DXVec3Dot(&referencePoints[iRef], &planeNormal) - planeOffset);
			if(distance > bestDistance)
			{
				bestDistance = distance;
				bestRefencePoint = referencePoints[iRef];
			}
		}
		a -= bestRefencePoint;
		b -= bestRefencePoint;
		c -= bestRefencePoint;

		D3DXMATRIX vertexMatrix, inverseVertexMatrix;
		D3DXMatrixIdentity(&vertexMatrix);

		vertexMatrix._11 = a.x;
		vertexMatrix._21 = a.y;
		vertexMatrix._31 = a.z;
		vertexMatrix._12 = b.x;
		vertexMatrix._22 = b.y;
		vertexMatrix._32 = b.z;
		vertexMatrix._13 = c.x;
		vertexMatrix._23 = c.y;
		vertexMatrix._33 = c.z;

		D3DXMatrixInverse(&inverseVertexMatrix, NULL, &vertexMatrix);

		D3DXVECTOR3 planePos = D3DXVECTOR3(
			inverseVertexMatrix._11 + inverseVertexMatrix._21 + inverseVertexMatrix._31,
			inverseVertexMatrix._12 + inverseVertexMatrix._22 + inverseVertexMatrix._32,
			inverseVertexMatrix._13 + inverseVertexMatrix._23 + inverseVertexMatrix._33
			);
		planePos /= D3DXVec3Dot(&planePos, &planePos);
		if(planePos.x < 0 && bestRefencePoint.x < 0.5)
			bool breki = true;
		if(planePos.x > 0 && bestRefencePoint.x > 0.5)
			bool breki = true;
		if(planePos.y < 0 && bestRefencePoint.y < 0.5)
			bool breki = true;
		if(planePos.y > 0 && bestRefencePoint.y > 0.5)
			bool breki = true;
		if(planePos.z < 0 && bestRefencePoint.z < 0.5)
			bool breki = true;
		if(planePos.z > 0 && bestRefencePoint.z > 0.5)
			bool breki = true;

		kdTriangleTableData[iFace + 8192 * 0] = D3DXVECTOR3(inverseVertexMatrix._11, inverseVertexMatrix._12, inverseVertexMatrix._13);
		kdTriangleTableData[iFace + 8192 * 1] = D3DXVECTOR3(inverseVertexMatrix._21, inverseVertexMatrix._22, inverseVertexMatrix._23);
		kdTriangleTableData[iFace + 8192 * 2] = D3DXVECTOR3(inverseVertexMatrix._31, inverseVertexMatrix._32, inverseVertexMatrix._33);
		kdTriangleTableData[iFace + 8192 * 3] = face.normals;

		//kdTriangleTableData[iFace / 16 + 8192 * (0 + 4*(iFace%16))] = D3DXVECTOR3(inverseVertexMatrix._11, inverseVertexMatrix._12, inverseVertexMatrix._13);
		//kdTriangleTableData[iFace / 16 + 8192 * (1 + 4*(iFace%16))] = D3DXVECTOR3(inverseVertexMatrix._21, inverseVertexMatrix._22, inverseVertexMatrix._23);
		//kdTriangleTableData[iFace / 16 + 8192 * (2 + 4*(iFace%16))] = D3DXVECTOR3(inverseVertexMatrix._31, inverseVertexMatrix._32, inverseVertexMatrix._33);
		//kdTriangleTableData[iFace / 16 + 8192 * (3 + 4*(iFace%16))] = face.normals;

/*		D3DXVECTOR3 np = D3DXVECTOR3(
			inverseVertexMatrix._11 + inverseVertexMatrix._21 + inverseVertexMatrix._31,
			inverseVertexMatrix._12 + inverseVertexMatrix._22 + inverseVertexMatrix._32,
			inverseVertexMatrix._13 + inverseVertexMatrix._23 + inverseVertexMatrix._33
			);
		np.x *= bestRefencePoint.x - 0.5;
		np.y *= bestRefencePoint.y - 0.5;
		np.z *= bestRefencePoint.z - 0.5;
		if(np.x > 0 || np.y > 0 || np.z > 0)
			bool kamugaz = true;*/

	}
	bool bruki = true;
}

BIHBuilder::~BIHBuilder(void)
{
	delete [] faces;
	delete sortedArrays[0];
	delete sortedArrays[1];
	delete sortedArrays[2];
}

BIHBuilder::Face* BIHBuilder::FaceIndex::faces = NULL;
unsigned int BIHBuilder::FaceIndex::sortAxis = 0;