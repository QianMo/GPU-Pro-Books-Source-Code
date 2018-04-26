#include "DXUT.h"
#include "RaytracingMesh.h"

RaytracingMesh::RaytracingMesh(ID3DX10Mesh* mesh, unsigned int index)
{
	this->mesh = mesh;
	this->index = index;
}

RaytracingMesh::~RaytracingMesh(void)
{
}

void RaytracingMesh::setMatrices(const D3DXMATRIX& unitizerMatrix, const D3DXMATRIX& deunitizerMatrix)
{
	this->unitizerMatrix = unitizerMatrix;
	this->deunitizerMatrix = deunitizerMatrix;
}

D3DXVECTOR4 RaytracingMesh::getRaytracingDiffuseBrdfParameter()
{
	return D3DXVECTOR4(1, 1, 1, 1);
}

D3DXVECTOR4 RaytracingMesh::getRaytracingSpecularBrdfParameter()
{
	return D3DXVECTOR4(0, 0, 0, 1);
}
