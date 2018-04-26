#pragma once

class RaytracingMesh
{
	unsigned int index;
	D3DXMATRIX unitizerMatrix;
	D3DXMATRIX deunitizerMatrix;

	ID3DX10Mesh* mesh;
public:
	RaytracingMesh(ID3DX10Mesh* mesh, unsigned int index);
	~RaytracingMesh(void);
	ID3DX10Mesh* getMesh(){return mesh;}
	void setMatrices(const D3DXMATRIX& unitizerMatrix, const D3DXMATRIX& deunitizerMatrix);

	unsigned int getIndex(){return index;}
	D3DXVECTOR4 getRaytracingDiffuseBrdfParameter();
	D3DXVECTOR4 getRaytracingSpecularBrdfParameter();
	const D3DXMATRIX& getUnitizerMatrix(){return unitizerMatrix;}

};
