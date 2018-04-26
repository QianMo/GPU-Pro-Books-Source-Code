#pragma once

/**
 *	Pedro Hermosilla
 *	
 *	Moving Group - UPC
 *	Mesh.h
 */

#include <d3d10.h>
#include <d3dx10.h>
#include <vector>

class Mesh
{
	private:

		ID3D10Device* _d3dDevice;
		ID3D10Buffer* _vertexBuffer;
		ID3D10Buffer* _indexBuffer;
		ID3D10Buffer* _adyIndexBuffer;
		std::vector<D3D10_INPUT_ELEMENT_DESC > _vertexDescr;

		float _xMin,_yMin,_zMin;
		float _xMax,_yMax,_zMax;

		unsigned int  _vertexSize;
		unsigned int  _numVertexs;
		unsigned int  _numFaces;

		void unifyVertexs(std::vector<float>& vertexs,std::vector<unsigned int>& faces);

		void computeNormals(std::vector<float>& vertexs, std::vector<float>& normals, 
			std::vector<unsigned int>& faces);

		void computeAdy(std::vector<float>& vertexs, std::vector<float>& normals, 
			std::vector<unsigned int>& faces,std::vector<unsigned int>& adyFaces);

		void computeCurv(std::vector<float>& vertexs, std::vector<float>& normals,
			std::vector<float>& curv, std::vector<unsigned int>& faces);

		void createBuffers(std::vector<float>& vertexs, std::vector<float>& normals,
			std::vector<float>& curv, std::vector<unsigned int>& faces, 
			std::vector<unsigned int>& adyFaces);

	public:

		Mesh(ID3D10Device* d3dDevice);

		~Mesh(void);

		void load(const char* fileName);

		const std::vector<D3D10_INPUT_ELEMENT_DESC>& getVertexDescription() const;

		void render();

		void renderAdy();

		unsigned int getNumTriangles()const;

		void getaabbMin(float& x, float& y, float& z);

		void getaabbMax(float& x, float& y, float& z);
};

inline const std::vector<D3D10_INPUT_ELEMENT_DESC>& Mesh::getVertexDescription() const
{
	return _vertexDescr;
}

inline void Mesh::render()
{
	UINT aux = 0;
	unsigned int auxVertexSize = sizeof(float)*_vertexSize;
	_d3dDevice->IASetVertexBuffers( 0, 1, &_vertexBuffer, &auxVertexSize,&aux);
	_d3dDevice->IASetIndexBuffer( _indexBuffer, DXGI_FORMAT_R32_UINT, 0 );
	_d3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
}

inline void Mesh::renderAdy()
{
	UINT aux = 0;
	unsigned int auxVertexSize = sizeof(float)*_vertexSize;
	_d3dDevice->IASetVertexBuffers( 0, 1, &_vertexBuffer, &auxVertexSize,&aux);
	_d3dDevice->IASetIndexBuffer(_adyIndexBuffer, DXGI_FORMAT_R32_UINT, 0 );
	_d3dDevice->IASetPrimitiveTopology(D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ);
}

inline unsigned int Mesh::getNumTriangles()const
{
	return _numFaces;
}

inline void Mesh::getaabbMin(float& x, float& y, float& z)
{
	x = _xMin;
	y = _yMin;
	z = _zMin;
}

inline void Mesh::getaabbMax(float& x, float& y, float& z)
{
	x = _xMax;
	y = _yMax;
	z = _zMax;
}