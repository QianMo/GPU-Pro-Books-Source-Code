#include "DXUT.h"
#include "Mesh/Importer.h"
#include "Mesh/VertexStream.h"
#include "Mesh/Indexed.h"
#include "Mesh/Instanced.h"
#include "ThrowOnFail.h"
#include <assimp.hpp>      // C++ importer interface
#include <aiScene.h>       // Output data structure
#include <aiPostProcess.h> // Post processing flags

Mesh::Geometry::P Mesh::Importer::fromAiMesh(ID3D11Device* device, aiMesh* assMesh)
{
	D3D11_INPUT_ELEMENT_DESC elements[64];
	unsigned int cElements = 0;
	unsigned int cOffset = 0;
	unsigned int positionOffset = 0;
	unsigned int normalOffset = 0;
	unsigned int tangentOffset = 0;
	unsigned int binormalOffset = 0;
	unsigned int texcoord0Offset = 0;

	if(assMesh->HasPositions())
	{
		elements[cElements].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		elements[cElements].AlignedByteOffset = positionOffset = cOffset;
		cOffset += sizeof(float) * 3;
		elements[cElements].InputSlot = 0;
		elements[cElements].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		elements[cElements].InstanceDataStepRate = 0;
		elements[cElements].SemanticIndex = 0;
//		const char* semanticNameLiteral = "POSITION";
//		char* semanticName = new char[strlen(semanticNameLiteral)+1];
//		strcpy(semanticName, semanticNameLiteral);
		elements[cElements].SemanticName = "POSITION";//semanticName;
		cElements++;
	}
	if(assMesh->HasNormals())
	{
		elements[cElements].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		elements[cElements].AlignedByteOffset = normalOffset = cOffset;
		cOffset += sizeof(float) * 3;
		elements[cElements].InputSlot = 0;
		elements[cElements].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		elements[cElements].InstanceDataStepRate = 0;
		elements[cElements].SemanticIndex = 0;
//		const char* semanticNameLiteral = "NORMAL";
//		char* semanticName = new char[strlen(semanticNameLiteral)+1];
//		strcpy(semanticName, semanticNameLiteral);
//		elements[cElements].SemanticName = semanticName;
		elements[cElements].SemanticName = "NORMAL";
		cElements++;
	}
	if(assMesh->HasTangentsAndBitangents())
	{
		elements[cElements].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		elements[cElements].AlignedByteOffset = tangentOffset = cOffset;
		cOffset += sizeof(float) * 3;
		elements[cElements].InputSlot = 0;
		elements[cElements].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		elements[cElements].InstanceDataStepRate = 0;
		elements[cElements].SemanticIndex = 0;
//		const char* semanticNameLiteral = "TANGENT";
//		char* semanticName = new char[strlen(semanticNameLiteral)+1];
//		strcpy(semanticName, semanticNameLiteral);
//		elements[cElements].SemanticName = semanticName;
		elements[cElements].SemanticName = "TANGENT";
		cElements++;
	}
	if(assMesh->HasTangentsAndBitangents())
	{
		elements[cElements].Format = DXGI_FORMAT_R32G32B32_FLOAT;
		elements[cElements].AlignedByteOffset = binormalOffset = cOffset;
		cOffset += sizeof(float) * 3;
		elements[cElements].InputSlot = 0;
		elements[cElements].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		elements[cElements].InstanceDataStepRate = 0;
		elements[cElements].SemanticIndex = 0;
//		const char* semanticNameLiteral = "BINORMAL";
//		char* semanticName = new char[strlen(semanticNameLiteral)+1];
//		strcpy(semanticName, semanticNameLiteral);
//		elements[cElements].SemanticName = semanticName;
		elements[cElements].SemanticName = "BINORMAL";
		cElements++;
	}
	if(assMesh->HasTextureCoords(0))
	{
		elements[cElements].Format = DXGI_FORMAT_R32G32_FLOAT;
		elements[cElements].AlignedByteOffset = texcoord0Offset = cOffset;
		cOffset += sizeof(float) * 2;
		elements[cElements].InputSlot = 0;
		elements[cElements].InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
		elements[cElements].InstanceDataStepRate = 0;
		elements[cElements].SemanticIndex = 0;
//		const char* semanticNameLiteral = "TEXCOORD";
//		char* semanticName = new char[strlen(semanticNameLiteral)+1];
//		strcpy(semanticName, semanticNameLiteral);
//		elements[cElements].SemanticName = semanticName;
		elements[cElements].SemanticName = "TEXCOORD";
		cElements++;
	}
	unsigned int vertexStride = cOffset;
	unsigned int nElements = cElements;
	unsigned int nVertices = assMesh->mNumVertices;

	char* sysMemVertices = new char[nVertices * vertexStride];

	for(unsigned int iVertex=0; iVertex < assMesh->mNumVertices; iVertex++)
	{
		memcpy(sysMemVertices + iVertex * vertexStride + positionOffset, &assMesh->mVertices[iVertex], sizeof(float) * 3);
		if(assMesh->HasNormals())
			memcpy(sysMemVertices + iVertex * vertexStride + normalOffset, &assMesh->mNormals[iVertex], sizeof(float) * 3);
		if(assMesh->HasTangentsAndBitangents())
		{
			memcpy(sysMemVertices + iVertex * vertexStride + tangentOffset, &assMesh->mTangents[iVertex], sizeof(float) * 3);
			memcpy(sysMemVertices + iVertex * vertexStride + binormalOffset, &assMesh->mBitangents[iVertex], sizeof(float) * 3);
		}
		if(assMesh->HasTextureCoords(0))
			memcpy(sysMemVertices + iVertex * vertexStride + texcoord0Offset, &assMesh->mTextureCoords[0][iVertex], sizeof(float) * 2);
	}
	
	Mesh::VertexStreamDesc vertexStreamDesc;
	vertexStreamDesc.elements = elements;
	vertexStreamDesc.nElements = nElements;
	vertexStreamDesc.nVertices = nVertices;
	vertexStreamDesc.vertexData = sysMemVertices;
	vertexStreamDesc.vertexStride = vertexStride;

	Mesh::VertexStream::P vertexStream = Mesh::VertexStream::make(device, vertexStreamDesc);

	unsigned int nPrimitives = assMesh->mNumFaces;
	bool wideIndexBuffer = nVertices > USHRT_MAX;

	Mesh::IndexBufferDesc indexBufferDesc;
	indexBufferDesc.nIndices = nPrimitives * 3;
	indexBufferDesc.nPrimitives = nPrimitives;
	indexBufferDesc.topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

	if(wideIndexBuffer)
	{
		unsigned int* sysMemIndices = new unsigned int[nPrimitives*3];
		for(int iFace=0; iFace < nPrimitives; iFace++)
		{
			sysMemIndices[iFace * 3 + 0] = assMesh->mFaces[iFace].mIndices[0];
			sysMemIndices[iFace * 3 + 1] = assMesh->mFaces[iFace].mIndices[1];
			sysMemIndices[iFace * 3 + 2] = assMesh->mFaces[iFace].mIndices[2];
		}
		indexBufferDesc.indexData = sysMemIndices;
		indexBufferDesc.indexFormat = DXGI_FORMAT_R32_UINT;
	}
	else
	{
		unsigned short* sysMemIndices = new unsigned short[nPrimitives*3];
		for(int iFace=0; iFace < nPrimitives; iFace++)
		{
			sysMemIndices[iFace * 3 + 0] = assMesh->mFaces[iFace].mIndices[0];
			sysMemIndices[iFace * 3 + 1] = assMesh->mFaces[iFace].mIndices[1];
			sysMemIndices[iFace * 3 + 2] = assMesh->mFaces[iFace].mIndices[2];
		}
		indexBufferDesc.indexData = sysMemIndices;
		indexBufferDesc.indexFormat = DXGI_FORMAT_R16_UINT;
	}

	Mesh::Indexed::P indexed = Mesh::Indexed::make(device, indexBufferDesc, vertexStream);

	delete vertexStreamDesc.vertexData;
	delete indexBufferDesc.indexData;

	return indexed;
}
