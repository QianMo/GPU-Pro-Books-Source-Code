#include "DXUT.h"
#include "Instanced.h"
#include "ThrowOnFail.h"


const D3D11_INPUT_ELEMENT_DESC Mesh::InstanceBufferDesc::defaultElements[3] = {
	{ "INSTANCEPOSE", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 0, D3D11_INPUT_PER_INSTANCE_DATA, 1},
	{ "INSTANCEPOSE", 1, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, 4*sizeof(float), D3D11_INPUT_PER_INSTANCE_DATA, 1},
	{ "INSTANCEOFFSET", 0, DXGI_FORMAT_R32G32B32A32_UINT, 1, 8*sizeof(float), D3D11_INPUT_PER_INSTANCE_DATA, 1},
};
const D3DXVECTOR4 Mesh::InstanceBufferDesc::defaultData[3] = { D3DXVECTOR4(0, 0, 0, 1), D3DXVECTOR4(0, 0, 0, 1), D3DXVECTOR4(0, 0, 0, 0)};


Mesh::Instanced::Instanced(ID3D11Device* device, unsigned int nInstances, InstanceBufferDesc* desc, unsigned int nInstanceBuffers, Mesh::Indexed::P indexed)
{
	this->nInstances = nInstances;

	nIndexedVertexBuffers = indexed->getVertexBufferCount();
	this->nVertexBuffers = indexed->getVertexBufferCount() + nInstanceBuffers;
	this->vertexBuffers = new ID3D11Buffer*[nVertexBuffers];
	this->vertexStrides = new unsigned int[nVertexBuffers];

	ID3D11Buffer** indexedVertexBuffers = indexed->getVertexBuffers();
	unsigned int* indexedVertexStrides = indexed->getVertexStrides();

	for(int iVertexBuffer=0; iVertexBuffer < indexed->getVertexBufferCount(); iVertexBuffer++)
	{
		this->vertexBuffers[iVertexBuffer] = indexedVertexBuffers[iVertexBuffer];
		this->vertexBuffers[iVertexBuffer]->AddRef();
		this->vertexStrides[iVertexBuffer] = indexedVertexStrides[iVertexBuffer];
	}
	for(int iInstanceBuffer=0; iInstanceBuffer < nInstanceBuffers; iInstanceBuffer++)
	{
		D3D11_BUFFER_DESC instanceBufferDesc;
		instanceBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
		instanceBufferDesc.ByteWidth = nInstances * desc[iInstanceBuffer].instanceStride;
		instanceBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		instanceBufferDesc.MiscFlags = 0;
		instanceBufferDesc.StructureByteStride = desc[iInstanceBuffer].instanceStride;
		instanceBufferDesc.Usage = D3D11_USAGE_DYNAMIC;

		D3D11_SUBRESOURCE_DATA initialInstanceData;
		initialInstanceData.pSysMem = desc[iInstanceBuffer].instanceData;

		ID3D11Buffer* instanceBuffer;
		ThrowOnFail("Could not create instance buffer.", __FILE__, __LINE__) = 
			device->CreateBuffer(&instanceBufferDesc, &initialInstanceData, &instanceBuffer);

		vertexBuffers[iInstanceBuffer + indexed->getVertexBufferCount()] = instanceBuffer;
		vertexStrides[iInstanceBuffer + indexed->getVertexBufferCount()] = desc[iInstanceBuffer].instanceStride;
	}

	const D3D11_INPUT_ELEMENT_DESC* indexedElements;
	unsigned int indexedElementCount;
	indexed->getElements(indexedElements, indexedElementCount);

	nElements = indexedElementCount;
	for(int iInstanceBuffer=0; iInstanceBuffer < nInstanceBuffers; iInstanceBuffer++)
	{
		nElements += desc[iInstanceBuffer].nElements;
	}
	D3D11_INPUT_ELEMENT_DESC* elements = new D3D11_INPUT_ELEMENT_DESC[nElements];

	nElements=0;
	for(int iElement=0; iElement<indexedElementCount; iElement++, nElements++)
	{
		elements[iElement] = indexedElements[iElement];
		char* semanticName = new char[strlen(indexedElements[iElement].SemanticName)+1];
		strcpy(semanticName, indexedElements[iElement].SemanticName);
		elements[iElement].SemanticName = semanticName;
	}
	for(int iInstanceBuffer=0; iInstanceBuffer<nInstanceBuffers; iInstanceBuffer++)
	{
		for(int iElement=0; iElement< desc[iInstanceBuffer].nElements; iElement++, nElements++)
		{
			elements[nElements] = desc[iInstanceBuffer].elements[iElement];
			elements[nElements].InputSlot = iInstanceBuffer + indexed->getVertexBufferCount();
			char* semanticName = new char[strlen(desc[iInstanceBuffer].elements[iElement].SemanticName)+1];
			strcpy(semanticName, desc[iInstanceBuffer].elements[iElement].SemanticName);
			elements[nElements].SemanticName = semanticName;
		}
	}
	this->elements = elements;

	this->nPrimitives = indexed->getPrimitiveCount();
	this->indexFormat = indexed->getIndexFormat();
	this->topology = indexed->getTopology();
	this->nIndices = indexed->getIndexCount();
	this->indexBuffer = indexed->getIndexBuffer();
	indexBuffer->AddRef();

}


Mesh::Instanced::~Instanced(void)
{
	for(int iVertexBuffer=0; iVertexBuffer < nVertexBuffers; iVertexBuffer++)
		vertexBuffers[iVertexBuffer]->Release();
	delete [] vertexBuffers;
	delete [] vertexStrides;
	if(indexBuffer)
		indexBuffer->Release();
	for(int i=0; i<nElements; i++)
		if(elements[i].SemanticName)
			delete elements[i].SemanticName;
	delete [] elements;
}

void Mesh::Instanced::getElements(const D3D11_INPUT_ELEMENT_DESC*& elements, unsigned int& nElements)
{
	elements = this->elements;
	nElements = this->nElements;
}

void Mesh::Instanced::draw(ID3D11DeviceContext* context)
{
	context->IASetPrimitiveTopology(topology);
	context->IASetIndexBuffer(indexBuffer, indexFormat, 0);
	static const unsigned int zeros[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	context->IASetVertexBuffers(0, nVertexBuffers, vertexBuffers, vertexStrides, zeros);
	context->DrawIndexedInstanced(nIndices, nInstances, 0, 0, 0);
}
