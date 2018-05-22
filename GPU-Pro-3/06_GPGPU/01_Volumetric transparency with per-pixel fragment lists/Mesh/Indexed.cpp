#include "DXUT.h"
#include "Mesh/Indexed.h"
#include "ThrowOnFail.h"

const unsigned short Mesh::IndexBufferDesc::defaultData[6] = { 0, 1, 2, 3, 2, 1};


Mesh::Indexed::Indexed(ID3D11Device* device, IndexBufferDesc& desc, VertexStream::A& vertexStreams)
{
	this->nVertexBuffers = vertexStreams.size();
	this->vertexBuffers = new ID3D11Buffer*[nVertexBuffers];
	this->vertexStrides = new unsigned int[nVertexBuffers];

	for(int iVertexBuffer=0; iVertexBuffer < nVertexBuffers; iVertexBuffer++)
	{
		this->vertexBuffers[iVertexBuffer] = vertexStreams.at(iVertexBuffer)->getBuffer();
		this->vertexBuffers[iVertexBuffer]->AddRef();
		this->vertexStrides[iVertexBuffer] = vertexStreams.at(iVertexBuffer)->getStride();
	}

	nElements = 0;
	for(int iVertexBuffer=0; iVertexBuffer<nVertexBuffers; iVertexBuffer++)
	{
		nElements += vertexStreams.at(iVertexBuffer)->getElementCount();
	}
	D3D11_INPUT_ELEMENT_DESC* elements = new D3D11_INPUT_ELEMENT_DESC[nElements];
	nElements = 0;
	for(int iVertexBuffer=0; iVertexBuffer<nVertexBuffers; iVertexBuffer++)
	{
		const D3D11_INPUT_ELEMENT_DESC* vbElements;
		unsigned int vbnElements;
		vertexStreams.at(iVertexBuffer)->getElements(vbElements, vbnElements);
		for(int iElement=0; iElement<vbnElements; iElement++, nElements++)
		{
			elements[nElements] = vbElements[iElement];
			elements[nElements].InputSlot = iVertexBuffer;
			char* semanticName = new char[strlen(vbElements[iElement].SemanticName)+1];
			strcpy(semanticName, vbElements[iElement].SemanticName);
			elements[nElements].SemanticName = semanticName;
		}
	}
	this->elements = elements;

	this->nPrimitives = desc.nPrimitives;
	this->indexFormat = desc.indexFormat;
	this->topology = desc.topology;

	switch(topology)
	{
	case D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED: throw HrException(E_INVALIDARG, "Index buffer primitive topology undefined.", __FILE__, __LINE__);
	case D3D11_PRIMITIVE_TOPOLOGY_POINTLIST: this->nIndices = nPrimitives; break;
	case D3D11_PRIMITIVE_TOPOLOGY_LINELIST: this->nIndices = nPrimitives*2; break;
	case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP: this->nIndices = nPrimitives+1; break; // WARNING
	case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST: this->nIndices = nPrimitives*3; break;
	case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP: this->nIndices = nPrimitives+2; break; // WARNING
	case D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ: this->nIndices = nPrimitives*4; break;
	case D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ: this->nIndices = nPrimitives+3; break;	 // WARNING
	case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ: this->nIndices = nPrimitives*6; break;
	case D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ: this->nIndices = nPrimitives*2+4; break; // WARNING
	case D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives; break;
	case D3D11_PRIMITIVE_TOPOLOGY_2_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*2; break;
	case D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*3; break;
	case D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*4; break;
	case D3D11_PRIMITIVE_TOPOLOGY_5_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*5; break;
	case D3D11_PRIMITIVE_TOPOLOGY_6_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*6; break;
	case D3D11_PRIMITIVE_TOPOLOGY_7_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*7; break;
	case D3D11_PRIMITIVE_TOPOLOGY_8_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*8; break;
	case D3D11_PRIMITIVE_TOPOLOGY_9_CONTROL_POINT_PATCHLIST	: this->nIndices = nPrimitives*9; break;
	case D3D11_PRIMITIVE_TOPOLOGY_10_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*10; break;
	case D3D11_PRIMITIVE_TOPOLOGY_11_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*11; break;
	case D3D11_PRIMITIVE_TOPOLOGY_12_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*12; break;
	case D3D11_PRIMITIVE_TOPOLOGY_13_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*13; break;
	case D3D11_PRIMITIVE_TOPOLOGY_14_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*14; break;
	case D3D11_PRIMITIVE_TOPOLOGY_15_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*15; break;
	case D3D11_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*16; break;
	case D3D11_PRIMITIVE_TOPOLOGY_17_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*17; break;
	case D3D11_PRIMITIVE_TOPOLOGY_18_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*18; break;
	case D3D11_PRIMITIVE_TOPOLOGY_19_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*19; break;
	case D3D11_PRIMITIVE_TOPOLOGY_20_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*20; break;
	case D3D11_PRIMITIVE_TOPOLOGY_21_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*21; break;
	case D3D11_PRIMITIVE_TOPOLOGY_22_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*22; break;
	case D3D11_PRIMITIVE_TOPOLOGY_23_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*23; break;
	case D3D11_PRIMITIVE_TOPOLOGY_24_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*24; break;
	case D3D11_PRIMITIVE_TOPOLOGY_25_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*25; break;
	case D3D11_PRIMITIVE_TOPOLOGY_26_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*26; break;
	case D3D11_PRIMITIVE_TOPOLOGY_27_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*27; break;
	case D3D11_PRIMITIVE_TOPOLOGY_28_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*28; break;
	case D3D11_PRIMITIVE_TOPOLOGY_29_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*29; break;
	case D3D11_PRIMITIVE_TOPOLOGY_30_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*30; break;
	case D3D11_PRIMITIVE_TOPOLOGY_31_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*31; break;
	case D3D11_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST: this->nIndices = nPrimitives*32; break;
	}
	
	if(desc.nIndices != 0)
	{
		if(desc.nIndices != this->nIndices
			&& topology != D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP
			&& topology != D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ
			&& topology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP
			&& topology != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ)
			throw HrException(E_INVALIDARG, "Index count is not consistent with topology and primitive count.", __FILE__, __LINE__);
		this->nIndices = desc.nIndices;
	}

	D3D11_BUFFER_DESC indexBufferDesc;
	indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
	indexBufferDesc.ByteWidth = nIndices * desc.getIndexStride();
	indexBufferDesc.CPUAccessFlags = 0;
	indexBufferDesc.MiscFlags = 0;
	indexBufferDesc.StructureByteStride = 0;
	indexBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;

	D3D11_SUBRESOURCE_DATA initialIndexData;
	initialIndexData.pSysMem = desc.indexData;

	ThrowOnFail("Could not create index buffer.", __FILE__, __LINE__) = 
		device->CreateBuffer(&indexBufferDesc, &initialIndexData, &indexBuffer);

}


Mesh::Indexed::~Indexed(void)
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


void Mesh::Indexed::getElements(const D3D11_INPUT_ELEMENT_DESC*& elements, unsigned int& nElements)
{
	elements = this->elements;
	nElements = this->nElements;
}

void Mesh::Indexed::draw(ID3D11DeviceContext* context)
{
	context->IASetPrimitiveTopology(topology);
	context->IASetIndexBuffer(indexBuffer, indexFormat, 0);

	static const unsigned int zeros[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	context->IASetVertexBuffers(0, nVertexBuffers, vertexBuffers, vertexStrides, zeros);
	context->DrawIndexed(nIndices, 0, 0);
}
