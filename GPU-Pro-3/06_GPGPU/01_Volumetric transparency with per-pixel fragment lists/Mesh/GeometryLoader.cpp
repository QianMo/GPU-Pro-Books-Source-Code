#include "DXUT.h"
#include "Mesh/GeometryLoader.h"
#include "Mesh/VertexStream.h"
#include "Mesh/Indexed.h"
#include "Mesh/Instanced.h"
#include "ThrowOnFail.h"

#include <fstream>


Mesh::Geometry::P Mesh::GeometryLoader::createGeometryFromFile(ID3D11Device* device, const char* filename)
{
    // Open the file
    HANDLE file = CreateFileA(filename, FILE_READ_DATA, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    if( INVALID_HANDLE_VALUE == file )
        throw HrException(DXUTERR_MEDIANOTFOUND, "File not found.", __FILE__, __LINE__);

    // Get the file size
    LARGE_INTEGER fileSize;
    GetFileSizeEx( file, &fileSize );
    UINT cBytes = fileSize.LowPart;

    // Allocate memory
    BYTE* meshData = new BYTE[ cBytes ];
    if( !meshData )
    {
        CloseHandle( file );
        throw HrException(E_OUTOFMEMORY, "Out of memory.", __FILE__, __LINE__);
    }

    // Read in the file
    DWORD dwBytesRead;
    if( !ReadFile( file, meshData, cBytes, &dwBytesRead, NULL ) )
	{
	    CloseHandle( file );
		throw HrException(E_FAIL, "File access error.", __FILE__, __LINE__);
	}

	CloseHandle( file );

	Mesh::Geometry::P geometry = createGeometryFromMemory(  device,
                                meshData,
                                cBytes);
	delete [] meshData;
    return geometry;
}

Mesh::Geometry::P Mesh::GeometryLoader::createGeometryFromMemory(ID3D11Device* device, BYTE* data, unsigned int nBytes)
{	
	BYTE* p = data;

	unsigned int gbsVersion = *(unsigned int*)p;
	if(gbsVersion == 0x1) // barebone .gbs
	{
		p += sizeof(unsigned int); // skip version info
		D3D11_INPUT_ELEMENT_DESC elements[20];

		unsigned int vbHeader = *(unsigned int*)p;
		p += sizeof(unsigned int);
		if(vbHeader != 0x004e77e3)
		{ MessageBox(NULL, L"Expected a D3D11 vertex buffer.", L"Error processing mesh file!", MB_OK); exit(-1); }

		unsigned int vertexStride = 0;
		unsigned int nElements = 0;
		while(*p)
		{
			CopyMemory((void*)&elements[nElements], (void*)p, sizeof(D3D11_INPUT_ELEMENT_DESC));
			p += sizeof(D3D11_INPUT_ELEMENT_DESC);
			unsigned int semanticNameLength = strlen( (char*)p) + 1;
			char* semanticName = new char[semanticNameLength];
			strcpy(semanticName, (char*)p);
			p += semanticNameLength;
			elements[nElements].SemanticName = semanticName;

			switch(elements[nElements].Format)
			{
				case DXGI_FORMAT_R32_SINT :
				case DXGI_FORMAT_R32_UINT :
				case DXGI_FORMAT_R32_FLOAT : vertexStride += sizeof(float); break;
				case DXGI_FORMAT_R32G32_SINT:
				case DXGI_FORMAT_R32G32_UINT:
				case DXGI_FORMAT_R32G32_FLOAT : vertexStride += 2 * sizeof(float); break;
				case DXGI_FORMAT_R32G32B32_SINT:
				case DXGI_FORMAT_R32G32B32_UINT:
				case DXGI_FORMAT_R32G32B32_FLOAT : vertexStride += 3 * sizeof(float); break;
				case DXGI_FORMAT_R32G32B32A32_SINT:
				case DXGI_FORMAT_R32G32B32A32_UINT:
				case DXGI_FORMAT_R32G32B32A32_FLOAT : vertexStride += 4 * sizeof(float); break;
			};

			nElements++;
		}
		p++;

		unsigned int inFileVertexStride = *(unsigned int*)p;
		p += sizeof(unsigned int);
		if(inFileVertexStride < vertexStride)
		{ MessageBox(NULL, L"Vertex stride cannot accomodate elements.", L"Error processing mesh file!", MB_OK); exit(-1); }
		vertexStride = inFileVertexStride;

		unsigned int nVertices = *(unsigned int*)p;
		p += sizeof(unsigned int);

		void* vertexData = p;
		p += vertexStride * nVertices;

		unsigned int ibHeader = *(unsigned int*)p;
		p += sizeof(unsigned int);
		if(ibHeader != 0x00010de3)
		{ MessageBox(NULL, L"Expected a D3D11 index buffer.", L"Error processing mesh file!", MB_OK); exit(-1); }

		bool wideIndexBuffer = *(bool*)p;
		p += sizeof(bool);

		if(nVertices >= 0xffff && !wideIndexBuffer)
		{ MessageBox(NULL, L"More vertices than indexable.", L"Error processing mesh file!", MB_OK); exit(-1); }

		D3D11_PRIMITIVE_TOPOLOGY topology = *(D3D11_PRIMITIVE_TOPOLOGY*)p;
		p += sizeof(D3D11_PRIMITIVE_TOPOLOGY);

		unsigned int nPrimitives = *(unsigned int*)p;
		p += sizeof(unsigned int);

		void* indexData = p;
		Mesh::VertexStreamDesc vertexStreamDesc;
		vertexStreamDesc.elements = elements;
		vertexStreamDesc.nElements = nElements;
		vertexStreamDesc.nVertices = nVertices;
		vertexStreamDesc.vertexData = vertexData;
		vertexStreamDesc.vertexStride = vertexStride;

		Mesh::VertexStream::P vertexStream = Mesh::VertexStream::make(device, vertexStreamDesc);
		Mesh::IndexBufferDesc indexedDesc;
		indexedDesc.topology = topology;
		indexedDesc.indexData = indexData;
		indexedDesc.indexFormat = wideIndexBuffer?DXGI_FORMAT_R32_UINT:DXGI_FORMAT_R16_UINT;
		indexedDesc.nPrimitives = nPrimitives;
		indexedDesc.nIndices = 0;
		Mesh::Geometry::P indexed = Mesh::Indexed::make(device, indexedDesc, vertexStream);

		for(int iElements=0; iElements<nElements; iElements++)
			delete elements[iElements].SemanticName;
		return indexed;
	}

	 // primitve .dgb
	D3D11_INPUT_ELEMENT_DESC elements[20];

	unsigned int vertexStride = 0;
	unsigned int nElements = 0;
	while(*p)
	{
		CopyMemory((void*)&elements[nElements], (void*)p, sizeof(D3D11_INPUT_ELEMENT_DESC));
		p += sizeof(D3D11_INPUT_ELEMENT_DESC);
		unsigned int semanticNameLength = strlen( (char*)p) + 1;
		char* semanticName = new char[semanticNameLength];
		strcpy(semanticName, (char*)p);
		p += semanticNameLength;
		elements[nElements].SemanticName = semanticName;

		switch(elements[nElements].Format)
		{
			case DXGI_FORMAT_R32_SINT :
			case DXGI_FORMAT_R32_UINT :
			case DXGI_FORMAT_R32_FLOAT : vertexStride += sizeof(float); break;
			case DXGI_FORMAT_R32G32_SINT:
			case DXGI_FORMAT_R32G32_UINT:
			case DXGI_FORMAT_R32G32_FLOAT : vertexStride += 2 * sizeof(float); break;
			case DXGI_FORMAT_R32G32B32_SINT:
			case DXGI_FORMAT_R32G32B32_UINT:
			case DXGI_FORMAT_R32G32B32_FLOAT : vertexStride += 3 * sizeof(float); break;
			case DXGI_FORMAT_R32G32B32A32_SINT:
			case DXGI_FORMAT_R32G32B32A32_UINT:
			case DXGI_FORMAT_R32G32B32A32_FLOAT : vertexStride += 4 * sizeof(float); break;
		};

		nElements++;
	}
	p++;

	unsigned int nVertices = *(unsigned int*)p;
	p += sizeof(unsigned int);
	unsigned int nPrimitives = *(unsigned int*)p;
	p += sizeof(unsigned int);

	bool wideIndexBuffer = false;
	if(nVertices >= 0xffff)
		wideIndexBuffer = true;

	void* vertexData = p;
	void* indexData = p + vertexStride * nVertices;

	Mesh::VertexStreamDesc vertexStreamDesc;
	vertexStreamDesc.elements = elements;
	vertexStreamDesc.nElements = nElements;
	vertexStreamDesc.nVertices = nVertices;
	vertexStreamDesc.vertexData = vertexData;
	vertexStreamDesc.vertexStride = vertexStride;
	Mesh::VertexStream::P vertexStream = Mesh::VertexStream::make(device, vertexStreamDesc);
	Mesh::IndexBufferDesc indexedDesc;
	indexedDesc.topology = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	indexedDesc.indexData = indexData;
	indexedDesc.indexFormat = wideIndexBuffer?DXGI_FORMAT_R32_UINT:DXGI_FORMAT_R16_UINT;
	indexedDesc.nPrimitives = nPrimitives;
	indexedDesc.nIndices = 0;
	Mesh::Geometry::P indexed = Mesh::Indexed::make(device, indexedDesc, vertexStream);

	for(int iElements=0; iElements<nElements; iElements++)
		delete elements[iElements].SemanticName;
	return indexed;
}

