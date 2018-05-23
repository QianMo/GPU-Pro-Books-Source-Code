/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/beMeshSerialization.h"
#include "beScene/DX11/beMesh.h"
#include <beGraphics/Any/beFormat.h>
#include <beGraphics/Any/beDevice.h>
#include <lean/io/mapped_file.h>
#include <lean/logging/errors.h>
#include <lean/logging/log.h>

namespace beScene
{

namespace
{

/// Converts the given data pointer into a chunk header pointer.
LEAN_INLINE const MeshDataChunkHeader& ToChunkHeader(const void *data)
{
	return *static_cast<const MeshDataChunkHeader*>(data);
}

/// Converts the given data pointer into a chunk header pointer.
template <class Data>
LEAN_INLINE const Data& ToData(const void *data)
{
	return *static_cast<const Data*>(data);
}

/// Gets the semantic name for the given vertex attribute.
LEAN_INLINE const char* GetAttributeSemantic(MeshVertexAttributes::T attribute)
{
	switch (attribute)
	{
	case MeshVertexAttributes::Position: return "POSITION";
	case MeshVertexAttributes::Normal: return "NORMAL";
	case MeshVertexAttributes::Color: return "COLOR";
	case MeshVertexAttributes::TexCoord: return "TEXCOORD";
	case MeshVertexAttributes::Tangent: return "TANGENT";
	case MeshVertexAttributes::BiTangent: return "BITANGENT";
	}

	LEAN_THROW_ERROR_MSG("Unknown vertex attribute");
	LEAN_ASSERT_UNREACHABLE();
}

/// Reads the vertex description.
void ReadVertexDesc(std::vector<D3D11_INPUT_ELEMENT_DESC> &vertexElementDescs, const char *vertexData, const char *vertexDataEnd)
{
	const MeshDataVertexElementDesc *elementDescs = &ToData<MeshDataVertexElementDesc>(vertexData);
	const MeshDataVertexElementDesc *elementDescsEnd = &ToData<MeshDataVertexElementDesc>(vertexDataEnd);

	for (const MeshDataVertexElementDesc *elementDesc = elementDescs; elementDesc < elementDescsEnd; ++elementDesc)
	{
		D3D11_INPUT_ELEMENT_DESC elementDescDX = { 0 };
		
		elementDescDX.SemanticName = GetAttributeSemantic( static_cast<MeshVertexAttributes::T>(elementDesc->Attribute) );
		elementDescDX.SemanticIndex = elementDesc->Index;
		elementDescDX.Format = beGraphics::Any::ToAPI( static_cast<beGraphics::Format::T>(elementDesc->Format) );

		elementDescDX.AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
		elementDescDX.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;

		vertexElementDescs.push_back(elementDescDX);
	}
}

/// Reads vertices.
void ReadVertices(std::vector<D3D11_INPUT_ELEMENT_DESC> &vertexElementDescs, uint4 &vertexCount,
	const char *&vertices, const char *&verticesEnd,
	const char *vertexData, const char *vertexDataEnd)
{
	vertexElementDescs.clear();
	vertexCount = 0;
	vertices = nullptr;
	verticesEnd = nullptr;

	for (const char *offset = vertexData; offset < vertexDataEnd; )
	{
		const MeshDataChunkHeader &header = ToChunkHeader(offset);
		const char *data = offset + sizeof(MeshDataChunkHeader);

		switch (header.ChunkID)
		{
		case MeshDataChunk::Count:
			vertexCount = ToData<uint4>(data);
			break;

		case MeshDataChunk::Desc:
			ReadVertexDesc(vertexElementDescs, data, data + header.ChunkSize);
			break;

		case MeshDataChunk::Data:
			vertices = data;
			verticesEnd = data + header.ChunkSize;
			break;
		}

		offset = data + header.ChunkSize;
	}

	if (vertexCount != 0)
	{
		if (vertices == verticesEnd)
			LEAN_THROW_ERROR_MSG("Actual vertex data missing");
		if (vertexElementDescs.empty())
			LEAN_THROW_ERROR_MSG("Vertex element description missing");
	}
	else if (vertices != verticesEnd)
		LEAN_THROW_ERROR_MSG("Vertex count missing");
}

/// Reads indices.
void ReadIndices(DXGI_FORMAT &indexFormat, uint4 &indexCount,
	const char *&indices, const char *&indicesEnd,
	const char *indexData, const char *indexDataEnd)
{
	indexFormat = DXGI_FORMAT_UNKNOWN;
	indexCount = 0;
	indices = nullptr;
	indicesEnd = nullptr;

	for (const char *offset = indexData; offset < indexDataEnd; )
	{
		const MeshDataChunkHeader &header = ToChunkHeader(offset);
		const char *data = offset + sizeof(MeshDataChunkHeader);

		switch (header.ChunkID)
		{
		case MeshDataChunk::Count:
			indexCount = ToData<uint4>(data);
			break;

		case MeshDataChunk::Desc:
			indexFormat = beGraphics::Any::ToAPI( static_cast<beGraphics::Format::T>(ToData<uint4>(data)) );
			break;

		case MeshDataChunk::Data:
			indices = data;
			indicesEnd = data + header.ChunkSize;
			break;
		}

		offset = data + header.ChunkSize;
	}

	if (indexCount != 0)
	{
		if (indices == indicesEnd)
			LEAN_THROW_ERROR_MSG("Actual index data missing");
		if (indexFormat == DXGI_FORMAT_UNKNOWN)
			LEAN_THROW_ERROR_MSG("Index format missing");
	}
	else if (indices != indicesEnd)
		LEAN_THROW_ERROR_MSG("Index count missing");
}

/// Reads a subset.
lean::resource_ptr<Mesh, true> ReadSubset(const char *subsetData, const char* subsetDataEnd, const beGraphics::Device &device, AssembledMesh *pCompound)
{
	utf8_string subsetName;

	uint4 vertexCount = 0;
	const char *vertices = nullptr, *verticesEnd = nullptr;
	std::vector<D3D11_INPUT_ELEMENT_DESC> vertexElementDescs;

	uint4 indexCount = 0;
	const char *indices = nullptr, *indicesEnd = nullptr;
	DXGI_FORMAT indexFormat = DXGI_FORMAT_UNKNOWN;

	for (const char *offset = subsetData; offset < subsetDataEnd; )
	{
		const MeshDataChunkHeader &header = ToChunkHeader(offset);
		const char *data = offset + sizeof(MeshDataChunkHeader);

		switch (header.ChunkID)
		{
		case MeshDataChunk::Name:
			subsetName.assign(data, data + header.ChunkSize);
			break;

		case MeshDataChunk::Vertices:
			ReadVertices(vertexElementDescs, vertexCount, vertices, verticesEnd, data, data + header.ChunkSize);
			break;

		case MeshDataChunk::Indices:
			ReadIndices(indexFormat, indexCount, indices, indicesEnd, data, data + header.ChunkSize);
			break;
		}

		offset = data + header.ChunkSize;
	}

	return new_resource DX11::Mesh(
			subsetName,
			&vertexElementDescs[0], static_cast<uint4>(vertexElementDescs.size()),
			static_cast<uint4>(static_cast<size_t>(verticesEnd - vertices) / vertexCount),
			vertices, vertexCount,
			indexFormat,
			indices, indexCount,
			ToImpl(device),
			pCompound
		);
}

/// Reads all subsets.
void ReadSubsets(AssembledMesh &compound, const char *subsetData, const char* subsetDataEnd, const beGraphics::Device &device, AssembledMesh *pCompound)
{
	uint4 subsetCount = 0;

	for (const char *offset = subsetData; offset < subsetDataEnd; )
	{
		const MeshDataChunkHeader &header = ToChunkHeader(offset);
		const char *data = offset + sizeof(MeshDataChunkHeader);

		switch (header.ChunkID)
		{
		case MeshDataChunk::Count:
			subsetCount = ToData<uint4>(data);
			break;

		case MeshDataChunk::Data:
			// TODO: Read LOD from somewhere?
			compound.AddMeshWithMaterial( ReadSubset(data, data + header.ChunkSize, device, pCompound).get(), nullptr, 0 );
			break;
		}

		offset = data + header.ChunkSize;
	}
}

} // namespace

// Loads a mesh compound from the given memory.
lean::resource_ptr<AssembledMesh, true> LoadMeshes(const char *meshData, uint8, beGraphics::Device &device)
{
	// Read spanning header
	const MeshDataChunkHeader &header = ToChunkHeader(meshData);
	if (header.ChunkID != MeshDataChunk::Header)
		LEAN_THROW_ERROR_MSG("Invalid mesh header (NOTE: swapped endian not supported yet)");

	meshData += sizeof(MeshDataChunkHeader);

	lean::resource_ptr<AssembledMesh> pCompound = new_resource AssembledMesh();
	utf8_string meshName;
	utf8_string meshSource;

	const char *meshDataEnd = meshData + header.ChunkSize;

	for (const char *offset = meshData; offset < meshDataEnd; )
	{
		const MeshDataChunkHeader &header = ToChunkHeader(offset);
		const char *data = offset + sizeof(MeshDataChunkHeader);

		switch (header.ChunkID)
		{
		case MeshDataChunk::Name:
			meshName.assign(data, data + header.ChunkSize);
			break;

		case MeshDataChunk::Source:
			meshSource.assign(data, data + header.ChunkSize);
			break;

		case MeshDataChunk::Subsets:
			ReadSubsets(*pCompound, data, data + header.ChunkSize, device, pCompound);
			break;
		}

		offset = data + header.ChunkSize;
	}

	return pCompound.transfer();
}

// Loads a mesh compound from the given file.
lean::resource_ptr<AssembledMesh, true> LoadMeshes(const utf8_ntri &file, beGraphics::Device &device)
{
	lean::resource_ptr<AssembledMesh, true>  pMesh;

	LEAN_LOG("Attempting to load mesh \"" << file.c_str() << "\"");

	lean::rmapped_file meshFile(file);
	pMesh = LoadMeshes(reinterpret_cast<const char*>(meshFile.data()), file.size(), device);

	LEAN_LOG("Mesh \"" << file.c_str() << "\" created successfully");

	return pMesh.transfer();
}

} // namespace