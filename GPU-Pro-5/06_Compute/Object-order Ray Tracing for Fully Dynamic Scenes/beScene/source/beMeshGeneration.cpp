/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#include "beSceneInternal/stdafx.h"
#include "beScene/DX11/beMeshGeneration.h"
#include "beScene/DX11/beMesh.h"
#include <beGraphics/Any/beFormat.h>
#include <beGraphics/Any/beDevice.h>

#include <beMath/beVector.h>

namespace beScene
{

namespace DX11
{

/// Computes a vertex description from the given mesh generation flags.
beCore::Exchange::vector_t<D3D11_INPUT_ELEMENT_DESC>::t ComputeVertexDesc(uint4 meshGenFlags)
{
	beCore::Exchange::vector_t<D3D11_INPUT_ELEMENT_DESC>::t vertexDesc;
	vertexDesc.reserve(5);

	D3D11_INPUT_ELEMENT_DESC defaultDesc;
	defaultDesc.SemanticIndex = 0;
	defaultDesc.InputSlot = 0;
	defaultDesc.AlignedByteOffset = D3D11_APPEND_ALIGNED_ELEMENT;
	defaultDesc.InputSlotClass = D3D11_INPUT_PER_VERTEX_DATA;
	defaultDesc.InstanceDataStepRate = 0;

	// Position
	{
		D3D11_INPUT_ELEMENT_DESC posDesc(defaultDesc);
		posDesc.SemanticName = "POSITION";
		posDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		vertexDesc.push_back(posDesc);
	}

	// Normals
	if (meshGenFlags & MeshGenFlags::Normal)
	{
		D3D11_INPUT_ELEMENT_DESC normalDesc(defaultDesc);
		normalDesc.SemanticName = "NORMAL";
		normalDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		vertexDesc.push_back(normalDesc);
	}

	// Tex coords
	if (meshGenFlags & MeshGenFlags::TexCoord)
	{
		D3D11_INPUT_ELEMENT_DESC texCoordDesc(defaultDesc);
		texCoordDesc.SemanticName = "TEXCOORD";
		texCoordDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
		vertexDesc.push_back(texCoordDesc);
	}

	// Tangent
	if (meshGenFlags & MeshGenFlags::Tangent)
	{
		D3D11_INPUT_ELEMENT_DESC tangentDesc(defaultDesc);
		tangentDesc.SemanticName = "TANGENT";
		tangentDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		vertexDesc.push_back(tangentDesc);
	}

	// Bi-Tangent
	if (meshGenFlags & MeshGenFlags::BiTangent)
	{
		D3D11_INPUT_ELEMENT_DESC biTangentDesc(defaultDesc);
		biTangentDesc.SemanticName = "BINORMAL";
		biTangentDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
		vertexDesc.push_back(biTangentDesc);
	}

	return vertexDesc;
}

/// Computes the index format from the given vertex count.
DXGI_FORMAT ComputeIndexFormat(uint4 vertexCount, uint4 meshGenFlags)
{
	return (vertexCount > static_cast<uint2>(-1)) || (meshGenFlags & MeshGenFlags::WideIndex)
		? DXGI_FORMAT_R32_UINT
		: DXGI_FORMAT_R16_UINT;
}

} // namespace

// Computes the size of one vertex.
uint4 ComputeVertexSize(uint4 meshGenFlags)
{
	uint4 size = sizeof(float) * 3;

	if (meshGenFlags & MeshGenFlags::Normal)
		size += sizeof(float) * 3;
	if (meshGenFlags & MeshGenFlags::TexCoord)
		size += sizeof(float) * 2;
	if (meshGenFlags & MeshGenFlags::Tangent)
		size += sizeof(float) * 3;
	if (meshGenFlags & MeshGenFlags::BiTangent)
		size += sizeof(float) * 3;

	return size;
}

// Computes the size of one index.
uint4 ComputeIndexSize(uint4 vertexCount, uint4 meshGenFlags)
{
	return (vertexCount > static_cast<uint2>(-1)) || (meshGenFlags & MeshGenFlags::WideIndex)
		? sizeof(uint4)
		: sizeof(uint2);
}

// Computes the number of vertices in a regular grid of the given size.
uint4 ComputeGridVertexCount(uint4 cellsU, uint4 cellsV)
{
	return (cellsU + 1) * (cellsV + 1);
}

// Computes the number of indices in a regular grid of the given size.
uint4 ComputeGridIndexCount(uint4 cellsU, uint4 cellsV)
{
	return 6 * cellsU * cellsV;
}

// Generates a regular grid of the given dimensions.
void GenerateGrid(void *vertices, void *indices,
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV,
	uint4 meshGenFlags,
	uint4 meshBaseIndex)
{
	const size_t vertexCount = ComputeGridVertexCount(cellsU, cellsV);
	const size_t indexCount = ComputeGridIndexCount(cellsU, cellsV);

	const size_t vertexSize = ComputeVertexSize(meshGenFlags);
	const size_t indexSize = ComputeIndexSize(vertexCount, meshGenFlags);

	const uint4 vertexIndexPitch = cellsU + 1;
	const size_t vertexPitch = vertexIndexPitch * vertexSize;

	const float oneOverCellsU = 1.0f / cellsU;
	const float oneOverCellsV = 1.0f / cellsV;

	size_t vertexDataOffset = 0;

	// Positions
	{
		char *vertexPositions = static_cast<char*>(vertices) + vertexDataOffset;
		vertexDataOffset += sizeof(float) * 3;

		for (uint4 v = 0; v <= cellsV; ++v)
			for (uint4 u = 0; u <= cellsU; ++u)
			{
				memcpy(vertexPositions,
					( b + x * (u * oneOverCellsU) + y * (v * oneOverCellsV) ).cdata(),
					sizeof(float) * 3);

				vertexPositions += vertexSize;
			}
	}

	// Normals
	if (meshGenFlags & MeshGenFlags::Normal)
	{
		const beMath::fvec3 normal = normalize( -cross(x, y) );

		char *vertexNormals = static_cast<char*>(vertices) + vertexDataOffset;
		vertexDataOffset += sizeof(float) * 3;

		for (uint4 v = 0; v <= cellsV; ++v)
			for (uint4 u = 0; u <= cellsU; ++u)
			{
				memcpy(vertexNormals, normal.cdata(), sizeof(float) * 3);

				vertexNormals += vertexSize;
			}
	}

	// Tex coords
	if (meshGenFlags & MeshGenFlags::TexCoord)
	{
		char *vertexTexCoords = static_cast<char*>(vertices) + vertexDataOffset;
		vertexDataOffset += sizeof(float) * 2;

		for (uint4 v = 0; v <= cellsV; ++v)
			for (uint4 u = 0; u <= cellsU; ++u)
			{
				memcpy(
					vertexTexCoords,
					( beMath::vec(u0, v0) + beMath::vec(du, dv) * beMath::vec(u * oneOverCellsU, 1 - v * oneOverCellsV) ).cdata(),
					sizeof(float) * 2);

				vertexTexCoords += vertexSize;
			}
	}

	// Tangent
	if (meshGenFlags & MeshGenFlags::Tangent)
	{
		const beMath::fvec3 tangent = normalize(x);

		char *vertexTangents = static_cast<char*>(vertices) + vertexDataOffset;
		vertexDataOffset += sizeof(float) * 3;

		for (uint4 v = 0; v <= cellsV; ++v)
			for (uint4 u = 0; u <= cellsU; ++u)
			{
				memcpy(vertexTangents, tangent.cdata(), sizeof(float) * 3);

				vertexTangents += vertexSize;
			}
	}

	// Bi-Tangent
	if (meshGenFlags & MeshGenFlags::BiTangent)
	{
		const beMath::fvec3 biTangent = normalize(-y);

		char *vertexBiTangents = static_cast<char*>(vertices) + vertexDataOffset;
		vertexDataOffset += sizeof(float) * 3;

		for (uint4 v = 0; v <= cellsV; ++v)
			for (uint4 u = 0; u <= cellsU; ++u)
			{
				memcpy(vertexBiTangents, biTangent.cdata(), sizeof(float) * 3);

				vertexBiTangents += vertexSize;
			}
	}

	// Indices
	if (indexSize == sizeof(uint2))
	{
		uint2 *shortIndices = static_cast<uint2*>(indices);

		for (uint4 v = 0; v < cellsV; ++v)
			for (uint4 u = 0; u < cellsU; ++u)
			{
				uint4 baseIndex = meshBaseIndex + v * vertexIndexPitch + u;
				uint4 topIndex = baseIndex + vertexIndexPitch;
				uint4 rightIndex = baseIndex + 1;
				uint4 topRightIndex = topIndex + 1;

				*shortIndices++ = static_cast<uint2>(baseIndex);
				*shortIndices++ = static_cast<uint2>(topIndex);
				*shortIndices++ = static_cast<uint2>(rightIndex);
				*shortIndices++ = static_cast<uint2>(rightIndex);
				*shortIndices++ = static_cast<uint2>(topIndex);
				*shortIndices++ = static_cast<uint2>(topRightIndex);
			}
	}
	else if (indexSize == sizeof(uint4))
	{
		uint4 *longIndices = static_cast<uint4*>(indices);

		for (uint4 v = 0; v < cellsV; ++v)
			for (uint4 u = 0; u < cellsU; ++u)
			{
				uint4 baseIndex = meshBaseIndex + v * vertexIndexPitch + u;
				uint4 topIndex = baseIndex + vertexIndexPitch;
				uint4 rightIndex = baseIndex + 1;
				uint4 topRightIndex = topIndex + 1;

				*longIndices++ = baseIndex;
				*longIndices++ = topIndex;
				*longIndices++ = rightIndex;
				*longIndices++ = rightIndex;
				*longIndices++ = topIndex;
				*longIndices++ = topRightIndex;
			}
	}
	else
		LEAN_ASSERT_UNREACHABLE();
}

// Generates a regular grid mesh of the given dimensions.
lean::resource_ptr<Mesh, true> GenerateGridMesh(
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV,
	uint4 meshGenFlags,
	const beGraphics::Device &device)
{
	const size_t vertexCount = ComputeGridVertexCount(cellsU, cellsV);
	const size_t indexCount = ComputeGridIndexCount(cellsU, cellsV);

	const size_t vertexSize = ComputeVertexSize(meshGenFlags);
	const size_t indexSize = ComputeIndexSize(vertexCount, meshGenFlags);

	std::vector<char> vertexMemory(vertexCount * vertexSize);
	std::vector<char> indexMemory(indexCount * indexSize);

	GenerateGrid(
		&vertexMemory[0],
		&indexMemory[0],
		b, x, y,
		u0, v0, du, dv,
		cellsU, cellsV,
		meshGenFlags);

	const beCore::Exchange::vector_t<D3D11_INPUT_ELEMENT_DESC>::t vertexDesc = DX11::ComputeVertexDesc(meshGenFlags);
	const DXGI_FORMAT indexFormat = DX11::ComputeIndexFormat(vertexCount, meshGenFlags);

	return new_resource DX11::Mesh(
			"Grid",
			&vertexDesc[0], vertexDesc.size(),
			vertexSize, &vertexMemory[0], vertexCount,
			indexFormat, &indexMemory[0], indexCount,
			ToImpl(device)
		);
}

// Computes the number of vertices in a regular cuboid of the given size.
uint4 ComputeCuboidVertexCount(uint4 cellsU, uint4 cellsV, uint4 cellsW)
{
	return 2 * ComputeGridVertexCount(cellsU, cellsV)
		+ 2 * ComputeGridVertexCount(cellsV, cellsW)
		+ 2 * ComputeGridVertexCount(cellsW, cellsU);
}

// Computes the number of indices in a regular cuboid of the given size.
uint4 ComputeCuboidIndexCount(uint4 cellsU, uint4 cellsV, uint4 cellsW)
{
	return 2 * ComputeGridIndexCount(cellsU, cellsV)
		+ 2 * ComputeGridIndexCount(cellsV, cellsW)
		+ 2 * ComputeGridIndexCount(cellsW, cellsU);
}

// Generates a regular cuboid of the given dimensions.
void GenerateCuboid(void *vertices, void *indices,
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y, const beMath::fvec3 &z,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV, uint4 cellsW,
	uint4 meshGenFlags)
{
	const size_t vertexCount = ComputeCuboidVertexCount(cellsU, cellsV, cellsW);
	const size_t indexCount = ComputeCuboidIndexCount(cellsU, cellsV, cellsW);

	const size_t vertexSize = ComputeVertexSize(meshGenFlags);
	const size_t indexSize = ComputeIndexSize(vertexCount, meshGenFlags);

	// Make sure wide indices are used when total vertex count > MAX_SHORT, but grid vertex count < MAX_SHORT
	if (indexSize != sizeof(uint2))
		meshGenFlags = MeshGenFlags::WideIndex;

	char *vertexBase = static_cast<char*>(vertices);
	char *indexBase = static_cast<char*>(indices);
	uint4 baseIndex = 0;

	GenerateGrid(vertexBase, indexBase, b, x, y, u0, v0, du, dv, cellsU, cellsV, meshGenFlags, baseIndex);
	vertexBase += vertexSize * ComputeGridVertexCount(cellsU, cellsV);
	baseIndex += ComputeGridVertexCount(cellsU, cellsV);
	indexBase += indexSize * ComputeGridIndexCount(cellsU, cellsV);

	GenerateGrid(vertexBase, indexBase, b + x, z, y, u0, v0, du, dv, cellsW, cellsV, meshGenFlags, baseIndex);
	vertexBase += vertexSize * ComputeGridVertexCount(cellsW, cellsV);
	baseIndex += ComputeGridVertexCount(cellsW, cellsV);
	indexBase += indexSize * ComputeGridIndexCount(cellsW, cellsV);

	GenerateGrid(vertexBase, indexBase, b + y, x, z, u0, v0, du, dv, cellsU, cellsW, meshGenFlags, baseIndex);
	vertexBase += vertexSize * ComputeGridVertexCount(cellsU, cellsW);
	baseIndex += ComputeGridVertexCount(cellsU, cellsW);
	indexBase += indexSize * ComputeGridIndexCount(cellsU, cellsW);

	GenerateGrid(vertexBase, indexBase, b + x + z, -x, y, u0, v0, du, dv, cellsU, cellsV, meshGenFlags, baseIndex);
	vertexBase += vertexSize * ComputeGridVertexCount(cellsU, cellsV);
	baseIndex += ComputeGridVertexCount(cellsU, cellsV);
	indexBase += indexSize * ComputeGridIndexCount(cellsU, cellsV);

	GenerateGrid(vertexBase, indexBase, b + z, -z, y, u0, v0, du, dv, cellsW, cellsV, meshGenFlags, baseIndex);
	vertexBase += vertexSize * ComputeGridVertexCount(cellsW, cellsV);
	baseIndex += ComputeGridVertexCount(cellsW, cellsV);
	indexBase += indexSize * ComputeGridIndexCount(cellsW, cellsV);

	GenerateGrid(vertexBase, indexBase, b + z, x, -z, u0, v0, du, dv, cellsU, cellsW, meshGenFlags, baseIndex);
	vertexBase += vertexSize * ComputeGridVertexCount(cellsU, cellsW);
	baseIndex += ComputeGridVertexCount(cellsU, cellsW);
	indexBase += indexSize * ComputeGridIndexCount(cellsU, cellsW);
}

// Generates a regular cuboid mesh of the given dimensions.
lean::resource_ptr<Mesh, true> GenerateCuboidMesh(
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y, const beMath::fvec3 &z,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV, uint4 cellsW,
	uint4 meshGenFlags,
	const beGraphics::Device &device)
{
	const size_t vertexCount = ComputeCuboidVertexCount(cellsU, cellsV, cellsW);
	const size_t indexCount = ComputeCuboidIndexCount(cellsU, cellsV, cellsW);

	const size_t vertexSize = ComputeVertexSize(meshGenFlags);
	const size_t indexSize = ComputeIndexSize(vertexCount, meshGenFlags);

	std::vector<char> vertexMemory(vertexCount * vertexSize);
	std::vector<char> indexMemory(indexCount * indexSize);

	GenerateCuboid(
		&vertexMemory[0],
		&indexMemory[0],
		b, x, y, z,
		u0, v0, du, dv,
		cellsU, cellsV, cellsW,
		meshGenFlags);

	const beCore::Exchange::vector_t<D3D11_INPUT_ELEMENT_DESC>::t vertexDesc = DX11::ComputeVertexDesc(meshGenFlags);
	const DXGI_FORMAT indexFormat = DX11::ComputeIndexFormat(vertexCount, meshGenFlags);

	return new_resource DX11::Mesh(
			"Cuboid",
			&vertexDesc[0], vertexDesc.size(),
			vertexSize, &vertexMemory[0], vertexCount,
			indexFormat, &indexMemory[0], indexCount,
			ToImpl(device)
		);
}

} // namespace