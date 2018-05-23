/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_SERIALIZATION
#define BE_SCENE_MESH_SERIALIZATION

#include "beScene.h"
#include "beAssembledMesh.h"
#include <lean/smart/resource_ptr.h>
#include <beGraphics/beDevice.h>

namespace beScene
{

/// Mesh chunk IDs.
struct MeshDataChunk
{
	/// Enumeration.
	enum T
	{
		Header	= 0xBE20,	///< Mesh header.

		Name = 0xA0A0,		///< Name attribute.
		Source = 0xA1A1,	///< Source attribute.
		Desc = 0xA2A2,		///< Description attribute.
		Count = 0xA3A3,		///< Count attribute.
		Data = 0xA4A4,		///< Data attribute.

		Subsets = 0xD0D0,	///< Subset data.
		Vertices = 0xD1D1,	///< Vertex data.
		Indices = 0xD2D2	///< Index data.
	};
	LEAN_MAKE_ENUM_STRUCT(MeshDataChunk)
};

/// Mesh data chunk.
struct MeshDataChunkHeader
{
	uint2 ChunkID;		///< Chunk ID.
	uint2 _Pad;
	uint4 ChunkSize;	///< Chunk size.

	/// Non-initializing constructor.
	MeshDataChunkHeader() { }
	/// Constructor.
	MeshDataChunkHeader(MeshDataChunk::T chunk, uint4 size)
		: ChunkID( static_cast<uint2>(chunk) ),
		ChunkSize(size) { }
};

/// Mesh vertex attributes.
struct MeshVertexAttributes
{
	/// Enumeration
	enum T
	{
		Position = 1 << 0,	///< Vertex position.
		
		Normal = 1 << 1,	///< Vertex normal.
		
		Color = 1 << 2,		///< Vertex color.
		TexCoord = 1 << 3,	///< Vertex tex coord.
		
		Tangent = 1 << 4,		///< Vertex tangent.
		BiTangent = 1 << 5,		///< Vertex bi-tangent.
		Handedness = 1 << 6,	///< Vertex tangent includes handedness.

		TangentFrame = Normal | Tangent | Handedness,		///< Normal-tangent frame.
		BiTangentFrame = Normal | BiTangent | Handedness,	///< Normal-bi-tangent frame.
		FullFrame = Normal | Tangent | BiTangent			///< Full normal-tangent-bi-tangent frame.
	};
	LEAN_MAKE_ENUM_STRUCT(MeshVertexAttributes)
};

/// Vertex element description.
struct MeshDataVertexElementDesc
{
	uint4 Attribute;	///< Vertex attribute (one of MeshVertexAttributes).
	uint4 Index;		///< Vertex attribute index.
	uint4 Format;		///< Vertex element format (one of beGraphics::Format).

	/// Non-initializing constructor.
	MeshDataVertexElementDesc() { }
	/// Constructor.
	MeshDataVertexElementDesc(MeshVertexAttributes::T attribute, uint4 index, beGraphics::Format::T format)
		: Attribute(attribute),
		Index(index),
		Format(format) { }
};

/// Mesh index flags.
struct MeshIndexFlags
{
	/// Enumeration
	enum T
	{
		WideIndex= 1 << 0	///< Wide indices.
	};
	LEAN_MAKE_ENUM_STRUCT(MeshIndexFlags)
};

/// Index description.
struct MesDataIndexDesc
{
	uint4 Flags;	///< Index flags.
};

/// Loads a mesh compound from the given file.
BE_SCENE_API lean::resource_ptr<AssembledMesh, true> LoadMeshes(const utf8_ntri &file, beGraphics::Device &device);
/// Loads a mesh compound from the given memory.
BE_SCENE_API lean::resource_ptr<AssembledMesh, true> LoadMeshes(const char *data, uint8 dataLength, beGraphics::Device &device);

} // namespace

#endif