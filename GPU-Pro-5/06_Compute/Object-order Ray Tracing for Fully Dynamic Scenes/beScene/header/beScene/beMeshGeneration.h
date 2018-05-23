/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_GENERATION
#define BE_SCENE_MESH_GENERATION

#include "beScene.h"
#include "beMesh.h"
#include <beMath/beVectorDef.h>
#include <lean/smart/resource_ptr.h>
#include <beGraphics/beDevice.h>

namespace beScene
{

/// Mesh flag enumeration.
struct MeshGenFlags
{
	/// Enumeration
	enum T
	{
		Normal = 0x1,		///< Generate normal vectors.
		TexCoord = 0x2,		///< Generate tex coords.
		Tangent = 0x4,		///< Generate tangent vectors.
		BiTangent = 0x8,	///< Generate bi-tangent vectors.
		
		TangentFrame = Normal | Tangent,			///< Generate normal-tangent frame.
		BiTangentFrame = Normal | BiTangent,		///< Generate normal-bi-tangent frame.
		FullFrame = Normal | Tangent | BiTangent,	///< Generate full normal-tangent-bi-tangent frame.

		WideIndex = 0x10000	///< Generate wide indices.
	};
	LEAN_MAKE_ENUM_STRUCT(MeshGenFlags)
};

/// Mesh generation vertex base.
template <bool>
struct MeshPosition
{
	float Pos[3];	///< Vertex position.
};
template <> struct MeshPosition<false> { };

/// Mesh generation vertex base.
template <bool>
struct MeshNormal
{
	float Norm[3];	///< Vertex normal.
};
template <> struct MeshNormal<false> { };

/// Mesh generation vertex base.
template <bool>
struct MeshTexCoord
{
	float Tex[2];	///< Vertex tex coord.
};
template <> struct MeshTexCoord<false> { };

/// Mesh generation vertex base.
template <bool>
struct MeshTangent
{
	float Tan[3];	///< Vertex tangent.
};
template <> struct MeshTangent<false> { };

/// Mesh generation vertex base.
template <bool>
struct MeshBiTangent
{
	float BiTan[3];	///< Vertex bi-tangent.
};
template <> struct MeshBiTangent<false> { };

/// Mesh generation vertex structure.
template <uint4 Flags>
struct MeshVertex
	: public MeshPosition<true>,
	public MeshNormal<(Flags & MeshGenFlags::Normal) != 0>,
	public MeshTexCoord<(Flags & MeshGenFlags::TexCoord) != 0>,
	public MeshTangent<(Flags & MeshGenFlags::Tangent) != 0>,
	public MeshBiTangent<(Flags & MeshGenFlags::BiTangent) != 0> { };

/// Computes the size of one vertex.
BE_SCENE_API uint4 ComputeVertexSize(uint4 meshGenFlags);
/// Computes the size of one index.
BE_SCENE_API uint4 ComputeIndexSize(uint4 vertexCount, uint4 meshGenFlags);

/// Computes the number of vertices in a regular grid of the given size.
BE_SCENE_API uint4 ComputeGridVertexCount(uint4 cellsU, uint4 cellsV);
/// Computes the number of indices in a regular grid of the given size.
BE_SCENE_API uint4 ComputeGridIndexCount(uint4 cellsU, uint4 cellsV);

/// Generates a regular grid of the given dimensions.
BE_SCENE_API void GenerateGrid(void *vertices, void *indices,
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV,
	uint4 meshGenFlags,
	uint4 meshBaseIndex = 0);

/// Generates a regular grid mesh of the given dimensions.
BE_SCENE_API lean::resource_ptr<Mesh, true> GenerateGridMesh(
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV,
	uint4 meshGenFlags,
	const beGraphics::Device &device);

/// Computes the number of vertices in a regular cuboid of the given size.
BE_SCENE_API uint4 ComputeCuboidVertexCount(uint4 cellsU, uint4 cellsV, uint4 cellsW);
/// Computes the number of indices in a regular cuboid of the given size.
BE_SCENE_API uint4 ComputeCuboidIndexCount(uint4 cellsU, uint4 cellsV, uint4 cellsW);

/// Generates a regular cuboid of the given dimensions.
BE_SCENE_API void GenerateCuboid(void *vertices, void *indices,
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y, const beMath::fvec3 &z,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV, uint4 cellsW,
	uint4 meshGenFlags);

/// Generates a regular cuboid mesh of the given dimensions.
BE_SCENE_API lean::resource_ptr<Mesh, true> GenerateCuboidMesh(
	const beMath::fvec3 &b, const beMath::fvec3 &x, const beMath::fvec3 &y, const beMath::fvec3 &z,
	float u0, float v0, float du, float dv,
	uint4 cellsU, uint4 cellsV, uint4 cellsW,
	uint4 meshGenFlags,
	const beGraphics::Device &device);

} // namespace

#endif