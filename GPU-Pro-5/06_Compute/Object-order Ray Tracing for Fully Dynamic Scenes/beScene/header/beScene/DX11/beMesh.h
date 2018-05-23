/****************************************************/
/* breeze Engine Scene Module  (c) Tobias Zirr 2011 */
/****************************************************/

#pragma once
#ifndef BE_SCENE_MESH_DX11
#define BE_SCENE_MESH_DX11

#include "../beScene.h"
#include "../beMesh.h"
#include <beGraphics/Any/beBuffer.h>
#include <beGraphics/Any/beAPI.h>
#include <vector>

namespace beScene
{

namespace DX11
{

/// Constructs a vertex buffer description form the given parameters.
BE_SCENE_API D3D11_BUFFER_DESC GetVertexBufferDesc(uint4 vertexSize, uint4 vertexCount);
// Constructs an index buffer description form the given parameters.
BE_SCENE_API D3D11_BUFFER_DESC GetIndexBufferDesc(uint4 indexSize, uint4 indexCount);
/// Computes a bounding box from the given vertices.
BE_SCENE_API beMath::faab3 ComputeBounds(const D3D11_INPUT_ELEMENT_DESC *pElementDescs, uint4 elementCount,
	uint4 vertexSize, const void *pVertices, uint4 vertexCount);

/// Mesh base.
class Mesh : public beScene::Mesh
{
protected:
	typedef std::vector<D3D11_INPUT_ELEMENT_DESC> vertex_element_vector;
	vertex_element_vector m_vertexElements;

	beGraphics::Any::Buffer m_vertexBuffer;
	beGraphics::Any::Buffer m_indexBuffer;

	uint4 m_vertexCount;
	uint4 m_vertexSize;

	uint4 m_indexCount;
	DXGI_FORMAT m_indexFormat;

public:
	/// Constructor.
	BE_SCENE_API Mesh(const utf8_ntri& name, const D3D11_INPUT_ELEMENT_DESC *pElementDescs, uint4 elementCount,
		uint4 vertexSize, const void *pVertices, uint4 vertexCount,
		DXGI_FORMAT indexFormat, const void *pIndices, uint4 indexCount,
		const beMath::faab3 &bounds, 
		beGraphics::Any::API::Device *pDevice, AssembledMesh *pCompound = nullptr);
	/// Constructor.
	BE_SCENE_API Mesh(const utf8_ntri& name, const D3D11_INPUT_ELEMENT_DESC *pElementDescs, uint4 elementCount,
		uint4 vertexSize, const void *pVertices, uint4 vertexCount,
		DXGI_FORMAT indexFormat, const void *pIndices, uint4 indexCount,
		beGraphics::Any::API::Device *pDevice, AssembledMesh *pCompound = nullptr);
	/// Destructor.
	BE_SCENE_API ~Mesh();

	/// Gets the implementation identifier.
	beGraphics::ImplementationID GetImplementationID() const { return beGraphics::DX11Implementation; };

	/// Gets the vertex buffer.
	LEAN_INLINE const beGraphics::Any::Buffer& GetVertexBuffer() const { return m_vertexBuffer; }
	/// Gets the vertex element descriptions.
	const D3D11_INPUT_ELEMENT_DESC* GetVertexElementDescs() const { return &m_vertexElements[0]; }
	/// Gets the vertex element descriptions.
	const uint4 GetVertexElementDescCount() const { return static_cast<uint4>(m_vertexElements.size()); }
	/// Gets the vertex size.
	LEAN_INLINE uint4 GetVertexSize() const { return m_vertexSize; }
	/// Gets the vertex count.
	LEAN_INLINE uint4 GetVertexCount() const { return m_vertexCount; }

	/// Gets the index buffer.
	LEAN_INLINE const beGraphics::Any::Buffer& GetIndexBuffer() const { return m_indexBuffer; }
	/// Gets the index format.
	LEAN_INLINE DXGI_FORMAT GetIndexFormat() const { return m_indexFormat; }
	/// Gets the index count.
	LEAN_INLINE uint4 GetIndexCount() const { return m_indexCount; }
};

} // namespace

using beGraphics::DX11::ToImpl;

} // namespace

namespace beGraphics
{
	namespace DX11
	{
		template <> struct ToImplementationDX11<beScene::Mesh> { typedef beScene::DX11::Mesh Type; };
	} // namespace
} // namespace

#endif