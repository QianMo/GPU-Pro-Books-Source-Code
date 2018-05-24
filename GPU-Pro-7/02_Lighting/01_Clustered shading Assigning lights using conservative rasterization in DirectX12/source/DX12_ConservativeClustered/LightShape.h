#pragma once
#include "KGraphicsDevice.h"
#include <vector>
#include "Vert.h"

struct VertexPIndexData
{
	std::vector<VertP8> vertexData;
	std::vector<uint16> indexData;
};

class LightShape
{
public:
	LightShape(VertexPIndexData* vert_ind_data, ID3D12GraphicsCommandList* gfx_command_list);
	~LightShape();
	void DrawIndexedInstanced(uint32 num_instances, ID3D12GraphicsCommandList* gfx_command_list);

private:

	D3D12_VERTEX_BUFFER_VIEW vertexBuffer;
	D3D12_INDEX_BUFFER_VIEW indexBuffer;

	ID3D12Resource* g_VertexBuffer;
	ID3D12Resource* g_VertexBufferUpload;
	ID3D12Resource* g_IndexBuffer;
	ID3D12Resource* g_IndexBufferUpload;

	size_t numVertices;
	size_t numIndices;
	uint32 offset;
	uint32 stride;
};