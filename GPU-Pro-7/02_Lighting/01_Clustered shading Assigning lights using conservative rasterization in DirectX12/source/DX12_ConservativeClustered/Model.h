#pragma once
#include "KGraphicsDevice.h"
#include "Vert.h"
#include <map>
#include <vector>
#include "Texture.h"

struct VertexIndexData
{
	std::vector<Vert> vertexData;
	std::vector<uint32> indexData;
};

struct Material
{
	Texture* diffuse;
};

struct MeshGroup 
{
	//The index where the draw command should start
	UINT startIndex;
	//Number of indices that will be rendered
	UINT numIndices;

	//Texture handles
	char material[64];
};

class Model
{
public:
	Model(VertexIndexData* vert_ind_data, std::vector<MeshGroup> pmeshGroups, std::map<std::string, Material> pmaterialMap, ID3D12GraphicsCommandList* gfx_command_list);
	~Model();

	void Apply(ID3D12GraphicsCommandList* gfx_command_list);

	void DrawIndexed(ID3D12GraphicsCommandList* gfx_command_list);

	void CleanupUploadData();

private:

	D3D12_VERTEX_BUFFER_VIEW vertexBuffer;
	D3D12_INDEX_BUFFER_VIEW indexBuffer;

	ID3D12Resource* g_VertexBuffer;
	ID3D12Resource* g_VertexBufferUpload;
	ID3D12Resource* g_IndexBuffer;
	ID3D12Resource* g_IndexBufferUpload;

	size_t numVertices;
	size_t numIndices;
	UINT offset;
	UINT stride;

	std::vector<MeshGroup> meshGroups;
	std::map<std::string, Material> materialMap;
};