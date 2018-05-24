#include "Model.h"
#include "SharedContext.h"
#include "d3dx12.h"

Model::Model(VertexIndexData* vert_ind_data, std::vector<MeshGroup> pmeshGroups, std::map<std::string, Material> pmaterialMap, ID3D12GraphicsCommandList* gfx_command_list)
{
	meshGroups = pmeshGroups;
	materialMap = pmaterialMap;

	numVertices = vert_ind_data->vertexData.size();
	numIndices = vert_ind_data->indexData.size();

	stride = sizeof(Vert);
	offset = 0;

	//HRESULT result;

	//Vertex buffer
	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(numVertices * sizeof(Vert)),
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&g_VertexBuffer));

	{
		shared_context.gfx_device->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(numVertices * sizeof(Vert)),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&g_VertexBufferUpload));

		D3D12_SUBRESOURCE_DATA vertData = {};
		vertData.pData = &vert_ind_data->vertexData[0];
		vertData.RowPitch = numVertices * sizeof(Vert);
		vertData.SlicePitch = vertData.RowPitch;

		shared_context.gfx_device->TransitionResource(gfx_command_list, g_VertexBuffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		UpdateSubresources<1>(gfx_command_list, g_VertexBuffer, g_VertexBufferUpload, 0, 0, 1, &vertData);
		shared_context.gfx_device->TransitionResource(gfx_command_list, g_VertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
	}

	vertexBuffer.BufferLocation = g_VertexBuffer->GetGPUVirtualAddress();
	vertexBuffer.StrideInBytes = sizeof(Vert);
	vertexBuffer.SizeInBytes = (UINT)numVertices * sizeof(Vert);

	//Index buffer
	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(numIndices * sizeof(uint32)),
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&g_IndexBuffer));

	{
		shared_context.gfx_device->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(numIndices * sizeof(uint32)),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&g_IndexBufferUpload));

		D3D12_SUBRESOURCE_DATA indexData = {};
		indexData.pData = &vert_ind_data->indexData[0];
		indexData.RowPitch = numIndices * sizeof(uint32);
		indexData.SlicePitch = indexData.RowPitch;

		shared_context.gfx_device->TransitionResource(gfx_command_list, g_IndexBuffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		UpdateSubresources<1>(gfx_command_list, g_IndexBuffer, g_IndexBufferUpload, 0, 0, 1, &indexData);
		shared_context.gfx_device->TransitionResource(gfx_command_list, g_IndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
	}

	indexBuffer.BufferLocation = g_IndexBuffer->GetGPUVirtualAddress();
	indexBuffer.Format = DXGI_FORMAT_R32_UINT;
	indexBuffer.SizeInBytes = (UINT)numIndices * sizeof(uint32);
	
}

Model::~Model()
{
	g_IndexBuffer->Release();
	g_IndexBufferUpload->Release();
	g_VertexBuffer->Release();
	g_VertexBufferUpload->Release();

	for(auto mat : materialMap)
	{
		delete mat.second.diffuse;
	}
}

void Model::Apply(ID3D12GraphicsCommandList* gfx_command_list)
{
	gfx_command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	gfx_command_list->IASetVertexBuffers(0, 1, &vertexBuffer);
	gfx_command_list->IASetIndexBuffer(&indexBuffer);
}

void Model::DrawIndexed(ID3D12GraphicsCommandList* gfx_command_list)
{
	for (auto mg : meshGroups)
	{
		Material mat = materialMap[mg.material];
		if (mat.diffuse != nullptr)
			gfx_command_list->SetGraphicsRootDescriptorTable(1, mat.diffuse->GetGPUDescriptorHandle());

		gfx_command_list->DrawIndexedInstanced(mg.numIndices, 1, mg.startIndex, 0, 0);
	}
}

void Model::CleanupUploadData()
{
	for (auto mat : materialMap)
	{
		if(mat.second.diffuse)
			mat.second.diffuse->DeleteUploadTexture();
	}
}


