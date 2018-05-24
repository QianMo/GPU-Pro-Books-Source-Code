#include "LightShape.h"
#include "SharedContext.h"

LightShape::LightShape(VertexPIndexData* vert_ind_data, ID3D12GraphicsCommandList* gfx_command_list)
{
	numVertices = vert_ind_data->vertexData.size();
	numIndices = vert_ind_data->indexData.size();

	stride = sizeof(VertP8);
	offset = 0;

	//HRESULT result;

	//Vertex buffer
	shared_context.gfx_device->GetDevice()->CreateCommittedResource(
		&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT, 0, 0),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(numVertices * sizeof(VertP8)),
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&g_VertexBuffer));

	{
		shared_context.gfx_device->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(numVertices * sizeof(VertP8)),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&g_VertexBufferUpload));

		D3D12_SUBRESOURCE_DATA vertData = {};
		vertData.pData = &vert_ind_data->vertexData[0];
		vertData.RowPitch = numVertices * sizeof(VertP8);
		vertData.SlicePitch = vertData.RowPitch;

		shared_context.gfx_device->TransitionResource(gfx_command_list, g_VertexBuffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		UpdateSubresources<1>(gfx_command_list, g_VertexBuffer, g_VertexBufferUpload, 0, 0, 1, &vertData);
		shared_context.gfx_device->TransitionResource(gfx_command_list, g_VertexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
	}

	vertexBuffer.BufferLocation = g_VertexBuffer->GetGPUVirtualAddress();
	vertexBuffer.StrideInBytes = sizeof(VertP8);
	vertexBuffer.SizeInBytes = (UINT)numVertices * sizeof(VertP8);

	//Index buffer
	shared_context.gfx_device->GetDevice()->CreateCommittedResource(&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
		D3D12_HEAP_FLAG_NONE,
		&CD3DX12_RESOURCE_DESC::Buffer(numIndices * sizeof(uint16)),
		D3D12_RESOURCE_STATE_COMMON,
		nullptr,
		IID_PPV_ARGS(&g_IndexBuffer));

	{
		shared_context.gfx_device->GetDevice()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_UPLOAD, 0, 0),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer(numIndices * sizeof(uint16)),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&g_IndexBufferUpload));

		D3D12_SUBRESOURCE_DATA indexData = {};
		indexData.pData = &vert_ind_data->indexData[0];
		indexData.RowPitch = numIndices * sizeof(uint16);
		indexData.SlicePitch = indexData.RowPitch;

		shared_context.gfx_device->TransitionResource(gfx_command_list, g_IndexBuffer, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
		UpdateSubresources<1>(gfx_command_list, g_IndexBuffer, g_IndexBufferUpload, 0, 0, 1, &indexData);
		shared_context.gfx_device->TransitionResource(gfx_command_list, g_IndexBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_GENERIC_READ);
	}


	indexBuffer.BufferLocation = g_IndexBuffer->GetGPUVirtualAddress();
	indexBuffer.Format = DXGI_FORMAT_R16_UINT;
	indexBuffer.SizeInBytes = (UINT)numIndices * sizeof(uint16);
}

LightShape::~LightShape()
{
	g_IndexBuffer->Release();
	g_IndexBufferUpload->Release();
	g_VertexBuffer->Release();
	g_VertexBufferUpload->Release();
}

void LightShape::DrawIndexedInstanced(uint32 num_instances, ID3D12GraphicsCommandList* gfx_command_list)
{
	gfx_command_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	gfx_command_list->IASetVertexBuffers(0, 1, &vertexBuffer);
	gfx_command_list->IASetIndexBuffer(&indexBuffer);
	gfx_command_list->DrawIndexedInstanced((UINT)numIndices, num_instances, 0, 0, 0);
}

