#pragma once
#include "KGraphicsDevice.h"

class Texture
{
public:
	Texture();
	~Texture();

	void CreateTextureFromFile(ID3D12GraphicsCommandList* gfx_command_list, const char* file);
	void DeleteUploadTexture();

	D3D12_GPU_DESCRIPTOR_HANDLE GetGPUDescriptorHandle() { return m_GPUhandle; }

private:
	ID3D12Resource* m_texture;
	ID3D12Resource* m_uploadTexture;

	D3D12_GPU_DESCRIPTOR_HANDLE m_GPUhandle;
};