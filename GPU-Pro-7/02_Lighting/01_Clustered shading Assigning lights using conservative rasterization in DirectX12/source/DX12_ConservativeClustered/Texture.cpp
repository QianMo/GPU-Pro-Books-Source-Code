#include "Texture.h"
#include "SharedContext.h"
#include "Log.h"
#include <fstream>
#include <iostream>
#include "DDSTextureLoader.h"

using namespace Log;

Texture::Texture()
	: m_texture(nullptr), m_uploadTexture(nullptr)
{
}

Texture::~Texture()
{
	m_texture->Release();
	if(m_uploadTexture)
		m_uploadTexture->Release();
}

void Texture::CreateTextureFromFile(ID3D12GraphicsCommandList* gfx_command_list, const char* file)
{
	std::streampos size;
	uint8* tex_data;

	PRINT(LogLevel::DEBUG_PRINT, "Loading texture: %s", file);

	std::ifstream file_stream(file, std::ios::in | std::ios::binary | std::ios::ate);
	if (file_stream.is_open())
	{
		size = file_stream.tellg();
		tex_data = new uint8[size];
		file_stream.seekg(0, std::ios::beg);
		file_stream.read((char*)tex_data, size);
		file_stream.close();

		CreateDDSTextureFromMemory(gfx_command_list, tex_data, (uint32)size, &m_texture, &m_uploadTexture);

		if(m_texture != nullptr && m_uploadTexture != nullptr)
		{
			m_GPUhandle = shared_context.gfx_device->GetDescHeapCBV_SRV()->GetGPUHandleAtHead();
			PRINT(LogLevel::SUCCESS, "Texture loaded: %s", file);
		}
		delete[] tex_data;
	}
	else 
	{
		PRINT(LogLevel::FATAL_ERROR, "Texture load failed: %s", file);
	}
}

void Texture::DeleteUploadTexture()
{
	if(m_uploadTexture)
	{
		m_uploadTexture->Release();
		m_uploadTexture = nullptr;
	}
}

