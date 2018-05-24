#pragma once
#include <d3d12.h>

void CreateDDSTextureFromMemory(
	_In_ ID3D12GraphicsCommandList* cmdList,
	_In_reads_bytes_(ddsDataSize) const uint8* ddsData,
	_In_ size_t ddsDataSize,
	_Out_opt_ ID3D12Resource** texture,
	_Out_opt_ ID3D12Resource** textureView,
	_In_ size_t maxsize = 0
	);