#pragma once

#include <d3d12.h>
#include <d3dx12.h>

namespace NGraphics
{
    ID3D12RootSignature* CreateRootSignature( ID3D12Device* device, UINT parameter_count, CD3DX12_ROOT_PARAMETER* parameters, D3D12_ROOT_SIGNATURE_FLAGS flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT );
}