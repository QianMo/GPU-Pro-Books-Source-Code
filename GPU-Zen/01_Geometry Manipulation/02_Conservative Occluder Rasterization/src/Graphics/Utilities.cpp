#include "Utilities.h"
#include "GraphicsDefines.h"

namespace NGraphics
{
    ID3D12RootSignature* CreateRootSignature( ID3D12Device* device, UINT parameter_count, CD3DX12_ROOT_PARAMETER* parameters, D3D12_ROOT_SIGNATURE_FLAGS flags )
    {
        CD3DX12_ROOT_SIGNATURE_DESC root_signature_desc;
        root_signature_desc.Init( parameter_count, parameters, 0, nullptr, flags );

        ID3DBlob* signature = nullptr;
        ID3DBlob* error = nullptr;
        if ( FAILED( D3D12SerializeRootSignature( &root_signature_desc, D3D_ROOT_SIGNATURE_VERSION_1, &signature, &error ) ) )
        {
            if ( error )
                LOG( ( char* )error->GetBufferPointer() );
            else
                LOG( "D3D12SerializeRootSignature failed." );
        }

        ID3D12RootSignature* root_signature = nullptr;
        HR( device->CreateRootSignature( 0, signature->GetBufferPointer(), signature->GetBufferSize(), __uuidof( ID3D12RootSignature ), ( void** )&root_signature ) );

        if ( signature != nullptr )
            signature->Release();
        if ( error != nullptr )
            error->Release();

        return root_signature;
    }
}