#include "Text.h"
#include "GraphicsDefines.h"

#include <d3dx12.h>
#include <assert.h>

namespace NGraphics
{
    CText::CText() :
        m_DefaultResource( nullptr ),
        m_UploadResource(nullptr),
        m_Size( 0 ),
        m_MappedData( nullptr ),
        m_Font( nullptr )
    {
    }

    void CText::Create( ID3D12Device* device, UINT max_width, UINT max_height,
                        const char* font_filepath, int font_size, SDL_Color font_color )
    {
        assert( m_DefaultResource == nullptr &&
                m_UploadResource == nullptr &&
                m_Font == nullptr );
        assert( max_width > 0 && max_height > 0 );

        m_Layout.Offset = 0;
        m_Layout.Footprint.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        m_Layout.Footprint.RowPitch = max_width * 4;
        m_Layout.Footprint.Width = max_width;
        m_Layout.Footprint.Height = max_height;
        m_Layout.Footprint.Depth = 1;
        m_Size = m_Layout.Footprint.RowPitch * m_Layout.Footprint.Height;

        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, max_width, max_height, 1, 1,
                DXGI_FORMAT_B8G8R8A8_UNORM, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS( &m_DefaultResource ) ) );

        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( GetRequiredIntermediateSize( m_DefaultResource, 0, 1 ) ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_UploadResource ) ) );

        HR( m_UploadResource->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &m_MappedData ) ) );

        m_Font = TTF_OpenFont( font_filepath, font_size );
        m_FontColor = font_color;
    }

    void CText::Destroy()
    {
        SAFE_RELEASE_UNMAP( m_UploadResource );
        SAFE_RELEASE( m_DefaultResource );

        m_Size = 0;

        if ( m_Font != nullptr )
        {
            TTF_CloseFont( m_Font );
            m_Font = nullptr;
        }
    }

    void CText::Srv( ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle )
    {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
        ZeroMemory( &srv_desc, sizeof( srv_desc ) );
        srv_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView( m_DefaultResource, &srv_desc, cpu_handle );
    }

    void CText::Update( ID3D12GraphicsCommandList* command_list, const char* text )
    {
        assert( m_Font != nullptr );

        SDL_Surface* text_surface = TTF_RenderText_Blended_Wrapped( m_Font, text, m_FontColor, m_Layout.Footprint.Width );

        ZeroMemory( m_MappedData, m_Size );

        UINT destination_row_pitch = m_Layout.Footprint.RowPitch;
        UINT source_row_pitch = static_cast< UINT >( text_surface->w ) > destination_row_pitch ? destination_row_pitch : text_surface->w * 4;
        UINT row_count = static_cast< UINT >( text_surface->h ) > m_Layout.Footprint.Height ? m_Layout.Footprint.Height : text_surface->h;
        for ( UINT i = 0; i < row_count; ++i )
        {
            BYTE* destination_row = m_MappedData + destination_row_pitch * i;
            const BYTE* source_row = reinterpret_cast< const BYTE* >( text_surface->pixels ) + source_row_pitch * i;
            memcpy( destination_row, source_row, source_row_pitch );
        }

        SDL_FreeSurface( text_surface );

        CD3DX12_TEXTURE_COPY_LOCATION destinationLocation( m_DefaultResource, 0 );
        CD3DX12_TEXTURE_COPY_LOCATION sourceLocation( m_UploadResource, m_Layout );

        command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DefaultResource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ) );
        command_list->CopyTextureRegion( &destinationLocation, 0, 0, 0, &sourceLocation, nullptr );
        command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DefaultResource, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );
    }

    DirectX::XMMATRIX CText::CalculateScaleMatrix( D3D12_VIEWPORT viewport )
    {
        return DirectX::XMMatrixScaling( m_Layout.Footprint.Width / viewport.Width,
                                         m_Layout.Footprint.Height / viewport.Height,
                                         1.0f );
    }
}