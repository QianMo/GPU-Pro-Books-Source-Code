#pragma once

#include <d3d12.h>
#include <SDL_ttf.h>
#include <DirectXMath.h>

namespace NGraphics
{
    class CText
    {
    private:
        ID3D12Resource* m_DefaultResource;
        ID3D12Resource* m_UploadResource;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT m_Layout;
        UINT m_Size;
        BYTE* m_MappedData;

        TTF_Font* m_Font;
        SDL_Color m_FontColor;

    public:
        CText();

        void Create( ID3D12Device* device, UINT max_width, UINT max_height,
                     const char* font_filepath, int font_size, SDL_Color font_color );
        void Destroy();

        void Srv( ID3D12Device* device, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle );

        void Update( ID3D12GraphicsCommandList* command_list, const char* text );

        DirectX::XMMATRIX CalculateScaleMatrix( D3D12_VIEWPORT viewport );
    };
}