#pragma once

#include "CommandContext.h"
#include "DescriptorHeap.h"
#include "Mesh.h"
#include "Shader.h"

#include "Input.h"

#include <d3d12.h>
#include <d3dx12.h>
#include <dxgi1_4.h>
#include <SDL.h>
#include <vector>
#include <AntTweakBar.h>

namespace NGraphics
{
    class CGraphicsBase
    {
    protected:
        SDL_Window* m_Window;
        HWND m_Hwnd;

        ID3D12Device* m_Device;

        CCommandContext m_GraphicsContext;
        
        IDXGIFactory4* m_Factory;
        IDXGISwapChain* m_SwapChain;
        UINT8 m_SwapIndex;
        UINT8 m_PreviousSwapIndex;

        CDescriptorHeap m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ];

        ID3D12Resource* m_RenderTargets[ 2 ];
        SDescriptorHandle m_RenderTargetRtvs[ 2 ];

        ID3D12Resource* m_DepthTargets[ 2 ];
        SDescriptorHandle m_DepthTargetDsvs[ 2 ];
        SDescriptorHandle m_DepthTargetSrvs[ 2 ];

        UINT m_Width;
        UINT m_Height;
        D3D12_VIEWPORT m_Viewport;
        D3D12_RECT m_ScissorRect;

        BOOL m_Fullscreen;

        D3D12_FEATURE_DATA_D3D12_OPTIONS m_FeatureSupport;

        CInput m_Input;

        std::vector< TwBar* > m_TwBars;
        std::vector< SDL_Event > m_Events;

    public:
        CGraphicsBase();

        void Initialize( UINT width, UINT height, const char* window_title );
        void Terminate();

        void Run();

    protected:
        virtual void Create() = 0;
        virtual void Update( float dt ) = 0;
        virtual void Draw() = 0;
        virtual void Destroy() = 0;

        virtual void Resize( UINT width, UINT height );

        void BeginFrame( const FLOAT* clear_color );
        void ResumeFrame();
        void EndFrame();
        void Present();

        void DrawBars();

    private:
        bool PollEvents();

        void PrintFeatureSupport();
    };
}