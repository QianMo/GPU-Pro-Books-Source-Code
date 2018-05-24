#include "GraphicsBase.h"
#include "GraphicsDefines.h"
#include "Shader.h"
#include "Timer.h"

#include <assert.h>
#include <stdlib.h>
#include <SDL_syswm.h>
#include <SDL_ttf.h>
#include <vector>
#include <dxgi1_4.h>

namespace NGraphics
{
    CGraphicsBase::CGraphicsBase() :
        m_Window( nullptr ),
        m_Device( nullptr ),
        m_Factory( nullptr ),
        m_SwapChain( nullptr ),
        m_SwapIndex( 0 ),
        m_PreviousSwapIndex( 1 ),
        m_RenderTargets{ nullptr, nullptr },
        m_DepthTargets{ nullptr, nullptr },
        m_Fullscreen( FALSE )
    {
    }

    void CGraphicsBase::Initialize( UINT width, UINT height, const char* window_title )
    {
        assert( SDL_Init( SDL_INIT_EVERYTHING ) == 0 && TTF_Init() == 0 );

        m_Window = SDL_CreateWindow( window_title, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                     width, height, SDL_WINDOW_SHOWN );

        SDL_SysWMinfo info;
        SDL_VERSION( &info.version );
        SDL_GetWindowWMInfo( m_Window, &info );
        m_Hwnd = info.info.win.window;

    #ifdef _DEBUG
        ID3D12Debug* debug_controller = nullptr;
        D3D12GetDebugInterface( IID_PPV_ARGS( &debug_controller ) );
        if ( debug_controller != nullptr )
        {
            debug_controller->EnableDebugLayer();
            debug_controller->Release();
        }
    #endif

        HR( D3D12CreateDevice( nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS( &m_Device ) ) );

        m_GraphicsContext.Create( m_Device, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FENCE_FLAG_NONE );

        HR( CreateDXGIFactory1( IID_PPV_ARGS( &m_Factory ) ) );

        DXGI_SWAP_CHAIN_DESC swap_chain_desc;
        ZeroMemory( &swap_chain_desc, sizeof( swap_chain_desc ) );
        swap_chain_desc.BufferCount = 2;
        swap_chain_desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        swap_chain_desc.BufferUsage = DXGI_USAGE_BACK_BUFFER | DXGI_USAGE_RENDER_TARGET_OUTPUT;
        swap_chain_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
        swap_chain_desc.OutputWindow = m_Hwnd;
        swap_chain_desc.SampleDesc.Count = 1;
        swap_chain_desc.Windowed = TRUE;
        swap_chain_desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
        HR( m_Factory->CreateSwapChain( m_GraphicsContext.GetCommandQueue(), &swap_chain_desc, &m_SwapChain ) );
        
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].Create( m_Device, 256, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE );
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].Create( m_Device, 32, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE );
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].Create( m_Device, 128, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE );
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].Create( m_Device, 32, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, D3D12_DESCRIPTOR_HEAP_FLAG_NONE );

        m_RenderTargetRtvs[ 0 ] = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
        m_RenderTargetRtvs[ 1 ] = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
        m_DepthTargetDsvs[ 0 ] = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GenerateHandle();
        m_DepthTargetDsvs[ 1 ] = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GenerateHandle();
        m_DepthTargetSrvs[ 0 ] = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_DepthTargetSrvs[ 1 ] = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        
        TwInit( TW_DIRECT3D12, reinterpret_cast< void* >( m_Device ), reinterpret_cast< void* >( m_GraphicsContext.GetCommandList() ) );

        Resize( width, height );

        HR( m_Device->SetStablePowerState( TRUE ) );

        PrintFeatureSupport();

        Create();

        LOG( "\n" );
        LOG( "------------------------------ DESCRIPTOR HEAP INFO ------------------------------\n" );
        LOG( "CBV/SRV/UAV: %d / %d\n", m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GetSize(), m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GetCapacity() );
        LOG( "SAMPLER: %d / %d\n", m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].GetSize(), m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].GetCapacity() );
        LOG( "RTV: %d / %d\n", m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GetSize(), m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GetCapacity() );
        LOG( "DSV: %d / %d\n", m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GetSize(), m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GetCapacity() );
        LOG( "----------------------------------------------------------------------------------\n" );
    }

    void CGraphicsBase::Resize( UINT width, UINT height )
    {
        m_SwapChain->SetFullscreenState( m_Fullscreen, nullptr );

        SDL_SetWindowFullscreen( m_Window, 0 );
        SDL_SetWindowSize( m_Window, width, height );
        SDL_SetWindowFullscreen( m_Window, m_Fullscreen );

        SAFE_RELEASE( m_DepthTargets[ 1 ] );
        SAFE_RELEASE( m_DepthTargets[ 0 ] );

        SAFE_RELEASE( m_RenderTargets[ 1 ] );
        SAFE_RELEASE( m_RenderTargets[ 0 ] );

        HR( m_SwapChain->ResizeBuffers( 2, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH ) );
        m_SwapIndex = 0;
        m_PreviousSwapIndex = 1 - m_SwapIndex;

        HR( m_SwapChain->GetBuffer( 0, IID_PPV_ARGS( &m_RenderTargets[ 0 ] ) ) );
        HR( m_SwapChain->GetBuffer( 1, IID_PPV_ARGS( &m_RenderTargets[ 1 ] ) ) );

        D3D12_CLEAR_VALUE depth_target_clear_value;
        ZeroMemory( &depth_target_clear_value, sizeof( depth_target_clear_value ) );
        depth_target_clear_value.Format = DXGI_FORMAT_D32_FLOAT;
        depth_target_clear_value.DepthStencil.Depth = 0.0f;
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
                DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL ),
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            &depth_target_clear_value,
            IID_PPV_ARGS( &m_DepthTargets[ 0 ] ) ) );
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
                DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            &depth_target_clear_value,
            IID_PPV_ARGS( &m_DepthTargets[ 1 ] ) ) );

        D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc;
        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;

        m_Device->CreateRenderTargetView( m_RenderTargets[ 0 ], nullptr, m_RenderTargetRtvs[ 0 ].m_Cpu );
        m_Device->CreateRenderTargetView( m_RenderTargets[ 1 ], nullptr, m_RenderTargetRtvs[ 1 ].m_Cpu );
        
        ZeroMemory( &dsv_desc, sizeof( dsv_desc ) );
        dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
        dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
        m_Device->CreateDepthStencilView( m_DepthTargets[ 0 ], &dsv_desc, m_DepthTargetDsvs[ 0 ].m_Cpu );
        m_Device->CreateDepthStencilView( m_DepthTargets[ 1 ], &dsv_desc, m_DepthTargetDsvs[ 1 ].m_Cpu );
        
        ZeroMemory( &srv_desc, sizeof( srv_desc ) );
        srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = 1;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        m_Device->CreateShaderResourceView( m_DepthTargets[ 0 ], &srv_desc, m_DepthTargetSrvs[ 0 ].m_Cpu );
        m_Device->CreateShaderResourceView( m_DepthTargets[ 1 ], &srv_desc, m_DepthTargetSrvs[ 1 ].m_Cpu );
        
        m_Width = width;
        m_Height = height;
        m_Viewport = { 0.0f, 0.0f, static_cast< float >( width ), static_cast< float >( height ), 0.0f, 1.0f };
        m_ScissorRect = { 0, 0, static_cast< long >( width ), static_cast< long >( height ) };

        TwWindowSize( width, height );
    }

    void CGraphicsBase::Terminate()
    {
        Destroy();

        SAFE_RELEASE( m_DepthTargets[ 1 ] );
        SAFE_RELEASE( m_DepthTargets[ 0 ] );

        SAFE_RELEASE( m_RenderTargets[ 1 ] );
        SAFE_RELEASE( m_RenderTargets[ 0 ] );

        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].Destroy();
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].Destroy();
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].Destroy();
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].Destroy();

        m_GraphicsContext.Destroy();

        m_SwapChain->SetFullscreenState( FALSE, nullptr );
        SAFE_RELEASE( m_SwapChain );
        SAFE_RELEASE( m_Factory );

        SAFE_RELEASE( m_Device );

        for ( TwBar* bar : m_TwBars )
        {
            TwDeleteBar( bar );
        }
        TwTerminate();

        SDL_DestroyWindow( m_Window );
        TTF_Quit();
        SDL_Quit();
    }

    void CGraphicsBase::Run()
    {
        CTimer timer;
        while ( PollEvents() )
        {
            timer.Tick();

            Update( timer.GetDeltaTime() );
            Draw();
        }
    }

    bool CGraphicsBase::PollEvents()
    {
        m_Input.Reset();

        SDL_Event event;
        while ( SDL_PollEvent( &event ) )
        {
            switch ( event.type )
            {
                case SDL_QUIT:
                    return false;
                case SDL_WINDOWEVENT:
                    if ( event.window.event == SDL_WINDOWEVENT_MINIMIZED )
                    {
                        m_Fullscreen = FALSE;
                        Resize( m_Width, m_Height );
                    }
                    break;
            }
            m_Input.ProcessEvent( event );

            m_Events.push_back( event );
        }

        if ( m_Input.IsKeyPressed( SDL_SCANCODE_ESCAPE ) )
        {
            return false;
        }

        return true;
    }

    void CGraphicsBase::BeginFrame( const FLOAT* clear_color )
    {
        ID3D12GraphicsCommandList* command_list = m_GraphicsContext.GetCommandList();

        D3D12_RESOURCE_BARRIER begin_frame_barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition( m_RenderTargets[ m_SwapIndex ], D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET ),
            CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargets[ m_SwapIndex ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE )
        };
        command_list->ResourceBarrier( _countof( begin_frame_barriers ), begin_frame_barriers );

        ResumeFrame();

        command_list->ClearRenderTargetView( m_RenderTargetRtvs[ m_SwapIndex ].m_Cpu, clear_color, 0, nullptr );
        command_list->ClearDepthStencilView( m_DepthTargetDsvs[ m_SwapIndex ].m_Cpu, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr );
    }

    void CGraphicsBase::ResumeFrame()
    {
        ID3D12GraphicsCommandList* command_list = m_GraphicsContext.GetCommandList();

        command_list->RSSetViewports( 1, &m_Viewport );
        command_list->RSSetScissorRects( 1, &m_ScissorRect );

        command_list->OMSetRenderTargets( 1, &m_RenderTargetRtvs[ m_SwapIndex ].m_Cpu, false, &m_DepthTargetDsvs[ m_SwapIndex ].m_Cpu );
    }

    void CGraphicsBase::EndFrame()
    {
        ID3D12GraphicsCommandList* command_list = m_GraphicsContext.GetCommandList();

        D3D12_RESOURCE_BARRIER end_frame_barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition( m_RenderTargets[ m_SwapIndex ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT ),
            CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargets[ m_SwapIndex ], D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE )
        };
        command_list->ResourceBarrier( _countof( end_frame_barriers ), end_frame_barriers );
    }

    void CGraphicsBase::Present()
    {
        HR( m_SwapChain->Present( 1, 0 ) );
        m_PreviousSwapIndex = m_SwapIndex;
        m_SwapIndex = 1 - m_SwapIndex;
    }

    void CGraphicsBase::DrawBars()
    {
        for ( SDL_Event& event : m_Events )
        {
            TwEventSDL( &event, SDL_MAJOR_VERSION, SDL_MINOR_VERSION );
        }
        m_Events.clear();
        for ( TwBar* bar : m_TwBars )
        {
            TwRefreshBar( bar );
        }
        TwDraw();
    }

    void CGraphicsBase::PrintFeatureSupport()
    {
        if ( FAILED( m_Device->CheckFeatureSupport( D3D12_FEATURE_D3D12_OPTIONS, &m_FeatureSupport, sizeof( D3D12_FEATURE_DATA_D3D12_OPTIONS ) ) ) )
        {
            LOG( "CheckFeatureSupport failed.\n" );
            return;
        }

        LOG( "-------------------------------- HARDWARE OPTIONS --------------------------------\n" );

        switch ( m_FeatureSupport.ConservativeRasterizationTier )
        {
            case D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED:
                LOG( "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_NOT_SUPPORTED\n" );
                break;
            case D3D12_CONSERVATIVE_RASTERIZATION_TIER_1:
                LOG( "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_TIER_1\n" );
                break;
            case D3D12_CONSERVATIVE_RASTERIZATION_TIER_2:
                LOG( "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_TIER_2\n" );
                break;
            case D3D12_CONSERVATIVE_RASTERIZATION_TIER_3:
                LOG( "ConservativeRasterizationTier: D3D12_CONSERVATIVE_RASTERIZATION_TIER_3\n" );
                break;
        }

        if ( m_FeatureSupport.CrossAdapterRowMajorTextureSupported )
            LOG( "CrossAdapterRowMajorTextureSupported: TRUE\n" );
        else
            LOG( "CrossAdapterRowMajorTextureSupported: FALSE\n" );

        switch ( m_FeatureSupport.CrossNodeSharingTier )
        {
            case D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED:
                LOG( "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARINm_NOT_SUPPORTED\n" );
                break;
            case D3D12_CROSS_NODE_SHARING_TIER_1_EMULATED:
                LOG( "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARINm_TIER_1_EMULATED\n" );
                break;
            case D3D12_CROSS_NODE_SHARING_TIER_1:
                LOG( "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARINm_TIER_1\n" );
                break;
            case D3D12_CROSS_NODE_SHARING_TIER_2:
                LOG( "CrossNodeSharingTier: D3D12_CROSS_NODE_SHARINm_TIER_2\n" );
                break;
        }

        if ( m_FeatureSupport.DoublePrecisionFloatShaderOps )
            LOG( "DoublePrecisionFloatShaderOps: TRUE\n" );
        else
            LOG( "DoublePrecisionFloatShaderOps: FALSE\n" );

        LOG( "MaxGPUVirtualAddressBitsPerResource: %d\n", m_FeatureSupport.MaxGPUVirtualAddressBitsPerResource );

        switch ( m_FeatureSupport.MinPrecisionSupport )
        {
            case D3D12_SHADER_MIN_PRECISION_SUPPORT_NONE:
                LOG( "MinPrecisionSupport: D3D12_SHADER_MIN_PRECISION_NONE\n" );
                break;
            case D3D12_SHADER_MIN_PRECISION_SUPPORT_10_BIT:
                LOG( "MinPrecisionSupport: D3D12_SHADER_MIN_PRECISION_10_BIT\n" );
                break;
            case D3D12_SHADER_MIN_PRECISION_SUPPORT_16_BIT:
                LOG( "MinPrecisionSupport: D3D12_SHADER_MIN_PRECISION_16_BIT\n" );
                break;
        }

        if ( m_FeatureSupport.OutputMergerLogicOp )
            LOG( "OutputMergerLogicOp: TRUE\n" );
        else
            LOG( "OutputMergerLogicOp: FALSE\n" );

        if ( m_FeatureSupport.PSSpecifiedStencilRefSupported )
            LOG( "PSSpecifiedStencilRefSupported: TRUE\n" );
        else
            LOG( "PSSpecifiedStencilRefSupported: FALSE\n" );

        switch ( m_FeatureSupport.ResourceBindingTier )
        {
            case D3D12_RESOURCE_BINDING_TIER_1:
                LOG( "ResourceBindingTier: D3D12_RESOURCE_BINDINm_TIER_1\n" );
                break;
            case D3D12_RESOURCE_BINDING_TIER_2:
                LOG( "ResourceBindingTier: D3D12_RESOURCE_BINDINm_TIER_2\n" );
                break;
            case D3D12_RESOURCE_BINDING_TIER_3:
                LOG( "ResourceBindingTier: D3D12_RESOURCE_BINDINm_TIER_3\n" );
                break;
        }

        switch ( m_FeatureSupport.ResourceHeapTier )
        {
            case D3D12_RESOURCE_HEAP_TIER_1:
                LOG( "ResourceHeapTier: D3D12_RESOURCE_HEAP_TIER_1\n" );
                break;
            case D3D12_RESOURCE_HEAP_TIER_2:
                LOG( "ResourceHeapTier: D3D12_RESOURCE_HEAP_TIER_2\n" );
                break;
        }

        if ( m_FeatureSupport.ROVsSupported )
            LOG( "ROVsSupported: TRUE\n" );
        else
            LOG( "ROVsSupported: FALSE\n" );

        if ( m_FeatureSupport.StandardSwizzle64KBSupported )
            LOG( "StandardSwizzle64KBSupported: TRUE\n" );
        else
            LOG( "StandardSwizzle64KBSupported: FALSE\n" );

        switch ( m_FeatureSupport.TiledResourcesTier )
        {
            case D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED:
                LOG( "TiledResourcesTier: D3D12_TILED_RESOURCES_NOT_SUPPORTED\n" );
                break;
            case D3D12_TILED_RESOURCES_TIER_1:
                LOG( "TiledResourcesTier: D3D12_TILED_RESOURCES_TIER_1\n" );
                break;
            case D3D12_TILED_RESOURCES_TIER_2:
                LOG( "TiledResourcesTier: D3D12_TILED_RESOURCES_TIER_2\n" );
                break;
            case D3D12_TILED_RESOURCES_TIER_3:
                LOG( "TiledResourcesTier: D3D12_TILED_RESOURCES_TIER_3\n" );
                break;
        }

        if ( m_FeatureSupport.TypedUAVLoadAdditionalFormats )
            LOG( "TypedUAVLoadAdditionalFormats: TRUE\n" );
        else
            LOG( "TypedUAVLoadAdditionalFormats: FALSE\n" );

        if ( m_FeatureSupport.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation )
            LOG( "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation: TRUE\n" );
        else
            LOG( "VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation: FALSE\n" );

        LOG( "----------------------------------------------------------------------------------\n" );
    }
}