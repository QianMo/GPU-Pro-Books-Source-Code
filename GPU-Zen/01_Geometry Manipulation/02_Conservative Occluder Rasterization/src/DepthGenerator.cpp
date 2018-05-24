#include "DepthGenerator.h"
#include "OccluderCollection.h"

#include <DirectXColors.h>
#include <string>

CDepthGenerator::CDepthGenerator() :
    m_RasterizerMode( ERasterizerMode::None ),
    m_ReprojectionMode( EReprojectionMode::Off ),

    m_Device( nullptr ),
    m_OccluderCollection( nullptr ),

    m_StandardDepthRootSignature( nullptr ),
    m_InnerConservativeDepthRootSignature( nullptr ),
    m_DownsamplingRootSignature( nullptr ),
    m_UpsamplingRootSignature( nullptr ),
    m_SilhouetteGenerationRootSignature( nullptr ),
    m_ReprojectionRootSignature( nullptr ),
    m_MergingRootSignature( nullptr ),
    m_DebugDrawRootSignature( nullptr ),

    m_StandardDepthPipelineState( nullptr ),
    m_StandardFullscreenDepthPipelineState( nullptr ),
    m_InnerConservativeDepthPipelineState( nullptr ),
    m_DownsamplingPipelineState( nullptr ),
    m_UpsamplingPipelineState( nullptr ),
    m_SilhouetteGenerationPipelineState( nullptr ),
    m_ReprojectionPipelineState( nullptr ),
    m_MergingPipelineState( nullptr ),
    m_DebugDrawPipelineState( nullptr ),

    m_DepthWidth( 0 ),
    m_DepthHeight( 0 ),
    m_FullscreenWidth( 0 ),
    m_FullscreenHeight( 0 ),
    m_DepthViewport{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f },
    m_FullscreenViewport{ 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f },
    m_DepthScissorRect{ 0, 0, 0, 0 },
    m_FullscreenScissorRect{ 0, 0, 0, 0 },

    m_OccluderCount( 0 ),

    m_ConstantsBuffer( nullptr ),
    m_MappedConstantsBuffer( nullptr ),

    m_DepthTarget( nullptr ),
    m_DepthTargetFullscreen( nullptr ),

    m_DownsampleTargetCount( 0 ),
    m_DownsampleViewports( nullptr ),
    m_DownsampleScissorRects( nullptr ),
    m_DownsampleTargets{ nullptr, nullptr },
    m_DownsampleTargetRtvs( nullptr ),
    m_DownsampleTargetSrvs( nullptr ),

    m_UpsampledDownsampleTargetCount( 0 ),
    m_UpsampledDownsampleViewports( nullptr ),
    m_UpsampledDownsampleScissorRects( nullptr ),
    m_UpsampledDownsampleTargets{ nullptr, nullptr },
    m_UpsampledDownsampleTargetRtvs( nullptr ),
    m_UpsampledDownsampleTargetSrvs( nullptr ),

    m_DepthTargetFullscreenDownsampled( nullptr ),

    m_PreviousDepthDownsampled( nullptr ),

    m_DepthReprojection( nullptr ),
    m_DepthReprojectionReset( nullptr ),

    m_DepthFinal( nullptr )
{
}

void CDepthGenerator::Create(
    ID3D12Device* device,
    NGraphics::CCommandContext* graphics_context,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
    COccluderCollection* occluder_collection,
    UINT depth_width, UINT depth_height,
    UINT fullscreen_width, UINT fullscreen_height )
{
    m_Device = device;
    m_OccluderCollection = occluder_collection;

    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();
    ID3D12CommandQueue* command_queue = graphics_context->GetCommandQueue();

    {
        CD3DX12_DESCRIPTOR_RANGE standard_depth_ranges_0[ 1 ];
        standard_depth_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE standard_depth_ranges_1[ 1 ];
        standard_depth_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_ROOT_PARAMETER standard_depth_root_parameters[ 3 ];
        standard_depth_root_parameters[ 0 ].InitAsConstants( 2, 1, 0, D3D12_SHADER_VISIBILITY_VERTEX );
        standard_depth_root_parameters[ 1 ].InitAsDescriptorTable( _countof( standard_depth_ranges_0 ), standard_depth_ranges_0, D3D12_SHADER_VISIBILITY_ALL );
        standard_depth_root_parameters[ 2 ].InitAsDescriptorTable( _countof( standard_depth_ranges_1 ), standard_depth_ranges_1, D3D12_SHADER_VISIBILITY_VERTEX );
        m_StandardDepthRootSignature = NGraphics::CreateRootSignature( device, _countof( standard_depth_root_parameters ), standard_depth_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE inner_conservative_depth_ranges_0[ 1 ];
        inner_conservative_depth_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE inner_conservative_depth_ranges_1[ 1 ];
        inner_conservative_depth_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE inner_conservative_depth_ranges_2[ 1 ];
        inner_conservative_depth_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1 );
        CD3DX12_ROOT_PARAMETER inner_conservative_depth_root_parameters[ 4 ];
        inner_conservative_depth_root_parameters[ 0 ].InitAsConstants( 2, 1, 0, D3D12_SHADER_VISIBILITY_VERTEX );
        inner_conservative_depth_root_parameters[ 1 ].InitAsDescriptorTable( _countof( inner_conservative_depth_ranges_0 ), inner_conservative_depth_ranges_0, D3D12_SHADER_VISIBILITY_ALL );
        inner_conservative_depth_root_parameters[ 2 ].InitAsDescriptorTable( _countof( inner_conservative_depth_ranges_1 ), inner_conservative_depth_ranges_1, D3D12_SHADER_VISIBILITY_VERTEX );
        inner_conservative_depth_root_parameters[ 3 ].InitAsDescriptorTable( _countof( inner_conservative_depth_ranges_2 ), inner_conservative_depth_ranges_2, D3D12_SHADER_VISIBILITY_PIXEL );
        m_InnerConservativeDepthRootSignature = NGraphics::CreateRootSignature( device, _countof( inner_conservative_depth_root_parameters ), inner_conservative_depth_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE downsampling_ranges_0[ 1 ];
        downsampling_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE downsampling_ranges_1[ 1 ];
        downsampling_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 );
        CD3DX12_ROOT_PARAMETER downsampling_root_parameters[ 3 ];
        downsampling_root_parameters[ 0 ].InitAsConstants( 3, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL );
        downsampling_root_parameters[ 1 ].InitAsDescriptorTable( _countof( downsampling_ranges_0 ), downsampling_ranges_0, D3D12_SHADER_VISIBILITY_PIXEL );
        downsampling_root_parameters[ 2 ].InitAsDescriptorTable( _countof( downsampling_ranges_1 ), downsampling_ranges_1, D3D12_SHADER_VISIBILITY_PIXEL );
        m_DownsamplingRootSignature = NGraphics::CreateRootSignature( device, _countof( downsampling_root_parameters ), downsampling_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE upsampling_ranges_0[ 1 ];
        upsampling_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE upsampling_ranges_1[ 1 ];
        upsampling_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 );
        CD3DX12_ROOT_PARAMETER upsampling_root_parameters[ 3 ];
        upsampling_root_parameters[ 0 ].InitAsConstants( 2, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL );
        upsampling_root_parameters[ 1 ].InitAsDescriptorTable( _countof( upsampling_ranges_0 ), upsampling_ranges_0, D3D12_SHADER_VISIBILITY_PIXEL );
        upsampling_root_parameters[ 2 ].InitAsDescriptorTable( _countof( upsampling_ranges_1 ), upsampling_ranges_1, D3D12_SHADER_VISIBILITY_PIXEL );
        m_UpsamplingRootSignature = NGraphics::CreateRootSignature( device, _countof( upsampling_root_parameters ), upsampling_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE silhouette_generation_ranges_0[ 1 ];
        silhouette_generation_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1 );
        CD3DX12_DESCRIPTOR_RANGE silhouette_generation_ranges_1[ 1 ];
        silhouette_generation_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE silhouette_generation_ranges_2[ 1 ];
        silhouette_generation_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE silhouette_generation_ranges_3[ 1 ];
        silhouette_generation_ranges_3[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0 );
        CD3DX12_ROOT_PARAMETER silhouette_generation_root_parameters[ 5 ];
        silhouette_generation_root_parameters[ 0 ].InitAsConstants( 2, 1 );
        silhouette_generation_root_parameters[ 1 ].InitAsDescriptorTable( _countof( silhouette_generation_ranges_0 ), silhouette_generation_ranges_0 );
        silhouette_generation_root_parameters[ 2 ].InitAsDescriptorTable( _countof( silhouette_generation_ranges_1 ), silhouette_generation_ranges_1 );
        silhouette_generation_root_parameters[ 3 ].InitAsDescriptorTable( _countof( silhouette_generation_ranges_2 ), silhouette_generation_ranges_2 );
        silhouette_generation_root_parameters[ 4 ].InitAsDescriptorTable( _countof( silhouette_generation_ranges_3 ), silhouette_generation_ranges_3 );
        m_SilhouetteGenerationRootSignature = NGraphics::CreateRootSignature( device, _countof( silhouette_generation_root_parameters ), silhouette_generation_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE reprojection_ranges_0[ 1 ];
        reprojection_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE reprojection_ranges_1[ 1 ];
        reprojection_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE reprojection_ranges_2[ 1 ];
        reprojection_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0 );
        CD3DX12_ROOT_PARAMETER reprojection_root_parameters[ 3 ];
        reprojection_root_parameters[ 0 ].InitAsDescriptorTable( _countof( reprojection_ranges_0 ), reprojection_ranges_0 );
        reprojection_root_parameters[ 1 ].InitAsDescriptorTable( _countof( reprojection_ranges_1 ), reprojection_ranges_1 );
        reprojection_root_parameters[ 2 ].InitAsDescriptorTable( _countof( reprojection_ranges_2 ), reprojection_ranges_2 );
        m_ReprojectionRootSignature = NGraphics::CreateRootSignature( device, _countof( reprojection_root_parameters ), reprojection_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE merging_ranges_0[ 1 ];
        merging_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE merging_ranges_1[ 1 ];
        merging_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1 );
        CD3DX12_ROOT_PARAMETER merging_root_parameters[ 2 ];
        merging_root_parameters[ 0 ].InitAsDescriptorTable( _countof( merging_ranges_0 ), merging_ranges_0 );
        merging_root_parameters[ 1 ].InitAsDescriptorTable( _countof( merging_ranges_1 ), merging_ranges_1 );
        m_MergingRootSignature = NGraphics::CreateRootSignature( device, _countof( merging_root_parameters ), merging_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE debug_draw_ranges_0[ 1 ];
        debug_draw_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE debug_draw_ranges_1[ 1 ];
        debug_draw_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 );
        CD3DX12_ROOT_PARAMETER debug_draw_root_parameters[ 2 ];
        debug_draw_root_parameters[ 0 ].InitAsDescriptorTable( _countof( debug_draw_ranges_0 ), debug_draw_ranges_0, D3D12_SHADER_VISIBILITY_PIXEL );
        debug_draw_root_parameters[ 1 ].InitAsDescriptorTable( _countof( debug_draw_ranges_1 ), debug_draw_ranges_1, D3D12_SHADER_VISIBILITY_PIXEL );
        m_DebugDrawRootSignature = NGraphics::CreateRootSignature( device, _countof( debug_draw_root_parameters ), debug_draw_root_parameters );

        std::string reprojection_block_size_x_string = std::to_string( m_ReprojectionBlockSizeX );
        std::string reprojection_block_size_y_string = std::to_string( m_ReprojectionBlockSizeY );
        std::string silhouette_generation_block_size_x_string = std::to_string( m_SilhouetteGenerationBlockSizeX );

        const D3D_SHADER_MACRO standard_fullscreen_depth_defines[] =
        {
            "FULLSCREEN", 0,
            0, 0
        };
        const D3D_SHADER_MACRO inner_conservative_depth_defines[] =
        {
            "INNER_CONSERVATIVE", 0,
            0, 0
        };
        const D3D_SHADER_MACRO reprojection_defines[] =
        {
            "BLOCK_SIZE_X", reprojection_block_size_x_string.c_str(),
            "BLOCK_SIZE_Y", reprojection_block_size_y_string.c_str(),
            0, 0
        };
        const D3D_SHADER_MACRO silhouette_generation_defines[] =
        {
            "BLOCK_SIZE_X", silhouette_generation_block_size_x_string.c_str(),
            0, 0
        };

        NGraphics::CShader standard_depth_vertex_shader( L"res/shaders/DepthShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader standard_fullscreen_depth_vertex_shader( L"res/shaders/DepthShader.hlsl", "VSMain", "vs_5_0", standard_fullscreen_depth_defines );
        NGraphics::CShader standard_fullscreen_depth_geometry_shader( L"res/shaders/DepthShader.hlsl", "GSMain", "gs_5_0", standard_fullscreen_depth_defines );
        NGraphics::CShader standard_fullscreen_depth_pixel_shader( L"res/shaders/DepthShader.hlsl", "PSMain", "ps_5_0", standard_fullscreen_depth_defines );
        NGraphics::CShader inner_conservative_depth_vertex_shader( L"res/shaders/DepthShader.hlsl", "VSMain", "vs_5_0", inner_conservative_depth_defines );
        NGraphics::CShader inner_conservative_depth_geometry_shader( L"res/shaders/DepthShader.hlsl", "GSMain", "gs_5_0", inner_conservative_depth_defines );
        NGraphics::CShader inner_conservative_depth_pixel_shader( L"res/shaders/DepthShader.hlsl", "PSMain", "ps_5_0", inner_conservative_depth_defines );
        NGraphics::CShader downsampling_vertex_shader( L"res/shaders/DepthDownsamplingShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader downsampling_pixel_shader( L"res/shaders/DepthDownsamplingShader.hlsl", "PSMain", "ps_5_0" );
        NGraphics::CShader upsampling_vertex_shader( L"res/shaders/DepthUpsamplingShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader upsampling_pixel_shader( L"res/shaders/DepthUpsamplingShader.hlsl", "PSMain", "ps_5_0" );
        NGraphics::CShader silhouette_generation_compute_shader( L"res/shaders/SilhouetteGenerationShader.hlsl", "CSMain", "cs_5_0", silhouette_generation_defines );
        NGraphics::CShader reprojection_compute_shader( L"res/shaders/DepthReprojectionShader.hlsl", "CSMain", "cs_5_0", reprojection_defines );
        NGraphics::CShader merging_vertex_shader( L"res/shaders/DepthMergingShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader merging_pixel_shader( L"res/shaders/DepthMergingShader.hlsl", "PSMain", "ps_5_0" );
        NGraphics::CShader debug_draw_vertex_shader( L"res/shaders/DebugDrawDepthShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader debug_draw_pixel_shader( L"res/shaders/DebugDrawDepthShader.hlsl", "PSMain", "ps_5_0" );
        
        D3D12_GRAPHICS_PIPELINE_STATE_DESC standard_depth_state_desc;
        ZeroMemory( &standard_depth_state_desc, sizeof( standard_depth_state_desc ) );
        standard_depth_state_desc.InputLayout = standard_depth_vertex_shader.GetInputLayout();
        standard_depth_state_desc.pRootSignature = m_StandardDepthRootSignature;
        standard_depth_state_desc.VS = standard_depth_vertex_shader.GetShaderBytecode();
        //standard_depth_state_desc.GS = standard_depth_geometry_shader.GetShaderBytecode();
        //standard_depth_state_desc.PS = standard_depth_pixel_shader.GetShaderBytecode();
        standard_depth_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        standard_depth_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        standard_depth_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        standard_depth_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        standard_depth_state_desc.SampleMask = UINT_MAX;
        standard_depth_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        standard_depth_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        standard_depth_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &standard_depth_state_desc, IID_PPV_ARGS( &m_StandardDepthPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC standard_fullscreen_depth_state_desc;
        ZeroMemory( &standard_fullscreen_depth_state_desc, sizeof( standard_fullscreen_depth_state_desc ) );
        standard_fullscreen_depth_state_desc.InputLayout = standard_fullscreen_depth_vertex_shader.GetInputLayout();
        standard_fullscreen_depth_state_desc.pRootSignature = m_StandardDepthRootSignature;
        standard_fullscreen_depth_state_desc.VS = standard_fullscreen_depth_vertex_shader.GetShaderBytecode();
        //standard_fullscreen_depth_state_desc.GS = standard_fullscreen_depth_geometry_shader.GetShaderBytecode();
        //standard_fullscreen_depth_state_desc.PS = standard_fullscreen_depth_pixel_shader.GetShaderBytecode();
        standard_fullscreen_depth_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        standard_fullscreen_depth_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        standard_fullscreen_depth_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        standard_fullscreen_depth_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        standard_fullscreen_depth_state_desc.SampleMask = UINT_MAX;
        standard_fullscreen_depth_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        standard_fullscreen_depth_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        standard_fullscreen_depth_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &standard_fullscreen_depth_state_desc, IID_PPV_ARGS( &m_StandardFullscreenDepthPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC inner_conservative_depth_state_desc;
        ZeroMemory( &inner_conservative_depth_state_desc, sizeof( inner_conservative_depth_state_desc ) );
        inner_conservative_depth_state_desc.InputLayout = inner_conservative_depth_vertex_shader.GetInputLayout();
        inner_conservative_depth_state_desc.pRootSignature = m_InnerConservativeDepthRootSignature;
        inner_conservative_depth_state_desc.VS = inner_conservative_depth_vertex_shader.GetShaderBytecode();
        inner_conservative_depth_state_desc.GS = inner_conservative_depth_geometry_shader.GetShaderBytecode();
        inner_conservative_depth_state_desc.PS = inner_conservative_depth_pixel_shader.GetShaderBytecode();
        inner_conservative_depth_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        inner_conservative_depth_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        inner_conservative_depth_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        inner_conservative_depth_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        inner_conservative_depth_state_desc.SampleMask = UINT_MAX;
        inner_conservative_depth_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        inner_conservative_depth_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        inner_conservative_depth_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &inner_conservative_depth_state_desc, IID_PPV_ARGS( &m_InnerConservativeDepthPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC downsampling_state_desc;
        ZeroMemory( &downsampling_state_desc, sizeof( downsampling_state_desc ) );
        downsampling_state_desc.InputLayout = downsampling_vertex_shader.GetInputLayout();
        downsampling_state_desc.pRootSignature = m_DownsamplingRootSignature;
        downsampling_state_desc.VS = downsampling_vertex_shader.GetShaderBytecode();
        downsampling_state_desc.PS = downsampling_pixel_shader.GetShaderBytecode();
        downsampling_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        downsampling_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        downsampling_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        downsampling_state_desc.DepthStencilState.DepthEnable = FALSE;
        downsampling_state_desc.SampleMask = UINT_MAX;
        downsampling_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        downsampling_state_desc.NumRenderTargets = 1;
        downsampling_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R32_FLOAT;
        downsampling_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &downsampling_state_desc, IID_PPV_ARGS( &m_DownsamplingPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC upsampling_state_desc;
        ZeroMemory( &upsampling_state_desc, sizeof( upsampling_state_desc ) );
        upsampling_state_desc.InputLayout = upsampling_vertex_shader.GetInputLayout();
        upsampling_state_desc.pRootSignature = m_UpsamplingRootSignature;
        upsampling_state_desc.VS = upsampling_vertex_shader.GetShaderBytecode();
        upsampling_state_desc.PS = upsampling_pixel_shader.GetShaderBytecode();
        upsampling_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        upsampling_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        upsampling_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        upsampling_state_desc.DepthStencilState.DepthEnable = FALSE;
        upsampling_state_desc.SampleMask = UINT_MAX;
        upsampling_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        upsampling_state_desc.NumRenderTargets = 1;
        upsampling_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R32_FLOAT;
        upsampling_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &upsampling_state_desc, IID_PPV_ARGS( &m_UpsamplingPipelineState ) ) );

        D3D12_COMPUTE_PIPELINE_STATE_DESC silhouette_generation_state_desc;
        ZeroMemory( &silhouette_generation_state_desc, sizeof( silhouette_generation_state_desc ) );
        silhouette_generation_state_desc.pRootSignature = m_SilhouetteGenerationRootSignature;
        silhouette_generation_state_desc.CS = silhouette_generation_compute_shader.GetShaderBytecode();
        HR( device->CreateComputePipelineState( &silhouette_generation_state_desc, IID_PPV_ARGS( &m_SilhouetteGenerationPipelineState ) ) );

        D3D12_COMPUTE_PIPELINE_STATE_DESC reprojection_state_desc;
        ZeroMemory( &reprojection_state_desc, sizeof( reprojection_state_desc ) );
        reprojection_state_desc.pRootSignature = m_ReprojectionRootSignature;
        reprojection_state_desc.CS = reprojection_compute_shader.GetShaderBytecode();
        HR( device->CreateComputePipelineState( &reprojection_state_desc, IID_PPV_ARGS( &m_ReprojectionPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC merging_state_desc;
        ZeroMemory( &merging_state_desc, sizeof( merging_state_desc ) );
        merging_state_desc.InputLayout = merging_vertex_shader.GetInputLayout();
        merging_state_desc.pRootSignature = m_MergingRootSignature;
        merging_state_desc.VS = merging_vertex_shader.GetShaderBytecode();
        merging_state_desc.PS = merging_pixel_shader.GetShaderBytecode();
        merging_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        merging_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        merging_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        merging_state_desc.DepthStencilState.DepthEnable = FALSE;
        merging_state_desc.SampleMask = UINT_MAX;
        merging_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        merging_state_desc.NumRenderTargets = 1;
        merging_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R32_FLOAT;
        merging_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &merging_state_desc, IID_PPV_ARGS( &m_MergingPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC debug_draw_state_desc;
        ZeroMemory( &debug_draw_state_desc, sizeof( debug_draw_state_desc ) );
        debug_draw_state_desc.InputLayout = debug_draw_vertex_shader.GetInputLayout();
        debug_draw_state_desc.pRootSignature = m_DebugDrawRootSignature;
        debug_draw_state_desc.VS = debug_draw_vertex_shader.GetShaderBytecode();
        debug_draw_state_desc.PS = debug_draw_pixel_shader.GetShaderBytecode();
        debug_draw_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        debug_draw_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        debug_draw_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        debug_draw_state_desc.DepthStencilState.DepthEnable = FALSE;
        debug_draw_state_desc.SampleMask = UINT_MAX;
        debug_draw_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        debug_draw_state_desc.NumRenderTargets = 1;
        debug_draw_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R8G8B8A8_UNORM;
        debug_draw_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &debug_draw_state_desc, IID_PPV_ARGS( &m_DebugDrawPipelineState ) ) );
    }

    {
        m_ConstantsBufferCbv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        
        m_DepthSampler = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].GenerateHandle();

        m_DepthTargetDsv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GenerateHandle();
        m_DepthTargetFullscreenDsv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GenerateHandle();
        m_DepthTargetSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_DepthTargetFullscreenSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        
        m_DownsampleTargetRtvs = new NGraphics::SDescriptorHandle[ m_MaxDownsampleTargetCount ];
        m_DownsampleTargetSrvs = new NGraphics::SDescriptorHandle[ m_MaxDownsampleTargetCount ];
        for ( UINT i = 0; i < m_MaxDownsampleTargetCount; ++i )
        {
            m_DownsampleTargetRtvs[ i ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
            m_DownsampleTargetSrvs[ i ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        }

        m_UpsampledDownsampleTargetRtvs = new NGraphics::SDescriptorHandle[ m_MaxDownsampleTargetCount ];
        m_UpsampledDownsampleTargetSrvs = new NGraphics::SDescriptorHandle[ m_MaxDownsampleTargetCount ];
        for ( UINT i = 0; i < m_MaxDownsampleTargetCount; ++i )
        {
            m_UpsampledDownsampleTargetRtvs[ i ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
            m_UpsampledDownsampleTargetSrvs[ i ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        }

        m_DepthTargetFullscreenDownsampledRtv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
        m_DepthTargetFullscreenDownsampledSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        m_PreviousDepthDownsampledRtv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
        m_PreviousDepthDownsampledSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        m_DepthReprojectionUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_DepthReprojectionSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        m_DepthFinalRtv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
        m_DepthFinalSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    }

    {
        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( sizeof( SConstants ) ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_ConstantsBuffer ) ) );
        HR( m_ConstantsBuffer->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &m_MappedConstantsBuffer ) ) );

        CreateQuadMesh( command_list );

        Resize( command_list, depth_width, depth_height, fullscreen_width, fullscreen_height );

        m_TimestampQueryHeap.Create( device, command_queue, 32 );
    }

    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc;
        D3D12_SAMPLER_DESC sampler_desc;

        ZeroMemory( &cbv_desc, sizeof( cbv_desc ) );
        cbv_desc.BufferLocation = m_ConstantsBuffer->GetGPUVirtualAddress();
        cbv_desc.SizeInBytes = sizeof( SConstants );
        device->CreateConstantBufferView( &cbv_desc, m_ConstantsBufferCbv.m_Cpu );

        ZeroMemory( &sampler_desc, sizeof( sampler_desc ) );
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MinLOD = -FLT_MAX;
        sampler_desc.MaxLOD = FLT_MAX;
        device->CreateSampler( &sampler_desc, m_DepthSampler.m_Cpu );
    }
}

void CDepthGenerator::CreateQuadMesh( ID3D12GraphicsCommandList* command_list )
{
    std::vector< DirectX::XMFLOAT4 > vertices;
    vertices.insert( vertices.begin(), {
        { DirectX::XMFLOAT4( -1.0f,  1.0f, 0.0f, 0.0f ) },
        { DirectX::XMFLOAT4( -1.0f, -1.0f, 0.0f, 1.0f ) },
        { DirectX::XMFLOAT4( 1.0f,  1.0f, 1.0f, 0.0f ) },
        { DirectX::XMFLOAT4( 1.0f, -1.0f, 1.0f, 1.0f ) }
    } );
    std::vector<Index> indices;
    indices.insert( indices.begin(), {
        0, 2, 1,
        3, 1, 2
    } );
    m_QuadMesh.Create( m_Device, command_list, &vertices, &indices );
}

void CDepthGenerator::Resize(
    ID3D12GraphicsCommandList* command_list,
    UINT depth_width, UINT depth_height,
    UINT fullscreen_width, UINT fullscreen_height )
{
    if ( depth_width == 0 || depth_height == 0 || fullscreen_width == 0 || fullscreen_height == 0 ||
         ( m_DepthWidth == depth_width && m_DepthHeight == depth_height &&
           m_FullscreenWidth == fullscreen_width && m_FullscreenHeight == fullscreen_height ) )
    {
        return;
    }

    SAFE_RELEASE( m_DepthTarget );
    SAFE_RELEASE( m_DepthTargetFullscreen );
    for ( UINT i = 0; i < m_DownsampleTargetCount; ++i )
    {
        SAFE_RELEASE( m_DownsampleTargets[ i ] );
    }
    for ( UINT i = 0; i < m_UpsampledDownsampleTargetCount; ++i )
    {
        SAFE_RELEASE( m_UpsampledDownsampleTargets[ i ] );
    }
    SAFE_RELEASE( m_DepthTargetFullscreenDownsampled );
    SAFE_RELEASE( m_PreviousDepthDownsampled );
    SAFE_RELEASE( m_DepthReprojection );
    SAFE_RELEASE( m_DepthReprojectionReset );
    SAFE_RELEASE( m_DepthFinal );

    SAFE_DELETE_ARRAY( m_DownsampleViewports );
    SAFE_DELETE_ARRAY( m_DownsampleScissorRects );

    SAFE_DELETE_ARRAY( m_UpsampledDownsampleViewports );
    SAFE_DELETE_ARRAY( m_UpsampledDownsampleScissorRects );

    D3D12_CLEAR_VALUE depth_target_clear_value;
    ZeroMemory( &depth_target_clear_value, sizeof( depth_target_clear_value ) );
    depth_target_clear_value.Format = DXGI_FORMAT_D32_FLOAT;
    depth_target_clear_value.DepthStencil.Depth = 0.0f;

    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, depth_width, depth_height, 1, 1,
            DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL ),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        &depth_target_clear_value,
        IID_PPV_ARGS( &m_DepthTarget ) ) );
    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, fullscreen_width, fullscreen_height, 1, 1,
            DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL ),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        &depth_target_clear_value,
        IID_PPV_ARGS( &m_DepthTargetFullscreen ) ) );
    m_DepthTarget->SetName( L"Depth Target" );
    m_DepthTargetFullscreen->SetName( L"Depth Target Fullscreen" );

    m_DownsampleTargetCount = 1;
    UINT x = fullscreen_width;
    UINT y = fullscreen_height;
    while ( ( x > depth_width || y > depth_height ) &&
            m_DownsampleTargetCount < m_MaxDownsampleTargetCount )
    {
        x /= 2;
        y /= 2;
        if ( x < depth_width ) x = depth_width;
        if ( y < depth_height ) y = depth_height;
        ++m_DownsampleTargetCount;
    }
    m_DownsampleTargetCount = max( m_DownsampleTargetCount, 2 );
    m_DownsampleViewports = new D3D12_VIEWPORT[ m_DownsampleTargetCount ];
    m_DownsampleScissorRects = new D3D12_RECT[ m_DownsampleTargetCount ];
    x = fullscreen_width;
    y = fullscreen_height;
    for ( UINT i = 0; i < m_DownsampleTargetCount; ++i )
    {
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, x, y, 1, 1,
                DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS( &m_DownsampleTargets[ i ] ) ) );
        m_DownsampleTargets[ i ]->SetName( std::wstring( L"Downsample Target " + std::to_wstring( i ) ).c_str() );

        m_DownsampleViewports[ i ] = { 0.0f, 0.0f, static_cast< FLOAT >( x ), static_cast< FLOAT >( y ), 0.0f, 1.0f };
        m_DownsampleScissorRects[ i ] = { 0, 0, static_cast< LONG >( x ), static_cast< LONG >( y ) };

        x /= 2;
        y /= 2;
        if ( x < depth_width ) x = depth_width;
        if ( y < depth_height ) y = depth_height;
    }

    m_UpsampledDownsampleTargetCount = 1;
    UINT upsampled_width = static_cast< UINT >( exp2f( ceilf( log2f( static_cast< FLOAT >( fullscreen_width ) ) ) ) );
    UINT upsampled_height = static_cast< UINT >( exp2f( ceilf( log2f( static_cast< FLOAT >( fullscreen_height ) ) ) ) );
    x = upsampled_width;
    y = upsampled_height;
    while ( ( x > depth_width || y > depth_height ) &&
            m_UpsampledDownsampleTargetCount < m_MaxDownsampleTargetCount )
    {
        x /= 2;
        y /= 2;
        if ( x < depth_width ) x = depth_width;
        if ( y < depth_height ) y = depth_height;
        ++m_UpsampledDownsampleTargetCount;
    }
    m_UpsampledDownsampleTargetCount = max( m_DownsampleTargetCount, 2 );
    m_UpsampledDownsampleViewports = new D3D12_VIEWPORT[ m_DownsampleTargetCount ];
    m_UpsampledDownsampleScissorRects = new D3D12_RECT[ m_DownsampleTargetCount ];
    x = upsampled_width;
    y = upsampled_height;
    for ( UINT i = 0; i < m_UpsampledDownsampleTargetCount; ++i )
    {
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC(
                D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, x, y, 1, 1,
                DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS( &m_UpsampledDownsampleTargets[ i ] ) ) );
        m_UpsampledDownsampleTargets[ i ]->SetName( std::wstring( L"Upsampled Downsample Target " + std::to_wstring( i ) ).c_str() );

        m_UpsampledDownsampleViewports[ i ] = { 0.0f, 0.0f, static_cast< FLOAT >( x ), static_cast< FLOAT >( y ), 0.0f, 1.0f };
        m_UpsampledDownsampleScissorRects[ i ] = { 0, 0, static_cast< LONG >( x ), static_cast< LONG >( y ) };

        x /= 2;
        y /= 2;
        if ( x < depth_width ) x = depth_width;
        if ( y < depth_height ) y = depth_height;
    }

    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, depth_width, depth_height, 1, 1,
            DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS( &m_DepthTargetFullscreenDownsampled ) ) );
    m_DepthTargetFullscreenDownsampled->SetName( L"Depth Target Fullscreen Downsampled" );

    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, depth_width, depth_height, 1, 1,
            DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS( &m_PreviousDepthDownsampled ) ) );
    m_PreviousDepthDownsampled->SetName( L"Previous Depth Downsampled" );

    m_DepthReprojectionResetLayout.Offset = 0;
    m_DepthReprojectionResetLayout.Footprint.Format = DXGI_FORMAT_R32_UINT;
    m_DepthReprojectionResetLayout.Footprint.Width = depth_width;
    m_DepthReprojectionResetLayout.Footprint.Height = depth_height;
    m_DepthReprojectionResetLayout.Footprint.Depth = 1;
    m_DepthReprojectionResetLayout.Footprint.RowPitch = depth_width * 4;
    m_DepthReprojectionResetLayout.Footprint.RowPitch += m_DepthReprojectionResetLayout.Footprint.RowPitch % D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;

    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, depth_width, depth_height, 1, 1,
            DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS( &m_DepthReprojection ) ) );
    UINT64 depth_reprojection_reset_size = GetRequiredIntermediateSize( m_DepthReprojection, 0, 1 );
    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( depth_reprojection_reset_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_DepthReprojectionReset ) ) );
    m_DepthReprojection->SetName( L"Depth Reprojection" );
    m_DepthReprojectionReset->SetName( L"Depth Reprojection Reset" );
    
    void* mapped_depth_reprojection_reset = nullptr;
    HR( m_DepthReprojectionReset->Map( 0, &CD3DX12_RANGE( 0, 0 ), &mapped_depth_reprojection_reset ) );
    memset( mapped_depth_reprojection_reset, 0xffffffff, depth_reprojection_reset_size );
    m_DepthReprojectionReset->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

    HR( m_Device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC(
            D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, depth_width, depth_height, 1, 1,
            DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
        D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS( &m_DepthFinal ) ) );
    m_DepthReprojectionReset->SetName( L"Depth Final" );

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;
    D3D12_RENDER_TARGET_VIEW_DESC rtv_desc;
    D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc;

    ZeroMemory( &dsv_desc, sizeof( dsv_desc ) );
    dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
    dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    m_Device->CreateDepthStencilView( m_DepthTarget, &dsv_desc, m_DepthTargetDsv.m_Cpu );
    m_Device->CreateDepthStencilView( m_DepthTargetFullscreen, &dsv_desc, m_DepthTargetFullscreenDsv.m_Cpu );

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    m_Device->CreateShaderResourceView( m_DepthTarget, &srv_desc, m_DepthTargetSrv.m_Cpu );
    m_Device->CreateShaderResourceView( m_DepthTargetFullscreen, &srv_desc, m_DepthTargetFullscreenSrv.m_Cpu );
    m_Device->CreateShaderResourceView( m_DepthTargetFullscreenDownsampled, &srv_desc, m_DepthTargetFullscreenDownsampledSrv.m_Cpu );
    m_Device->CreateShaderResourceView( m_PreviousDepthDownsampled, &srv_desc, m_PreviousDepthDownsampledSrv.m_Cpu );
    m_Device->CreateShaderResourceView( m_DepthReprojection, &srv_desc, m_DepthReprojectionSrv.m_Cpu );
    m_Device->CreateShaderResourceView( m_DepthFinal, &srv_desc, m_DepthFinalSrv.m_Cpu );

    ZeroMemory( &rtv_desc, sizeof( rtv_desc ) );
    rtv_desc.Format = DXGI_FORMAT_R32_FLOAT;
    rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    m_Device->CreateRenderTargetView( m_DepthTargetFullscreenDownsampled, &rtv_desc, m_DepthTargetFullscreenDownsampledRtv.m_Cpu );
    m_Device->CreateRenderTargetView( m_PreviousDepthDownsampled, &rtv_desc, m_PreviousDepthDownsampledRtv.m_Cpu );
    m_Device->CreateRenderTargetView( m_DepthFinal, &rtv_desc, m_DepthFinalRtv.m_Cpu );

    ZeroMemory( &uav_desc, sizeof( uav_desc ) );
    uav_desc.Format = DXGI_FORMAT_R32_UINT;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    m_Device->CreateUnorderedAccessView( m_DepthReprojection, nullptr, &uav_desc, m_DepthReprojectionUav.m_Cpu );

    ZeroMemory( &rtv_desc, sizeof( rtv_desc ) );
    rtv_desc.Format = DXGI_FORMAT_R32_FLOAT;
    rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv_desc.Texture2D.MipLevels = 1;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    for ( UINT i = 0; i < m_DownsampleTargetCount; ++i )
    {
        m_Device->CreateRenderTargetView( m_DownsampleTargets[ i ], &rtv_desc, m_DownsampleTargetRtvs[ i ].m_Cpu );
        m_Device->CreateShaderResourceView( m_DownsampleTargets[ i ], &srv_desc, m_DownsampleTargetSrvs[ i ].m_Cpu );
    }
    for ( UINT i = 0; i < m_UpsampledDownsampleTargetCount; ++i )
    {
        m_Device->CreateRenderTargetView( m_UpsampledDownsampleTargets[ i ], &rtv_desc, m_UpsampledDownsampleTargetRtvs[ i ].m_Cpu );
        m_Device->CreateShaderResourceView( m_UpsampledDownsampleTargets[ i ], &srv_desc, m_UpsampledDownsampleTargetSrvs[ i ].m_Cpu );
    }

    m_DepthWidth = depth_width;
    m_DepthHeight = depth_height;
    m_FullscreenWidth = fullscreen_width;
    m_FullscreenHeight = fullscreen_height;
    m_DepthViewport = { 0.0f, 0.0f, static_cast< FLOAT >( depth_width ), static_cast< FLOAT >( depth_height ), 0.0f, 1.0f };
    m_FullscreenViewport = { 0.0f, 0.0f, static_cast< FLOAT >( fullscreen_width ), static_cast< FLOAT >( fullscreen_height ),0.0f, 1.0f };
    m_DepthScissorRect = { 0, 0, static_cast< LONG >( depth_width ), static_cast< LONG >( depth_height ) };
    m_FullscreenScissorRect = { 0, 0, static_cast< LONG >( fullscreen_width ), static_cast< LONG >( fullscreen_height ) };
}

void CDepthGenerator::Destroy()
{
    m_TimestampQueryHeap.Destroy();

    SAFE_RELEASE( m_DepthFinal );

    SAFE_RELEASE( m_DepthReprojectionReset );
    SAFE_RELEASE( m_DepthReprojection );
    ZeroMemory( &m_DepthReprojectionResetLayout, sizeof( m_DepthReprojectionResetLayout ) );

    SAFE_RELEASE( m_PreviousDepthDownsampled );

    SAFE_RELEASE( m_DepthTargetFullscreenDownsampled );

    SAFE_DELETE_ARRAY( m_UpsampledDownsampleTargetSrvs );
    SAFE_DELETE_ARRAY( m_UpsampledDownsampleTargetRtvs );
    for ( UINT i = 0; i < m_UpsampledDownsampleTargetCount; ++i )
    {
        SAFE_RELEASE( m_UpsampledDownsampleTargets[ i ] );
    }
    SAFE_DELETE_ARRAY( m_UpsampledDownsampleScissorRects );
    SAFE_DELETE_ARRAY( m_UpsampledDownsampleViewports );
    m_UpsampledDownsampleTargetCount = 0;

    SAFE_DELETE_ARRAY( m_DownsampleTargetSrvs );
    SAFE_DELETE_ARRAY( m_DownsampleTargetRtvs );
    for ( UINT i = 0; i < m_DownsampleTargetCount; ++i )
    {
        SAFE_RELEASE( m_DownsampleTargets[ i ] );
    }
    SAFE_DELETE_ARRAY( m_DownsampleScissorRects );
    SAFE_DELETE_ARRAY( m_DownsampleViewports );
    m_DownsampleTargetCount = 0;

    SAFE_RELEASE( m_DepthTargetFullscreen );
    SAFE_RELEASE( m_DepthTarget );

    m_QuadMesh.Destroy();

    m_MappedConstantsBuffer = nullptr;
    SAFE_RELEASE_UNMAP( m_ConstantsBuffer );

    m_PreviousCamera = NGraphics::CCamera();
    m_CurrentCamera = NGraphics::CCamera();

    m_FullscreenScissorRect = { 0, 0, 0, 0 };
    m_DepthScissorRect = { 0, 0, 0, 0 };
    m_FullscreenViewport = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    m_DepthViewport = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    m_FullscreenHeight = 0;
    m_FullscreenWidth = 0;
    m_DepthHeight = 0;
    m_DepthWidth = 0;

    SAFE_RELEASE( m_DebugDrawPipelineState );
    SAFE_RELEASE( m_MergingPipelineState );
    SAFE_RELEASE( m_ReprojectionPipelineState );
    SAFE_RELEASE( m_SilhouetteGenerationPipelineState );
    SAFE_RELEASE( m_UpsamplingPipelineState );
    SAFE_RELEASE( m_DownsamplingPipelineState );
    SAFE_RELEASE( m_InnerConservativeDepthPipelineState );
    SAFE_RELEASE( m_StandardFullscreenDepthPipelineState );
    SAFE_RELEASE( m_StandardDepthPipelineState );

    SAFE_RELEASE( m_DebugDrawRootSignature );
    SAFE_RELEASE( m_MergingRootSignature );
    SAFE_RELEASE( m_ReprojectionRootSignature );
    SAFE_RELEASE( m_SilhouetteGenerationRootSignature );
    SAFE_RELEASE( m_UpsamplingRootSignature );
    SAFE_RELEASE( m_DownsamplingRootSignature );
    SAFE_RELEASE( m_InnerConservativeDepthRootSignature );
    SAFE_RELEASE( m_StandardDepthRootSignature );

    m_OccluderCollection = nullptr;
    m_Device = nullptr;
}

void CDepthGenerator::SetRasterizerMode( ERasterizerMode rasterizer_mode )
{
    m_RasterizerMode = rasterizer_mode;
}
void CDepthGenerator::SetReprojectionMode( EReprojectionMode reprojection_mode )
{
    m_ReprojectionMode = reprojection_mode;
}

void CDepthGenerator::Execute(
    ID3D12GraphicsCommandList* command_list,
    NGraphics::CCamera* camera,
    NGraphics::SDescriptorHandle* previous_depth_srv )
{
    m_PreviousCamera = m_CurrentCamera;
    m_CurrentCamera = *camera;

    DirectX::XMVECTOR determinant;
    DirectX::XMMATRIX previous_inverse_view_projection_flipped_z = DirectX::XMMatrixInverse( &determinant, m_PreviousCamera.GetViewProjectionFlippedZ() );
    DirectX::XMMATRIX previous_to_current_view_projection_flipped_z = DirectX::XMMatrixMultiply( previous_inverse_view_projection_flipped_z, m_CurrentCamera.GetViewProjectionFlippedZ() );

    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjectionFlippedZ,
        DirectX::XMMatrixTranspose( m_CurrentCamera.GetViewProjectionFlippedZ() ) );
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_PreviousToCurrentViewProjectionFlippedZ,
        DirectX::XMMatrixTranspose( previous_to_current_view_projection_flipped_z ) );
    DirectX::XMStoreFloat4(
        &m_MappedConstantsBuffer->m_CameraPosition,
        m_CurrentCamera.GetPosition() );
    m_MappedConstantsBuffer->m_DepthWidth = m_DepthWidth;
    m_MappedConstantsBuffer->m_DepthHeight = m_DepthHeight;
    m_MappedConstantsBuffer->m_FullscreenWidth = m_FullscreenWidth;
    m_MappedConstantsBuffer->m_FullscreenHeight = m_FullscreenHeight;
    m_MappedConstantsBuffer->m_NearZ = m_CurrentCamera.GetNearZ();
    m_MappedConstantsBuffer->m_FarZ = m_CurrentCamera.GetFarZ();
    m_MappedConstantsBuffer->m_SilhouetteEdgeBufferOffset = m_OccluderCollection->m_SilhouetteEdgeBufferOffset;

    std::vector< COccluderCollection::SOccluderModel >* occluder_models = &m_OccluderCollection->m_OccluderModels;

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 0 );

    ID3D12Resource* rasterized_depth_resource = nullptr;
    NGraphics::SDescriptorHandle* rasterized_depth_srv = nullptr;
    switch ( m_RasterizerMode )
    {
        case None:
        {
            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE ) );
            command_list->ClearDepthStencilView( m_DepthTargetDsv.m_Cpu, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr );
            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

            rasterized_depth_resource = m_DepthTarget;
            rasterized_depth_srv = &m_DepthTargetSrv;
            break;
        }
        // Draw occluders at depth resolution directly
        // Approximate culling results
        case Standard:
        {
            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE ) );

            command_list->SetPipelineState( m_StandardDepthPipelineState );
            command_list->SetGraphicsRootSignature( m_StandardDepthRootSignature );

            command_list->RSSetViewports( 1, &m_DepthViewport );
            command_list->RSSetScissorRects( 1, &m_DepthScissorRect );

            command_list->OMSetRenderTargets( 0, nullptr, false, &m_DepthTargetDsv.m_Cpu );
            command_list->ClearDepthStencilView( m_DepthTargetDsv.m_Cpu, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr );

            command_list->SetGraphicsRootDescriptorTable( 1, m_ConstantsBufferCbv.m_Gpu );
            command_list->SetGraphicsRootDescriptorTable( 2, m_OccluderCollection->m_WorldMatrixBufferSrv.m_Gpu );

            UINT instance_offset = 0;
            for ( COccluderCollection::SOccluderModel& occluder_model : *occluder_models )
            {
                if ( occluder_model.m_SelectedOccluderCount > 0 )
                {
                    command_list->SetGraphicsRoot32BitConstant( 0, instance_offset, 0 );
                    occluder_model.m_OccluderMesh->Draw( command_list, occluder_model.m_SelectedOccluderCount );
                }
                instance_offset += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
            }

            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

            rasterized_depth_resource = m_DepthTarget;
            rasterized_depth_srv = &m_DepthTargetSrv;
            break;
        }
        // Draw occluders at fullscreen resolution and downsample successively into depth resolution
        // Conservative culling results
        case StandardFullscreen:
        {
            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreen, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE ) );

            command_list->SetPipelineState( m_StandardFullscreenDepthPipelineState );
            command_list->SetGraphicsRootSignature( m_StandardDepthRootSignature );

            command_list->RSSetViewports( 1, &m_FullscreenViewport );
            command_list->RSSetScissorRects( 1, &m_FullscreenScissorRect );

            command_list->OMSetRenderTargets( 0, nullptr, false, &m_DepthTargetFullscreenDsv.m_Cpu );
            command_list->ClearDepthStencilView( m_DepthTargetFullscreenDsv.m_Cpu, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr );

            command_list->SetGraphicsRootDescriptorTable( 1, m_ConstantsBufferCbv.m_Gpu );
            command_list->SetGraphicsRootDescriptorTable( 2, m_OccluderCollection->m_WorldMatrixBufferSrv.m_Gpu );

            UINT instance_offset = 0;
            for ( COccluderCollection::SOccluderModel& occluder_model : *occluder_models )
            {
                if ( occluder_model.m_SelectedOccluderCount > 0 )
                {
                    command_list->SetGraphicsRoot32BitConstant( 0, instance_offset, 0 );
                    occluder_model.m_OccluderMesh->Draw( command_list, occluder_model.m_SelectedOccluderCount );
                }
                instance_offset += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
            }

            m_TimestampQueryHeap.SetTimestampQuery( command_list, 1 );

            D3D12_RESOURCE_BARRIER pre_downsampling_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreen, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreenDownsampled, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
            };
            command_list->ResourceBarrier( _countof( pre_downsampling_barriers ), pre_downsampling_barriers );

            Downsample( command_list, &m_DepthTargetFullscreenSrv, &m_DepthTargetFullscreenDownsampledRtv );

            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreenDownsampled, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

            rasterized_depth_resource = m_DepthTargetFullscreenDownsampled;
            rasterized_depth_srv = &m_DepthTargetFullscreenDownsampledSrv;
            break;
        }
        // Draw occluders at fullscreen resolution and downsample successively into depth resolution
        // Conservative culling results
        case StandardFullscreenUpsample:
        {
            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreen, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE ) );

            command_list->SetPipelineState( m_StandardFullscreenDepthPipelineState );
            command_list->SetGraphicsRootSignature( m_StandardDepthRootSignature );

            command_list->RSSetViewports( 1, &m_FullscreenViewport );
            command_list->RSSetScissorRects( 1, &m_FullscreenScissorRect );

            command_list->OMSetRenderTargets( 0, nullptr, false, &m_DepthTargetFullscreenDsv.m_Cpu );
            command_list->ClearDepthStencilView( m_DepthTargetFullscreenDsv.m_Cpu, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr );

            command_list->SetGraphicsRootDescriptorTable( 1, m_ConstantsBufferCbv.m_Gpu );
            command_list->SetGraphicsRootDescriptorTable( 2, m_OccluderCollection->m_WorldMatrixBufferSrv.m_Gpu );

            UINT instance_offset = 0;
            for ( COccluderCollection::SOccluderModel& occluder_model : *occluder_models )
            {
                if ( occluder_model.m_SelectedOccluderCount > 0 )
                {
                    command_list->SetGraphicsRoot32BitConstant( 0, instance_offset, 0 );
                    occluder_model.m_OccluderMesh->Draw( command_list, occluder_model.m_SelectedOccluderCount );
                }
                instance_offset += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
            }

            m_TimestampQueryHeap.SetTimestampQuery( command_list, 1 );

            D3D12_RESOURCE_BARRIER pre_downsampling_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreen, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreenDownsampled, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
            };
            command_list->ResourceBarrier( _countof( pre_downsampling_barriers ), pre_downsampling_barriers );

            UpsampledDownsample( command_list, &m_DepthTargetFullscreenSrv, &m_DepthTargetFullscreenDownsampledRtv );

            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTargetFullscreenDownsampled, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

            rasterized_depth_resource = m_DepthTargetFullscreenDownsampled;
            rasterized_depth_srv = &m_DepthTargetFullscreenDownsampledSrv;
            break;
        }
        // Draw occluders at depth resolution using inner conservative rasterization
        // Discards pixels that are intersected by silhouette edges
        // Conservative culling results
        case InnerConservative:
        {
            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_OccluderCollection->m_SilhouetteEdgeCountBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ) );

            command_list->CopyBufferRegion( m_OccluderCollection->m_SilhouetteEdgeCountBuffer, 0, m_OccluderCollection->m_SilhouetteEdgeCountBufferReset, 0, m_OccluderCollection->m_SilhouetteEdgeCountBufferSize );
            
            D3D12_RESOURCE_BARRIER post_silhouette_reset_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_OccluderCollection->m_SilhouetteEdgeCountBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_OccluderCollection->m_SilhouetteEdgeBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS )
            };
            command_list->ResourceBarrier( _countof( post_silhouette_reset_barriers ), post_silhouette_reset_barriers );
            
            command_list->SetComputeRootSignature( m_SilhouetteGenerationRootSignature );
            command_list->SetPipelineState( m_SilhouetteGenerationPipelineState );

            command_list->SetComputeRootDescriptorTable( 2, m_ConstantsBufferCbv.m_Gpu );
            command_list->SetComputeRootDescriptorTable( 3, m_OccluderCollection->m_WorldMatrixBufferSrv.m_Gpu );
            command_list->SetComputeRootDescriptorTable( 4, m_OccluderCollection->m_SilhouetteEdgeBufferUav.m_Gpu );

            UINT instance_offset = 0;
            for ( COccluderCollection::SOccluderModel& occluder_model : *occluder_models )
            {
                if ( occluder_model.m_SelectedOccluderCount > 0 )
                {
                    command_list->SetComputeRoot32BitConstant( 0, instance_offset, 0 );
                    command_list->SetComputeRoot32BitConstant( 0, occluder_model.m_OccluderMesh->GetFaceCount(), 1 );
                    command_list->SetComputeRootDescriptorTable( 1, occluder_model.m_OccluderMesh->GetVertexSrvBufferSrv().m_Gpu );
                    
                    command_list->Dispatch( ( occluder_model.m_SelectedOccluderCount * occluder_model.m_OccluderMesh->GetFaceCount() * 3 + m_SilhouetteGenerationBlockSizeX - 1 ) / m_SilhouetteGenerationBlockSizeX, 1, 1 );
                }
                instance_offset += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
            }

            D3D12_RESOURCE_BARRIER post_silhouette_generation_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_OccluderCollection->m_SilhouetteEdgeBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_OccluderCollection->m_SilhouetteEdgeCountBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_DEPTH_WRITE )
            };
            command_list->ResourceBarrier( _countof( post_silhouette_generation_barriers ), post_silhouette_generation_barriers );

            m_TimestampQueryHeap.SetTimestampQuery( command_list, 1 );

            command_list->SetPipelineState( m_InnerConservativeDepthPipelineState );
            command_list->SetGraphicsRootSignature( m_InnerConservativeDepthRootSignature );

            command_list->RSSetViewports( 1, &m_DepthViewport );
            command_list->RSSetScissorRects( 1, &m_DepthScissorRect );

            command_list->OMSetRenderTargets( 0, nullptr, false, &m_DepthTargetDsv.m_Cpu );
            command_list->ClearDepthStencilView( m_DepthTargetDsv.m_Cpu, D3D12_CLEAR_FLAG_DEPTH, 0.0f, 0, 0, nullptr );

            command_list->SetGraphicsRootDescriptorTable( 1, m_ConstantsBufferCbv.m_Gpu );
            command_list->SetGraphicsRootDescriptorTable( 2, m_OccluderCollection->m_WorldMatrixBufferSrv.m_Gpu );
            command_list->SetGraphicsRootDescriptorTable( 3, m_OccluderCollection->m_SilhouetteEdgeBufferSrv.m_Gpu );

            instance_offset = 0;
            for ( COccluderCollection::SOccluderModel& occluder_model : *occluder_models )
            {
                if ( occluder_model.m_SelectedOccluderCount > 0 )
                {
                    command_list->SetGraphicsRoot32BitConstant( 0, instance_offset, 0 );
                    occluder_model.m_OccluderMesh->Draw( command_list, occluder_model.m_SelectedOccluderCount );
                }
                instance_offset += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
            }

            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

            rasterized_depth_resource = m_DepthTarget;
            rasterized_depth_srv = &m_DepthTargetSrv;
            break;
        }
    }

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 2 );

    switch ( m_ReprojectionMode )
    {
        case Off:
        {
            D3D12_RESOURCE_BARRIER pre_depth_copy_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( rasterized_depth_resource, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthFinal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST )
            };
            command_list->ResourceBarrier( _countof( pre_depth_copy_barriers ), pre_depth_copy_barriers );

            command_list->CopyTextureRegion( &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthFinal, 0 ), 0, 0, 0, &CD3DX12_TEXTURE_COPY_LOCATION( rasterized_depth_resource, 0 ), nullptr );

            D3D12_RESOURCE_BARRIER post_depth_copy_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( rasterized_depth_resource, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthFinal, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE )
            };
            command_list->ResourceBarrier( _countof( post_depth_copy_barriers ), post_depth_copy_barriers );

            break;
        }
        case On:
        {
            D3D12_RESOURCE_BARRIER pre_reprojection_reset_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_PreviousDepthDownsampled, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthReprojection, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST )
            };
            command_list->ResourceBarrier( _countof( pre_reprojection_reset_barriers ), pre_reprojection_reset_barriers );

            Downsample( command_list, previous_depth_srv, &m_PreviousDepthDownsampledRtv );

            command_list->CopyTextureRegion( &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthReprojection, 0 ), 0, 0, 0, &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthReprojectionReset, m_DepthReprojectionResetLayout ), nullptr );

            D3D12_RESOURCE_BARRIER post_reprojection_reset_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_PreviousDepthDownsampled, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthReprojection, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS )
            };
            command_list->ResourceBarrier( _countof( post_reprojection_reset_barriers ), post_reprojection_reset_barriers );

            m_TimestampQueryHeap.SetTimestampQuery( command_list, 3 );

            command_list->SetPipelineState( m_ReprojectionPipelineState );
            command_list->SetComputeRootSignature( m_ReprojectionRootSignature );

            command_list->SetComputeRootDescriptorTable( 0, m_ConstantsBufferCbv.m_Gpu );
            command_list->SetComputeRootDescriptorTable( 1, m_PreviousDepthDownsampledSrv.m_Gpu );
            command_list->SetComputeRootDescriptorTable( 2, m_DepthReprojectionUav.m_Gpu );

            command_list->Dispatch(
                ( m_DepthWidth + m_ReprojectionBlockSizeX - 1 ) / m_ReprojectionBlockSizeX,
                ( m_DepthHeight + m_ReprojectionBlockSizeY - 1 ) / m_ReprojectionBlockSizeY,
                1 );

            D3D12_RESOURCE_BARRIER post_reprojection_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthReprojection, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthFinal, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
            };
            command_list->ResourceBarrier( _countof( post_reprojection_barriers ), post_reprojection_barriers );

            m_TimestampQueryHeap.SetTimestampQuery( command_list, 4 );

            command_list->SetPipelineState( m_MergingPipelineState );
            command_list->SetGraphicsRootSignature( m_MergingRootSignature );

            command_list->RSSetViewports( 1, &m_DepthViewport );
            command_list->RSSetScissorRects( 1, &m_DepthScissorRect );

            command_list->OMSetRenderTargets( 1, &m_DepthFinalRtv.m_Cpu, true, nullptr );

            command_list->SetGraphicsRootDescriptorTable( 0, rasterized_depth_srv->m_Gpu );
            command_list->SetGraphicsRootDescriptorTable( 1, m_DepthReprojectionSrv.m_Gpu );

            m_QuadMesh.Draw( command_list );

            command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DepthFinal, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

            break;
        }
    }

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 5 );
}

void CDepthGenerator::Downsample(
    ID3D12GraphicsCommandList* command_list,
    NGraphics::SDescriptorHandle* input_srv,
    NGraphics::SDescriptorHandle* output_rtv )
{
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DownsampleTargets[ 1 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET ) );

    command_list->SetPipelineState( m_DownsamplingPipelineState );
    command_list->SetGraphicsRootSignature( m_DownsamplingRootSignature );

    command_list->SetGraphicsRootDescriptorTable( 2, m_DepthSampler.m_Gpu );

    struct DownsamplingConstants
    {
        BOOL m_IsInputSizeEven;
        FLOAT m_InputTexelSizeX;
        FLOAT m_InputTexelSizeY;
    } downsampling_constants;

    for ( UINT i = 1; i < m_DownsampleTargetCount; ++i )
    {
        downsampling_constants.m_IsInputSizeEven =
            ( static_cast< UINT >( m_DownsampleViewports[ i - 1 ].Width ) % 2 == 0 &&
              static_cast< UINT >( m_DownsampleViewports[ i - 1 ].Height ) % 2 == 0 ) ? TRUE : FALSE;
        downsampling_constants.m_InputTexelSizeX = 1.0f / m_DownsampleViewports[ i - 1 ].Width;
        downsampling_constants.m_InputTexelSizeY = 1.0f / m_DownsampleViewports[ i - 1 ].Height;

        command_list->RSSetViewports( 1, &m_DownsampleViewports[ i ] );
        command_list->RSSetScissorRects( 1, &m_DownsampleScissorRects[ i ] );

        command_list->OMSetRenderTargets( 1, i == m_DownsampleTargetCount - 1 ? &output_rtv->m_Cpu : &m_DownsampleTargetRtvs[ i ].m_Cpu, true, nullptr );

        command_list->SetGraphicsRoot32BitConstants( 0, 3, &downsampling_constants, 0 );
        command_list->SetGraphicsRootDescriptorTable( 1, i == 1 ? input_srv->m_Gpu : m_DownsampleTargetSrvs[ i - 1 ].m_Gpu );

        m_QuadMesh.Draw( command_list );

        if ( i < m_DownsampleTargetCount - 1 )
        {
            D3D12_RESOURCE_BARRIER downsampling_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_DownsampleTargets[ i ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DownsampleTargets[ i + 1 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
            };
            command_list->ResourceBarrier( _countof( downsampling_barriers ), downsampling_barriers );
        }
    }

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_DownsampleTargets[ m_DownsampleTargetCount - 1 ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );
}

void CDepthGenerator::UpsampledDownsample(
    ID3D12GraphicsCommandList* command_list,
    NGraphics::SDescriptorHandle* input_srv,
    NGraphics::SDescriptorHandle* output_rtv )
{
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_UpsampledDownsampleTargets[ 0 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET ) );

    command_list->SetPipelineState( m_UpsamplingPipelineState );
    command_list->SetGraphicsRootSignature( m_UpsamplingRootSignature );

    struct UpsamplingConstants
    {
        FLOAT m_HalfInputTexelSizeX;
        FLOAT m_HalfInputTexelSizeY;
    } upsampling_constants;
    upsampling_constants.m_HalfInputTexelSizeX = 0.5f / m_FullscreenViewport.Width;
    upsampling_constants.m_HalfInputTexelSizeY = 0.5f / m_FullscreenViewport.Height;

    command_list->RSSetViewports( 1, &m_UpsampledDownsampleViewports[ 0 ] );
    command_list->RSSetScissorRects( 1, &m_UpsampledDownsampleScissorRects[ 0 ] );

    command_list->OMSetRenderTargets( 1, &m_UpsampledDownsampleTargetRtvs[ 0 ].m_Cpu, true, nullptr );

    command_list->SetGraphicsRoot32BitConstants( 0, 2, &upsampling_constants, 0 );
    command_list->SetGraphicsRootDescriptorTable( 1, input_srv->m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 2, m_DepthSampler.m_Gpu );

    m_QuadMesh.Draw( command_list );

    D3D12_RESOURCE_BARRIER pre_downsampling_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_UpsampledDownsampleTargets[ 0 ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_UpsampledDownsampleTargets[ 1 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET ),
    };
    command_list->ResourceBarrier( _countof( pre_downsampling_barriers ), pre_downsampling_barriers );

    command_list->SetPipelineState( m_DownsamplingPipelineState );
    command_list->SetGraphicsRootSignature( m_DownsamplingRootSignature );

    command_list->SetGraphicsRootDescriptorTable( 2, m_DepthSampler.m_Gpu );

    struct DownsamplingConstants
    {
        BOOL m_IsInputSizeEven;
        FLOAT m_InputTexelSizeX;
        FLOAT m_InputTexelSizeY;
    } downsampling_constants;

    for ( UINT i = 1; i < m_UpsampledDownsampleTargetCount; ++i )
    {
        downsampling_constants.m_IsInputSizeEven =
            ( static_cast< UINT >( m_UpsampledDownsampleViewports[ i - 1 ].Width ) % 2 == 0 &&
              static_cast< UINT >( m_UpsampledDownsampleViewports[ i - 1 ].Height ) % 2 == 0 ) ? TRUE : FALSE;
        downsampling_constants.m_InputTexelSizeX = 1.0f / m_UpsampledDownsampleViewports[ i - 1 ].Width;
        downsampling_constants.m_InputTexelSizeY = 1.0f / m_UpsampledDownsampleViewports[ i - 1 ].Height;

        command_list->RSSetViewports( 1, &m_UpsampledDownsampleViewports[ i ] );
        command_list->RSSetScissorRects( 1, &m_UpsampledDownsampleScissorRects[ i ] );

        command_list->OMSetRenderTargets( 1, i == m_UpsampledDownsampleTargetCount - 1 ? &output_rtv->m_Cpu : &m_UpsampledDownsampleTargetRtvs[ i ].m_Cpu, true, nullptr );

        command_list->SetGraphicsRoot32BitConstants( 0, 3, &downsampling_constants, 0 );
        command_list->SetGraphicsRootDescriptorTable( 1, m_UpsampledDownsampleTargetSrvs[ i - 1 ].m_Gpu );

        m_QuadMesh.Draw( command_list );

        if ( i < m_UpsampledDownsampleTargetCount - 1 )
        {
            D3D12_RESOURCE_BARRIER downsampling_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_UpsampledDownsampleTargets[ i ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_UpsampledDownsampleTargets[ i + 1 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
            };
            command_list->ResourceBarrier( _countof( downsampling_barriers ), downsampling_barriers );
        }
    }

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_UpsampledDownsampleTargets[ m_UpsampledDownsampleTargetCount - 1 ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );
}

void CDepthGenerator::DebugDraw( ID3D12GraphicsCommandList* command_list )
{
    command_list->SetPipelineState( m_DebugDrawPipelineState );
    command_list->SetGraphicsRootSignature( m_DebugDrawRootSignature );

    command_list->SetGraphicsRootDescriptorTable( 0, m_DepthFinalSrv.m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 1, m_DepthSampler.m_Gpu );

    m_QuadMesh.Draw( command_list );
}

void CDepthGenerator::GetTimings( UINT timings[ 8 ] )
{
    timings[ 0 ] = static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 0, 5 ) * 1000000.0f ) );
    timings[ 1 ] = static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 0, 1 ) * 1000000.0f ) );
    timings[ 2 ] = static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 1, 2 ) * 1000000.0f ) );
    timings[ 3 ] = static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 2, 3 ) * 1000000.0f ) );
    timings[ 4 ] = static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 3, 4 ) * 1000000.0f ) );
    timings[ 5 ] = static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 4, 5 ) * 1000000.0f ) );
    timings[ 6 ] = 0;
    timings[ 7 ] = 0;
}

ID3D12Resource* CDepthGenerator::GetDepth() const
{
    return m_DepthFinal;
}

const UINT CDepthGenerator::GetDepthWidth() const
{
    return m_DepthWidth;
}
const UINT CDepthGenerator::GetDepthHeight() const
{
    return m_DepthHeight;
}