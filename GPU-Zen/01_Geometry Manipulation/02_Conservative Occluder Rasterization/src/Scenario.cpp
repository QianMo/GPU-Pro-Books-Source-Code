#include "Scenario.h"

#include <DirectXColors.h>

CScenario::CScenario() :
    m_ModelRootSignature( nullptr ),
    m_InstanceMappingsUpdateRootSignature( nullptr ),
    m_CommandBufferGenerationRootSignature( nullptr ),
    m_ModelCommandSignature( nullptr ),
    m_DefaultPipelineState( nullptr ),
    m_GroundPipelineState( nullptr ),
    m_SkyPipelineState( nullptr ),
    m_InstanceMappingsUpdatePipelineState( nullptr ),
    m_CommandBufferGenerationPipelineState( nullptr ),

    m_FrameConstantsBuffer( nullptr ),
    m_MappedFrameConstantsBuffer( nullptr ),

    m_MainBar( nullptr ),
    m_ShowDepth( FALSE ),
    m_ShowOccluders( FALSE ),
    m_ShowOccludees( FALSE ),
    m_PrevFullscreen( FALSE ),
    m_DepthRasterizerSelection( 4 ),
    m_DepthReprojectionSelection( 0 ),
    m_OcclusionAlgorithmSelection( 0 ),
    m_DrawModeSelection( 0 ),
    m_PrevDepthRasterizerSelection( 0 ),
    m_PrevDepthReprojectionSelection( 0 ),
    m_PrevOcclusionQuerySelection( 0 ),
    m_PrevDrawModeSelection( 0 ),
    m_DepthSize{ 256, 128 },
    m_OccluderSizeThreshold( 0.0f ),

    m_OccluderBar( nullptr ),
    m_SelectedModel( nullptr )
{
}

void CScenario::Create()
{
    ID3D12GraphicsCommandList* command_list = m_GraphicsContext.GetCommandList();

    // Create meshes and other resources
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc;
        D3D12_SAMPLER_DESC sampler_desc;

        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( sizeof( FrameConstants ) ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_FrameConstantsBuffer ) ) );
        m_FrameConstantsBuffer->SetName( L"Frame Constants Buffer" );

        HR( m_FrameConstantsBuffer->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &m_MappedFrameConstantsBuffer ) ) );

        m_FrameConstantsCbv = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        ZeroMemory( &cbv_desc, sizeof( cbv_desc ) );
        cbv_desc.BufferLocation = m_FrameConstantsBuffer->GetGPUVirtualAddress();
        cbv_desc.SizeInBytes = sizeof( FrameConstants );
        m_Device->CreateConstantBufferView( &cbv_desc, m_FrameConstantsCbv.m_Cpu );

        m_Sampler = m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].GenerateHandle();

        ZeroMemory( &sampler_desc, sizeof( sampler_desc ) );
        sampler_desc.Filter = D3D12_FILTER_ANISOTROPIC;
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.MaxAnisotropy = 16;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MinLOD = 0;
        sampler_desc.MaxLOD = FLT_MAX;
        m_Device->CreateSampler( &sampler_desc, m_Sampler.m_Cpu );

        m_CastleLargeOccluders.Load( "res/models", "castle_large_occluders.mds" );
        m_CastleLargeOccluders.Create( m_Device, command_list, m_DescriptorHeaps );
        m_Ground.Load( "res/models", "ground.mds" );
        m_Ground.Create( m_Device, command_list, m_DescriptorHeaps );
    #ifdef _DEBUG
        m_MarketStalls.Load( "res/models", "market_stalls_debug.mds" );
        m_MarketStalls.Create( m_Device, command_list, m_DescriptorHeaps );
        m_CastleSmallDecorations.Load( "res/models", "castle_small_decorations_debug.mds" );
        m_CastleSmallDecorations.Create( m_Device, command_list, m_DescriptorHeaps );
    #else
        m_MarketStalls.Load( "res/models", "market_stalls.mds" );
        m_MarketStalls.Create( m_Device, command_list, m_DescriptorHeaps );
        m_CastleSmallDecorations.Load( "res/models", "castle_small_decorations.mds" );
        m_CastleSmallDecorations.Create( m_Device, command_list, m_DescriptorHeaps );
    #endif
        m_Sky.Load( "res/models", "sky.mds" );
        m_Sky.Create( m_Device, command_list, m_DescriptorHeaps );
    }

    // Create pipelines
    {
        UINT texture_count = 1;
        texture_count = max( texture_count, m_CastleLargeOccluders.GetTextureCount() );
        texture_count = max( texture_count, m_Ground.GetTextureCount() );
        texture_count = max( texture_count, m_MarketStalls.GetTextureCount() );
        texture_count = max( texture_count, m_CastleSmallDecorations.GetTextureCount() );
        texture_count = max( texture_count, m_Sky.GetTextureCount() );

        // Create root signatures
        CD3DX12_DESCRIPTOR_RANGE model_ranges_0[ 1 ];
        model_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE model_ranges_1[ 1 ];
        model_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1 );
        CD3DX12_DESCRIPTOR_RANGE model_ranges_2[ 1 ];
        model_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 2 );
        CD3DX12_DESCRIPTOR_RANGE model_ranges_3[ 1 ];
        model_ranges_3[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, texture_count, 3 );
        CD3DX12_DESCRIPTOR_RANGE model_ranges_4[ 1 ];
        model_ranges_4[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1 );
        CD3DX12_DESCRIPTOR_RANGE model_ranges_5[ 1 ];
        model_ranges_5[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 );
        CD3DX12_ROOT_PARAMETER model_root_parameters[ 7 ];
        model_root_parameters[ 0 ].InitAsConstants( 2, 0 );
        model_root_parameters[ 1 ].InitAsDescriptorTable( _countof( model_ranges_0 ), model_ranges_0, D3D12_SHADER_VISIBILITY_VERTEX );
        model_root_parameters[ 2 ].InitAsDescriptorTable( _countof( model_ranges_1 ), model_ranges_1, D3D12_SHADER_VISIBILITY_VERTEX );
        model_root_parameters[ 3 ].InitAsDescriptorTable( _countof( model_ranges_2 ), model_ranges_2, D3D12_SHADER_VISIBILITY_PIXEL );
        model_root_parameters[ 4 ].InitAsDescriptorTable( _countof( model_ranges_3 ), model_ranges_3, D3D12_SHADER_VISIBILITY_PIXEL );
        model_root_parameters[ 5 ].InitAsDescriptorTable( _countof( model_ranges_4 ), model_ranges_4, D3D12_SHADER_VISIBILITY_ALL );
        model_root_parameters[ 6 ].InitAsDescriptorTable( _countof( model_ranges_5 ), model_ranges_5, D3D12_SHADER_VISIBILITY_PIXEL );
        m_ModelRootSignature = NGraphics::CreateRootSignature( m_Device, _countof( model_root_parameters ), model_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE instance_mappings_update_ranges_0[ 2 ];
        instance_mappings_update_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1 );
        instance_mappings_update_ranges_0[ 1 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0 );
        CD3DX12_DESCRIPTOR_RANGE instance_mappings_update_ranges_1[ 1 ];
        instance_mappings_update_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_ROOT_PARAMETER instance_mappings_update_root_parameters[ 3 ];
        instance_mappings_update_root_parameters[ 0 ].InitAsConstants( 1, 0 );
        instance_mappings_update_root_parameters[ 1 ].InitAsDescriptorTable( _countof( instance_mappings_update_ranges_0 ), instance_mappings_update_ranges_0 );
        instance_mappings_update_root_parameters[ 2 ].InitAsDescriptorTable( _countof( instance_mappings_update_ranges_1 ), instance_mappings_update_ranges_1 );
        m_InstanceMappingsUpdateRootSignature = NGraphics::CreateRootSignature( m_Device, _countof( instance_mappings_update_root_parameters ), instance_mappings_update_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE command_buffer_generation_ranges_0[ 1 ];
        command_buffer_generation_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE command_buffer_generation_ranges_1[ 2 ];
        command_buffer_generation_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1 );
        command_buffer_generation_ranges_1[ 1 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0 );
        CD3DX12_ROOT_PARAMETER command_buffer_generation_root_parameters[ 2 ];
        command_buffer_generation_root_parameters[ 0 ].InitAsDescriptorTable( _countof( command_buffer_generation_ranges_0 ), command_buffer_generation_ranges_0 );
        command_buffer_generation_root_parameters[ 1 ].InitAsDescriptorTable( _countof( command_buffer_generation_ranges_1 ), command_buffer_generation_ranges_1 );
        m_CommandBufferGenerationRootSignature = NGraphics::CreateRootSignature( m_Device, _countof( command_buffer_generation_root_parameters ), command_buffer_generation_root_parameters );

        // Create command signatures
        D3D12_INDIRECT_ARGUMENT_DESC model_argument_descs[ 4 ];
        model_argument_descs[ 0 ].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
        model_argument_descs[ 0 ].Constant.RootParameterIndex = 0;
        model_argument_descs[ 0 ].Constant.Num32BitValuesToSet = 2;
        model_argument_descs[ 0 ].Constant.DestOffsetIn32BitValues = 0;
        model_argument_descs[ 1 ].Type = D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW;
        model_argument_descs[ 1 ].VertexBuffer.Slot = 0;
        model_argument_descs[ 2 ].Type = D3D12_INDIRECT_ARGUMENT_TYPE_INDEX_BUFFER_VIEW;
        model_argument_descs[ 3 ].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED;
        D3D12_COMMAND_SIGNATURE_DESC model_command_signature_desc;
        ZeroMemory( &model_command_signature_desc, sizeof( model_command_signature_desc ) );
        model_command_signature_desc.pArgumentDescs = model_argument_descs;
        model_command_signature_desc.NumArgumentDescs = _countof( model_argument_descs );
        model_command_signature_desc.ByteStride = sizeof( CModelSet::SIndirectCommand );
        HR( m_Device->CreateCommandSignature( &model_command_signature_desc, m_ModelRootSignature, IID_PPV_ARGS( &m_ModelCommandSignature ) ) );

        std::string texture_count_string = std::to_string( texture_count );
        std::string instance_mappings_update_block_size_x_string = std::to_string( m_InstanceMappingsUpdateBlockSizeX );
        std::string command_buffer_generation_block_size_x_string = std::to_string( m_CommandBufferGenerationBlockSizeX );

        const D3D_SHADER_MACRO model_defines[] =
        {
            "TEXTURE_COUNT", texture_count_string.c_str(),
            0, 0
        };
        const D3D_SHADER_MACRO instance_mappings_update_defines[] =
        {
            "BLOCK_SIZE_X", instance_mappings_update_block_size_x_string.c_str(),
            0, 0
        };
        const D3D_SHADER_MACRO command_buffer_generation_defines[] =
        {
            "BLOCK_SIZE_X", command_buffer_generation_block_size_x_string.c_str(),
            0, 0
        };

        // Load shaders
        NGraphics::CShader default_vertex_shader( L"res/shaders/DefaultShader.hlsl", "VSMain", "vs_5_1", model_defines );
        NGraphics::CShader default_pixel_shader( L"res/shaders/DefaultShader.hlsl", "PSMain", "ps_5_1", model_defines );
        NGraphics::CShader ground_vertex_shader( L"res/shaders/GroundShader.hlsl", "VSMain", "vs_5_1", model_defines );
        NGraphics::CShader ground_pixel_shader( L"res/shaders/GroundShader.hlsl", "PSMain", "ps_5_1", model_defines );
        NGraphics::CShader sky_vertex_shader( L"res/shaders/SkyShader.hlsl", "VSMain", "vs_5_1", model_defines );
        NGraphics::CShader sky_pixel_shader( L"res/shaders/SkyShader.hlsl", "PSMain", "ps_5_1", model_defines );
        NGraphics::CShader instance_mappings_update_compute_shader( L"res/shaders/InstanceMappingsUpdateShader.hlsl", "CSMain", "cs_5_0", instance_mappings_update_defines );
        NGraphics::CShader command_buffer_generation_compute_shader( L"res/shaders/CommandBufferGenerationShader.hlsl", "CSMain", "cs_5_0", command_buffer_generation_defines );

        // Create pipeline states
        D3D12_GRAPHICS_PIPELINE_STATE_DESC default_state_desc;
        ZeroMemory( &default_state_desc, sizeof( default_state_desc ) );
        default_state_desc.InputLayout = default_vertex_shader.GetInputLayout();
        default_state_desc.pRootSignature = m_ModelRootSignature;
        default_state_desc.VS = default_vertex_shader.GetShaderBytecode();
        default_state_desc.PS = default_pixel_shader.GetShaderBytecode();
        default_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        default_state_desc.RasterizerState.FrontCounterClockwise = TRUE;
        default_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        default_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        default_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        default_state_desc.SampleMask = UINT_MAX;
        default_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        default_state_desc.NumRenderTargets = 1;
        default_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R8G8B8A8_UNORM;
        default_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        default_state_desc.SampleDesc.Count = 1;
        HR( m_Device->CreateGraphicsPipelineState( &default_state_desc, IID_PPV_ARGS( &m_DefaultPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC ground_state_desc;
        ZeroMemory( &ground_state_desc, sizeof( ground_state_desc ) );
        ground_state_desc.InputLayout = ground_vertex_shader.GetInputLayout();
        ground_state_desc.pRootSignature = m_ModelRootSignature;
        ground_state_desc.VS = ground_vertex_shader.GetShaderBytecode();
        ground_state_desc.PS = ground_pixel_shader.GetShaderBytecode();
        ground_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        ground_state_desc.RasterizerState.FrontCounterClockwise = TRUE;
        ground_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        ground_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        ground_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        ground_state_desc.SampleMask = UINT_MAX;
        ground_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        ground_state_desc.NumRenderTargets = 1;
        ground_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R8G8B8A8_UNORM;
        ground_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        ground_state_desc.SampleDesc.Count = 1;
        HR( m_Device->CreateGraphicsPipelineState( &ground_state_desc, IID_PPV_ARGS( &m_GroundPipelineState ) ) );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC sky_state_desc;
        ZeroMemory( &sky_state_desc, sizeof( sky_state_desc ) );
        sky_state_desc.InputLayout = sky_vertex_shader.GetInputLayout();
        sky_state_desc.pRootSignature = m_ModelRootSignature;
        sky_state_desc.VS = sky_vertex_shader.GetShaderBytecode();
        sky_state_desc.PS = sky_pixel_shader.GetShaderBytecode();
        sky_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        sky_state_desc.RasterizerState.FrontCounterClockwise = TRUE;
        sky_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        sky_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        sky_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        sky_state_desc.SampleMask = UINT_MAX;
        sky_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        sky_state_desc.NumRenderTargets = 1;
        sky_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R8G8B8A8_UNORM;
        sky_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        sky_state_desc.SampleDesc.Count = 1;
        HR( m_Device->CreateGraphicsPipelineState( &sky_state_desc, IID_PPV_ARGS( &m_SkyPipelineState ) ) );

        D3D12_COMPUTE_PIPELINE_STATE_DESC instance_mappings_update_state_desc;
        ZeroMemory( &instance_mappings_update_state_desc, sizeof( instance_mappings_update_state_desc ) );
        instance_mappings_update_state_desc.pRootSignature = m_InstanceMappingsUpdateRootSignature;
        instance_mappings_update_state_desc.CS = instance_mappings_update_compute_shader.GetShaderBytecode();
        HR( m_Device->CreateComputePipelineState( &instance_mappings_update_state_desc, IID_PPV_ARGS( &m_InstanceMappingsUpdatePipelineState ) ) );

        D3D12_COMPUTE_PIPELINE_STATE_DESC command_buffer_generation_state_desc;
        ZeroMemory( &command_buffer_generation_state_desc, sizeof( command_buffer_generation_state_desc ) );
        command_buffer_generation_state_desc.pRootSignature = m_CommandBufferGenerationRootSignature;
        command_buffer_generation_state_desc.CS = command_buffer_generation_compute_shader.GetShaderBytecode();
        HR( m_Device->CreateComputePipelineState( &command_buffer_generation_state_desc, IID_PPV_ARGS( &m_CommandBufferGenerationPipelineState ) ) );
    }

    m_TimestampQueryHeap.Create( m_Device, m_GraphicsContext.GetCommandQueue(), 16 );
    
    m_OccluderCollection.AddOccluderObbs( m_CastleLargeOccluders.GetOccluderObbs() );
    m_OccluderCollection.AddOccluderCylinders( m_CastleLargeOccluders.GetOccluderCylinders() );
    m_OccluderCollection.AddOccluderObbs( m_Ground.GetOccluderObbs() );
    m_OccluderCollection.AddOccluderCylinders( m_Ground.GetOccluderCylinders() );
    m_OccluderCollection.Create( m_Device, &m_GraphicsContext, m_DescriptorHeaps );

    m_OccludeeCollection.AddOccludees( m_CastleLargeOccluders.GetOccludeeAabbs() );
    m_OccludeeCollection.AddOccludees( m_Ground.GetOccludeeAabbs() );
    m_OccludeeCollection.AddOccludees( m_MarketStalls.GetOccludeeAabbs() );
    m_OccludeeCollection.AddOccludees( m_CastleSmallDecorations.GetOccludeeAabbs() );
    m_OccludeeCollection.Create( m_Device, &m_GraphicsContext, m_DescriptorHeaps );

    m_DepthGenerator.Create(
        m_Device, &m_GraphicsContext, m_DescriptorHeaps, &m_OccluderCollection,
        m_DepthSize[ 0 ], m_DepthSize[ 1 ], m_Width, m_Height );

    m_HiZOcclusionAlgorithm.Create(
        m_Device, &m_GraphicsContext, m_DescriptorHeaps,
        &m_OccludeeCollection, &m_DepthGenerator );

    m_IsRasterCullingSupported = m_FeatureSupport.ConservativeRasterizationTier != D3D12_CONSERVATIVE_RASTERIZATION_TIER_NOT_SUPPORTED;
    if ( m_IsRasterCullingSupported )
        m_RasterOcclusionAlgorithm.Create(
            m_Device, &m_GraphicsContext, m_DescriptorHeaps,
            &m_OccludeeCollection, &m_DepthGenerator );
    else
        SDL_ShowSimpleMessageBox( SDL_MESSAGEBOX_INFORMATION, "Info", "This GPU does not support conservative rasterization. Raster culling is disabled.", m_Window );

    m_GraphicsContext.CloseCommandList();
    m_GraphicsContext.ExecuteCommandList();
    m_GraphicsContext.WaitForGpu();

    float fov_y = 60.0f * DirectX::XM_PI / 180.0f;
    m_Camera.SetPerspective( fov_y, m_Viewport.Width, m_Viewport.Height, 1.0f, 2000.0f );
    m_Camera.SetPosition( DirectX::XMFLOAT3( 29.0f, 2.0f, 44.0f ) );
    m_Camera.Yaw( DirectX::XM_PI );

    m_Camera.UpdateViewMatrix();

    m_MainBar = TwNewBar( "MainBar" );
    TwDefine( " MainBar label='Main' position='10 10' size='320 480' visible=true " );
    m_TwBars.push_back( m_MainBar );
    UpdateMainBar();

    m_OccluderBar = TwNewBar( "OccluderBar" );
    TwDefine( " OccluderBar label='Occluder' position='340 10' size='320 480' visible=false " );
    m_TwBars.push_back( m_OccluderBar );

    m_ScreenSize[ 0 ] = m_Width;
    m_ScreenSize[ 1 ] = m_Height;

    m_Fullscreen = FALSE;
}

void CScenario::Destroy()
{
    if ( m_IsRasterCullingSupported )
        m_RasterOcclusionAlgorithm.Destroy();
    m_HiZOcclusionAlgorithm.Destroy();
    m_DepthGenerator.Destroy();
    m_OccludeeCollection.Destroy();
    m_OccluderCollection.Destroy();

    m_TimestampQueryHeap.Destroy();

    m_Sky.Destroy();
    m_CastleSmallDecorations.Destroy();
    m_MarketStalls.Destroy();
    m_Ground.Destroy();
    m_CastleLargeOccluders.Destroy();

    SAFE_RELEASE_UNMAP( m_FrameConstantsBuffer );

    SAFE_RELEASE( m_CommandBufferGenerationPipelineState );
    SAFE_RELEASE( m_InstanceMappingsUpdatePipelineState );
    SAFE_RELEASE( m_SkyPipelineState );
    SAFE_RELEASE( m_GroundPipelineState );
    SAFE_RELEASE( m_DefaultPipelineState );
    SAFE_RELEASE( m_ModelCommandSignature );
    SAFE_RELEASE( m_CommandBufferGenerationRootSignature );
    SAFE_RELEASE( m_InstanceMappingsUpdateRootSignature );
    SAFE_RELEASE( m_ModelRootSignature );
}
void CScenario::Update( float dt )
{
    if ( m_Input.IsMouseButtonDown( SDL_BUTTON_RIGHT ) )
    {
        const Sint32* delta_mouse_position = m_Input.GetDeltaMousePosition();
        m_Camera.Yaw( -delta_mouse_position[ 0 ] * 0.01f );
        m_Camera.Pitch( delta_mouse_position[ 1 ] * 0.01f );
    }

    if ( m_Input.IsKeyUp( SDL_SCANCODE_LCTRL ) )
    {
        const float speed = 20.0f;
        if ( m_Input.IsKeyDown( SDL_SCANCODE_W ) )
            m_Camera.Walk( dt * speed );
        if ( m_Input.IsKeyDown( SDL_SCANCODE_S ) )
            m_Camera.Walk( -dt * speed );
        if ( m_Input.IsKeyDown( SDL_SCANCODE_A ) )
            m_Camera.Strafe( dt * speed );
        if ( m_Input.IsKeyDown( SDL_SCANCODE_D ) )
            m_Camera.Strafe( -dt * speed );
    }
    m_Camera.UpdateViewMatrix();

    if ( m_Input.IsMouseButtonPressed( SDL_BUTTON_MIDDLE ) )
    {
        SelectModel();
    }

    if ( m_Input.IsKeyDown( SDL_SCANCODE_LCTRL ) && m_Input.IsKeyPressed( SDL_SCANCODE_S ) )
    {
        m_CastleLargeOccluders.Save();
        m_Ground.Save();
    }

    if ( m_ScreenSize[ 0 ] != m_Width ||
         m_ScreenSize[ 1 ] != m_Height )
    {
        Resize( m_ScreenSize[ 0 ], m_ScreenSize[ 1 ] );
    }
    
    if ( m_Fullscreen != m_PrevFullscreen )
    {
        Resize( m_ScreenSize[ 0 ], m_ScreenSize[ 1 ] );
        m_PrevFullscreen = m_Fullscreen;
    }

    if ( m_DepthRasterizerSelection != m_PrevDepthRasterizerSelection ||
         m_DepthReprojectionSelection != m_PrevDepthReprojectionSelection ||
         m_OcclusionAlgorithmSelection != m_PrevOcclusionQuerySelection ||
         m_DrawModeSelection != m_PrevDrawModeSelection )
    {
        UpdateMainBar();

        m_PrevDepthRasterizerSelection = m_DepthRasterizerSelection;
        m_PrevDepthReprojectionSelection = m_DepthReprojectionSelection;
        m_PrevOcclusionQuerySelection = m_OcclusionAlgorithmSelection;
        m_PrevDrawModeSelection = m_DrawModeSelection;

        for ( UINT i = 0; i < 16; ++i )
            m_AvgTimings[ i ].Reset();
    }

    DirectX::XMStoreFloat4x4(
        &m_MappedFrameConstantsBuffer->m_ViewProjection,
        DirectX::XMMatrixTranspose( m_Camera.GetViewProjectionFlippedZ() ) );
    DirectX::XMStoreFloat4(
        &m_MappedFrameConstantsBuffer->m_CameraPosition,
        m_Camera.GetPosition() );
    m_MappedFrameConstantsBuffer->m_LightDirection = DirectX::XMFLOAT4( -0.273201f, -0.961691f, -0.0225958f, 0.0f );
}

void CScenario::Draw()
{
    COcclusionAlgorithm* selected_occlusion_algorithm = m_OcclusionAlgorithmSelection == 0 ?
        reinterpret_cast< COcclusionAlgorithm* >( &m_HiZOcclusionAlgorithm ) :
        reinterpret_cast< COcclusionAlgorithm* >( &m_RasterOcclusionAlgorithm );

    ID3D12GraphicsCommandList* command_list = m_GraphicsContext.GetCommandList();

    m_GraphicsContext.ResetCommandList();
    ID3D12DescriptorHeap* descriptor_heaps[] =
    {
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GetHeap(),
        m_DescriptorHeaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].GetHeap()
    };
    command_list->SetDescriptorHeaps( _countof( descriptor_heaps ), descriptor_heaps );

    m_CastleLargeOccluders.CalculateOccluders();
    m_Ground.CalculateOccluders();
    
    m_OccluderCollection.ClearOccluders();
    m_OccluderCollection.AddOccluderObbs( m_CastleLargeOccluders.GetOccluderObbs() );
    m_OccluderCollection.AddOccluderCylinders( m_CastleLargeOccluders.GetOccluderCylinders() );
    m_OccluderCollection.AddOccluderObbs( m_Ground.GetOccluderObbs() );
    m_OccluderCollection.AddOccluderCylinders( m_Ground.GetOccluderCylinders() );
    m_OccluderCollection.Update( command_list );
    
    m_DepthGenerator.Resize( command_list, m_DepthSize[ 0 ], m_DepthSize[ 1 ], m_Width, m_Height );
    selected_occlusion_algorithm->Update( command_list );
    
    m_TimestampQueryHeap.SetTimestampQuery( command_list, 0 );

    m_OccluderCollection.SelectOccluders( command_list, &m_Camera, m_OccluderSizeThreshold );

    m_DepthGenerator.SetRasterizerMode( ( CDepthGenerator::ERasterizerMode )m_DepthRasterizerSelection );
    m_DepthGenerator.SetReprojectionMode( ( CDepthGenerator::EReprojectionMode )m_PrevDepthReprojectionSelection );
    m_DepthGenerator.Execute( command_list, &m_Camera, &m_DepthTargetSrvs[ m_PreviousSwapIndex ] );
    
    selected_occlusion_algorithm->Execute( command_list, &m_Camera );

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 1 );

    switch ( m_DrawModeSelection )
    {
        case 0:
            m_GraphicsContext.CloseCommandList();
            m_GraphicsContext.ExecuteCommandList();
            m_GraphicsContext.WaitForGpu();

            selected_occlusion_algorithm->Readback();

            m_GraphicsContext.ResetCommandList();
            command_list->SetDescriptorHeaps( _countof( descriptor_heaps ), descriptor_heaps );

            BeginFrame( DirectX::Colors::Black );

            DrawScene( command_list, selected_occlusion_algorithm );
            break;
        case 1:
            BeginFrame( DirectX::Colors::Black );

            DrawSceneIndirect( command_list, selected_occlusion_algorithm );
            break;
    }

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 3 );

    m_GraphicsContext.CloseCommandList();
    m_GraphicsContext.ExecuteCommandList();
    m_GraphicsContext.WaitForGpu();

    m_AvgTimings[ 0 ].AddSample( static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 0, 3 ) * 1000000.0f ) ) );
    m_AvgTimings[ 1 ].AddSample( static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 0, 1 ) * 1000000.0f ) ) );
    m_AvgTimings[ 2 ].AddSample( static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 1, 2 ) * 1000000.0f ) ) );
    m_AvgTimings[ 3 ].AddSample( static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 2, 3 ) * 1000000.0f ) ) );

    UINT depth_generator_timings[ 8 ];
    m_DepthGenerator.GetTimings( depth_generator_timings );
    for ( UINT i = 0; i < 8; ++i )
        m_AvgTimings[ i + 4 ].AddSample( depth_generator_timings[ i ] );

    switch ( m_OcclusionAlgorithmSelection )
    {
        case 0:
            m_AvgTimings[ 13 ].AddSample( m_HiZOcclusionAlgorithm.GetDownsampleTime() );
            m_AvgTimings[ 14 ].AddSample( m_HiZOcclusionAlgorithm.GetOcclusionQueryTime() );
            break;
        case 1:
            m_AvgTimings[ 13 ].AddSample( m_RasterOcclusionAlgorithm.GetOcclusionQueryTime() );
            break;
    }

    if ( m_DrawModeSelection == 1 )
    {
        selected_occlusion_algorithm->Readback();
    }

    m_OccludersDrawnString = std::to_string( m_OccluderCollection.GetSelectedOccluderCount() ) + "/" + std::to_string( m_OccluderCollection.GetOccluderInstanceCount() );

    m_VisibleOccludeeCount = 0;
    for ( UINT i = 0; i < m_OccludeeCollection.GetAabbCount(); ++i )
        if ( selected_occlusion_algorithm->IsOccludeeVisible( i ) )
            ++m_VisibleOccludeeCount;
    m_VisibleOccludeeCountString = std::to_string( m_VisibleOccludeeCount ) + "/" + std::to_string( m_OccludeeCollection.GetAabbCount() );

    m_GraphicsContext.ResetCommandList();
    command_list->SetDescriptorHeaps( _countof( descriptor_heaps ), descriptor_heaps );

    ResumeFrame();

    if ( m_ShowDepth ) m_DepthGenerator.DebugDraw( command_list );
    if ( m_ShowOccludees ) m_OccludeeCollection.DebugDraw( command_list, &m_Camera, selected_occlusion_algorithm );
    if ( m_ShowOccluders ) m_OccluderCollection.DebugDraw( command_list, &m_Camera );
    
    DrawBars();

    EndFrame();
    m_GraphicsContext.CloseCommandList();
    m_GraphicsContext.ExecuteCommandList();
    Present();
    m_GraphicsContext.WaitForGpu();
}

void CScenario::DrawScene( ID3D12GraphicsCommandList* command_list, COcclusionAlgorithm* occlusion_algorithm )
{
    std::vector< CModelSet* > model_sets;
    model_sets.push_back( &m_CastleLargeOccluders );
    model_sets.push_back( &m_Ground );
    model_sets.push_back( &m_MarketStalls );
    model_sets.push_back( &m_CastleSmallDecorations );

    UINT occludee_offset = 0;
    for ( CModelSet* model_set : model_sets )
    {
        model_set->UpdateInstanceMappings( command_list, occlusion_algorithm, occludee_offset );
        occludee_offset += model_set->GetInstanceCount();
    }

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 2 );

    command_list->SetGraphicsRootSignature( m_ModelRootSignature );

    command_list->SetGraphicsRootDescriptorTable( 5, m_FrameConstantsCbv.m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 6, m_Sampler.m_Gpu );

    command_list->SetPipelineState( m_GroundPipelineState );
    DrawModelSet( command_list, &m_Ground );

    command_list->SetPipelineState( m_DefaultPipelineState );
    DrawModelSet( command_list, &m_CastleLargeOccluders );
    DrawModelSet( command_list, &m_MarketStalls );
    DrawModelSet( command_list, &m_CastleSmallDecorations );

    command_list->SetPipelineState( m_SkyPipelineState );
    DrawModelSet( command_list, &m_Sky );
}

void CScenario::DrawSceneIndirect( ID3D12GraphicsCommandList* command_list, COcclusionAlgorithm* occlusion_algorithm )
{
    std::vector< CModelSet* > model_sets;
    model_sets.push_back( &m_CastleLargeOccluders );
    model_sets.push_back( &m_Ground );
    model_sets.push_back( &m_MarketStalls );
    model_sets.push_back( &m_CastleSmallDecorations );

    const D3D12_RESOURCE_BARRIER pre_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_DEST )
    };
    command_list->ResourceBarrier( _countof( pre_copy_barriers ), pre_copy_barriers );

    for ( CModelSet* model_set : model_sets )
    {
        command_list->CopyBufferRegion( model_set->GetModelInstanceCountBuffer(), 0, model_set->GetModelInstanceCountBufferReset(), 0, model_set->GetModelInstanceCountBufferSize() );
        command_list->CopyBufferRegion( model_set->GetOutputCommandBuffer(), model_set->GetOutputCommandBufferCounterOffset(), model_set->GetOutputCommandBufferCounterReset(), 0, sizeof( UINT ) );
    }

    const D3D12_RESOURCE_BARRIER post_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS )
    };
    command_list->ResourceBarrier( _countof( post_copy_barriers ), post_copy_barriers );

    command_list->SetComputeRootSignature( m_InstanceMappingsUpdateRootSignature );
    command_list->SetPipelineState( m_InstanceMappingsUpdatePipelineState );

    command_list->SetComputeRootDescriptorTable( 2, occlusion_algorithm->GetVisibilityBufferSrv().m_Gpu );

    UINT instance_offset = 0;
    for ( CModelSet* model_set : model_sets )
    {
        command_list->SetComputeRoot32BitConstant( 0, instance_offset, 0 );
        command_list->SetComputeRootDescriptorTable( 1, model_set->GetInstanceModelMappingsBufferSrv().m_Gpu );
        command_list->Dispatch( ( model_set->GetInstanceCount() + m_InstanceMappingsUpdateBlockSizeX - 1 ) / m_InstanceMappingsUpdateBlockSizeX, 1, 1 );

        instance_offset += model_set->GetInstanceCount();
    }

    const D3D12_RESOURCE_BARRIER post_instance_mappings_update_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetModelInstanceCountBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE )
    };
    command_list->ResourceBarrier( _countof( post_instance_mappings_update_barriers ), post_instance_mappings_update_barriers );

    command_list->SetComputeRootSignature( m_CommandBufferGenerationRootSignature );
    command_list->SetPipelineState( m_CommandBufferGenerationPipelineState );

    for ( CModelSet* model_set : model_sets )
    {
        command_list->SetComputeRootDescriptorTable( 0, model_set->GetModelInstanceCountBufferSrv().m_Gpu );
        command_list->SetComputeRootDescriptorTable( 1, model_set->GetInputCommandBufferSrv().m_Gpu );
        command_list->Dispatch( ( static_cast< UINT >( model_set->GetModels()->size() ) + m_CommandBufferGenerationBlockSizeX - 1 ) / m_CommandBufferGenerationBlockSizeX, 1, 1 );
    }

    const D3D12_RESOURCE_BARRIER post_command_buffer_generation_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetOutputCommandBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleLargeOccluders.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Ground.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_MarketStalls.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_CastleSmallDecorations.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_Sky.GetInstanceIndexMappingsBuffer(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE )
    };
    command_list->ResourceBarrier( _countof( post_command_buffer_generation_barriers ), post_command_buffer_generation_barriers );

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 2 );

    command_list->IASetPrimitiveTopology( D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

    command_list->SetGraphicsRootSignature( m_ModelRootSignature );

    command_list->SetGraphicsRootDescriptorTable( 5, m_FrameConstantsCbv.m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 6, m_Sampler.m_Gpu );

    command_list->SetPipelineState( m_GroundPipelineState );
    DrawModelSetIndirect( command_list, &m_Ground );

    command_list->SetPipelineState( m_DefaultPipelineState );
    DrawModelSetIndirect( command_list, &m_CastleLargeOccluders );
    DrawModelSetIndirect( command_list, &m_MarketStalls );
    DrawModelSetIndirect( command_list, &m_CastleSmallDecorations );

    command_list->SetPipelineState( m_SkyPipelineState );
    DrawModelSet( command_list, &m_Sky );
}

void CScenario::DrawModelSet( ID3D12GraphicsCommandList* command_list, CModelSet* model_set )
{
    std::vector<CModelSet::SModel*>* models = model_set->GetModels();

    command_list->SetGraphicsRootDescriptorTable( 1, model_set->GetInstanceIndexMappingsBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 2, model_set->GetWorldMatrixBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 3, model_set->GetMaterialBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 4, model_set->GetTexturesSrv().m_Gpu );

    UINT instance_offset = 0;
    for ( CModelSet::SModel* model : *models )
    {
        if ( model->m_VisibleInstanceCount > 0 )
        {
            command_list->SetGraphicsRoot32BitConstant( 0, instance_offset, 0 );
            command_list->SetGraphicsRoot32BitConstant( 0, model->m_MaterialIndex, 1 );
            model->m_Mesh.Draw( command_list, model->m_VisibleInstanceCount );
        }

        instance_offset += static_cast< UINT >( model->m_Instances.size() );
    }
}

void CScenario::DrawModelSetIndirect( ID3D12GraphicsCommandList* command_list, CModelSet* model_set )
{
    command_list->SetGraphicsRootDescriptorTable( 1, model_set->GetInstanceIndexMappingsBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 2, model_set->GetWorldMatrixBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 3, model_set->GetMaterialBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 4, model_set->GetTexturesSrv().m_Gpu );
    command_list->ExecuteIndirect(
        m_ModelCommandSignature,
        model_set->GetModelCount(),
        model_set->GetOutputCommandBuffer(),
        0,
        model_set->GetOutputCommandBuffer(),
        model_set->GetOutputCommandBufferCounterOffset() );
}

void CScenario::Resize( UINT width, UINT height )
{
    CGraphicsBase::Resize( width, height );

    float fov_y = 60.0f * DirectX::XM_PI / 180.0f;
    m_Camera.SetPerspective( fov_y, m_Viewport.Width, m_Viewport.Height, 1.0f, 2000.0f );

    m_ScreenSize[ 0 ] = width;
    m_ScreenSize[ 1 ] = height;
}

void CScenario::UpdateMainBar()
{
    TwRemoveAllVars( m_MainBar );

    TwType depth_rasterizer_type = TwDefineEnumFromString( "DepthRasterizerMode", "None,Standard,Standard Fullscreen,Standard Fullscreen Upsample,Inner Conservative" );
    TwType depth_reprojection_type = TwDefineEnumFromString( "DepthReprojectionMode", "Off,On" );
    TwType occlusion_query_type = TwDefineEnumFromString( "OcclusionQueryType", m_IsRasterCullingSupported ? "HiZ,Raster" : "HiZ" );
    TwType draw_mode_type = TwDefineEnumFromString( "DrawModeType", "Readback,Indirect" );
    TwType window_mode_type = TwDefineEnumFromString( "WindowModeType", "Windowed,Fullscreen" );

    TwAddVarRW( m_MainBar, "DepthRasterizerMode", depth_rasterizer_type, &m_DepthRasterizerSelection, " label='Rasterizer' " );
    TwAddVarRW( m_MainBar, "DepthReprojectionMode", depth_reprojection_type, &m_DepthReprojectionSelection, " label='Reprojection' " );
    TwAddVarRW( m_MainBar, "OcclusionQuerySelection", occlusion_query_type, &m_OcclusionAlgorithmSelection, " label='Algorithm' " );
    TwAddVarRW( m_MainBar, "DrawModeSelection", draw_mode_type, &m_DrawModeSelection, " label='Draw Mode' " );
    TwAddVarRW( m_MainBar, "ShowDepth", TW_TYPE_BOOL32, &m_ShowDepth, " label='Show Depth' " );
    TwAddVarRW( m_MainBar, "ShowOccluders", TW_TYPE_BOOL32, &m_ShowOccluders, " label='Show Occluders' " );
    TwAddVarRW( m_MainBar, "ShowOccludees", TW_TYPE_BOOL32, &m_ShowOccludees, " label='Show Occludees' " );
    TwAddVarRW( m_MainBar, "OccluderSizeThreshold", TW_TYPE_FLOAT, &m_OccluderSizeThreshold, " label='Occluder Size Threshold' " );
    TwAddVarRW( m_MainBar, "OccludersDrawn", TW_TYPE_STDSTRING, &m_OccludersDrawnString, " label='Occluders Drawn' " );
    TwAddVarRW( m_MainBar, "VisibleOccludeeCount", TW_TYPE_STDSTRING, &m_VisibleOccludeeCountString, " label='Visible Occludees' " );
    TwAddVarRW( m_MainBar, "WindowMode", window_mode_type, &m_Fullscreen, " label='Window Mode' " );
    TwAddVarRW( m_MainBar, "ScreenWidth", TW_TYPE_UINT32, &m_ScreenSize[ 0 ], " label='Screen Width' " );
    TwAddVarRW( m_MainBar, "ScreenHeight", TW_TYPE_UINT32, &m_ScreenSize[ 1 ], " label='Screen Height' " );
    TwAddVarRW( m_MainBar, "DepthWidth", TW_TYPE_UINT32, &m_DepthSize[ 0 ], " label='Depth Width' " );
    TwAddVarRW( m_MainBar, "DepthHeight", TW_TYPE_UINT32, &m_DepthSize[ 1 ], " label='Depth Height' " );

    TwAddVarRO( m_MainBar, "TotalGpuFrameTime", TW_TYPE_UINT32, m_AvgTimings[ 0 ].GetAveragePtr(), " label='Total GPU Frame Time (\xB5s)' " );
    TwAddVarRO( m_MainBar, "TotalOcclusionCullingTime", TW_TYPE_UINT32, m_AvgTimings[ 1 ].GetAveragePtr(), " label='Total GPU Occlusion Culling Time (\xB5s)' " );
    TwAddVarRO( m_MainBar, "TotalCommandProcessingTime", TW_TYPE_UINT32, m_AvgTimings[ 2 ].GetAveragePtr(), " label='Total GPU Command Processing Time (\xB5s)' " );
    TwAddVarRO( m_MainBar, "TotalGpuDrawTime", TW_TYPE_UINT32, m_AvgTimings[ 3 ].GetAveragePtr(), " label='Total GPU Draw Time (\xB5s)' " );
    
    TwAddVarRO( m_MainBar, "OcclusionCullingDepthTime", TW_TYPE_UINT32, m_AvgTimings[ 4 ].GetAveragePtr(), " label='  Depth Generation (\xB5s)' group='Occlusion Culling Stats' " );
    switch ( m_DepthRasterizerSelection )
    {
        case 0:
            break;
        case 1:
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthRasterTime", TW_TYPE_UINT32, m_AvgTimings[ 5 ].GetAveragePtr(), " label='    Depth Rendering (\xB5s)' group='Occlusion Culling Stats' " );
            break;
        case 2:
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthRasterTime", TW_TYPE_UINT32, m_AvgTimings[ 5 ].GetAveragePtr(), " label='    Depth Rendering (\xB5s)' group='Occlusion Culling Stats' " );
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthDownsampleTime", TW_TYPE_UINT32, m_AvgTimings[ 6 ].GetAveragePtr(), " label='    Depth Downsampling (\xB5s)' group='Occlusion Culling Stats' " );
            break;
        case 3:
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthRasterTime", TW_TYPE_UINT32, m_AvgTimings[ 5 ].GetAveragePtr(), " label='    Depth Rendering (\xB5s)' group='Occlusion Culling Stats' " );
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthDownsampleTime", TW_TYPE_UINT32, m_AvgTimings[ 6 ].GetAveragePtr(), " label='    Depth Upsampling + Downsampling (\xB5s)' group='Occlusion Culling Stats' " );
            break;
        case 4:
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthSilhouetteTime", TW_TYPE_UINT32, m_AvgTimings[ 5 ].GetAveragePtr(), " label='    Silhouette Generation (\xB5s)' group='Occlusion Culling Stats' " );
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthRasterTime", TW_TYPE_UINT32, m_AvgTimings[ 6 ].GetAveragePtr(), " label='    Depth Rendering (\xB5s)' group='Occlusion Culling Stats' " );
            break;
    }
    switch ( m_DepthReprojectionSelection )
    {
        case 0:
            break;
        case 1:
            TwAddVarRO( m_MainBar, "OcclusionCullingPreviousDepthDownsampleTime", TW_TYPE_UINT32, m_AvgTimings[ 7 ].GetAveragePtr(), " label='    Previous Depth Downsampling (\xB5s)' group='Occlusion Culling Stats' " );
            TwAddVarRO( m_MainBar, "OcclusionCullingDepthReprojectionTime", TW_TYPE_UINT32, m_AvgTimings[ 8 ].GetAveragePtr(), " label='    Previous Depth Reprojection (\xB5s)' group='Occlusion Culling Stats' " );
            break;
    }
    switch ( m_OcclusionAlgorithmSelection )
    {
        case 0:
            TwAddVarRO( m_MainBar, "HiZDownsampleTime", TW_TYPE_UINT32, m_AvgTimings[ 13 ].GetAveragePtr(), " label='  HiZ Downsampling (\xB5s)' group='Occlusion Culling Stats' " );
            TwAddVarRO( m_MainBar, "HiZOcclusionQueryTime", TW_TYPE_UINT32, m_AvgTimings[ 14 ].GetAveragePtr(), " label='  HiZ Culling (\xB5s)' group='Occlusion Culling Stats' " );
            break;
        case 1:
            TwAddVarRO( m_MainBar, "RasterOcclusionQueryTime", TW_TYPE_UINT32, m_AvgTimings[ 13 ].GetAveragePtr(), " label='  Raster Culling (\xB5s)' group='Occlusion Culling Stats' " );
            break;
    }
}

void TW_CALL AddOccluderObbCallback( void* client_data )
{
    CScenario* scenario = reinterpret_cast< CScenario* >( client_data );
    scenario->AddOccluderObb();
}
void TW_CALL AddOccluderCylinderCallback( void* client_data )
{
    CScenario* scenario = reinterpret_cast< CScenario* >( client_data );
    scenario->AddOccluderCylinder();
}
void TW_CALL RemoveOccluderObbCallback( void* client_data )
{
    std::pair< CScenario*, size_t >* scenario_occluder_index = reinterpret_cast< std::pair< CScenario*, size_t >* >( client_data );
    scenario_occluder_index->first->RemoveOccluderObb( scenario_occluder_index->second );
}
void TW_CALL RemoveOccluderCylinderCallback( void* client_data )
{
    std::pair< CScenario*, size_t >* scenario_occluder_index = reinterpret_cast< std::pair< CScenario*, size_t >* >( client_data );
    scenario_occluder_index->first->RemoveOccluderCylinder( scenario_occluder_index->second );
}

void CScenario::AddOccluderObb()
{
    m_SelectedModel->m_OccluderObbs.push_back( CModelSet::SOccluderObb() );
    UpdateOccluderBar();
}
void CScenario::AddOccluderCylinder()
{
    m_SelectedModel->m_OccluderCylinders.push_back( CModelSet::SOccluderCylinder() );
    UpdateOccluderBar();
}
void CScenario::RemoveOccluderObb( size_t index )
{
    m_SelectedModel->m_OccluderObbs.erase( m_SelectedModel->m_OccluderObbs.begin() + index );
    UpdateOccluderBar();
}
void CScenario::RemoveOccluderCylinder( size_t index )
{
    m_SelectedModel->m_OccluderCylinders.erase( m_SelectedModel->m_OccluderCylinders.begin() + index );
    UpdateOccluderBar();
}

void CScenario::UpdateOccluderBar()
{
    TwRemoveAllVars( m_OccluderBar );
    if ( m_SelectedModel != nullptr )
    {
        TwAddVarRO( m_OccluderBar, "Model Name", TW_TYPE_STDSTRING, &m_SelectedModel->m_Name, "" );
        
        m_ScenarioOccluderObbIndices.resize( m_SelectedModel->m_OccluderObbs.size() );
        for ( size_t i = 0; i < m_SelectedModel->m_OccluderObbs.size(); ++i )
        {
            TwAddVarRW( m_OccluderBar, std::string( "ObbCenterX" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Center.x, std::string( "label='Center X' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbCenterY" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Center.y, std::string( "label='Center Y' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbCenterZ" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Center.z, std::string( "label='Center Z' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbExtentX" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Extent.x, std::string( "label='Extent X' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbExtentY" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Extent.y, std::string( "label='Extent Y' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbExtentZ" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Extent.z, std::string( "label='Extent Z' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbRotationX" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Rotation.x, std::string( "label='Rotation X' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbRotationY" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Rotation.y, std::string( "label='Rotation Y' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "ObbRotationZ" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderObbs[ i ].m_Rotation.z, std::string( "label='Rotation Z' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );

            m_ScenarioOccluderObbIndices[ i ] = std::pair< CScenario*, size_t >( this, i );
            TwAddButton( m_OccluderBar, std::string( "ObbRemove" + std::to_string( i ) ).c_str(), RemoveOccluderObbCallback, &m_ScenarioOccluderObbIndices[ i ], std::string( "label='Remove' group='Occluder OBB " + std::to_string( i + 1 ) + "'" ).c_str() );
        }

        m_ScenarioOccluderCylinderIndices.resize( m_SelectedModel->m_OccluderCylinders.size() );
        for ( size_t i = 0; i < m_SelectedModel->m_OccluderCylinders.size(); ++i )
        {
            TwAddVarRW( m_OccluderBar, std::string( "CylinderCenterX" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Center.x, std::string( "label='Center X' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderCenterY" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Center.y, std::string( "label='Center Y' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderCenterZ" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Center.z, std::string( "label='Center Z' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderRadius" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Radius, std::string( "label='Radius' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderHeight" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Height, std::string( "label='Height' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderRotationX" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Rotation.x, std::string( "label='Rotation X' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderRotationY" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Rotation.y, std::string( "label='Rotation Y' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
            TwAddVarRW( m_OccluderBar, std::string( "CylinderRotationZ" + std::to_string( i ) ).c_str(), TW_TYPE_FLOAT, &m_SelectedModel->m_OccluderCylinders[ i ].m_Rotation.z, std::string( "label='Rotation Z' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );

            m_ScenarioOccluderCylinderIndices[ i ] = std::pair< CScenario*, size_t >( this, i );
            TwAddButton( m_OccluderBar, std::string( "CylinderRemove" + std::to_string( i ) ).c_str(), RemoveOccluderCylinderCallback, &m_ScenarioOccluderCylinderIndices[ i ], std::string( "label='Remove' group='Occluder Cylinder " + std::to_string( i + 1 ) + "'" ).c_str() );
        }

        TwAddButton( m_OccluderBar, "Add Occluder OBB", AddOccluderObbCallback, this, "" );
        TwAddButton( m_OccluderBar, "Add Occluder Cylinder", AddOccluderCylinderCallback, this, "" );

        TwDefine( " OccluderBar visible=true " );
    }
    else
    {
        TwDefine( " OccluderBar visible=false " );
    }
}

void CScenario::SelectModel()
{
    NGraphics::CCamera::SRay ray = m_Camera.ComputeRay( m_Input.GetMousePosition() );

    DirectX::XMFLOAT3 inverse_direction = DirectX::XMFLOAT3(
        1.0f / ray.m_Direction.x,
        1.0f / ray.m_Direction.y,
        1.0f / ray.m_Direction.z );

    float t_closest = FLT_MAX;
    m_SelectedModel = nullptr;

    std::vector<CModelSet::SModel*> models;
    models.insert( models.begin(), m_CastleLargeOccluders.GetModels()->begin(), m_CastleLargeOccluders.GetModels()->end() );
    //models.insert( models.begin(), m_Ground.GetModels()->begin(), m_Ground.GetModels()->end() );
    for ( CModelSet::SModel* model : models )
    {
        for ( CModelSet::SInstance& instance : model->m_Instances )
        {
            DirectX::XMFLOAT3 aabb_min = instance.m_OccludeeAabbMin;
            DirectX::XMFLOAT3 aabb_max = instance.m_OccludeeAabbMax;

            float t_0 = ( aabb_min.x - ray.m_Origin.x ) * inverse_direction.x;
            float t_1 = ( aabb_max.x - ray.m_Origin.x ) * inverse_direction.x;
            float t_2 = ( aabb_min.y - ray.m_Origin.y ) * inverse_direction.y;
            float t_3 = ( aabb_max.y - ray.m_Origin.y ) * inverse_direction.y;
            float t_4 = ( aabb_min.z - ray.m_Origin.z ) * inverse_direction.z;
            float t_5 = ( aabb_max.z - ray.m_Origin.z ) * inverse_direction.z;

            float t_min = max( max( min( t_0, t_1 ), min( t_2, t_3 ) ), min( t_4, t_5 ) );
            float t_max = min( min( max( t_0, t_1 ), max( t_2, t_3 ) ), max( t_4, t_5 ) );

            if ( t_max >= 0.0f && t_min <= t_max && t_min < t_closest )
            {
                t_closest = t_min;
                m_SelectedModel = model;
            }
        }
    }

    UpdateOccluderBar();
}