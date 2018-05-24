#include "OcclusionAlgorithm.h"
#include "DepthGenerator.h"
#include "OccludeeCollection.h"

#include <DirectXColors.h>
#include <string>

COcclusionAlgorithm::COcclusionAlgorithm() :
    m_OccludeeCollection( nullptr ),
    m_DepthGenerator( nullptr ),

    m_Device( nullptr ),

    m_ConstantsBuffer( nullptr ),
    m_MappedConstantsBuffer( nullptr ),

    m_VisibilityBufferSize( 0 ),
    m_Visibility( nullptr ),
    m_VisibilityBuffer( nullptr ),
    m_VisibilityBufferUpload( nullptr ),
    m_VisibilityBufferReadback( nullptr )
{
}

void COcclusionAlgorithm::Create(
    ID3D12Device* device,
    NGraphics::CCommandContext* graphics_context,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
    COccludeeCollection* occludee_collection,
    CDepthGenerator* depth_generator )
{
    m_OccludeeCollection = occludee_collection;
    m_DepthGenerator = depth_generator;

    m_Device = device;

    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();
    ID3D12CommandQueue* command_queue = graphics_context->GetCommandQueue();

    {
        HR( device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( sizeof( SConstants ) ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_ConstantsBuffer ) ) );
        HR( m_ConstantsBuffer->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &m_MappedConstantsBuffer ) ) );

        m_TimestampQueryHeap.Create( device, command_queue, 32 );
    }

    {
        m_ConstantsBufferCbv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_VisibilityBufferUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_VisibilityBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    }

    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc;

        ZeroMemory( &cbv_desc, sizeof( cbv_desc ) );
        cbv_desc.BufferLocation = m_ConstantsBuffer->GetGPUVirtualAddress();
        cbv_desc.SizeInBytes = sizeof( SConstants );
        device->CreateConstantBufferView( &cbv_desc, m_ConstantsBufferCbv.m_Cpu );
    }

    UpdateVisibilityBuffer( command_list );
}

void COcclusionAlgorithm::UpdateVisibilityBuffer( ID3D12GraphicsCommandList* command_list )
{
    if ( ( m_OccludeeCollection->GetAabbCount() - 1 ) / 32 + 1 != m_VisibilityBufferSize )
    {
        SAFE_RELEASE( m_VisibilityBufferReadback );
        SAFE_RELEASE( m_VisibilityBufferUpload );
        SAFE_RELEASE( m_VisibilityBuffer );
        
        UINT visibility_count = ( m_OccludeeCollection->GetAabbCount() - 1 ) / 32 + 1;
        m_VisibilityBufferSize = visibility_count * sizeof( UINT );

        SAFE_DELETE_ARRAY( m_Visibility );
        m_Visibility = new UINT[ visibility_count ];
        memset( m_Visibility, 1, m_VisibilityBufferSize );

        if ( m_VisibilityBufferSize > 0 )
        {
            HR( m_Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer( m_VisibilityBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                nullptr,
                IID_PPV_ARGS( &m_VisibilityBuffer ) ) );
            HR( m_Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer( m_VisibilityBufferSize ),
                D3D12_RESOURCE_STATE_GENERIC_READ,
                nullptr,
                IID_PPV_ARGS( &m_VisibilityBufferUpload ) ) );
            HR( m_Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_READBACK ),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC::Buffer( m_VisibilityBufferSize ),
                D3D12_RESOURCE_STATE_COPY_DEST,
                nullptr,
                IID_PPV_ARGS( &m_VisibilityBufferReadback ) ) );

            BYTE* mapped_visibility_buffer_upload = nullptr;
            HR( m_VisibilityBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_visibility_buffer_upload ) ) );
            ZeroMemory( mapped_visibility_buffer_upload, m_VisibilityBufferSize );
            m_VisibilityBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
            D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;

            ZeroMemory( &uav_desc, sizeof( uav_desc ) );
            uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            uav_desc.Buffer.NumElements = visibility_count;
            uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            m_Device->CreateUnorderedAccessView( m_VisibilityBuffer, nullptr, &uav_desc, m_VisibilityBufferUav.m_Cpu );

            ZeroMemory( &srv_desc, sizeof( srv_desc ) );
            srv_desc.Format = DXGI_FORMAT_R32_TYPELESS;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            srv_desc.Buffer.NumElements = visibility_count;
            srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            m_Device->CreateShaderResourceView( m_VisibilityBuffer, &srv_desc, m_VisibilityBufferSrv.m_Cpu );
        }
    }
}

void COcclusionAlgorithm::Destroy()
{
    m_TimestampQueryHeap.Destroy();

    SAFE_RELEASE( m_VisibilityBufferReadback );
    SAFE_RELEASE( m_VisibilityBufferUpload );
    SAFE_RELEASE( m_VisibilityBuffer );
    SAFE_DELETE_ARRAY( m_Visibility );
    m_VisibilityBufferSize = 0;

    m_MappedConstantsBuffer = nullptr;
    SAFE_RELEASE_UNMAP( m_ConstantsBuffer );

    SAFE_DELETE_ARRAY( m_Visibility );

    m_Device = nullptr;

    m_DepthGenerator = nullptr;
    m_OccludeeCollection = nullptr;
}

void COcclusionAlgorithm::Update( ID3D12GraphicsCommandList* command_list )
{
    UpdateVisibilityBuffer( command_list );
}

void COcclusionAlgorithm::Readback()
{
    BYTE* mapped_visibility_buffer_readback = nullptr;
    HR( m_VisibilityBufferReadback->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_visibility_buffer_readback ) ) );
    memcpy( m_Visibility, mapped_visibility_buffer_readback, m_VisibilityBufferSize );
    m_VisibilityBufferReadback->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
}

BOOL COcclusionAlgorithm::IsOccludeeVisible( UINT occludee_index )
{
    assert( m_Visibility != nullptr );
    return ( m_Visibility[ occludee_index / 32 ] >> ( occludee_index % 32 ) ) & 1;
}

NGraphics::SDescriptorHandle COcclusionAlgorithm::GetVisibilityBufferSrv() const
{
    return m_VisibilityBufferSrv;
}

CHiZOcclusionAlgorithm::CHiZOcclusionAlgorithm() :
    m_DownsamplingRootSignature( nullptr ),
    m_OcclusionAlgorithmRootSignature( nullptr ),
    m_DownsamplingPipelineState( nullptr ),
    m_OcclusionAlgorithmPipelineState( nullptr ),

    m_DepthHierarchyWidth( 0 ),
    m_DepthHierarchyHeight( 0 ),
    m_DepthHierarchyMipCount( 0 ),
    m_DepthHierarchy{ nullptr, nullptr },
    m_DepthHierarchyViewports( nullptr ),
    m_DepthHierarchyScissorRects( nullptr ),
    m_DepthHierarchyMipRtvs( nullptr ),
    m_DepthHierarchyMipSrvs( nullptr )
{
}

void CHiZOcclusionAlgorithm::Create(
    ID3D12Device* device,
    NGraphics::CCommandContext* graphics_context,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
    COccludeeCollection* occludee_collection,
    CDepthGenerator* depth_generator )
{
    COcclusionAlgorithm::Create( device, graphics_context, descriptor_heaps, occludee_collection, depth_generator );

    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();

    {
        CD3DX12_DESCRIPTOR_RANGE downsampling_ranges_0[ 1 ];
        downsampling_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE downsampling_ranges_1[ 1 ];
        downsampling_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 );
        CD3DX12_ROOT_PARAMETER downsampling_root_parameters[ 3 ];
        downsampling_root_parameters[ 0 ].InitAsConstants( 3, 0, 0, D3D12_SHADER_VISIBILITY_PIXEL );
        downsampling_root_parameters[ 1 ].InitAsDescriptorTable( _countof( downsampling_ranges_0 ), downsampling_ranges_0, D3D12_SHADER_VISIBILITY_PIXEL );
        downsampling_root_parameters[ 2 ].InitAsDescriptorTable( _countof( downsampling_ranges_1 ), downsampling_ranges_1, D3D12_SHADER_VISIBILITY_PIXEL );
        m_DownsamplingRootSignature = NGraphics::CreateRootSignature( device, _countof( downsampling_root_parameters ), downsampling_root_parameters );

        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_0[ 1 ];
        occlusion_algorithm_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_1[ 1 ];
        occlusion_algorithm_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_2[ 1 ];
        occlusion_algorithm_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 1 );
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_3[ 1 ];
        occlusion_algorithm_ranges_3[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_4[ 1 ];
        occlusion_algorithm_ranges_4[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0 );
        CD3DX12_ROOT_PARAMETER occlusion_algorithm_root_parameters[ 5 ];
        occlusion_algorithm_root_parameters[ 0 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_0 ), occlusion_algorithm_ranges_0 );
        occlusion_algorithm_root_parameters[ 1 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_1 ), occlusion_algorithm_ranges_1 );
        occlusion_algorithm_root_parameters[ 2 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_2 ), occlusion_algorithm_ranges_2 );
        occlusion_algorithm_root_parameters[ 3 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_3 ), occlusion_algorithm_ranges_3 );
        occlusion_algorithm_root_parameters[ 4 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_4 ), occlusion_algorithm_ranges_4 );
        m_OcclusionAlgorithmRootSignature = NGraphics::CreateRootSignature( m_Device, _countof( occlusion_algorithm_root_parameters ), occlusion_algorithm_root_parameters );

        std::string occlusion_query_block_size_x_string = std::to_string( m_OcclusionQueryBlockSizeX );

        const D3D_SHADER_MACRO occlusion_query_defines[] =
        {
            "BLOCK_SIZE_X", occlusion_query_block_size_x_string.c_str(),
            0, 0
        };

        NGraphics::CShader downsampling_vertex_shader( L"res/shaders/DepthDownsamplingShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader downsampling_pixel_shader( L"res/shaders/DepthDownsamplingShader.hlsl", "PSMain", "ps_5_0" );
        NGraphics::CShader occlusion_algorithm_compute_shader( L"res/shaders/OcclusionAlgorithmHiZShader.hlsl", "CSMain", "cs_5_0", occlusion_query_defines );

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
        HR( m_Device->CreateGraphicsPipelineState( &downsampling_state_desc, IID_PPV_ARGS( &m_DownsamplingPipelineState ) ) );

        D3D12_COMPUTE_PIPELINE_STATE_DESC occlusion_algorithm_state_desc;
        ZeroMemory( &occlusion_algorithm_state_desc, sizeof( occlusion_algorithm_state_desc ) );
        occlusion_algorithm_state_desc.pRootSignature = m_OcclusionAlgorithmRootSignature;
        occlusion_algorithm_state_desc.CS = occlusion_algorithm_compute_shader.GetShaderBytecode();
        HR( m_Device->CreateComputePipelineState( &occlusion_algorithm_state_desc, IID_PPV_ARGS( &m_OcclusionAlgorithmPipelineState ) ) );
    }

    {
        m_DepthHierarchyMipRtvs = new NGraphics::SDescriptorHandle[ m_MaxDepthHierarchyMipCount ];
        m_DepthHierarchyMipSrvs = new NGraphics::SDescriptorHandle[ m_MaxDepthHierarchyMipCount ];
        for ( UINT i = 0; i < m_MaxDepthHierarchyMipCount; ++i )
        {
            m_DepthHierarchyMipRtvs[ i ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_RTV ].GenerateHandle();
            m_DepthHierarchyMipSrvs[ i ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        }

        m_DepthHierarchySrvs[ 0 ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_DepthHierarchySrvs[ 1 ] = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        m_DepthHierarchySampler = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER ].GenerateHandle();
    }

    {
        CreateQuadMesh( command_list );
    }

    {
        D3D12_SAMPLER_DESC sampler_desc;

        ZeroMemory( &sampler_desc, sizeof( sampler_desc ) );
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MinLOD = -FLT_MAX;
        sampler_desc.MaxLOD = FLT_MAX;
        m_Device->CreateSampler( &sampler_desc, m_DepthHierarchySampler.m_Cpu );
    }

    UpdateDepthHierarchy( command_list );
}

void CHiZOcclusionAlgorithm::CreateQuadMesh( ID3D12GraphicsCommandList* command_list )
{
    std::vector< DirectX::XMFLOAT4 > vertices;
    vertices.insert( vertices.begin(), {
        { DirectX::XMFLOAT4( -1.0f,  1.0f, 0.0f, 0.0f ) },
        { DirectX::XMFLOAT4( -1.0f, -1.0f, 0.0f, 1.0f ) },
        { DirectX::XMFLOAT4(  1.0f,  1.0f, 1.0f, 0.0f ) },
        { DirectX::XMFLOAT4(  1.0f, -1.0f, 1.0f, 1.0f ) }
    } );
    std::vector<Index> indices;
    indices.insert( indices.begin(), {
        0, 2, 1,
        3, 1, 2
    } );
    m_QuadMesh.Create( m_Device, command_list, &vertices, &indices );
}

void CHiZOcclusionAlgorithm::Destroy()
{
    m_QuadMesh.Destroy();

    SAFE_DELETE_ARRAY( m_DepthHierarchyMipSrvs );
    SAFE_DELETE_ARRAY( m_DepthHierarchyMipRtvs );
    SAFE_RELEASE( m_DepthHierarchy[ 1 ] );
    SAFE_RELEASE( m_DepthHierarchy[ 0 ] );
    SAFE_DELETE_ARRAY( m_DepthHierarchyScissorRects );
    SAFE_DELETE_ARRAY( m_DepthHierarchyViewports );
    m_DepthHierarchyMipCount = 0;
    m_DepthHierarchyHeight = 0;
    m_DepthHierarchyWidth = 0;

    SAFE_RELEASE( m_OcclusionAlgorithmPipelineState );
    SAFE_RELEASE( m_DownsamplingPipelineState );
    SAFE_RELEASE( m_OcclusionAlgorithmRootSignature );
    SAFE_RELEASE( m_DownsamplingRootSignature );

    COcclusionAlgorithm::Destroy();
}

void CHiZOcclusionAlgorithm::Update( ID3D12GraphicsCommandList* command_list )
{
    COcclusionAlgorithm::Update( command_list );

    UpdateDepthHierarchy( command_list );
}

void CHiZOcclusionAlgorithm::Execute( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera )
{
    camera->ExtractFrustumPlanes( m_MappedConstantsBuffer->m_FrustumPlanes );
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjection,
        DirectX::XMMatrixTranspose( camera->GetViewProjection() ) );
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjectionFlippedZ,
        DirectX::XMMatrixTranspose( camera->GetViewProjectionFlippedZ() ) );
    DirectX::XMStoreFloat4( &m_MappedConstantsBuffer->m_CameraPosition, camera->GetPosition() );
    m_MappedConstantsBuffer->m_Width = m_DepthHierarchyWidth;
    m_MappedConstantsBuffer->m_Height = m_DepthHierarchyHeight;
    m_MappedConstantsBuffer->m_NearZ = camera->GetNearZ();
    m_MappedConstantsBuffer->m_FarZ = camera->GetFarZ();

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 0 );

    D3D12_RESOURCE_BARRIER pre_depth_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthHierarchy[ 0 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthGenerator->GetDepth(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE )
    };
    command_list->ResourceBarrier( _countof( pre_depth_copy_barriers ), pre_depth_copy_barriers );

    command_list->CopyTextureRegion( &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthHierarchy[ 0 ], 0 ), 0, 0, 0, &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthGenerator->GetDepth(), 0 ), nullptr );

    D3D12_RESOURCE_BARRIER post_depth_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthGenerator->GetDepth(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthHierarchy[ 0 ], D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthHierarchy[ 1 ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
    };
    command_list->ResourceBarrier( _countof( post_depth_copy_barriers ), post_depth_copy_barriers );
    
    command_list->SetPipelineState( m_DownsamplingPipelineState );
    command_list->SetGraphicsRootSignature( m_DownsamplingRootSignature );

    command_list->SetGraphicsRootDescriptorTable( 2, m_DepthHierarchySampler.m_Gpu );

    struct DownsampleConstants
    {
        BOOL m_IsInputSizeEven;
        FLOAT m_InputTexelSizeX;
        FLOAT m_InputTexelSizeY;
    } downsample_constants;

    UINT rtv_index = 0;
    UINT srv_index = 1;
    for ( UINT i = 1; i < m_DepthHierarchyMipCount; ++i )
    {
        downsample_constants.m_IsInputSizeEven =
            ( static_cast< UINT >( m_DepthHierarchyViewports[ i - 1 ].Width ) % 2 == 0 &&
              static_cast< UINT >( m_DepthHierarchyViewports[ i - 1 ].Height ) % 2 == 0 ) ? TRUE : FALSE;
        downsample_constants.m_InputTexelSizeX = 1.0f / m_DepthHierarchyViewports[ i - 1 ].Width;
        downsample_constants.m_InputTexelSizeY = 1.0f / m_DepthHierarchyViewports[ i - 1 ].Height;
       
        command_list->RSSetViewports( 1, &m_DepthHierarchyViewports[ i ] );
        command_list->RSSetScissorRects( 1, &m_DepthHierarchyScissorRects[ i ] );

        command_list->OMSetRenderTargets( 1, &m_DepthHierarchyMipRtvs[ i ].m_Cpu, true, nullptr );

        command_list->SetGraphicsRoot32BitConstants( 0, 3, &downsample_constants, 0 );
        command_list->SetGraphicsRootDescriptorTable( 1, m_DepthHierarchyMipSrvs[ i - 1 ].m_Gpu );

        m_QuadMesh.Draw( command_list );

        rtv_index = 1 - rtv_index;
        srv_index = 1 - srv_index;

        if ( i < m_DepthHierarchyMipCount - 1 )
        {
            D3D12_RESOURCE_BARRIER downsampling_barriers[] =
            {
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthHierarchy[ rtv_index ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
                CD3DX12_RESOURCE_BARRIER::Transition( m_DepthHierarchy[ srv_index ], D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET )
            };
            command_list->ResourceBarrier( _countof( downsampling_barriers ), downsampling_barriers );
        }
    }

    D3D12_RESOURCE_BARRIER post_downsampling_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthHierarchy[ rtv_index ], D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST )
    };
    command_list->ResourceBarrier( _countof( post_downsampling_barriers ), post_downsampling_barriers );

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 1 );

    command_list->CopyBufferRegion( m_VisibilityBuffer, 0, m_VisibilityBufferUpload, 0, m_VisibilityBufferSize );
    
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ) );

    command_list->SetPipelineState( m_OcclusionAlgorithmPipelineState );
    command_list->SetComputeRootSignature( m_OcclusionAlgorithmRootSignature );

    command_list->SetComputeRootDescriptorTable( 0, m_ConstantsBufferCbv.m_Gpu );
    command_list->SetComputeRootDescriptorTable( 1, m_OccludeeCollection->GetAabbBufferSrv().m_Gpu );
    command_list->SetComputeRootDescriptorTable( 2, m_DepthHierarchySrvs[ 0 ].m_Gpu );
    command_list->SetComputeRootDescriptorTable( 3, m_VisibilityBufferUav.m_Gpu );
    command_list->SetComputeRootDescriptorTable( 4, m_DepthHierarchySampler.m_Gpu );

    command_list->Dispatch( ( m_OccludeeCollection->GetAabbCount() + m_OcclusionQueryBlockSizeX - 1 ) / m_OcclusionQueryBlockSizeX, 1, 1 );

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 2 );

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE ) );
    command_list->CopyBufferRegion( m_VisibilityBufferReadback, 0, m_VisibilityBuffer, 0, m_VisibilityBufferSize );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );
}

void CHiZOcclusionAlgorithm::UpdateDepthHierarchy( ID3D12GraphicsCommandList* command_list )
{
    if ( m_DepthHierarchyWidth != m_DepthGenerator->GetDepthWidth() || m_DepthHierarchyHeight != m_DepthGenerator->GetDepthHeight() )
    {
        SAFE_RELEASE( m_DepthHierarchy[ 0 ] );
        SAFE_RELEASE( m_DepthHierarchy[ 1 ] );

        SAFE_DELETE_ARRAY( m_DepthHierarchyViewports );
        SAFE_DELETE_ARRAY( m_DepthHierarchyScissorRects );

        m_DepthHierarchyWidth = m_DepthGenerator->GetDepthWidth();
        m_DepthHierarchyHeight = m_DepthGenerator->GetDepthHeight();
        m_DepthHierarchyMipCount = 0;

        if ( m_DepthHierarchyWidth > 0 && m_DepthHierarchyHeight > 0 )
        {
            m_DepthHierarchyMipCount = static_cast< UINT >( ceilf( log2f( static_cast< FLOAT >( max( m_DepthHierarchyWidth, m_DepthHierarchyHeight ) ) ) ) ) + 1;
            m_DepthHierarchyViewports = new D3D12_VIEWPORT[ m_DepthHierarchyMipCount ];
            m_DepthHierarchyScissorRects = new D3D12_RECT[ m_DepthHierarchyMipCount ];
            UINT x = m_DepthHierarchyWidth;
            UINT y = m_DepthHierarchyHeight;
            for ( UINT i = 0; i < m_DepthHierarchyMipCount; ++i )
            {
                m_DepthHierarchyViewports[ i ] = { 0.0f, 0.0f, static_cast< FLOAT >( x ), static_cast< FLOAT >( y ), 0.0f, 1.0f };
                m_DepthHierarchyScissorRects[ i ] = { 0, 0, static_cast< LONG >( x ), static_cast< LONG >( y ) };

                x /= 2;
                y /= 2;
                if ( x < 1 ) x = 1;
                if ( y < 1 ) y = 1;
            }

            HR( m_Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC(
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, m_DepthHierarchyWidth, m_DepthHierarchyHeight, 1, m_DepthHierarchyMipCount,
                    DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                nullptr,
                IID_PPV_ARGS( &m_DepthHierarchy[ 0 ] ) ) );
            HR( m_Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC(
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, m_DepthHierarchyWidth, m_DepthHierarchyHeight, 1, m_DepthHierarchyMipCount,
                    DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ),
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                nullptr,
                IID_PPV_ARGS( &m_DepthHierarchy[ 1 ] ) ) );

            D3D12_RENDER_TARGET_VIEW_DESC rtv_desc;
            D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;

            ZeroMemory( &rtv_desc, sizeof( rtv_desc ) );
            rtv_desc.Format = DXGI_FORMAT_R32_FLOAT;
            rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            ZeroMemory( &srv_desc, sizeof( srv_desc ) );
            srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = 1;
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            for ( UINT i = 0; i < m_DepthHierarchyMipCount; ++i )
            {
                rtv_desc.Texture2D.MipSlice = i;
                srv_desc.Texture2D.MostDetailedMip = i;
                m_Device->CreateRenderTargetView( m_DepthHierarchy[ i % 2 ], &rtv_desc, m_DepthHierarchyMipRtvs[ i ].m_Cpu );
                m_Device->CreateShaderResourceView( m_DepthHierarchy[ i % 2 ], &srv_desc, m_DepthHierarchyMipSrvs[ i ].m_Cpu );
            }

            ZeroMemory( &srv_desc, sizeof( srv_desc ) );
            srv_desc.Format = DXGI_FORMAT_R32_FLOAT;
            srv_desc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv_desc.Texture2D.MipLevels = m_DepthHierarchyMipCount;
            srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            m_Device->CreateShaderResourceView( m_DepthHierarchy[ 0 ], &srv_desc, m_DepthHierarchySrvs[ 0 ].m_Cpu );
            m_Device->CreateShaderResourceView( m_DepthHierarchy[ 1 ], &srv_desc, m_DepthHierarchySrvs[ 1 ].m_Cpu );
        }
    }
}

UINT CHiZOcclusionAlgorithm::GetDownsampleTime()
{
    return static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 0, 1 ) * 1000000.0f ) );
}
UINT CHiZOcclusionAlgorithm::GetOcclusionQueryTime()
{
    return static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 1, 2 ) * 1000000.0f ) );
}

CRasterOcclusionAlgorithm::CRasterOcclusionAlgorithm() :
    m_OcclusionAlgorithmRootSignature( nullptr ),
    m_OcclusionAlgorithmPipelineState( nullptr ),

    m_DepthTarget( nullptr )
{
}

void CRasterOcclusionAlgorithm::Create( ID3D12Device* device,
                                        NGraphics::CCommandContext* graphics_context,
                                        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
                                        COccludeeCollection* occludee_collection,
                                        CDepthGenerator* depth_generator )
{
    COcclusionAlgorithm::Create( device, graphics_context, descriptor_heaps, occludee_collection, depth_generator );

    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();

    {
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_0[ 1 ];
        occlusion_algorithm_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_1[ 1 ];
        occlusion_algorithm_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE occlusion_algorithm_ranges_2[ 1 ];
        occlusion_algorithm_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0 );
        CD3DX12_ROOT_PARAMETER occlusion_algorithm_root_parameters[ 3 ];
        occlusion_algorithm_root_parameters[ 0 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_0 ), occlusion_algorithm_ranges_0, D3D12_SHADER_VISIBILITY_VERTEX );
        occlusion_algorithm_root_parameters[ 1 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_1 ), occlusion_algorithm_ranges_1, D3D12_SHADER_VISIBILITY_VERTEX );
        occlusion_algorithm_root_parameters[ 2 ].InitAsDescriptorTable( _countof( occlusion_algorithm_ranges_2 ), occlusion_algorithm_ranges_2, D3D12_SHADER_VISIBILITY_ALL );
        m_OcclusionAlgorithmRootSignature = NGraphics::CreateRootSignature( m_Device, _countof( occlusion_algorithm_root_parameters ), occlusion_algorithm_root_parameters );

        NGraphics::CShader occlusion_algorithm_vertex_shader( L"res/shaders/OcclusionAlgorithmRasterShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader occlusion_algorithm_pixel_shader( L"res/shaders/OcclusionAlgorithmRasterShader.hlsl", "PSMain", "ps_5_0" );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC occlusion_algorithm_state_desc;
        ZeroMemory( &occlusion_algorithm_state_desc, sizeof( occlusion_algorithm_state_desc ) );
        occlusion_algorithm_state_desc.InputLayout = occlusion_algorithm_vertex_shader.GetInputLayout();
        occlusion_algorithm_state_desc.pRootSignature = m_OcclusionAlgorithmRootSignature;
        occlusion_algorithm_state_desc.VS = occlusion_algorithm_vertex_shader.GetShaderBytecode();
        occlusion_algorithm_state_desc.PS = occlusion_algorithm_pixel_shader.GetShaderBytecode();
        occlusion_algorithm_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        occlusion_algorithm_state_desc.RasterizerState.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON;
        occlusion_algorithm_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        occlusion_algorithm_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        occlusion_algorithm_state_desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_GREATER;
        occlusion_algorithm_state_desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        occlusion_algorithm_state_desc.SampleMask = UINT_MAX;
        occlusion_algorithm_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        occlusion_algorithm_state_desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        occlusion_algorithm_state_desc.SampleDesc.Count = 1;
        HR( m_Device->CreateGraphicsPipelineState( &occlusion_algorithm_state_desc, IID_PPV_ARGS( &m_OcclusionAlgorithmPipelineState ) ) );
    }

    {
        m_DepthTargetDsv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_DSV ].GenerateHandle();
    }

    UpdateDepthTarget( command_list );
}

void CRasterOcclusionAlgorithm::Destroy()
{
    SAFE_RELEASE( m_DepthTarget );

    SAFE_RELEASE( m_OcclusionAlgorithmPipelineState );
    SAFE_RELEASE( m_OcclusionAlgorithmRootSignature );

    COcclusionAlgorithm::Destroy();
}

void CRasterOcclusionAlgorithm::Update( ID3D12GraphicsCommandList* command_list )
{
    COcclusionAlgorithm::Update( command_list );

    UpdateDepthTarget( command_list );
}

void CRasterOcclusionAlgorithm::Execute( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera )
{
    camera->ExtractFrustumPlanes( m_MappedConstantsBuffer->m_FrustumPlanes );
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjection,
        DirectX::XMMatrixTranspose( camera->GetViewProjection() ) );
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjectionFlippedZ,
        DirectX::XMMatrixTranspose( camera->GetViewProjectionFlippedZ() ) );
    DirectX::XMStoreFloat4( &m_MappedConstantsBuffer->m_CameraPosition, camera->GetPosition() );
    m_MappedConstantsBuffer->m_Width = m_DepthTargetWidth;
    m_MappedConstantsBuffer->m_Height = m_DepthTargetHeight;
    m_MappedConstantsBuffer->m_NearZ = camera->GetNearZ();
    m_MappedConstantsBuffer->m_FarZ = camera->GetFarZ();

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 0 );

    D3D12_RESOURCE_BARRIER pre_depth_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_DEPTH_READ, D3D12_RESOURCE_STATE_COPY_DEST ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthGenerator->GetDepth(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE )
    };
    command_list->ResourceBarrier( _countof( pre_depth_copy_barriers ), pre_depth_copy_barriers );
    
    command_list->CopyTextureRegion( &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthTarget, 0 ), 0, 0, 0, &CD3DX12_TEXTURE_COPY_LOCATION( m_DepthGenerator->GetDepth(), 0 ), nullptr );
    
    D3D12_RESOURCE_BARRIER post_depth_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthGenerator->GetDepth(), D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_DepthTarget, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_DEPTH_READ )
    };
    command_list->ResourceBarrier( _countof( post_depth_copy_barriers ), post_depth_copy_barriers );

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ) );
    command_list->CopyBufferRegion( m_VisibilityBuffer, 0, m_VisibilityBufferUpload, 0, m_VisibilityBufferSize );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS ) );

    command_list->SetGraphicsRootSignature( m_OcclusionAlgorithmRootSignature );
    command_list->SetPipelineState( m_OcclusionAlgorithmPipelineState );

    command_list->RSSetViewports( 1, &m_Viewport );
    command_list->RSSetScissorRects( 1, &m_ScissorRect );

    command_list->OMSetRenderTargets( 0, nullptr, false, &m_DepthTargetDsv.m_Cpu );

    command_list->SetGraphicsRootDescriptorTable( 0, m_ConstantsBufferCbv.m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 1, m_OccludeeCollection->GetAabbBufferSrv().m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 2, m_VisibilityBufferUav.m_Gpu );

    m_OccludeeCollection->DrawAabbs( command_list );

    m_TimestampQueryHeap.SetTimestampQuery( command_list, 1 );

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE ) );
    command_list->CopyBufferRegion( m_VisibilityBufferReadback, 0, m_VisibilityBuffer, 0, m_VisibilityBufferSize );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_VisibilityBuffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );
}

void CRasterOcclusionAlgorithm::UpdateDepthTarget( ID3D12GraphicsCommandList* command_list )
{
    if ( m_DepthTargetWidth != m_DepthGenerator->GetDepthWidth() || m_DepthTargetHeight != m_DepthGenerator->GetDepthHeight() )
    {
        SAFE_RELEASE( m_DepthTarget );

        m_DepthTargetWidth = m_DepthGenerator->GetDepthWidth();
        m_DepthTargetHeight = m_DepthGenerator->GetDepthHeight();
        m_Viewport = { 0.0f, 0.0f, static_cast< FLOAT >( m_DepthTargetWidth ), static_cast< FLOAT >( m_DepthTargetHeight ), 0.0f, 1.0f };
        m_ScissorRect = { 0, 0, static_cast< LONG >( m_DepthTargetWidth ), static_cast< LONG >( m_DepthTargetHeight ) };

        if ( m_DepthTargetWidth > 0 && m_DepthTargetHeight > 0 )
        {
            HR( m_Device->CreateCommittedResource(
                &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
                D3D12_HEAP_FLAG_NONE,
                &CD3DX12_RESOURCE_DESC(
                    D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, m_DepthTargetWidth, m_DepthTargetHeight, 1, 1,
                    DXGI_FORMAT_R32_TYPELESS, 1, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL ),
                D3D12_RESOURCE_STATE_DEPTH_READ,
                nullptr,
                IID_PPV_ARGS( &m_DepthTarget ) ) );

            D3D12_DEPTH_STENCIL_VIEW_DESC dsv_desc;

            ZeroMemory( &dsv_desc, sizeof( dsv_desc ) );
            dsv_desc.Format = DXGI_FORMAT_D32_FLOAT;
            dsv_desc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
            m_Device->CreateDepthStencilView( m_DepthTarget, &dsv_desc, m_DepthTargetDsv.m_Cpu );
        }
    }
}

UINT CRasterOcclusionAlgorithm::GetOcclusionQueryTime()
{
    return static_cast< UINT >( roundf( m_TimestampQueryHeap.GetTimeDifference( 0, 1 ) * 1000000.0f ) );
}