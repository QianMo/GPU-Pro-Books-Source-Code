#include "OccludeeCollection.h"
#include "OcclusionAlgorithm.h"

COccludeeCollection::COccludeeCollection() :
    m_Device( nullptr ),

    m_DebugDrawOccludeeRootSignature( nullptr ),
    m_DebugDrawOccludeeAabbPipelineState( nullptr ),

    m_AabbBuffer( nullptr ),
    m_AabbBufferUpload( nullptr ),

    m_ConstantsBuffer( nullptr ),
    m_MappedConstantsBuffer( nullptr )
{
}

void COccludeeCollection::AddOccludees( std::vector< COccludeeCollection::SAabb >* occludees )
{
    m_Aabbs.insert( m_Aabbs.end(), occludees->begin(), occludees->end() );
}

void COccludeeCollection::ClearOccludees()
{
    m_Aabbs.clear();
}

void COccludeeCollection::Create( ID3D12Device* device,
                                  NGraphics::CCommandContext* graphics_context,
                                  NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    m_Device = device;

    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();

    {
        CD3DX12_DESCRIPTOR_RANGE debug_draw_occludee_ranges_0[ 1 ];
        debug_draw_occludee_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE debug_draw_occludee_ranges_1[ 1 ];
        debug_draw_occludee_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE debug_draw_occludee_ranges_2[ 1 ];
        debug_draw_occludee_ranges_2[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 1 );
        CD3DX12_ROOT_PARAMETER debug_draw_occludee_root_parameters[ 3 ];
        debug_draw_occludee_root_parameters[ 0 ].InitAsDescriptorTable( _countof( debug_draw_occludee_ranges_0 ), debug_draw_occludee_ranges_0, D3D12_SHADER_VISIBILITY_VERTEX );
        debug_draw_occludee_root_parameters[ 1 ].InitAsDescriptorTable( _countof( debug_draw_occludee_ranges_1 ), debug_draw_occludee_ranges_1, D3D12_SHADER_VISIBILITY_VERTEX );
        debug_draw_occludee_root_parameters[ 2 ].InitAsDescriptorTable( _countof( debug_draw_occludee_ranges_2 ), debug_draw_occludee_ranges_2, D3D12_SHADER_VISIBILITY_PIXEL );
        m_DebugDrawOccludeeRootSignature = NGraphics::CreateRootSignature( device, _countof( debug_draw_occludee_root_parameters ), debug_draw_occludee_root_parameters );

        NGraphics::CShader debug_draw_occludee_aabb_vertex_shader( L"res/shaders/DebugDrawOccludeeAabbShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader debug_draw_occludee_aabb_pixel_shader( L"res/shaders/DebugDrawOccludeeAabbShader.hlsl", "PSMain", "ps_5_0" );
        
        D3D12_GRAPHICS_PIPELINE_STATE_DESC debug_draw_occludee_aabb_state_desc;
        ZeroMemory( &debug_draw_occludee_aabb_state_desc, sizeof( debug_draw_occludee_aabb_state_desc ) );
        debug_draw_occludee_aabb_state_desc.InputLayout = debug_draw_occludee_aabb_vertex_shader.GetInputLayout();
        debug_draw_occludee_aabb_state_desc.pRootSignature = m_DebugDrawOccludeeRootSignature;
        debug_draw_occludee_aabb_state_desc.VS = debug_draw_occludee_aabb_vertex_shader.GetShaderBytecode();
        debug_draw_occludee_aabb_state_desc.PS = debug_draw_occludee_aabb_pixel_shader.GetShaderBytecode();
        debug_draw_occludee_aabb_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        debug_draw_occludee_aabb_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        debug_draw_occludee_aabb_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        debug_draw_occludee_aabb_state_desc.DepthStencilState.DepthEnable = FALSE;
        debug_draw_occludee_aabb_state_desc.SampleMask = UINT_MAX;
        debug_draw_occludee_aabb_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        debug_draw_occludee_aabb_state_desc.NumRenderTargets = 1;
        debug_draw_occludee_aabb_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R8G8B8A8_UNORM;
        debug_draw_occludee_aabb_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &debug_draw_occludee_aabb_state_desc, IID_PPV_ARGS( &m_DebugDrawOccludeeAabbPipelineState ) ) );
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
    
        CreateCubeMesh( command_list );
    }

    {
        m_AabbBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        m_ConstantsBufferCbv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    }

    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv_desc;

        ZeroMemory( &cbv_desc, sizeof( cbv_desc ) );
        cbv_desc.BufferLocation = m_ConstantsBuffer->GetGPUVirtualAddress();
        cbv_desc.SizeInBytes = sizeof( SConstants );
        device->CreateConstantBufferView( &cbv_desc, m_ConstantsBufferCbv.m_Cpu );
    }

    Update( command_list );
}

void COccludeeCollection::CreateCubeMesh( ID3D12GraphicsCommandList* command_list )
{
    std::vector< DirectX::XMFLOAT3 > vertices;
    vertices.insert( vertices.begin(), {
        DirectX::XMFLOAT3( -1.0f, -1.0f, -1.0f ),
        DirectX::XMFLOAT3( -1.0f, -1.0f,  1.0f ),
        DirectX::XMFLOAT3( -1.0f,  1.0f, -1.0f ),
        DirectX::XMFLOAT3( -1.0f,  1.0f,  1.0f ),
        DirectX::XMFLOAT3( 1.0f, -1.0f, -1.0f ),
        DirectX::XMFLOAT3( 1.0f, -1.0f,  1.0f ),
        DirectX::XMFLOAT3( 1.0f,  1.0f, -1.0f ),
        DirectX::XMFLOAT3( 1.0f,  1.0f,  1.0f )
    } );
    std::vector<Index> indices;
    indices.insert( indices.begin(), {
        0, 2, 1,
        1, 2, 3,
        4, 5, 6,
        5, 7, 6,
        0, 1, 5,
        0, 5, 4,
        2, 6, 7,
        2, 7, 3,
        0, 4, 6,
        0, 6, 2,
        1, 3, 7,
        1, 7, 5
    } );
    std::vector<Index> indices_line;
    indices_line.insert( indices_line.begin(), {
        0, 1,
        0, 2,
        0, 4,
        1, 3,
        1, 5,
        2, 3,
        2, 6,
        3, 7,
        4, 5,
        4, 6,
        5, 7,
        6, 7
    } );
    m_CubeMesh.Create( m_Device, command_list, &vertices, &indices );
    m_CubeMeshLine.Create( m_Device, command_list, &vertices, &indices_line, D3D_PRIMITIVE_TOPOLOGY_LINELIST );
}

void COccludeeCollection::Update( ID3D12GraphicsCommandList* command_list )
{
    SAFE_RELEASE( m_AabbBufferUpload );
    SAFE_RELEASE( m_AabbBuffer );

    if ( !m_Aabbs.empty() )
    {
        UINT aabb_buffer_size = static_cast< UINT >( sizeof( SAabb ) * m_Aabbs.size() );
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( aabb_buffer_size ),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS( &m_AabbBuffer ) ) );
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( aabb_buffer_size ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_AabbBufferUpload ) ) );

        SAabb* mapped_aabb_buffer_upload = nullptr;
        HR( m_AabbBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_aabb_buffer_upload ) ) );
        memcpy( mapped_aabb_buffer_upload, &m_Aabbs[ 0 ], aabb_buffer_size );
        m_AabbBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

        command_list->CopyBufferRegion( m_AabbBuffer, 0, m_AabbBufferUpload, 0, aabb_buffer_size );
        command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
            m_AabbBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ) );

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;

        ZeroMemory( &srv_desc, sizeof( srv_desc ) );
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Buffer.NumElements = static_cast< UINT >( m_Aabbs.size() );
        srv_desc.Buffer.StructureByteStride = sizeof( SAabb );
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        m_Device->CreateShaderResourceView( m_AabbBuffer, &srv_desc, m_AabbBufferSrv.m_Cpu );
    }
}

void COccludeeCollection::Destroy()
{
    m_CubeMeshLine.Destroy();
    m_CubeMesh.Destroy();

    m_MappedConstantsBuffer = nullptr;
    SAFE_RELEASE_UNMAP( m_ConstantsBuffer );

    SAFE_RELEASE( m_AabbBufferUpload );
    SAFE_RELEASE( m_AabbBuffer );

    SAFE_RELEASE( m_DebugDrawOccludeeAabbPipelineState );
    SAFE_RELEASE( m_DebugDrawOccludeeRootSignature );

    m_Device = nullptr;
}

void COccludeeCollection::DrawAabbs( ID3D12GraphicsCommandList* command_list )
{
    m_CubeMesh.Draw( command_list, static_cast< UINT >( m_Aabbs.size() ) );
}
void COccludeeCollection::DrawAabbsLine( ID3D12GraphicsCommandList* command_list )
{
    m_CubeMeshLine.Draw( command_list, static_cast< UINT >( m_Aabbs.size() ) );
}

void COccludeeCollection::DebugDraw( ID3D12GraphicsCommandList* command_list,
                                     NGraphics::CCamera* camera,
                                     COcclusionAlgorithm* occlusion_algorithm )
{
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjection,
        DirectX::XMMatrixTranspose( camera->GetViewProjection() ) );

    command_list->SetGraphicsRootSignature( m_DebugDrawOccludeeRootSignature );

    UINT aabb_count = static_cast< UINT >( m_Aabbs.size() );
    if ( aabb_count > 0 )
    {
        command_list->SetPipelineState( m_DebugDrawOccludeeAabbPipelineState );

        command_list->SetGraphicsRootDescriptorTable( 0, m_ConstantsBufferCbv.m_Gpu );
        command_list->SetGraphicsRootDescriptorTable( 1, m_AabbBufferSrv.m_Gpu );
        command_list->SetGraphicsRootDescriptorTable( 2, occlusion_algorithm->GetVisibilityBufferSrv().m_Gpu );

        m_CubeMeshLine.Draw( command_list, aabb_count );
    }
}

UINT COccludeeCollection::GetAabbCount()
{
    return static_cast< UINT >( m_Aabbs.size() );
}

NGraphics::SDescriptorHandle COccludeeCollection::GetAabbBufferSrv() const
{
    return m_AabbBufferSrv;
}