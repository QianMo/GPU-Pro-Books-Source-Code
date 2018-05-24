#include "OccluderCollection.h"

#include <iostream>

COccluderCollection::COccluderCollection() :
    m_Device( nullptr ),

    m_DebugDrawOccluderRootSignature( nullptr ),
    m_DebugDrawOccluderPipelineState( nullptr ),

    m_ConstantsBuffer( nullptr ),
    m_MappedConstantsBuffer( nullptr ),

    m_WorldMatrixBuffer( nullptr ),
    m_WorldMatrixBufferUpload( nullptr ),
    m_MappedWorldMatrixBufferUpload( nullptr ),

    m_SilhouetteEdgeCountBufferSize( 0 ),
    m_SilhouetteEdgeBuffer( nullptr ),
    m_SilhouetteEdgeCountBuffer( nullptr ),
    m_SilhouetteEdgeCountBufferReset( nullptr ),

    m_SelectedOccluderCount( 0 )
{
    m_OccluderModels.resize( 2 );
    m_OccluderModels[ 0 ].m_OccluderMesh = new CCubeOccluderMesh();
    m_OccluderModels[ 1 ].m_OccluderMesh = new CCylinderOccluderMesh();
}

COccluderCollection::~COccluderCollection()
{
    for ( SOccluderModel& occluder_model : m_OccluderModels )
    {
        delete occluder_model.m_OccluderMesh;
    }
}

void COccluderCollection::AddOccluderObbs( std::vector< DirectX::XMFLOAT4X4 >* occluders )
{
    m_OccluderModels[ 0 ].m_WorldMatrices.insert( m_OccluderModels[ 0 ].m_WorldMatrices.end(), occluders->begin(), occluders->end() );
}

void COccluderCollection::AddOccluderCylinders( std::vector< DirectX::XMFLOAT4X4 >* occluders )
{
    m_OccluderModels[ 1 ].m_WorldMatrices.insert( m_OccluderModels[ 1 ].m_WorldMatrices.end(), occluders->begin(), occluders->end() );
}

void COccluderCollection::ClearOccluders()
{
    for ( SOccluderModel& occluder_model : m_OccluderModels )
    {
        occluder_model.m_WorldMatrices.clear();
    }
}

void COccluderCollection::Create( ID3D12Device* device,
                                  NGraphics::CCommandContext* graphics_context,
                                  NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    m_Device = device;
    ID3D12GraphicsCommandList* command_list = graphics_context->GetCommandList();

    {
        CD3DX12_DESCRIPTOR_RANGE debug_draw_occluder_ranges_0[ 1 ];
        debug_draw_occluder_ranges_0[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0 );
        CD3DX12_DESCRIPTOR_RANGE debug_draw_occluder_ranges_1[ 1 ];
        debug_draw_occluder_ranges_1[ 0 ].Init( D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0 );
        CD3DX12_ROOT_PARAMETER debug_draw_occluder_root_parameters[ 3 ];
        debug_draw_occluder_root_parameters[ 0 ].InitAsConstants( 1, 1, 0, D3D12_SHADER_VISIBILITY_VERTEX );
        debug_draw_occluder_root_parameters[ 1 ].InitAsDescriptorTable( _countof( debug_draw_occluder_ranges_0 ), debug_draw_occluder_ranges_0, D3D12_SHADER_VISIBILITY_VERTEX );
        debug_draw_occluder_root_parameters[ 2 ].InitAsDescriptorTable( _countof( debug_draw_occluder_ranges_1 ), debug_draw_occluder_ranges_1, D3D12_SHADER_VISIBILITY_VERTEX );
        m_DebugDrawOccluderRootSignature = NGraphics::CreateRootSignature( device, _countof( debug_draw_occluder_root_parameters ), debug_draw_occluder_root_parameters );

        NGraphics::CShader debug_draw_occluder_vertex_shader( L"res/shaders/DebugDrawOccluderShader.hlsl", "VSMain", "vs_5_0" );
        NGraphics::CShader debug_draw_occluder_pixel_shader( L"res/shaders/DebugDrawOccluderShader.hlsl", "PSMain", "ps_5_0" );

        D3D12_GRAPHICS_PIPELINE_STATE_DESC debug_draw_occluder_state_desc;
        ZeroMemory( &debug_draw_occluder_state_desc, sizeof( debug_draw_occluder_state_desc ) );
        debug_draw_occluder_state_desc.InputLayout = debug_draw_occluder_vertex_shader.GetInputLayout();
        debug_draw_occluder_state_desc.pRootSignature = m_DebugDrawOccluderRootSignature;
        debug_draw_occluder_state_desc.VS = debug_draw_occluder_vertex_shader.GetShaderBytecode();
        debug_draw_occluder_state_desc.PS = debug_draw_occluder_pixel_shader.GetShaderBytecode();
        debug_draw_occluder_state_desc.RasterizerState = CD3DX12_RASTERIZER_DESC( D3D12_DEFAULT );
        debug_draw_occluder_state_desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        debug_draw_occluder_state_desc.BlendState = CD3DX12_BLEND_DESC( D3D12_DEFAULT );
        debug_draw_occluder_state_desc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC( D3D12_DEFAULT );
        debug_draw_occluder_state_desc.DepthStencilState.DepthEnable = FALSE;
        debug_draw_occluder_state_desc.SampleMask = UINT_MAX;
        debug_draw_occluder_state_desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
        debug_draw_occluder_state_desc.NumRenderTargets = 1;
        debug_draw_occluder_state_desc.RTVFormats[ 0 ] = DXGI_FORMAT_R8G8B8A8_UNORM;
        debug_draw_occluder_state_desc.SampleDesc.Count = 1;
        HR( device->CreateGraphicsPipelineState( &debug_draw_occluder_state_desc, IID_PPV_ARGS( &m_DebugDrawOccluderPipelineState ) ) );
    }

    {
        m_ConstantsBufferCbv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
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

        m_WorldMatrixBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        m_SilhouetteEdgeBufferUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_SilhouetteEdgeCountBufferUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_SilhouetteEdgeBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
        m_SilhouetteEdgeCountBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

        for ( SOccluderModel& occluder_model : m_OccluderModels )
        {
            occluder_model.m_OccluderMesh->Create( device, graphics_context, descriptor_heaps );
            occluder_model.m_SelectedOccluderCount = 0;
        }
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

void COccluderCollection::Update( ID3D12GraphicsCommandList* command_list )
{
    m_OccluderInstanceCount = 0;
    for ( SOccluderModel& occluder_model : m_OccluderModels )
    {
        m_OccluderInstanceCount += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
    }

    {
        m_MappedWorldMatrixBufferUpload = nullptr;
        SAFE_RELEASE_UNMAP( m_WorldMatrixBufferUpload );
        SAFE_RELEASE( m_WorldMatrixBuffer );

        m_WorldMatrixBufferSize = static_cast< UINT >( sizeof( DirectX::XMFLOAT4X4 ) ) * m_OccluderInstanceCount;
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( m_WorldMatrixBufferSize ),
            D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr,
            IID_PPV_ARGS( &m_WorldMatrixBuffer ) ) );
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( m_WorldMatrixBufferSize ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_WorldMatrixBufferUpload ) ) );
        m_WorldMatrixBuffer->SetName( L"World Matrix Buffer" );
        m_WorldMatrixBufferUpload->SetName( L"World Matrix Buffer Upload" );

        HR( m_WorldMatrixBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &m_MappedWorldMatrixBufferUpload ) ) );
        DirectX::XMFLOAT4X4* mapped_world_matrix_buffer_upload_ptr = m_MappedWorldMatrixBufferUpload;
        for ( SOccluderModel& occluder_model : m_OccluderModels )
        {
            memcpy( mapped_world_matrix_buffer_upload_ptr, &occluder_model.m_WorldMatrices[ 0 ], occluder_model.m_WorldMatrices.size() * sizeof( DirectX::XMFLOAT4X4 ) );
            mapped_world_matrix_buffer_upload_ptr += occluder_model.m_WorldMatrices.size();
        }
        
        command_list->CopyBufferRegion( m_WorldMatrixBuffer, 0, m_WorldMatrixBufferUpload, 0, m_WorldMatrixBufferSize );
        const D3D12_RESOURCE_BARRIER post_copy_barriers[] =
        {
            CD3DX12_RESOURCE_BARRIER::Transition( m_WorldMatrixBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE )
        };
        command_list->ResourceBarrier( _countof( post_copy_barriers ), post_copy_barriers );

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;

        ZeroMemory( &srv_desc, sizeof( srv_desc ) );
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Buffer.NumElements = m_OccluderInstanceCount;
        srv_desc.Buffer.StructureByteStride = sizeof( DirectX::XMFLOAT4X4 );
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        m_Device->CreateShaderResourceView( m_WorldMatrixBuffer, &srv_desc, m_WorldMatrixBufferSrv.m_Cpu );
    }

    {
        SAFE_RELEASE( m_SilhouetteEdgeCountBufferReset );
        SAFE_RELEASE( m_SilhouetteEdgeCountBuffer );
        SAFE_RELEASE( m_SilhouetteEdgeBuffer );

        UINT silhouette_edge_buffer_size = m_SilhouetteEdgeBufferOffset * sizeof( DirectX::XMFLOAT2 ) * m_OccluderInstanceCount;
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( silhouette_edge_buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS( &m_SilhouetteEdgeBuffer ) ) );
        m_SilhouetteEdgeCountBufferSize = sizeof( UINT ) * m_OccluderInstanceCount;
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( m_SilhouetteEdgeCountBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
            nullptr,
            IID_PPV_ARGS( &m_SilhouetteEdgeCountBuffer ) ) );
        HR( m_Device->CreateCommittedResource(
            &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
            D3D12_HEAP_FLAG_NONE,
            &CD3DX12_RESOURCE_DESC::Buffer( m_SilhouetteEdgeCountBufferSize ),
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS( &m_SilhouetteEdgeCountBufferReset ) ) );
        m_SilhouetteEdgeBuffer->SetName( L"Silhouette Edge Buffer" );
        m_SilhouetteEdgeCountBuffer->SetName( L"Silhouette Edge Count Buffer" );
        m_SilhouetteEdgeCountBufferReset->SetName( L"Silhouette Edge Count Buffer Reset" );

        BYTE* mapped_silhouette_edge_count_buffer_reset = nullptr;
        HR( m_SilhouetteEdgeCountBufferReset->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_silhouette_edge_count_buffer_reset ) ) );
        ZeroMemory( mapped_silhouette_edge_count_buffer_reset, m_SilhouetteEdgeCountBufferSize );
        m_SilhouetteEdgeCountBufferReset->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

        D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;

        ZeroMemory( &srv_desc, sizeof( srv_desc ) );
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Buffer.NumElements = m_SilhouetteEdgeBufferOffset * m_OccluderInstanceCount;
        srv_desc.Buffer.StructureByteStride = sizeof( DirectX::XMFLOAT2 );
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        m_Device->CreateShaderResourceView( m_SilhouetteEdgeBuffer, &srv_desc, m_SilhouetteEdgeBufferSrv.m_Cpu );

        ZeroMemory( &srv_desc, sizeof( srv_desc ) );
        srv_desc.Format = DXGI_FORMAT_UNKNOWN;
        srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv_desc.Buffer.NumElements = m_OccluderInstanceCount;
        srv_desc.Buffer.StructureByteStride = sizeof( UINT );
        srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
        srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        m_Device->CreateShaderResourceView( m_SilhouetteEdgeCountBuffer, &srv_desc, m_SilhouetteEdgeCountBufferSrv.m_Cpu );

        ZeroMemory( &uav_desc, sizeof( uav_desc ) );
        uav_desc.Format = DXGI_FORMAT_UNKNOWN;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.NumElements = m_SilhouetteEdgeBufferOffset * m_OccluderInstanceCount;
        uav_desc.Buffer.StructureByteStride = sizeof( DirectX::XMFLOAT2 );
        uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
        m_Device->CreateUnorderedAccessView( m_SilhouetteEdgeBuffer, nullptr, &uav_desc, m_SilhouetteEdgeBufferUav.m_Cpu );

        ZeroMemory( &uav_desc, sizeof( uav_desc ) );
        uav_desc.Format = DXGI_FORMAT_R32_TYPELESS;
        uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav_desc.Buffer.NumElements = m_OccluderInstanceCount;
        uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        m_Device->CreateUnorderedAccessView( m_SilhouetteEdgeCountBuffer, nullptr, &uav_desc, m_SilhouetteEdgeCountBufferUav.m_Cpu );
    }
}

void COccluderCollection::Destroy()
{
    for ( SOccluderModel& occluder_model : m_OccluderModels )
    {
        occluder_model.m_WorldMatrices.clear();
        occluder_model.m_OccluderMesh->Destroy();
    }

    SAFE_RELEASE( m_SilhouetteEdgeCountBufferReset );
    SAFE_RELEASE( m_SilhouetteEdgeCountBuffer );
    SAFE_RELEASE( m_SilhouetteEdgeBuffer );
    m_SilhouetteEdgeCountBufferSize = 0;

    m_MappedWorldMatrixBufferUpload = nullptr;
    SAFE_RELEASE_UNMAP( m_WorldMatrixBufferUpload );
    SAFE_RELEASE( m_WorldMatrixBuffer );

    m_MappedConstantsBuffer = nullptr;
    SAFE_RELEASE_UNMAP( m_ConstantsBuffer );

    SAFE_RELEASE( m_DebugDrawOccluderPipelineState );
    SAFE_RELEASE( m_DebugDrawOccluderRootSignature );

    m_Device = nullptr;
}

BOOL COccluderCollection::IsAabbInsideFrustum( DirectX::XMFLOAT3 aabb_center, DirectX::XMFLOAT3 aabb_extent, DirectX::XMFLOAT4 frustum_planes[ 6 ] )
{
    for ( UINT i = 0; i < 6; ++i )
    {
        DirectX::XMFLOAT4 plane = frustum_planes[ i ];

        FLOAT d =
            aabb_center.x * plane.x +
            aabb_center.y * plane.y +
            aabb_center.z * plane.z;

        FLOAT r =
            aabb_extent.x * abs( plane.x ) +
            aabb_extent.y * abs( plane.y ) +
            aabb_extent.z * abs( plane.z );

        if ( d + r < -plane.w )
        {
            return FALSE;
        }
    }
    
    return TRUE;
}

void COccluderCollection::SelectOccluders( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera, FLOAT size_threshold )
{
    DirectX::XMFLOAT4X4 view_projection;
    DirectX::XMStoreFloat4x4( &view_projection, camera->GetViewProjection() );

    DirectX::XMFLOAT4 frustum_planes[ 6 ];
    camera->ExtractFrustumPlanes( frustum_planes );

    m_SelectedOccluderCount = 0;

    DirectX::XMFLOAT4X4* mapped_world_matrix_buffer_upload_ptr = m_MappedWorldMatrixBufferUpload;

    for ( size_t i = 0; i < m_OccluderModels.size(); ++i )
    {
        m_OccluderModels[ i ].m_SelectedOccluderCount = 0;
        for ( size_t j = 0; j < m_OccluderModels[ i ].m_WorldMatrices.size(); ++j )
        {
            DirectX::XMMATRIX world_matrix = DirectX::XMMatrixTranspose( DirectX::XMLoadFloat4x4( &m_OccluderModels[ i ].m_WorldMatrices[ j ] ) );

            const DirectX::XMVECTOR positions[ 8 ] =
            {
                DirectX::XMVectorSet( -1.0f, -1.0f, -1.0f, 1.0f ),
                DirectX::XMVectorSet(  1.0f, -1.0f, -1.0f, 1.0f ),
                DirectX::XMVectorSet( -1.0f,  1.0f, -1.0f, 1.0f ),
                DirectX::XMVectorSet(  1.0f,  1.0f, -1.0f, 1.0f ),
                DirectX::XMVectorSet( -1.0f, -1.0f,  1.0f, 1.0f ),
                DirectX::XMVectorSet(  1.0f, -1.0f,  1.0f, 1.0f ),
                DirectX::XMVectorSet( -1.0f,  1.0f,  1.0f, 1.0f ),
                DirectX::XMVectorSet(  1.0f,  1.0f,  1.0f, 1.0f )
            };

            DirectX::XMVECTOR aabb_min = DirectX::XMVectorSet( FLT_MAX, FLT_MAX, FLT_MAX, 1.0f );
            DirectX::XMVECTOR aabb_max = DirectX::XMVectorSet( -FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f );

            for ( size_t k = 0; k < 8; ++k )
            {
                DirectX::XMVECTOR position = DirectX::XMVector3TransformCoord( positions[ k ], world_matrix );
                aabb_min = DirectX::XMVectorMin( aabb_min, position );
                aabb_max = DirectX::XMVectorMax( aabb_max, position );
            }

            DirectX::XMFLOAT3 aabb_center, aabb_extent;
            DirectX::XMStoreFloat3( &aabb_center, DirectX::XMVectorScale( DirectX::XMVectorAdd( aabb_max, aabb_min ), 0.5f ) );
            DirectX::XMStoreFloat3( &aabb_extent, DirectX::XMVectorScale( DirectX::XMVectorSubtract( aabb_max, aabb_min ), 0.5f ) );

            if ( IsAabbInsideFrustum( aabb_center, aabb_extent, frustum_planes ) )
            {
                FLOAT w =
                    aabb_center.x * view_projection._14 +
                    aabb_center.y * view_projection._24 +
                    aabb_center.z * view_projection._34 +
                    view_projection._44;

                FLOAT radius_sq =
                    aabb_extent.x * aabb_extent.x +
                    aabb_extent.y * aabb_extent.y +
                    aabb_extent.z * aabb_extent.z;

                if ( w <= 1.0f || radius_sq >= w * size_threshold )
                {
                    mapped_world_matrix_buffer_upload_ptr[ m_OccluderModels[ i ].m_SelectedOccluderCount++ ] = m_OccluderModels[ i ].m_WorldMatrices[ j ];
                    ++m_SelectedOccluderCount;
                }
            }
        }
        mapped_world_matrix_buffer_upload_ptr += m_OccluderModels[ i ].m_WorldMatrices.size();
    }

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_WorldMatrixBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ) );
    command_list->CopyBufferRegion( m_WorldMatrixBuffer, 0, m_WorldMatrixBufferUpload, 0, m_WorldMatrixBufferSize );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition( m_WorldMatrixBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ) );
}

void COccluderCollection::DebugDraw( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera )
{
    DirectX::XMStoreFloat4x4(
        &m_MappedConstantsBuffer->m_ViewProjection,
        DirectX::XMMatrixTranspose( camera->GetViewProjection() ) );

    command_list->SetGraphicsRootSignature( m_DebugDrawOccluderRootSignature );
    command_list->SetPipelineState( m_DebugDrawOccluderPipelineState );

    command_list->SetGraphicsRootDescriptorTable( 1, m_ConstantsBufferCbv.m_Gpu );
    command_list->SetGraphicsRootDescriptorTable( 2, m_WorldMatrixBufferSrv.m_Gpu );

    UINT instance_offset = 0;
    for ( SOccluderModel occluder_model : m_OccluderModels )
    {
        if ( occluder_model.m_SelectedOccluderCount > 0 )
        {
            command_list->SetGraphicsRoot32BitConstant( 0, instance_offset, 0 );
            occluder_model.m_OccluderMesh->DrawLine( command_list, occluder_model.m_SelectedOccluderCount );
        }
        instance_offset += static_cast< UINT >( occluder_model.m_WorldMatrices.size() );
    }
}

const UINT COccluderCollection::GetOccluderInstanceCount() const
{
    return m_OccluderInstanceCount;
}

const UINT COccluderCollection::GetSelectedOccluderCount() const
{
    return m_SelectedOccluderCount;
}