#include "ModelSet.h"
#include "OcclusionAlgorithm.h"

#include <DDSTextureLoader.h>
#include <fstream>
#include <locale>
#include <codecvt>

CModelSet::CModelSet() :
    m_InstanceCount( 0 ),

    m_MaterialBuffer( nullptr ),
    m_MaterialBufferUpload( nullptr ),

    m_WorldMatrixBufferSize( 0 ),
    m_WorldMatrixBuffer( nullptr ),
    m_WorldMatrixBufferUpload( nullptr ),

    m_ModelInstanceCountBufferSize( 0 ),
    m_InstanceIndexMappingsBufferSize( 0 ),
    m_InstanceModelMappingsBuffer( nullptr ),
    m_InstanceModelMappingsBufferUpload( nullptr ),
    m_ModelInstanceOffsetBuffer( nullptr ),
    m_ModelInstanceOffsetBufferUpload( nullptr ),
    m_ModelInstanceCountBuffer( nullptr ),
    m_ModelInstanceCountBufferReset( nullptr ),
    m_InstanceIndexMappingsBuffer( nullptr ),

    m_OutputCommandBufferCounterOffset( 0 ),
    m_InputCommandBuffer( nullptr ),
    m_InputCommandBufferUpload( nullptr ),
    m_OutputCommandBuffer( nullptr ),
    m_OutputCommandBufferCounterReset( nullptr )
{
}

void CModelSet::Load( const std::string& directory, const std::string& filename )
{
    m_Directory = directory;
    m_Filename = filename;
    LoadModels();
}

void CModelSet::Save()
{
    SaveModels();
}

void CModelSet::Create(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* command_list,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    m_InstanceCount = 0;
    for ( SModel* model : m_Models )
    {
        m_InstanceCount += static_cast< UINT >( model->m_Instances.size() );
    }

    CreateMeshes( device, command_list );
    CreateTexturesAndMaterials( device, command_list, descriptor_heaps );
    CreateWorldMatrixBuffer( device, command_list, descriptor_heaps );
    CreateInstanceMappingsBuffers( device, command_list, descriptor_heaps );
    CreateCommandBuffers( device, command_list, descriptor_heaps );

    CalculateOccludees();
    CalculateOccluders();
}

void CModelSet::Destroy()
{
    m_InstanceCount = 0;

    for ( SModel* model : m_Models )
    {
        model->m_Mesh.Destroy();
        delete model;
    }
    m_Models.clear();

    SAFE_RELEASE( m_MaterialBufferUpload );
    SAFE_RELEASE( m_MaterialBuffer );

    for ( STexture texture_array : m_Textures )
    {
        texture_array.m_Resource->Release();
        texture_array.m_ResourceUpload->Release();
    }
    m_Textures.clear();

    m_WorldMatrixBufferSize = 0;
    SAFE_RELEASE( m_WorldMatrixBuffer );
    SAFE_RELEASE_UNMAP( m_WorldMatrixBufferUpload );

    m_ModelInstanceCountBufferSize = 0;
    m_InstanceIndexMappingsBufferSize = 0;
    SAFE_RELEASE( m_InstanceModelMappingsBuffer );
    SAFE_RELEASE( m_InstanceModelMappingsBufferUpload );
    SAFE_RELEASE( m_ModelInstanceOffsetBuffer );
    SAFE_RELEASE( m_ModelInstanceOffsetBufferUpload );
    SAFE_RELEASE( m_ModelInstanceCountBuffer );
    SAFE_RELEASE( m_ModelInstanceCountBufferReset );
    SAFE_RELEASE( m_InstanceIndexMappingsBuffer );
    SAFE_RELEASE( m_InstanceIndexMappingsBufferUpload );

    m_OutputCommandBufferCounterOffset = 0;
    SAFE_RELEASE( m_InputCommandBuffer );
    SAFE_RELEASE( m_InputCommandBufferUpload );
    SAFE_RELEASE( m_OutputCommandBuffer );
    SAFE_RELEASE( m_OutputCommandBufferCounterReset );
}

void CModelSet::UpdateWorldMatrixBuffer( ID3D12GraphicsCommandList* command_list )
{
    for ( SModel* model : m_Models )
    {
        model->m_VisibleInstanceCount = static_cast< UINT >( model->m_Instances.size() );
        DirectX::XMFLOAT4X4* mapped_world_matrix_buffer = model->m_MappedWorldMatrixBuffer;
        for ( SInstance instance : model->m_Instances )
        {
            DirectX::XMStoreFloat4x4(
                mapped_world_matrix_buffer,
                DirectX::XMMatrixTranspose( DirectX::XMLoadFloat4x4( &instance.m_World ) ) );
            ++mapped_world_matrix_buffer;
        }
    }

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_WorldMatrixBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ) );
    command_list->CopyBufferRegion( m_WorldMatrixBuffer, 0, m_WorldMatrixBufferUpload, 0, m_WorldMatrixBufferSize );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_WorldMatrixBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ) );
}

void CModelSet::UpdateInstanceMappings(
    ID3D12GraphicsCommandList* command_list,
    COcclusionAlgorithm* occlusion_algorithm,
    UINT occludee_offset )
{
    UINT* mapped_instance_index_mappings_upload = nullptr;
    HR( m_InstanceIndexMappingsBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_instance_index_mappings_upload ) ) );

    UINT instance_index = 0;
    for ( SModel* model : m_Models )
    {
        model->m_VisibleInstanceCount = 0;
        for ( SInstance instance : model->m_Instances )
        {
            if ( occlusion_algorithm->IsOccludeeVisible( occludee_offset + instance_index ) )
            {
                mapped_instance_index_mappings_upload[ model->m_VisibleInstanceCount++ ] = instance_index;
            }
            ++instance_index;
        }
        mapped_instance_index_mappings_upload += model->m_Instances.size();
    }

    m_InstanceIndexMappingsBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_InstanceIndexMappingsBuffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST ) );
    command_list->CopyBufferRegion( m_InstanceIndexMappingsBuffer, 0, m_InstanceIndexMappingsBufferUpload, 0, m_InstanceIndexMappingsBufferSize );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_InstanceIndexMappingsBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ) );
}

void CModelSet::CalculateOccluders()
{
    size_t total_occluder_obb_count = 0;
    size_t total_occluder_cylinder_count = 0;
    for ( SModel* model : m_Models )
    {
        total_occluder_obb_count += model->m_OccluderObbs.size() * model->m_Instances.size();
        total_occluder_cylinder_count += model->m_OccluderCylinders.size() * model->m_Instances.size();
    }
    m_OccluderObbs.resize( total_occluder_obb_count );
    m_OccluderCylinders.resize( total_occluder_cylinder_count );

    size_t occluder_obb_index = 0;
    size_t occluder_cylinder_index = 0;
    for ( SModel* model : m_Models )
    {
        for ( size_t i = 0; i < model->m_Instances.size(); ++i )
        {
            DirectX::XMMATRIX instance_world = DirectX::XMLoadFloat4x4( &model->m_Instances[ i ].m_World );
            DirectX::XMVECTOR scale, rotation, translation;
            DirectX::XMMatrixDecompose( &scale, &rotation, &translation, instance_world );
            instance_world =
                DirectX::XMMatrixRotationQuaternion( rotation ) *
                DirectX::XMMatrixTranslationFromVector( translation );

            for ( size_t j = 0; j < model->m_OccluderObbs.size(); ++j )
            {
                DirectX::XMMATRIX obb_center =
                    DirectX::XMMatrixScaling(
                        model->m_OccluderObbs[ j ].m_Extent.x,
                        model->m_OccluderObbs[ j ].m_Extent.y,
                        model->m_OccluderObbs[ j ].m_Extent.z ) *
                    DirectX::XMMatrixRotationX( DirectX::XMConvertToRadians( model->m_OccluderObbs[ j ].m_Rotation.x ) ) *
                    DirectX::XMMatrixRotationY( DirectX::XMConvertToRadians( model->m_OccluderObbs[ j ].m_Rotation.y ) ) *
                    DirectX::XMMatrixRotationZ( DirectX::XMConvertToRadians( model->m_OccluderObbs[ j ].m_Rotation.z ) ) *
                    DirectX::XMMatrixTranslation(
                        model->m_OccluderObbs[ j ].m_Center.x,
                        model->m_OccluderObbs[ j ].m_Center.y,
                        model->m_OccluderObbs[ j ].m_Center.z );

                DirectX::XMStoreFloat4x4(
                    &m_OccluderObbs[ occluder_obb_index++ ],
                    DirectX::XMMatrixTranspose( DirectX::XMMatrixMultiply( obb_center, instance_world ) ) );
            }

            for ( size_t j = 0; j < model->m_OccluderCylinders.size(); ++j )
            {
                DirectX::XMMATRIX cylinder_center =
                    DirectX::XMMatrixScaling( 
                        model->m_OccluderCylinders[ j ].m_Radius, 
                        model->m_OccluderCylinders[ j ].m_Height, 
                        model->m_OccluderCylinders[ j ].m_Radius ) *
                    DirectX::XMMatrixRotationX( DirectX::XMConvertToRadians( model->m_OccluderCylinders[ j ].m_Rotation.x ) ) *
                    DirectX::XMMatrixRotationY( DirectX::XMConvertToRadians( model->m_OccluderCylinders[ j ].m_Rotation.y ) ) *
                    DirectX::XMMatrixRotationZ( DirectX::XMConvertToRadians( model->m_OccluderCylinders[ j ].m_Rotation.z ) ) *
                    DirectX::XMMatrixTranslation( 
                        model->m_OccluderCylinders[ j ].m_Center.x, 
                        model->m_OccluderCylinders[ j ].m_Center.y, 
                        model->m_OccluderCylinders[ j ].m_Center.z );

                DirectX::XMStoreFloat4x4(
                    &m_OccluderCylinders[ occluder_cylinder_index++ ],
                    DirectX::XMMatrixTranspose( DirectX::XMMatrixMultiply( cylinder_center, instance_world ) ) );
            }
        }
    }
}

void CModelSet::CalculateOccludees()
{
    m_OccludeeAabbs.resize( m_InstanceCount );

    UINT occludee_aabb_index = 0;
    for ( SModel* model : m_Models )
    {
        DirectX::XMVECTOR center = XMLoadFloat3( &model->m_OccludeeBoundingBoxCenter );
        DirectX::XMVECTOR extent = XMLoadFloat3( &model->m_OccludeeBoundingBoxExtent );

        DirectX::XMVECTOR positions[ 8 ] =
        {
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( 1.0f,  1.0f,  1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( 1.0f,  1.0f, -1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( 1.0f, -1.0f,  1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( 1.0f, -1.0f, -1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( -1.0f,  1.0f,  1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( -1.0f,  1.0f, -1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( -1.0f, -1.0f,  1.0f, 0.0f ), extent ) ),
            DirectX::XMVectorAdd( center, DirectX::XMVectorMultiply( DirectX::XMVectorSet( -1.0f, -1.0f, -1.0f, 0.0f ), extent ) )
        };

        for ( SInstance& instance : model->m_Instances )
        {
            DirectX::XMMATRIX instance_world = DirectX::XMLoadFloat4x4( &instance.m_World );

            DirectX::XMVECTOR aabb_min = DirectX::XMVectorSet( FLT_MAX, FLT_MAX, FLT_MAX, 1.0f );
            DirectX::XMVECTOR aabb_max = DirectX::XMVectorSet( -FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f );

            for ( UINT i = 0; i < 8; ++i )
            {
                DirectX::XMVECTOR position = DirectX::XMVector3TransformCoord( positions[ i ], instance_world );
                aabb_min = DirectX::XMVectorMin( aabb_min, position );
                aabb_max = DirectX::XMVectorMax( aabb_max, position );
            }

            DirectX::XMStoreFloat3( &instance.m_OccludeeAabbMin, aabb_min );
            DirectX::XMStoreFloat3( &instance.m_OccludeeAabbMax, aabb_max );

            DirectX::XMStoreFloat3( &m_OccludeeAabbs[ occludee_aabb_index ].m_Center, ( aabb_max + aabb_min ) * 0.5f );
            DirectX::XMStoreFloat3( &m_OccludeeAabbs[ occludee_aabb_index ].m_Extent, ( aabb_max - aabb_min ) * 0.5f );
            ++occludee_aabb_index;
        }
    }
}

const UINT CModelSet::GetModelCount() const
{
    return static_cast< UINT >( m_Models.size() );
}
const UINT CModelSet::GetInstanceCount() const
{
    return m_InstanceCount;
}
const UINT CModelSet::GetTextureCount() const
{
    return static_cast< UINT >( m_Textures.size() );
}

std::vector< CModelSet::SModel* >* CModelSet::GetModels()
{
    return &m_Models;
}

std::vector< DirectX::XMFLOAT4X4 >* CModelSet::GetOccluderObbs()
{
    return &m_OccluderObbs;
}
std::vector< DirectX::XMFLOAT4X4 >* CModelSet::GetOccluderCylinders()
{
    return &m_OccluderCylinders;
}
std::vector< COccludeeCollection::SAabb >* CModelSet::GetOccludeeAabbs()
{
    return &m_OccludeeAabbs;
}

NGraphics::SDescriptorHandle CModelSet::GetMaterialBufferSrv() const
{
    return m_MaterialBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetWorldMatrixBufferSrv() const
{
    return m_WorldMatrixBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetTexturesSrv() const
{
    return m_Textures[ 0 ].m_Handle;
}

NGraphics::SDescriptorHandle CModelSet::GetInstanceModelMappingsBufferSrv() const
{
    return m_InstanceModelMappingsBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetModelInstanceOffsetBufferSrv() const
{
    return m_ModelInstanceOffsetBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetModelInstanceCountBufferUav() const
{
    return m_ModelInstanceCountBufferUav;
}
NGraphics::SDescriptorHandle CModelSet::GetModelInstanceCountBufferSrv() const
{
    return m_ModelInstanceCountBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetInstanceIndexMappingsBufferUav() const
{
    return m_InstanceIndexMappingsBufferUav;
}
NGraphics::SDescriptorHandle CModelSet::GetInstanceIndexMappingsBufferSrv() const
{
    return m_InstanceIndexMappingsBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetInputCommandBufferSrv() const
{
    return m_InputCommandBufferSrv;
}
NGraphics::SDescriptorHandle CModelSet::GetOutputCommandBufferUav() const
{
    return m_OutputCommandBufferUav;
}

ID3D12Resource* CModelSet::GetModelInstanceCountBuffer() const
{
    return m_ModelInstanceCountBuffer;
}
ID3D12Resource* CModelSet::GetModelInstanceCountBufferReset() const
{
    return m_ModelInstanceCountBufferReset;
}
ID3D12Resource* CModelSet::GetInstanceIndexMappingsBuffer() const
{
    return m_InstanceIndexMappingsBuffer;
}
ID3D12Resource* CModelSet::GetOutputCommandBuffer() const
{
    return m_OutputCommandBuffer;
}
ID3D12Resource* CModelSet::GetOutputCommandBufferCounterReset() const
{
    return m_OutputCommandBufferCounterReset;
}

const UINT CModelSet::GetModelInstanceCountBufferSize() const
{
    return m_ModelInstanceCountBufferSize;
}
const UINT CModelSet::GetOutputCommandBufferCounterOffset() const
{
    return m_OutputCommandBufferCounterOffset;
}

void CModelSet::LoadModels()
{
    std::fstream file( m_Directory + "/" + m_Filename, std::ios::in | std::ios::binary );
    assert( file.is_open() && file.good() );

    size_t size;
    file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
    m_Models.resize( size );

    for ( size_t i = 0; i < m_Models.size(); ++i )
    {
        SModel* model = new SModel;

        file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        model->m_Name.resize( size );
        file.read( reinterpret_cast< char* >( &model->m_Name[ 0 ] ), size * sizeof( char ) );

        file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        model->m_Vertices.resize( size );
        file.read( reinterpret_cast< char* >( &model->m_Vertices[ 0 ] ), size * sizeof( SVertex ) );

        file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        model->m_Indices.resize( size );
        file.read( reinterpret_cast< char* >( &model->m_Indices[ 0 ] ), size * sizeof( UINT ) );

        size_t texture_filepath_count;
        file.read( reinterpret_cast< char* >( &texture_filepath_count ), sizeof( size_t ) );
        for ( size_t j = 0; j < texture_filepath_count; ++j )
        {
            std::string texture_filepath_key;
            std::string texture_filepath_value;

            file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
            texture_filepath_key.resize( size );
            file.read( reinterpret_cast< char* >( &texture_filepath_key[ 0 ] ), size * sizeof( char ) );

            file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
            texture_filepath_value.resize( size );
            file.read( reinterpret_cast< char* >( &texture_filepath_value[ 0 ] ), size * sizeof( char ) );

            model->m_TextureFilepathMap[ texture_filepath_key ] = texture_filepath_value;
        }

        file.read( reinterpret_cast< char* >( &model->m_OccludeeBoundingBoxCenter ), sizeof( DirectX::XMFLOAT3 ) );
        file.read( reinterpret_cast< char* >( &model->m_OccludeeBoundingBoxExtent ), sizeof( DirectX::XMFLOAT3 ) );

        file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        model->m_OccluderObbs.resize( size );
        if ( size > 0 )
        {
            file.read( reinterpret_cast< char* >( &model->m_OccluderObbs[ 0 ] ), sizeof( SOccluderObb ) * size );
        }

        file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        model->m_OccluderCylinders.resize( size );
        if ( size > 0 )
        {
            file.read( reinterpret_cast< char* >( &model->m_OccluderCylinders[ 0 ] ), sizeof( SOccluderCylinder ) * size );
        }

        file.read( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        model->m_Instances.resize( size );
        for ( size_t j = 0; j < model->m_Instances.size(); ++j )
        {
            file.read( reinterpret_cast< char* >( &model->m_Instances[ j ].m_World ), sizeof( DirectX::XMFLOAT4X4 ) );
        }

        m_Models[ i ] = model;
    }

    file.close();
}

void CModelSet::SaveModels()
{
    std::fstream file( m_Directory + "/" + m_Filename, std::ios::out | std::ios::binary );

    size_t size = m_Models.size();
    file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );

    for ( size_t i = 0; i < m_Models.size(); ++i )
    {
        SModel* model = m_Models[ i ];

        size = model->m_Name.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        file.write( reinterpret_cast< char* >( &model->m_Name[ 0 ] ), model->m_Name.size() * sizeof( char ) );

        size = model->m_Vertices.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        file.write( reinterpret_cast< char* >( &model->m_Vertices[ 0 ] ), model->m_Vertices.size() * sizeof( SVertex ) );

        size = model->m_Indices.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        file.write( reinterpret_cast< char* >( &model->m_Indices[ 0 ] ), model->m_Indices.size() * sizeof( UINT ) );

        size = model->m_TextureFilepathMap.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        for ( std::pair<std::string, std::string> texture_filepath_entry : model->m_TextureFilepathMap )
        {
            size = texture_filepath_entry.first.size();
            file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
            file.write( reinterpret_cast< char* >( &texture_filepath_entry.first[ 0 ] ), texture_filepath_entry.first.size() * sizeof( char ) );
            size = texture_filepath_entry.second.size();
            file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
            file.write( reinterpret_cast< char* >( &texture_filepath_entry.second[ 0 ] ), texture_filepath_entry.second.size() * sizeof( char ) );
        }

        file.write( reinterpret_cast< char* >( &model->m_OccludeeBoundingBoxCenter ), sizeof( DirectX::XMFLOAT3 ) );
        file.write( reinterpret_cast< char* >( &model->m_OccludeeBoundingBoxExtent ), sizeof( DirectX::XMFLOAT3 ) );

        size = model->m_OccluderObbs.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        if ( size > 0 )
        {
            file.write( reinterpret_cast< char* >( &model->m_OccluderObbs[ 0 ] ), sizeof( SOccluderObb ) * size );
        }

        size = model->m_OccluderCylinders.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        if ( size > 0 )
        {
            file.write( reinterpret_cast< char* >( &model->m_OccluderCylinders[ 0 ] ), sizeof( SOccluderCylinder ) * size );
        }

        size = model->m_Instances.size();
        file.write( reinterpret_cast< char* >( &size ), sizeof( size_t ) );
        for ( size_t j = 0; j < model->m_Instances.size(); ++j )
        {
            file.write( reinterpret_cast< char* >( &model->m_Instances[ j ].m_World ), sizeof( DirectX::XMFLOAT4X4 ) );
        }
    }

    file.close();
}

void CModelSet::CreateMeshes(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* command_list )
{
    for ( SModel* model : m_Models )
    {
        model->m_Mesh.Create( device, command_list,
                              &model->m_Vertices, &model->m_Indices );
    }
}

void CModelSet::CreateTexturesAndMaterials(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* command_list,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    assert( !m_Models.empty() );

    size_t material_texture_count = m_Models[ 0 ]->m_TextureFilepathMap.size();

    std::vector< std::wstring > texture_filepaths;
    std::vector< UINT > materials;
    std::vector< UINT > material_textures( material_texture_count );
    for ( SModel* model : m_Models )
    {
        size_t material_texture_index = 0;
        for ( std::pair<std::string, std::string> texture_filepath_entry : model->m_TextureFilepathMap )
        {
            std::wstring_convert< std::codecvt_utf8_utf16< wchar_t > > converter;
            std::wstring texture_filepath = converter.from_bytes( m_Directory + "/" + texture_filepath_entry.second );
            
            bool is_texture_found = false;
            for ( size_t i = 0; i < texture_filepaths.size(); ++i )
            {
                if ( texture_filepaths[ i ] == texture_filepath )
                {
                    material_textures[ material_texture_index++ ] = static_cast< UINT >( i );

                    is_texture_found = true;
                    break;
                }
            }
            if ( !is_texture_found )
            {
                STexture texture;
                texture.m_Resource = nullptr;
                texture.m_ResourceUpload = nullptr;
                texture.m_Handle = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
                CreateDDSTextureFromFile( device, command_list, texture_filepath.c_str(), 0, false,
                                          &texture.m_Resource, &texture.m_ResourceUpload,
                                          texture.m_Handle.m_Cpu, nullptr );

                m_Textures.push_back( texture );
                texture_filepaths.push_back( texture_filepath );

                material_textures[ material_texture_index++ ] = static_cast< UINT >( texture_filepaths.size() - 1 );
            }
        }

        bool is_material_found = false;
        UINT material_index = 0;
        for ( size_t i = 0; i < materials.size(); i += material_texture_count )
        {
            is_material_found = true;
            for ( size_t j = 0; j < material_texture_count; ++j )
            {
                if ( materials[ i + j ] != material_textures[ j ] )
                {
                    is_material_found = false;
                    break;
                }
            }
            if ( is_material_found )
            {
                break;
            }

            ++material_index;
        }
        if ( !is_material_found )
        {
            materials.insert( materials.end(), material_textures.begin(), material_textures.end() );
        }
        model->m_MaterialIndex = material_index;
    }

    UINT material_buffer_size = static_cast< UINT >( sizeof( UINT ) * materials.size() );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( material_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_MaterialBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( material_buffer_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_MaterialBufferUpload ) ) );

    BYTE* mapped_material_buffer_upload = nullptr;
    HR( m_MaterialBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_material_buffer_upload ) ) );
    memcpy( mapped_material_buffer_upload, materials.data(), material_buffer_size );
    m_MaterialBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

    command_list->CopyBufferRegion( m_MaterialBuffer, 0, m_MaterialBufferUpload, 0, material_buffer_size );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_MaterialBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE ) );

    m_MaterialBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = static_cast< UINT >( materials.size() / material_texture_count );
    srv_desc.Buffer.StructureByteStride = static_cast< UINT >( material_texture_count * sizeof( UINT ) );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_MaterialBuffer, &srv_desc, m_MaterialBufferSrv.m_Cpu );
}

void CModelSet::CreateWorldMatrixBuffer(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* command_list,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    m_WorldMatrixBufferSize = sizeof( DirectX::XMFLOAT4X4 ) * m_InstanceCount;
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( m_WorldMatrixBufferSize ),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS( &m_WorldMatrixBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( m_WorldMatrixBufferSize ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_WorldMatrixBufferUpload ) ) );

    DirectX::XMFLOAT4X4* mapped_world_matrix_buffer_upload = nullptr;
    HR( m_WorldMatrixBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_world_matrix_buffer_upload ) ) );

    UINT world_matrix_buffer_offset = 0;
    for ( SModel* model : m_Models )
    {
        model->m_MappedWorldMatrixBuffer = mapped_world_matrix_buffer_upload;
        mapped_world_matrix_buffer_upload += static_cast< UINT >( model->m_Instances.size() );
        world_matrix_buffer_offset += static_cast< UINT >( model->m_Instances.size() );
    }

    m_WorldMatrixBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = m_InstanceCount;
    srv_desc.Buffer.StructureByteStride = sizeof( DirectX::XMFLOAT4X4 );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_WorldMatrixBuffer, &srv_desc, m_WorldMatrixBufferSrv.m_Cpu );

    UpdateWorldMatrixBuffer( command_list );
}

void CModelSet::CreateInstanceMappingsBuffers(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* command_list,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    UINT instance_model_mappings_buffer_size = m_InstanceCount * sizeof( UINT );
    UINT model_instance_offset_buffer_size = static_cast< UINT >( m_Models.size() ) * sizeof( UINT );
    m_ModelInstanceCountBufferSize = static_cast< UINT >( m_Models.size() ) * sizeof( UINT );
    m_InstanceIndexMappingsBufferSize = m_InstanceCount * sizeof( UINT );

    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( instance_model_mappings_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_InstanceModelMappingsBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( instance_model_mappings_buffer_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_InstanceModelMappingsBufferUpload ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( model_instance_offset_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_ModelInstanceOffsetBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( model_instance_offset_buffer_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_ModelInstanceOffsetBufferUpload ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( m_ModelInstanceCountBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        nullptr,
        IID_PPV_ARGS( &m_ModelInstanceCountBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( m_ModelInstanceCountBufferSize ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_ModelInstanceCountBufferReset ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( m_InstanceIndexMappingsBufferSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_InstanceIndexMappingsBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( m_InstanceIndexMappingsBufferSize ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_InstanceIndexMappingsBufferUpload ) ) );

    UINT* mapped_instance_model_mappings_buffer_upload = nullptr;
    UINT* mapped_model_instance_offset_buffer_upload = nullptr;
    UINT* mapped_model_instance_count_buffer_reset = nullptr;
    UINT* mapped_instance_index_mappings_upload = nullptr;
    HR( m_InstanceModelMappingsBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_instance_model_mappings_buffer_upload ) ) );
    HR( m_ModelInstanceOffsetBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_model_instance_offset_buffer_upload ) ) );
    HR( m_ModelInstanceCountBufferReset->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_model_instance_count_buffer_reset ) ) );
    HR( m_InstanceIndexMappingsBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_instance_index_mappings_upload ) ) );

    UINT model_instance_offset = 0;
    for ( UINT i = 0; i < static_cast< UINT >( m_Models.size() ); ++i )
    {
        for ( UINT j = 0; j < static_cast< UINT >( m_Models[ i ]->m_Instances.size() ); ++j )
        {
            *mapped_instance_model_mappings_buffer_upload = i;
            ++mapped_instance_model_mappings_buffer_upload;

            mapped_instance_index_mappings_upload[ model_instance_offset + j ] = model_instance_offset + j;
        }

        *mapped_model_instance_offset_buffer_upload = model_instance_offset;
        ++mapped_model_instance_offset_buffer_upload;

        model_instance_offset += static_cast< UINT >( m_Models[ i ]->m_Instances.size() );
    }

    ZeroMemory( mapped_model_instance_count_buffer_reset, m_ModelInstanceCountBufferSize );

    m_InstanceModelMappingsBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_ModelInstanceOffsetBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_ModelInstanceCountBufferReset->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_InstanceIndexMappingsBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

    command_list->CopyBufferRegion( m_InstanceModelMappingsBuffer, 0, m_InstanceModelMappingsBufferUpload, 0, instance_model_mappings_buffer_size );
    command_list->CopyBufferRegion( m_ModelInstanceOffsetBuffer, 0, m_ModelInstanceOffsetBufferUpload, 0, model_instance_offset_buffer_size );
    command_list->CopyBufferRegion( m_InstanceIndexMappingsBuffer, 0, m_InstanceIndexMappingsBufferUpload, 0, m_InstanceIndexMappingsBufferSize );
    const D3D12_RESOURCE_BARRIER post_copy_barriers[] =
    {
        CD3DX12_RESOURCE_BARRIER::Transition( m_InstanceModelMappingsBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_ModelInstanceOffsetBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
        CD3DX12_RESOURCE_BARRIER::Transition( m_InstanceIndexMappingsBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ),
    };
    command_list->ResourceBarrier( _countof( post_copy_barriers ), post_copy_barriers );

    m_InstanceModelMappingsBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_ModelInstanceOffsetBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_ModelInstanceCountBufferUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_InstanceIndexMappingsBufferUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_ModelInstanceCountBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_InstanceIndexMappingsBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = m_InstanceCount;
    srv_desc.Buffer.StructureByteStride = sizeof( UINT );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_InstanceModelMappingsBuffer, &srv_desc, m_InstanceModelMappingsBufferSrv.m_Cpu );

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = static_cast< UINT >( m_Models.size() );
    srv_desc.Buffer.StructureByteStride = sizeof( UINT );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_ModelInstanceOffsetBuffer, &srv_desc, m_ModelInstanceOffsetBufferSrv.m_Cpu );

    ZeroMemory( &uav_desc, sizeof( uav_desc ) );
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = static_cast< UINT >( m_Models.size() );
    uav_desc.Buffer.StructureByteStride = sizeof( UINT );
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    device->CreateUnorderedAccessView( m_ModelInstanceCountBuffer, nullptr, &uav_desc, m_ModelInstanceCountBufferUav.m_Cpu );

    ZeroMemory( &uav_desc, sizeof( uav_desc ) );
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = m_InstanceCount;
    uav_desc.Buffer.StructureByteStride = sizeof( UINT );
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    device->CreateUnorderedAccessView( m_InstanceIndexMappingsBuffer, nullptr, &uav_desc, m_InstanceIndexMappingsBufferUav.m_Cpu );

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = static_cast< UINT >( m_Models.size() );
    srv_desc.Buffer.StructureByteStride = sizeof( UINT );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_ModelInstanceCountBuffer, &srv_desc, m_ModelInstanceCountBufferSrv.m_Cpu );

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = m_InstanceCount;
    srv_desc.Buffer.StructureByteStride = sizeof( UINT );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_InstanceIndexMappingsBuffer, &srv_desc, m_InstanceIndexMappingsBufferSrv.m_Cpu );
}

void CModelSet::CreateCommandBuffers(
    ID3D12Device* device,
    ID3D12GraphicsCommandList* command_list,
    NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] )
{
    UINT input_command_buffer_size = static_cast< UINT >( sizeof( SIndirectCommand ) * m_Models.size() );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( input_command_buffer_size ),
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS( &m_InputCommandBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( input_command_buffer_size ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_InputCommandBufferUpload ) ) );
    UINT output_command_buffer_misalignment = input_command_buffer_size % D3D12_UAV_COUNTER_PLACEMENT_ALIGNMENT;
    m_OutputCommandBufferCounterOffset = input_command_buffer_size + ( output_command_buffer_misalignment > 0 ? D3D12_UAV_COUNTER_PLACEMENT_ALIGNMENT - output_command_buffer_misalignment : 0 );
    UINT output_command_buffer_size = m_OutputCommandBufferCounterOffset + sizeof( UINT );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( output_command_buffer_size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        nullptr,
        IID_PPV_ARGS( &m_OutputCommandBuffer ) ) );
    HR( device->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
        D3D12_HEAP_FLAG_NONE,
        &CD3DX12_RESOURCE_DESC::Buffer( sizeof( UINT ) ),
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS( &m_OutputCommandBufferCounterReset ) ) );

    SIndirectCommand* mapped_input_command_buffer_upload = nullptr;
    UINT* mapped_output_command_buffer_counter_reset = nullptr;
    HR( m_InputCommandBufferUpload->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_input_command_buffer_upload ) ) );
    HR( m_OutputCommandBufferCounterReset->Map( 0, &CD3DX12_RANGE( 0, 0 ), reinterpret_cast< void** >( &mapped_output_command_buffer_counter_reset ) ) );
    
    UINT instance_offset = 0;
    for ( size_t i = 0; i < m_Models.size(); ++i )
    {
        mapped_input_command_buffer_upload[ i ].m_InstanceOffset = instance_offset;
        mapped_input_command_buffer_upload[ i ].m_MaterialIndex = m_Models[ i ]->m_MaterialIndex;
        mapped_input_command_buffer_upload[ i ].m_VertexBufferView = m_Models[ i ]->m_Mesh.GetVertexBufferView();
        mapped_input_command_buffer_upload[ i ].m_IndexBufferView = m_Models[ i ]->m_Mesh.GetIndexBufferView();
        mapped_input_command_buffer_upload[ i ].m_DrawArguments.IndexCountPerInstance = static_cast< UINT >( m_Models[ i ]->m_Indices.size() );
        mapped_input_command_buffer_upload[ i ].m_DrawArguments.InstanceCount = static_cast< UINT >( m_Models[ i ]->m_Instances.size() );
        mapped_input_command_buffer_upload[ i ].m_DrawArguments.StartIndexLocation = 0;
        mapped_input_command_buffer_upload[ i ].m_DrawArguments.BaseVertexLocation = 0;
        mapped_input_command_buffer_upload[ i ].m_DrawArguments.StartInstanceLocation = 0;

        instance_offset += static_cast< UINT >( m_Models[ i ]->m_Instances.size() );
    }
    
    *mapped_output_command_buffer_counter_reset = 0;

    m_InputCommandBufferUpload->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );
    m_OutputCommandBufferCounterReset->Unmap( 0, &CD3DX12_RANGE( 0, 0 ) );

    command_list->CopyBufferRegion( m_InputCommandBuffer, 0, m_InputCommandBufferUpload, 0, input_command_buffer_size );
    command_list->ResourceBarrier( 1, &CD3DX12_RESOURCE_BARRIER::Transition(
        m_InputCommandBuffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE ) );

    m_InputCommandBufferSrv = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();
    m_OutputCommandBufferUav = descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV ].GenerateHandle();

    D3D12_SHADER_RESOURCE_VIEW_DESC srv_desc;
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav_desc;

    ZeroMemory( &srv_desc, sizeof( srv_desc ) );
    srv_desc.Format = DXGI_FORMAT_UNKNOWN;
    srv_desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv_desc.Buffer.NumElements = static_cast< UINT >( m_Models.size() );
    srv_desc.Buffer.StructureByteStride = sizeof( SIndirectCommand );
    srv_desc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_NONE;
    srv_desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    device->CreateShaderResourceView( m_InputCommandBuffer, &srv_desc, m_InputCommandBufferSrv.m_Cpu );

    ZeroMemory( &uav_desc, sizeof( uav_desc ) );
    uav_desc.Format = DXGI_FORMAT_UNKNOWN;
    uav_desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav_desc.Buffer.NumElements = static_cast< UINT >( m_Models.size() );
    uav_desc.Buffer.StructureByteStride = sizeof( SIndirectCommand );
    uav_desc.Buffer.CounterOffsetInBytes = m_OutputCommandBufferCounterOffset;
    uav_desc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_NONE;
    device->CreateUnorderedAccessView( m_OutputCommandBuffer, m_OutputCommandBuffer, &uav_desc, m_OutputCommandBufferUav.m_Cpu );
}