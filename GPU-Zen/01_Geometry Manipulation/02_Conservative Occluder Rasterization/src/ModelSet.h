#pragma once

#include "Graphics/Graphics.h"
#include "OccluderCollection.h"
#include "OccludeeCollection.h"

#include <vector>
#include <map>
#include <string>

class COcclusionAlgorithm;

class CModelSet
{
public:
    struct SVertex
    {
        DirectX::XMFLOAT3 m_Position;
        DirectX::XMFLOAT3 m_Normal;
        DirectX::XMFLOAT2 m_Uv;
        DirectX::XMFLOAT3 m_Tangent;
        DirectX::XMFLOAT3 m_Binormal;
    };

    struct SOccluderObb
    {
        DirectX::XMFLOAT3 m_Center;
        DirectX::XMFLOAT3 m_Extent;
        DirectX::XMFLOAT3 m_Rotation; // Euler angles, for simplicity
    };

    struct SOccluderCylinder
    {
        DirectX::XMFLOAT3 m_Center;
        FLOAT m_Radius;
        FLOAT m_Height;
        DirectX::XMFLOAT3 m_Rotation; // Euler angles, for simplicity
    };

    struct SInstance
    {
        DirectX::XMFLOAT4X4 m_World;

        DirectX::XMFLOAT3 m_OccludeeAabbMin;
        DirectX::XMFLOAT3 m_OccludeeAabbMax;
    };

    struct SModel
    {
        std::string m_Name;

        std::vector< SVertex > m_Vertices;
        std::vector< UINT > m_Indices;

        std::map< std::string, std::string > m_TextureFilepathMap;

        DirectX::XMFLOAT3 m_OccludeeBoundingBoxCenter;
        DirectX::XMFLOAT3 m_OccludeeBoundingBoxExtent;

        std::vector< SOccluderObb > m_OccluderObbs;
        std::vector< SOccluderCylinder > m_OccluderCylinders;

        UINT m_VisibleInstanceCount;
        std::vector< SInstance > m_Instances;

        NGraphics::CMesh< SVertex > m_Mesh;

        UINT m_MaterialIndex;

        DirectX::XMFLOAT4X4* m_MappedWorldMatrixBuffer;
    };

    struct STexture
    {
        ID3D12Resource* m_Resource;
        ID3D12Resource* m_ResourceUpload;
        NGraphics::SDescriptorHandle m_Handle;
    };

    struct SIndirectCommand
    {
        UINT m_InstanceOffset;
        UINT m_MaterialIndex;
        D3D12_VERTEX_BUFFER_VIEW m_VertexBufferView;
        D3D12_INDEX_BUFFER_VIEW m_IndexBufferView;
        D3D12_DRAW_INDEXED_ARGUMENTS m_DrawArguments;
    };

private:
    UINT m_InstanceCount;

    std::vector< SModel* > m_Models;

    std::vector< STexture > m_Textures;

    ID3D12Resource* m_MaterialBuffer;
    ID3D12Resource* m_MaterialBufferUpload;
    NGraphics::SDescriptorHandle m_MaterialBufferSrv;

    UINT m_WorldMatrixBufferSize;
    ID3D12Resource* m_WorldMatrixBuffer;
    ID3D12Resource* m_WorldMatrixBufferUpload;
    NGraphics::SDescriptorHandle m_WorldMatrixBufferSrv;

    UINT m_ModelInstanceCountBufferSize;
    UINT m_InstanceIndexMappingsBufferSize;
    ID3D12Resource* m_InstanceModelMappingsBuffer;
    ID3D12Resource* m_InstanceModelMappingsBufferUpload;
    ID3D12Resource* m_ModelInstanceOffsetBuffer;
    ID3D12Resource* m_ModelInstanceOffsetBufferUpload;
    ID3D12Resource* m_ModelInstanceCountBuffer;
    ID3D12Resource* m_ModelInstanceCountBufferReset;
    ID3D12Resource* m_InstanceIndexMappingsBuffer;
    ID3D12Resource* m_InstanceIndexMappingsBufferUpload;
    NGraphics::SDescriptorHandle m_InstanceModelMappingsBufferSrv;
    NGraphics::SDescriptorHandle m_ModelInstanceOffsetBufferSrv;
    NGraphics::SDescriptorHandle m_ModelInstanceCountBufferUav;
    NGraphics::SDescriptorHandle m_InstanceIndexMappingsBufferUav;
    NGraphics::SDescriptorHandle m_ModelInstanceCountBufferSrv;
    NGraphics::SDescriptorHandle m_InstanceIndexMappingsBufferSrv;

    UINT m_OutputCommandBufferCounterOffset;
    ID3D12Resource* m_InputCommandBuffer;
    ID3D12Resource* m_InputCommandBufferUpload;
    ID3D12Resource* m_OutputCommandBuffer;
    ID3D12Resource* m_OutputCommandBufferCounterReset;
    NGraphics::SDescriptorHandle m_InputCommandBufferSrv;
    NGraphics::SDescriptorHandle m_OutputCommandBufferUav;

    std::string m_Directory;
    std::string m_Filename;

    std::vector< DirectX::XMFLOAT4X4 > m_OccluderObbs;
    std::vector< DirectX::XMFLOAT4X4 > m_OccluderCylinders;
    std::vector< COccludeeCollection::SAabb > m_OccludeeAabbs;

public:
    CModelSet();

    void Load( const std::string& directory, const std::string& filename );
    void Save();

    void Create(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
    void Destroy();

    void UpdateWorldMatrixBuffer( ID3D12GraphicsCommandList* command_list );
    void UpdateInstanceMappings(
        ID3D12GraphicsCommandList* command_list,
        COcclusionAlgorithm* occlusion_algorithm,
        UINT occludee_offset );

    void CalculateOccluders();
    void CalculateOccludees();

    const UINT GetModelCount() const;
    const UINT GetInstanceCount() const;
    const UINT GetTextureCount() const;

    std::vector< SModel* >* GetModels();
                 
    std::vector< DirectX::XMFLOAT4X4 >* GetOccluderObbs();
    std::vector< DirectX::XMFLOAT4X4 >* GetOccluderCylinders();
    std::vector< COccludeeCollection::SAabb >* GetOccludeeAabbs();

    NGraphics::SDescriptorHandle GetMaterialBufferSrv() const;
    NGraphics::SDescriptorHandle GetWorldMatrixBufferSrv() const;
    NGraphics::SDescriptorHandle GetTexturesSrv() const;

    NGraphics::SDescriptorHandle GetInstanceModelMappingsBufferSrv() const;
    NGraphics::SDescriptorHandle GetModelInstanceOffsetBufferSrv() const;
    NGraphics::SDescriptorHandle GetModelInstanceCountBufferUav() const;
    NGraphics::SDescriptorHandle GetModelInstanceCountBufferSrv() const;
    NGraphics::SDescriptorHandle GetInstanceIndexMappingsBufferUav() const;
    NGraphics::SDescriptorHandle GetInstanceIndexMappingsBufferSrv() const;
    NGraphics::SDescriptorHandle GetInputCommandBufferSrv() const;
    NGraphics::SDescriptorHandle GetOutputCommandBufferUav() const;

    ID3D12Resource* GetModelInstanceCountBuffer() const;
    ID3D12Resource* GetModelInstanceCountBufferReset() const;
    ID3D12Resource* GetInstanceIndexMappingsBuffer() const;
    ID3D12Resource* GetOutputCommandBuffer() const;
    ID3D12Resource* GetOutputCommandBufferCounterReset() const;

    const UINT GetModelInstanceCountBufferSize() const;
    const UINT GetOutputCommandBufferCounterOffset() const;

private:
    void LoadModels();
    void SaveModels();

    void CreateMeshes(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* command_list );
    void CreateTexturesAndMaterials(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
    void CreateWorldMatrixBuffer(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
    void CreateInstanceMappingsBuffers(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
    void CreateCommandBuffers(
        ID3D12Device* device,
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
};