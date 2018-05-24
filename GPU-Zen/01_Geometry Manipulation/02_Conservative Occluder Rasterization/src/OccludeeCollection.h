#pragma once

#include "Graphics/Graphics.h"

#include <vector>

class COcclusionAlgorithm;

class COccludeeCollection
{
public:
    struct SAabb
    {
        DirectX::XMFLOAT3 m_Center;
        DirectX::XMFLOAT3 m_Extent;
    };

private:
    struct SConstants
    {
        DirectX::XMFLOAT4X4 m_ViewProjection;

        FLOAT m_Padding[ 48 ];
    };

private:
    std::vector< SAabb > m_Aabbs;

    ID3D12Device* m_Device;

    ID3D12RootSignature* m_DebugDrawOccludeeRootSignature;
    ID3D12PipelineState* m_DebugDrawOccludeeAabbPipelineState;

    ID3D12Resource* m_AabbBuffer;
    ID3D12Resource* m_AabbBufferUpload;
    NGraphics::SDescriptorHandle m_AabbBufferSrv;

    ID3D12Resource* m_ConstantsBuffer;
    SConstants* m_MappedConstantsBuffer;
    NGraphics::SDescriptorHandle m_ConstantsBufferCbv;

    NGraphics::CMesh< DirectX::XMFLOAT3 > m_CubeMesh;
    NGraphics::CMesh< DirectX::XMFLOAT3 > m_CubeMeshLine;

public:
    COccludeeCollection();

    void AddOccludees( std::vector< SAabb >* occludees );
    void ClearOccludees();

    void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
    void Update( ID3D12GraphicsCommandList* command_list );
    void Destroy();

    void DrawAabbs( ID3D12GraphicsCommandList* command_list );
    void DrawAabbsLine( ID3D12GraphicsCommandList* command_list );

    void DebugDraw(
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CCamera* camera,
        COcclusionAlgorithm* occlusion_algorithm );

    UINT GetAabbCount();

    NGraphics::SDescriptorHandle GetAabbBufferSrv() const;

private:
    void CreateCubeMesh( ID3D12GraphicsCommandList* command_list );
};