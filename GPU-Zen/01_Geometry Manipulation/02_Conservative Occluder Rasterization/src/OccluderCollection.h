#pragma once

#include "Graphics/Graphics.h"
#include "OccluderMesh.h"

#include <vector>

class COccluderCollection
{
    friend class CDepthGenerator;

private:
    struct SOccluderModel
    {
        COccluderMesh* m_OccluderMesh;
        std::vector< DirectX::XMFLOAT4X4 > m_WorldMatrices;
        UINT m_SelectedOccluderCount;
    };

    struct SConstants
    {
        DirectX::XMFLOAT4X4 m_ViewProjection;

        FLOAT m_Padding[ 48 ];
    };

private:
    ID3D12Device* m_Device;

    ID3D12RootSignature* m_DebugDrawOccluderRootSignature;
    ID3D12PipelineState* m_DebugDrawOccluderPipelineState;

    ID3D12Resource* m_ConstantsBuffer;
    SConstants* m_MappedConstantsBuffer;
    NGraphics::SDescriptorHandle m_ConstantsBufferCbv;

    UINT m_WorldMatrixBufferSize;
    ID3D12Resource* m_WorldMatrixBuffer;
    ID3D12Resource* m_WorldMatrixBufferUpload;
    DirectX::XMFLOAT4X4* m_MappedWorldMatrixBufferUpload;
    NGraphics::SDescriptorHandle m_WorldMatrixBufferSrv;

    static const UINT m_SilhouetteEdgeBufferOffset = 32;
    UINT m_SilhouetteEdgeCountBufferSize;
    ID3D12Resource* m_SilhouetteEdgeBuffer;
    ID3D12Resource* m_SilhouetteEdgeCountBuffer;
    ID3D12Resource* m_SilhouetteEdgeCountBufferReset;
    NGraphics::SDescriptorHandle m_SilhouetteEdgeBufferSrv;
    NGraphics::SDescriptorHandle m_SilhouetteEdgeBufferUav;
    NGraphics::SDescriptorHandle m_SilhouetteEdgeCountBufferSrv;
    NGraphics::SDescriptorHandle m_SilhouetteEdgeCountBufferUav;

    std::vector< SOccluderModel > m_OccluderModels;
    UINT m_OccluderInstanceCount;

    UINT m_SelectedOccluderCount;

public:
    COccluderCollection();
    ~COccluderCollection();

    void AddOccluderObbs( std::vector< DirectX::XMFLOAT4X4 >* occluders );
    void AddOccluderCylinders( std::vector< DirectX::XMFLOAT4X4 >* occluders );
    void ClearOccluders();

    void Create( ID3D12Device* device,
                 NGraphics::CCommandContext* graphics_context,
                 NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ] );
    void Update( ID3D12GraphicsCommandList* command_list );
    void Destroy();

    void SelectOccluders( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera, FLOAT size_threshold );

    void DebugDraw( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera );

    const UINT GetOccluderInstanceCount() const;
    const UINT GetSelectedOccluderCount() const;

private:
    BOOL IsAabbInsideFrustum( DirectX::XMFLOAT3 aabb_center, DirectX::XMFLOAT3 aabb_extent, DirectX::XMFLOAT4 frustum_planes[ 6 ] );
};