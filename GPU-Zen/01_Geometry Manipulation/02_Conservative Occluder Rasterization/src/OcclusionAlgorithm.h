#pragma once

#include "Graphics/Graphics.h"

class CDepthGenerator;
class COccludeeCollection;

class COcclusionAlgorithm
{
protected:
    struct SConstants
    {
        DirectX::XMFLOAT4 m_FrustumPlanes[ 6 ];
        DirectX::XMFLOAT4X4 m_ViewProjection;
        DirectX::XMFLOAT4X4 m_ViewProjectionFlippedZ;
        DirectX::XMFLOAT4 m_CameraPosition;
        UINT m_Width;
        UINT m_Height;
        FLOAT m_NearZ;
        FLOAT m_FarZ;
    };

protected:
    COccludeeCollection* m_OccludeeCollection;
    CDepthGenerator* m_DepthGenerator;

    ID3D12Device* m_Device;

    ID3D12Resource* m_ConstantsBuffer;
    SConstants* m_MappedConstantsBuffer;
    NGraphics::SDescriptorHandle m_ConstantsBufferCbv;

    UINT m_VisibilityBufferSize;
    UINT* m_Visibility;
    ID3D12Resource* m_VisibilityBuffer;
    ID3D12Resource* m_VisibilityBufferUpload;
    ID3D12Resource* m_VisibilityBufferReadback;
    NGraphics::SDescriptorHandle m_VisibilityBufferUav;
    NGraphics::SDescriptorHandle m_VisibilityBufferSrv;

    NGraphics::CTimestampQueryHeap m_TimestampQueryHeap;

public:
    COcclusionAlgorithm();

    virtual void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
        COccludeeCollection* occludee_collection,
        CDepthGenerator* depth_generator );
    virtual void Destroy();

    virtual void Update( ID3D12GraphicsCommandList* command_list );

    virtual void Execute( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera ) = 0;

    void Readback();

    BOOL IsOccludeeVisible( UINT occludee_index );

    NGraphics::SDescriptorHandle GetVisibilityBufferSrv() const;

protected:
    virtual void UpdateVisibilityBuffer( ID3D12GraphicsCommandList* command_list );
};

class CHiZOcclusionAlgorithm : public COcclusionAlgorithm
{
private:
    ID3D12RootSignature* m_DownsamplingRootSignature;
    ID3D12RootSignature* m_OcclusionAlgorithmRootSignature;
    ID3D12PipelineState* m_DownsamplingPipelineState;
    ID3D12PipelineState* m_OcclusionAlgorithmPipelineState;

    const UINT m_OcclusionQueryBlockSizeX = 32;

    const UINT m_MaxDepthHierarchyMipCount = 16;
    UINT m_DepthHierarchyWidth;
    UINT m_DepthHierarchyHeight;
    UINT m_DepthHierarchyMipCount;
    D3D12_VIEWPORT* m_DepthHierarchyViewports;
    D3D12_RECT* m_DepthHierarchyScissorRects;
    ID3D12Resource* m_DepthHierarchy[ 2 ];
    NGraphics::SDescriptorHandle* m_DepthHierarchyMipRtvs;
    NGraphics::SDescriptorHandle* m_DepthHierarchyMipSrvs;
    NGraphics::SDescriptorHandle m_DepthHierarchySrvs[ 2 ];
    NGraphics::SDescriptorHandle m_DepthHierarchySampler;

    NGraphics::CMesh< DirectX::XMFLOAT4 > m_QuadMesh;

public:
    CHiZOcclusionAlgorithm();

    void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
        COccludeeCollection* occludee_collection,
        CDepthGenerator* depth_generator );
    void Destroy();

    void Update( ID3D12GraphicsCommandList* command_list );

    void Execute( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera );

    UINT GetDownsampleTime();
    UINT GetOcclusionQueryTime();

private:
    void CreateQuadMesh( ID3D12GraphicsCommandList* command_list );

    void UpdateDepthHierarchy( ID3D12GraphicsCommandList* command_list );
};

class CRasterOcclusionAlgorithm : public COcclusionAlgorithm
{
private:
    ID3D12RootSignature* m_OcclusionAlgorithmRootSignature;
    ID3D12PipelineState* m_OcclusionAlgorithmPipelineState;

    UINT m_DepthTargetWidth;
    UINT m_DepthTargetHeight;
    D3D12_VIEWPORT m_Viewport;
    D3D12_RECT m_ScissorRect;
    ID3D12Resource* m_DepthTarget;
    NGraphics::SDescriptorHandle m_DepthTargetDsv;

public:
    CRasterOcclusionAlgorithm();

    void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
        COccludeeCollection* occludee_collection,
        CDepthGenerator* depth_generator );
    void Destroy();

    void Update( ID3D12GraphicsCommandList* command_list );

    void Execute( ID3D12GraphicsCommandList* command_list, NGraphics::CCamera* camera );

    UINT GetOcclusionQueryTime();

private:
    void UpdateDepthTarget( ID3D12GraphicsCommandList* command_list );
};