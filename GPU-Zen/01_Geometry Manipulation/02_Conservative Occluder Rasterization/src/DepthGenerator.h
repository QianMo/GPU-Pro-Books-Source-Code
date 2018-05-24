#pragma once

#include "Graphics/Graphics.h"

class COccluderCollection;

class CDepthGenerator
{
public:
    enum ERasterizerMode
    {
        None,
        Standard,
        StandardFullscreen,
        StandardFullscreenUpsample,
        InnerConservative
    };

    enum EReprojectionMode
    {
        Off,
        On
    };

private:
    struct SConstants
    {
        DirectX::XMFLOAT4X4 m_ViewProjectionFlippedZ;
        DirectX::XMFLOAT4X4 m_PreviousToCurrentViewProjectionFlippedZ;
        DirectX::XMFLOAT4 m_CameraPosition;
        UINT m_DepthWidth;
        UINT m_DepthHeight;
        UINT m_FullscreenWidth;
        UINT m_FullscreenHeight;
        FLOAT m_NearZ;
        FLOAT m_FarZ;
        UINT m_SilhouetteEdgeBufferOffset;

        FLOAT m_Padding[ 21 ];
    };

private:
    ERasterizerMode m_RasterizerMode;
    EReprojectionMode m_ReprojectionMode;

    ID3D12Device* m_Device;
    COccluderCollection* m_OccluderCollection;

    ID3D12RootSignature* m_StandardDepthRootSignature;
    ID3D12RootSignature* m_InnerConservativeDepthRootSignature;
    ID3D12RootSignature* m_DownsamplingRootSignature;
    ID3D12RootSignature* m_UpsamplingRootSignature;
    ID3D12RootSignature* m_SilhouetteGenerationRootSignature;
    ID3D12RootSignature* m_ReprojectionRootSignature;
    ID3D12RootSignature* m_MergingRootSignature;
    ID3D12RootSignature* m_DebugDrawRootSignature;

    ID3D12PipelineState* m_StandardDepthPipelineState;
    ID3D12PipelineState* m_StandardFullscreenDepthPipelineState;
    ID3D12PipelineState* m_InnerConservativeDepthPipelineState;
    ID3D12PipelineState* m_DownsamplingPipelineState;
    ID3D12PipelineState* m_UpsamplingPipelineState;
    ID3D12PipelineState* m_SilhouetteGenerationPipelineState;
    ID3D12PipelineState* m_ReprojectionPipelineState;
    ID3D12PipelineState* m_MergingPipelineState;
    ID3D12PipelineState* m_DebugDrawPipelineState;

    static const UINT m_ReprojectionBlockSizeX = 32;
    static const UINT m_ReprojectionBlockSizeY = 32;
    static const UINT m_SilhouetteGenerationBlockSizeX = 32;

    UINT m_DepthWidth;
    UINT m_DepthHeight;
    UINT m_FullscreenWidth;
    UINT m_FullscreenHeight;
    D3D12_VIEWPORT m_DepthViewport;
    D3D12_VIEWPORT m_FullscreenViewport;
    D3D12_RECT m_DepthScissorRect;
    D3D12_RECT m_FullscreenScissorRect;

    NGraphics::CCamera m_CurrentCamera;
    NGraphics::CCamera m_PreviousCamera;

    UINT m_OccluderCount;

    ID3D12Resource* m_ConstantsBuffer;
    SConstants* m_MappedConstantsBuffer;
    NGraphics::SDescriptorHandle m_ConstantsBufferCbv;

    NGraphics::SDescriptorHandle m_DepthSampler;

    NGraphics::CMesh< DirectX::XMFLOAT4 > m_QuadMesh;

    ID3D12Resource* m_DepthTarget;
    ID3D12Resource* m_DepthTargetFullscreen;
    NGraphics::SDescriptorHandle m_DepthTargetDsv;
    NGraphics::SDescriptorHandle m_DepthTargetFullscreenDsv;
    NGraphics::SDescriptorHandle m_DepthTargetSrv;
    NGraphics::SDescriptorHandle m_DepthTargetFullscreenSrv;

    static const UINT m_MaxDownsampleTargetCount = 16;

    UINT m_DownsampleTargetCount;
    D3D12_VIEWPORT* m_DownsampleViewports;
    D3D12_RECT* m_DownsampleScissorRects;
    ID3D12Resource* m_DownsampleTargets[ m_MaxDownsampleTargetCount ];
    NGraphics::SDescriptorHandle* m_DownsampleTargetRtvs;
    NGraphics::SDescriptorHandle* m_DownsampleTargetSrvs;

    UINT m_UpsampledDownsampleTargetCount;
    D3D12_VIEWPORT* m_UpsampledDownsampleViewports;
    D3D12_RECT* m_UpsampledDownsampleScissorRects;
    ID3D12Resource* m_UpsampledDownsampleTargets[ m_MaxDownsampleTargetCount ];
    NGraphics::SDescriptorHandle* m_UpsampledDownsampleTargetRtvs;
    NGraphics::SDescriptorHandle* m_UpsampledDownsampleTargetSrvs;

    ID3D12Resource* m_DepthTargetFullscreenDownsampled;
    NGraphics::SDescriptorHandle m_DepthTargetFullscreenDownsampledRtv;
    NGraphics::SDescriptorHandle m_DepthTargetFullscreenDownsampledSrv;

    ID3D12Resource* m_PreviousDepthDownsampled;
    NGraphics::SDescriptorHandle m_PreviousDepthDownsampledRtv;
    NGraphics::SDescriptorHandle m_PreviousDepthDownsampledSrv;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT m_DepthReprojectionResetLayout;
    ID3D12Resource* m_DepthReprojection;
    ID3D12Resource* m_DepthReprojectionReset;
    NGraphics::SDescriptorHandle m_DepthReprojectionUav;
    NGraphics::SDescriptorHandle m_DepthReprojectionSrv;

    ID3D12Resource* m_DepthFinal;
    NGraphics::SDescriptorHandle m_DepthFinalRtv;
    NGraphics::SDescriptorHandle m_DepthFinalSrv;

    NGraphics::CTimestampQueryHeap m_TimestampQueryHeap;

public:
    CDepthGenerator();

    void Create(
        ID3D12Device* device,
        NGraphics::CCommandContext* graphics_context,
        NGraphics::CDescriptorHeap descriptor_heaps[ D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES ],
        COccluderCollection* occluder_collection,
        UINT depth_width, UINT depth_height,
        UINT fullscreen_width, UINT fullscreen_height );
    void Destroy();

    void Resize(
        ID3D12GraphicsCommandList* command_list,
        UINT depth_width, UINT depth_height,
        UINT fullscreen_width, UINT fullscreen_height );

    void SetRasterizerMode( ERasterizerMode rasterizer_mode );
    void SetReprojectionMode( EReprojectionMode reprojection_mode );
    void Execute(
        ID3D12GraphicsCommandList* command_list,
        NGraphics::CCamera* camera,
        NGraphics::SDescriptorHandle* previous_depth_srv );

    void DebugDraw( ID3D12GraphicsCommandList* command_list );

    void GetTimings( UINT timings[ 8 ] );

    ID3D12Resource* GetDepth() const;

    const UINT GetDepthWidth() const;
    const UINT GetDepthHeight() const;

private:
    void CreateQuadMesh( ID3D12GraphicsCommandList* command_list );

    void Downsample(
        ID3D12GraphicsCommandList* command_list,
        NGraphics::SDescriptorHandle* input_srv,
        NGraphics::SDescriptorHandle* output_rtv );
    void UpsampledDownsample(
        ID3D12GraphicsCommandList* command_list,
        NGraphics::SDescriptorHandle* input_srv,
        NGraphics::SDescriptorHandle* output_rtv );
};