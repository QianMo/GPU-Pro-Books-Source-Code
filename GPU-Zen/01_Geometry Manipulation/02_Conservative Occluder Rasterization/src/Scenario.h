#pragma once

#include "Graphics/Graphics.h"
#include "ModelSet.h"
#include "DepthGenerator.h"
#include "OcclusionAlgorithm.h"
#include "AvgUint.h"

class CScenario : public NGraphics::CGraphicsBase
{
public:
    struct FrameConstants
    {
        DirectX::XMFLOAT4X4 m_ViewProjection;
        DirectX::XMFLOAT4 m_CameraPosition;
        DirectX::XMFLOAT4 m_LightDirection;

        float m_Padding[ 40 ];
    };

    struct ViewVertex
    {
        DirectX::XMFLOAT2 m_Position;
        DirectX::XMFLOAT2 m_Uv;
    };

private:
    ID3D12RootSignature* m_ModelRootSignature;
    ID3D12RootSignature* m_InstanceMappingsUpdateRootSignature;
    ID3D12RootSignature* m_CommandBufferGenerationRootSignature;
    ID3D12CommandSignature* m_ModelCommandSignature;
    ID3D12PipelineState* m_DefaultPipelineState;
    ID3D12PipelineState* m_GroundPipelineState;
    ID3D12PipelineState* m_SkyPipelineState;
    ID3D12PipelineState* m_InstanceMappingsUpdatePipelineState;
    ID3D12PipelineState* m_CommandBufferGenerationPipelineState;

    const UINT m_InstanceMappingsUpdateBlockSizeX = 1;
    const UINT m_CommandBufferGenerationBlockSizeX = 1;

    ID3D12Resource* m_FrameConstantsBuffer;
    FrameConstants* m_MappedFrameConstantsBuffer;
    NGraphics::SDescriptorHandle m_FrameConstantsCbv;

    NGraphics::SDescriptorHandle m_Sampler;

    CModelSet m_CastleLargeOccluders;
    CModelSet m_Ground;
    CModelSet m_MarketStalls;
    CModelSet m_CastleSmallDecorations;
    CModelSet m_Sky;

    NGraphics::CTimestampQueryHeap m_TimestampQueryHeap;

    NGraphics::CCamera m_Camera;

    COccluderCollection m_OccluderCollection;
    COccludeeCollection m_OccludeeCollection;
    CDepthGenerator m_DepthGenerator;
    CHiZOcclusionAlgorithm m_HiZOcclusionAlgorithm;
    CRasterOcclusionAlgorithm m_RasterOcclusionAlgorithm;

    BOOL m_IsRasterCullingSupported;

    TwBar* m_MainBar;
    BOOL m_ShowDepth;
    BOOL m_ShowOccluders;
    BOOL m_ShowOccludees;
    BOOL m_PrevFullscreen;
    UINT m_DepthRasterizerSelection;
    UINT m_DepthReprojectionSelection;
    UINT m_OcclusionAlgorithmSelection;
    UINT m_DrawModeSelection;
    UINT m_PrevDepthRasterizerSelection;
    UINT m_PrevDepthReprojectionSelection;
    UINT m_PrevOcclusionQuerySelection;
    UINT m_PrevDrawModeSelection;
    UINT m_ScreenSize[ 2 ];
    UINT m_DepthSize[ 2 ];
    AvgUint m_AvgTimings[ 16 ];
    UINT m_VisibleOccludeeCount;
    std::string m_VisibleOccludeeCountString;
    std::string m_OccludersDrawnString;
    FLOAT m_OccluderSizeThreshold;

    TwBar* m_OccluderBar;
    CModelSet::SModel* m_SelectedModel;
    std::vector< std::pair< CScenario*, size_t > > m_ScenarioOccluderObbIndices;
    std::vector< std::pair< CScenario*, size_t > > m_ScenarioOccluderCylinderIndices;

public:
    CScenario();

    void AddOccluderObb();
    void AddOccluderCylinder();
    void RemoveOccluderObb( size_t index );
    void RemoveOccluderCylinder( size_t index );

protected:
    void Create();
    void Update( float dt );
    void Draw();
    void Destroy();

    void Resize( UINT width, UINT height );

private:
    void DrawScene( ID3D12GraphicsCommandList* command_list, COcclusionAlgorithm* occlusion_algorithm );
    void DrawSceneIndirect( ID3D12GraphicsCommandList* command_list, COcclusionAlgorithm* occlusion_algorithm );

    void DrawModelSet( ID3D12GraphicsCommandList* command_list, CModelSet* model_set );
    void DrawModelSetIndirect( ID3D12GraphicsCommandList* command_list, CModelSet* model_set );

    void UpdateMainBar();
    void UpdateOccluderBar();

    void SelectModel();
};