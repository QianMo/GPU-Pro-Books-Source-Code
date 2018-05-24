//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the "Materials") pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: CommonUtil.h
//
// Helper functions for the ComputeBasedTiledCulling sample.
//--------------------------------------------------------------------------------------

#pragma once

#include "..\\DXUT\\Core\\DXUT.h"
#include "CommonConstants.h"

// Forward declarations
namespace AMD
{
    class ShaderCache;
}
class CDXUTSDKMesh;
class CDXUTTextHelper;
class CFirstPersonCamera;

namespace ComputeBasedTiledCulling
{
    enum DebugDrawType
    {
        DEBUG_DRAW_NONE,
        DEBUG_DRAW_RADAR_COLORS,
        DEBUG_DRAW_GRAYSCALE,
    };

    enum TriangleDensityType
    {
        TRIANGLE_DENSITY_LOW = 0,
        TRIANGLE_DENSITY_MEDIUM,
        TRIANGLE_DENSITY_HIGH,
        TRIANGLE_DENSITY_NUM_TYPES
    };

    enum DepthStencilStateType
    {
        DEPTH_STENCIL_STATE_DISABLE_DEPTH_WRITE = 0,
        DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST,
        DEPTH_STENCIL_STATE_DEPTH_GREATER,
        DEPTH_STENCIL_STATE_DEPTH_GREATER_AND_DISABLE_DEPTH_WRITE,
        DEPTH_STENCIL_STATE_DEPTH_EQUAL_AND_DISABLE_DEPTH_WRITE,
        DEPTH_STENCIL_STATE_NUM_TYPES
    };

    enum RasterizerStateType
    {
        RASTERIZER_STATE_DISABLE_CULLING = 0,
        RASTERIZER_STATE_WIREFRAME,
        RASTERIZER_STATE_WIREFRAME_DISABLE_CULLING,
        RASTERIZER_STATE_NUM_TYPES
    };

    enum SamplerStateType
    {
        SAMPLER_STATE_POINT = 0,
        SAMPLER_STATE_LINEAR,
        SAMPLER_STATE_ANISO,
        SAMPLER_STATE_NUM_TYPES
    };

    struct GuiState
    {
        unsigned m_uNumPointLights;
        unsigned m_uNumSpotLights;
        int m_nDebugDrawType;
        bool m_bDoTiledDeferredWithSeparateCulling;
        bool m_bLightDrawingEnabled;
        int m_nGridObjectTriangleDensity;
        int m_nNumGridObjects;
        int m_nNumGBufferRenderTargets;
    };

    struct DepthStencilBuffer
    {
        ID3D11Texture2D*          m_pDepthStencilTexture;
        ID3D11DepthStencilView*   m_pDepthStencilView;
        ID3D11ShaderResourceView* m_pDepthStencilSRV;
    };

    struct Scene
    {
        CDXUTSDKMesh* m_pSceneMesh;
        CDXUTSDKMesh* m_pAlphaMesh;
        CFirstPersonCamera* m_pCamera;
    };

    class CommonUtil
    {
    public:
        // Constructor / destructor
        CommonUtil();
        ~CommonUtil();

        static void InitStaticData();
        static void CalculateSceneMinMax( CDXUTSDKMesh &Mesh, DirectX::XMVECTOR *pBBoxMinOut, DirectX::XMVECTOR *pBBoxMaxOut );

        void AddShadersToCache( AMD::ShaderCache *pShaderCache );

        void RenderLegend( CDXUTTextHelper *pTxtHelper, int nLineHeight, DirectX::XMFLOAT4 Color, int nDebugDrawType ) const;

        // Various hook functions
        HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
        void OnDestroyDevice();
        HRESULT OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, int nLineHeight );
        void OnReleasingSwapChain();

        unsigned GetNumTilesX() const;
        unsigned GetNumTilesY() const;
        unsigned GetMaxNumLightsPerTile() const;
        unsigned GetMaxNumElementsPerTile() const;

        ID3D11ShaderResourceView * const * GetLightIndexBufferSRVParam() const { return &m_pLightIndexBufferSRV; }
        ID3D11UnorderedAccessView * const * GetLightIndexBufferUAVParam() const { return &m_pLightIndexBufferUAV; }

        ID3D11ShaderResourceView * const * GetSpotIndexBufferSRVParam() const { return &m_pSpotIndexBufferSRV; }
        ID3D11UnorderedAccessView * const * GetSpotIndexBufferUAVParam() const { return &m_pSpotIndexBufferUAV; }

        void DrawGrid(int nGridNumber, int nTriangleDensity, bool bWithTextures = true) const;

        ID3D11ComputeShader * GetLightCullCS() const { return m_pLightCullCS; }

        ID3D11PixelShader * GetDebugDrawNumLightsPerTilePS( int nDebugDrawType ) const;

        ID3D11VertexShader * GetFullScreenVS() const { return m_pFullScreenVS; }
        ID3D11PixelShader * GetFullScreenPS() const { return m_pFullScreenPS; }

        ID3D11DepthStencilState * GetDepthStencilState( int nDepthStencilStateType ) const { return m_pDepthStencilState[nDepthStencilStateType]; }
        ID3D11RasterizerState * GetRasterizerState( int nRasterizerStateType ) const { return m_pRasterizerState[nRasterizerStateType]; }

        ID3D11SamplerState * const * GetSamplerStateParam( int nSamplerStateType ) const { return &m_pSamplerState[nSamplerStateType]; }

    private:

        // Light culling constants.
        // These must match their counterparts in CommonHeader.h
        static const unsigned TILE_RES = 8;
        static const unsigned MAX_NUM_LIGHTS_PER_TILE = 272;

        // back buffer width and height
        unsigned                    m_uWidth;
        unsigned                    m_uHeight;

        // buffers for light culling
        ID3D11Buffer*               m_pLightIndexBuffer;
        ID3D11ShaderResourceView*   m_pLightIndexBufferSRV;
        ID3D11UnorderedAccessView*  m_pLightIndexBufferUAV;

        // buffers for spot light culling
        ID3D11Buffer*               m_pSpotIndexBuffer;
        ID3D11ShaderResourceView*   m_pSpotIndexBufferSRV;
        ID3D11UnorderedAccessView*  m_pSpotIndexBufferUAV;

        // grid VB and IB (for different triangle densities)
        ID3D11Buffer*               m_pGridVB[TRIANGLE_DENSITY_NUM_TYPES][MAX_NUM_GRID_OBJECTS];
        ID3D11Buffer*               m_pGridIB[TRIANGLE_DENSITY_NUM_TYPES];

        // grid diffuse and normal map textures
        ID3D11ShaderResourceView*   m_pGridDiffuseTextureSRV;
        ID3D11ShaderResourceView*   m_pGridNormalMapSRV;

        // sprite quad VB (for debug drawing the lights-per-tile legend texture)
        ID3D11Buffer*               m_pQuadForLegendVB;

        // compute shader for tiled culling
        ID3D11ComputeShader*        m_pLightCullCS;

        // debug draw shaders for the lights-per-tile visualization modes
        ID3D11PixelShader*          m_pDebugDrawNumLightsPerTileRadarColorsPS;
        ID3D11PixelShader*          m_pDebugDrawNumLightsPerTileGrayscalePS;

        // debug draw shaders for the lights-per-tile legend
        ID3D11VertexShader*         m_pDebugDrawLegendForNumLightsPerTileVS;
        ID3D11PixelShader*          m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS;
        ID3D11PixelShader*          m_pDebugDrawLegendForNumLightsPerTileGrayscalePS;
        ID3D11InputLayout*          m_pDebugDrawLegendForNumLightsLayout11;

        // shaders for full-screen blit/downsample
        ID3D11VertexShader*         m_pFullScreenVS;
        ID3D11PixelShader*          m_pFullScreenPS;

        // state
        ID3D11DepthStencilState*    m_pDepthStencilState[DEPTH_STENCIL_STATE_NUM_TYPES];
        ID3D11RasterizerState*      m_pRasterizerState[RASTERIZER_STATE_NUM_TYPES];
        ID3D11SamplerState*         m_pSamplerState[SAMPLER_STATE_NUM_TYPES];
    };

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
