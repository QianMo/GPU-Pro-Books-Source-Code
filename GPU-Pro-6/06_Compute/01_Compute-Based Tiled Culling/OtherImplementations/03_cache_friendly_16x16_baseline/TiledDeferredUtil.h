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
// File: TiledDeferredUtil.h
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
namespace ComputeBasedTiledCulling
{
    struct GuiState;
    struct DepthStencilBuffer;
    struct Scene;
    class CommonUtil;
    class LightUtil;
}

namespace ComputeBasedTiledCulling
{
    static const int MAX_NUM_GBUFFER_RENDER_TARGETS = 5;

    class TiledDeferredUtil
    {
    public:
        // Constructor / destructor
        TiledDeferredUtil();
        ~TiledDeferredUtil();

        void AddShadersToCache( AMD::ShaderCache *pShaderCache );

        // Various hook functions
        HRESULT OnCreateDevice( ID3D11Device* pd3dDevice );
        void OnDestroyDevice();
        HRESULT OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc );
        void OnReleasingSwapChain();
        void OnRender( float fElapsedTime, const GuiState& CurrentGuiState, const DepthStencilBuffer& DepthStencilBuffer, const Scene& Scene, const CommonUtil& CommonUtil, const LightUtil& LightUtil );

    private:
        ID3D11ComputeShader * GetLightCullAndShadeCS( int nNumGBufferRenderTargets ) const;
        ID3D11ComputeShader * GetDebugDrawNumLightsPerTileCS( int nDebugDrawType ) const;

    private:
        // G-Buffer
        ID3D11Texture2D* m_pGBuffer[MAX_NUM_GBUFFER_RENDER_TARGETS];
        ID3D11ShaderResourceView* m_pGBufferSRV[MAX_NUM_GBUFFER_RENDER_TARGETS];
        ID3D11RenderTargetView* m_pGBufferRTV[MAX_NUM_GBUFFER_RENDER_TARGETS];

        // off-screen buffer for shading (the compute shader writes to this)
        ID3D11Texture2D*            m_pOffScreenBuffer;
        ID3D11ShaderResourceView*   m_pOffScreenBufferSRV;
        ID3D11RenderTargetView*     m_pOffScreenBufferRTV;
        ID3D11UnorderedAccessView*  m_pOffScreenBufferUAV;

        // shaders for Tiled Deferred (G-Buffer)
        static const int NUM_GBUFFER_PIXEL_SHADERS = 2*(MAX_NUM_GBUFFER_RENDER_TARGETS-1);
        ID3D11VertexShader*         m_pSceneDeferredBuildGBufferVS;
        ID3D11PixelShader*          m_pSceneDeferredBuildGBufferPS[NUM_GBUFFER_PIXEL_SHADERS];
        ID3D11InputLayout*          m_pLayoutDeferredBuildGBuffer11;

        // compute shaders for tiled culling and shading
        static const int NUM_DEFERRED_LIGHTING_COMPUTE_SHADERS = (MAX_NUM_GBUFFER_RENDER_TARGETS-1);
        ID3D11ComputeShader*        m_pLightCullAndShadeCS[NUM_DEFERRED_LIGHTING_COMPUTE_SHADERS];

        // debug draw shaders for the lights-per-tile visualization modes
        static const int NUM_DEBUG_DRAW_COMPUTE_SHADERS = 2;                      // 2 for radar vs. grayscale
        ID3D11ComputeShader*        m_pDebugDrawNumLightsPerTileCS[NUM_DEBUG_DRAW_COMPUTE_SHADERS];

        // pixel shaders for separate lighting
        static const int NUM_DEFERRED_LIGHTING_PIXEL_SHADERS = (MAX_NUM_GBUFFER_RENDER_TARGETS-1);
        ID3D11PixelShader*          m_pDeferredLightingPS[NUM_DEFERRED_LIGHTING_PIXEL_SHADERS];

        // debug draw shaders for the lights-per-tile visualization modes
        static const int NUM_DEBUG_DRAW_PIXEL_SHADERS = 2;                        // 2 for radar vs. grayscale
        ID3D11PixelShader*          m_pDebugDrawNumLightsPerTilePS[NUM_DEBUG_DRAW_PIXEL_SHADERS];

        // state for Tiled Deferred
        ID3D11BlendState*           m_pBlendStateOpaque;
        ID3D11BlendState*           m_pBlendStateAlphaToCoverage;
        ID3D11BlendState*           m_pBlendStateAlpha;
    };

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
