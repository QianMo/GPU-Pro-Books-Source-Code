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
// File: TiledDeferredUtil.cpp
//
// Helper functions for the ComputeBasedTiledCulling sample.
//--------------------------------------------------------------------------------------

#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"
#include "..\\DXUT\\Optional\\DXUTCamera.h"

#include "..\\AMD_SDK\\AMD_SDK.h"

#include "TiledDeferredUtil.h"
#include "CommonUtil.h"
#include "LightUtil.h"

#pragma warning( disable : 4100 ) // disable unreference formal parameter warnings for /W4 builds

// there should only be one TiledDeferredUtil object
static int TiledDeferredUtilObjectCounter = 0;

namespace ComputeBasedTiledCulling
{

    //--------------------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------------------
    TiledDeferredUtil::TiledDeferredUtil()
        :m_pOffScreenBuffer(NULL)
        ,m_pOffScreenBufferSRV(NULL)
        ,m_pOffScreenBufferRTV(NULL)
        ,m_pOffScreenBufferUAV(NULL)
        ,m_pSceneDeferredBuildGBufferVS(NULL)
        ,m_pLayoutDeferredBuildGBuffer11(NULL)
        ,m_pBlendStateOpaque(NULL)
        ,m_pBlendStateAlphaToCoverage(NULL)
        ,m_pBlendStateAlpha(NULL)
    {
        assert( TiledDeferredUtilObjectCounter == 0 );
        TiledDeferredUtilObjectCounter++;

        for( int i = 0; i < MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            m_pGBuffer[i] = NULL;
            m_pGBufferSRV[i] = NULL;
            m_pGBufferRTV[i] = NULL;
        }

        for( int i = 0; i < NUM_GBUFFER_PIXEL_SHADERS; i++ )
        {
            m_pSceneDeferredBuildGBufferPS[i] = NULL;
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_COMPUTE_SHADERS; i++ )
        {
            m_pLightCullAndShadeCS[i] = NULL;
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_COMPUTE_SHADERS; i++ )
        {
            m_pDebugDrawNumLightsPerTileCS[i] = NULL;
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_PIXEL_SHADERS; i++ )
        {
            m_pDeferredLightingPS[i] = NULL;
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_PIXEL_SHADERS; i++ )
        {
            m_pDebugDrawNumLightsPerTilePS[i] = NULL;
        }
    }


    //--------------------------------------------------------------------------------------
    // Destructor
    //--------------------------------------------------------------------------------------
    TiledDeferredUtil::~TiledDeferredUtil()
    {
        for( int i = 0; i < MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            SAFE_RELEASE(m_pGBuffer[i]);
            SAFE_RELEASE(m_pGBufferSRV[i]);
            SAFE_RELEASE(m_pGBufferRTV[i]);
        }

        SAFE_RELEASE(m_pOffScreenBuffer);
        SAFE_RELEASE(m_pOffScreenBufferSRV);
        SAFE_RELEASE(m_pOffScreenBufferRTV);
        SAFE_RELEASE(m_pOffScreenBufferUAV);

        SAFE_RELEASE(m_pSceneDeferredBuildGBufferVS);
        SAFE_RELEASE(m_pLayoutDeferredBuildGBuffer11);

        for( int i = 0; i < NUM_GBUFFER_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pSceneDeferredBuildGBufferPS[i]);
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_COMPUTE_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pLightCullAndShadeCS[i]);
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_COMPUTE_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDebugDrawNumLightsPerTileCS[i]);
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDeferredLightingPS[i]);
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDebugDrawNumLightsPerTilePS[i]);
        }

        SAFE_RELEASE(m_pBlendStateOpaque);
        SAFE_RELEASE(m_pBlendStateAlphaToCoverage);
        SAFE_RELEASE(m_pBlendStateAlpha);

        assert( TiledDeferredUtilObjectCounter == 1 );
        TiledDeferredUtilObjectCounter--;
    }

    //--------------------------------------------------------------------------------------
    // Device creation hook function
    //--------------------------------------------------------------------------------------
    HRESULT TiledDeferredUtil::OnCreateDevice( ID3D11Device* pd3dDevice )
    {
        HRESULT hr;

        // Create blend states 
        D3D11_BLEND_DESC BlendStateDesc;
        ZeroMemory( &BlendStateDesc, sizeof( D3D11_BLEND_DESC ) );
        BlendStateDesc.AlphaToCoverageEnable = FALSE;
        BlendStateDesc.IndependentBlendEnable = FALSE;
        D3D11_RENDER_TARGET_BLEND_DESC RTBlendDesc;
        RTBlendDesc.BlendEnable = FALSE;
        RTBlendDesc.SrcBlend = D3D11_BLEND_ONE; 
        RTBlendDesc.DestBlend = D3D11_BLEND_ZERO; 
        RTBlendDesc.BlendOp = D3D11_BLEND_OP_ADD;
        RTBlendDesc.SrcBlendAlpha = D3D11_BLEND_ONE; 
        RTBlendDesc.DestBlendAlpha = D3D11_BLEND_ZERO; 
        RTBlendDesc.BlendOpAlpha = D3D11_BLEND_OP_ADD;
        RTBlendDesc.RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        BlendStateDesc.RenderTarget[0] = RTBlendDesc;
        BlendStateDesc.RenderTarget[1] = RTBlendDesc;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateOpaque ) );
        BlendStateDesc.AlphaToCoverageEnable = TRUE;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateAlphaToCoverage ) );
        BlendStateDesc.AlphaToCoverageEnable = FALSE;
        RTBlendDesc.BlendEnable = TRUE;
        RTBlendDesc.SrcBlend = D3D11_BLEND_SRC_ALPHA;
        RTBlendDesc.DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        RTBlendDesc.SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
        RTBlendDesc.DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        BlendStateDesc.RenderTarget[0] = RTBlendDesc;
        BlendStateDesc.RenderTarget[1] = RTBlendDesc;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateAlpha ) );

        return S_OK;
    }


    //--------------------------------------------------------------------------------------
    // Device destruction hook function
    //--------------------------------------------------------------------------------------
    void TiledDeferredUtil::OnDestroyDevice()
    {
        for( int i = 0; i < MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            SAFE_RELEASE(m_pGBuffer[i]);
            SAFE_RELEASE(m_pGBufferSRV[i]);
            SAFE_RELEASE(m_pGBufferRTV[i]);
        }

        SAFE_RELEASE(m_pOffScreenBuffer);
        SAFE_RELEASE(m_pOffScreenBufferSRV);
        SAFE_RELEASE(m_pOffScreenBufferRTV);
        SAFE_RELEASE(m_pOffScreenBufferUAV);

        SAFE_RELEASE(m_pSceneDeferredBuildGBufferVS);
        SAFE_RELEASE(m_pLayoutDeferredBuildGBuffer11);

        for( int i = 0; i < NUM_GBUFFER_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pSceneDeferredBuildGBufferPS[i]);
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_COMPUTE_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pLightCullAndShadeCS[i]);
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_COMPUTE_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDebugDrawNumLightsPerTileCS[i]);
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDeferredLightingPS[i]);
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDebugDrawNumLightsPerTilePS[i]);
        }

        SAFE_RELEASE(m_pBlendStateOpaque);
        SAFE_RELEASE(m_pBlendStateAlphaToCoverage);
        SAFE_RELEASE(m_pBlendStateAlpha)
    }


    //--------------------------------------------------------------------------------------
    // Resized swap chain hook function
    //--------------------------------------------------------------------------------------
    HRESULT TiledDeferredUtil::OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc )
    {
        HRESULT hr;

        // create the G-Buffer
        V_RETURN( AMD::CreateSurface( &m_pGBuffer[0], &m_pGBufferSRV[0], &m_pGBufferRTV[0], NULL, DXGI_FORMAT_R8G8B8A8_UNORM, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, pBackBufferSurfaceDesc->SampleDesc.Count ) );
        V_RETURN( AMD::CreateSurface( &m_pGBuffer[1], &m_pGBufferSRV[1], &m_pGBufferRTV[1], NULL, DXGI_FORMAT_R8G8B8A8_UNORM, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, pBackBufferSurfaceDesc->SampleDesc.Count ) );

        // create extra dummy G-Buffer render targets to test performance 
        // as the G-Buffer gets fatter
        for( int i = 2; i < MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            V_RETURN( AMD::CreateSurface( &m_pGBuffer[i], &m_pGBufferSRV[i], &m_pGBufferRTV[i], NULL, DXGI_FORMAT_R8G8B8A8_UNORM, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, pBackBufferSurfaceDesc->SampleDesc.Count ) );
        }

        // create the offscreen buffer for shading
        // (note, multisampling is not supported with UAVs, 
        // so scale the resolution instead)
        unsigned uCorrectedWidth = (pBackBufferSurfaceDesc->SampleDesc.Count > 1)  ? 2*pBackBufferSurfaceDesc->Width : pBackBufferSurfaceDesc->Width;
        unsigned uCorrectedHeight = (pBackBufferSurfaceDesc->SampleDesc.Count == 4) ? 2*pBackBufferSurfaceDesc->Height : pBackBufferSurfaceDesc->Height;
        V_RETURN( AMD::CreateSurface( &m_pOffScreenBuffer, &m_pOffScreenBufferSRV, &m_pOffScreenBufferRTV, &m_pOffScreenBufferUAV, DXGI_FORMAT_R16G16B16A16_FLOAT, uCorrectedWidth, uCorrectedHeight, 1 ) );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Releasing swap chain hook function
    //--------------------------------------------------------------------------------------
    void TiledDeferredUtil::OnReleasingSwapChain()
    {
        for( int i = 0; i < MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            SAFE_RELEASE(m_pGBuffer[i]);
            SAFE_RELEASE(m_pGBufferSRV[i]);
            SAFE_RELEASE(m_pGBufferRTV[i]);
        }

        SAFE_RELEASE(m_pOffScreenBuffer);
        SAFE_RELEASE(m_pOffScreenBufferSRV);
        SAFE_RELEASE(m_pOffScreenBufferRTV);
        SAFE_RELEASE(m_pOffScreenBufferUAV);
    }

    //--------------------------------------------------------------------------------------
    // Render hook function, to draw the scene using Tiled Deferred
    //--------------------------------------------------------------------------------------
    void TiledDeferredUtil::OnRender( float fElapsedTime, const GuiState& CurrentGuiState, const DepthStencilBuffer& DepthStencilBuffer, const Scene& Scene, const CommonUtil& CommonUtil, const LightUtil& LightUtil )
    {
        assert(CurrentGuiState.m_nNumGBufferRenderTargets >=2 && CurrentGuiState.m_nNumGBufferRenderTargets <= MAX_NUM_GBUFFER_RENDER_TARGETS);

        ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

        // no need to clear DXUT's main RT, because we do a full-screen blit to it later

        float ClearColorGBuffer[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for( int i = 0; i < CurrentGuiState.m_nNumGBufferRenderTargets; i++ )
        {
            pd3dImmediateContext->ClearRenderTargetView( m_pGBufferRTV[i], ClearColorGBuffer );
        }
        pd3dImmediateContext->ClearDepthStencilView( DepthStencilBuffer.m_pDepthStencilView, D3D11_CLEAR_DEPTH, 0.0f, 0 );  // we are using inverted depth, so clear to zero

        bool bDebugDrawingEnabled = ( CurrentGuiState.m_nDebugDrawType == DEBUG_DRAW_RADAR_COLORS ) || ( CurrentGuiState.m_nDebugDrawType == DEBUG_DRAW_GRAYSCALE );
        bool bDoCullingSeparately = CurrentGuiState.m_bDoTiledDeferredWithSeparateCulling;

        // Light culling compute shader
        ID3D11ComputeShader* pLightCullCS = NULL;
        if( bDoCullingSeparately )
        {
            pLightCullCS = CommonUtil.GetLightCullCS();
        }
        else
        {
            pLightCullCS = bDebugDrawingEnabled ? GetDebugDrawNumLightsPerTileCS( CurrentGuiState.m_nDebugDrawType ) : GetLightCullAndShadeCS( CurrentGuiState.m_nNumGBufferRenderTargets );
        }

        // Deferred lighting pixel shader
        ID3D11PixelShader* pDeferredLightingPS = NULL;
        if( bDoCullingSeparately )
        {
            pDeferredLightingPS = m_pDeferredLightingPS[CurrentGuiState.m_nNumGBufferRenderTargets-2];
            if( bDebugDrawingEnabled )
            {
                const int nOffset = (CurrentGuiState.m_nDebugDrawType == DEBUG_DRAW_GRAYSCALE) ? 0 : 1;
                pDeferredLightingPS = m_pDebugDrawNumLightsPerTilePS[nOffset];
            }
        }

        // Depth buffer as shader input
        ID3D11ShaderResourceView* pDepthSRV = DepthStencilBuffer.m_pDepthStencilSRV;

        // Switch off alpha blending
        float BlendFactor[1] = { 0.0f };
        pd3dImmediateContext->OMSetBlendState( m_pBlendStateOpaque, BlendFactor, 0xffffffff );

        // Render objects here...
        {
            ID3D11RenderTargetView* pNULLRTVs[MAX_NUM_GBUFFER_RENDER_TARGETS] = { NULL, NULL, NULL, NULL, NULL };
            ID3D11DepthStencilView* pNULLDSV = NULL;
            ID3D11ShaderResourceView* pNULLSRV = NULL;
            ID3D11ShaderResourceView* pNULLSRVs[MAX_NUM_GBUFFER_RENDER_TARGETS] = { NULL, NULL, NULL, NULL, NULL };
            ID3D11UnorderedAccessView* pNULLUAV = NULL;
            ID3D11SamplerState* pNULLSampler = NULL;

            TIMER_Begin( 0, L"Core algorithm" );

            TIMER_Begin( 0, L"G-Buffer" );
            {
                // Set render targets to GBuffer RTs
                ID3D11RenderTargetView* pRTViews[MAX_NUM_GBUFFER_RENDER_TARGETS] = { NULL, NULL, NULL, NULL, NULL };
                for( int i = 0; i < CurrentGuiState.m_nNumGBufferRenderTargets; i++ )
                {
                    pRTViews[i] = m_pGBufferRTV[i];
                }
                pd3dImmediateContext->OMSetRenderTargets( (unsigned)CurrentGuiState.m_nNumGBufferRenderTargets, pRTViews, DepthStencilBuffer.m_pDepthStencilView );
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_GREATER), 0x00 );  // we are using inverted 32-bit float depth for better precision
                pd3dImmediateContext->IASetInputLayout( m_pLayoutDeferredBuildGBuffer11 );
                pd3dImmediateContext->VSSetShader( m_pSceneDeferredBuildGBufferVS, NULL, 0 );
                pd3dImmediateContext->PSSetShader( m_pSceneDeferredBuildGBufferPS[CurrentGuiState.m_nNumGBufferRenderTargets-2], NULL, 0 );
                pd3dImmediateContext->PSSetSamplers( 0, 1, CommonUtil.GetSamplerStateParam(SAMPLER_STATE_ANISO) );

                // Draw the grid objects (i.e. the "lots of triangles" system)
                for( int i = 0; i < CurrentGuiState.m_nNumGridObjects; i++ )
                {
                    CommonUtil.DrawGrid(i, CurrentGuiState.m_nGridObjectTriangleDensity);
                }

                // Draw the main scene
                Scene.m_pSceneMesh->Render( pd3dImmediateContext, 0, 1 );

                // Draw the alpha test geometry
                pd3dImmediateContext->RSSetState( CommonUtil.GetRasterizerState(RASTERIZER_STATE_DISABLE_CULLING) );
                pd3dImmediateContext->PSSetShader( m_pSceneDeferredBuildGBufferPS[(MAX_NUM_GBUFFER_RENDER_TARGETS-1) + (CurrentGuiState.m_nNumGBufferRenderTargets-2)], NULL, 0 );
                Scene.m_pAlphaMesh->Render( pd3dImmediateContext, 0, 1 );
                pd3dImmediateContext->RSSetState( NULL );
            }
            TIMER_End(); // G-Buffer

            TIMER_Begin( 0, L"Cull and light" );
            {
                pd3dImmediateContext->OMSetRenderTargets( (unsigned)CurrentGuiState.m_nNumGBufferRenderTargets, pNULLRTVs, pNULLDSV );  // null color buffers and depth-stencil
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST), 0x00 );
                pd3dImmediateContext->VSSetShader( NULL, NULL, 0 );  // null vertex shader
                pd3dImmediateContext->PSSetShader( NULL, NULL, 0 );  // null pixel shader
                pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );
            }
            if( !bDoCullingSeparately )
            {
                // Cull lights and do lighting on the GPU, using a single Compute Shader
                pd3dImmediateContext->CSSetShader( pLightCullCS, NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, LightUtil.GetPointLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 1, 1, LightUtil.GetSpotLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 2, 1, &pDepthSRV );
                pd3dImmediateContext->CSSetShaderResources( 4, 1, LightUtil.GetPointLightBufferColorSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 5, 1, LightUtil.GetSpotLightBufferColorSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 6, 1, LightUtil.GetSpotLightBufferSpotParamsSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 7, (unsigned)CurrentGuiState.m_nNumGBufferRenderTargets, &m_pGBufferSRV[0] );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1,  &m_pOffScreenBufferUAV, NULL );

                pd3dImmediateContext->Dispatch(CommonUtil.GetNumTilesX(),CommonUtil.GetNumTilesY(),1);

                pd3dImmediateContext->CSSetShader( NULL, NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 2, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 4, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 5, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 6, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 7, (unsigned)CurrentGuiState.m_nNumGBufferRenderTargets, &pNULLSRVs[0] );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &pNULLUAV, NULL );
            }
            else
            {
                // Do culling and lighting separately
                pd3dImmediateContext->CSSetShader( pLightCullCS, NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, LightUtil.GetPointLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 1, 1, LightUtil.GetSpotLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 2, 1, &pDepthSRV );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1,  CommonUtil.GetLightIndexBufferUAVParam(), NULL );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1,  CommonUtil.GetSpotIndexBufferUAVParam(), NULL );
                pd3dImmediateContext->Dispatch(CommonUtil.GetNumTilesX(),CommonUtil.GetNumTilesY(),1);

                pd3dImmediateContext->CSSetShader( NULL, NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 2, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &pNULLUAV, NULL );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1, &pNULLUAV, NULL );

                // Now that culling is done, do lighting with fullscreen quad
                pd3dImmediateContext->OMSetRenderTargets( 1, &m_pOffScreenBufferRTV, pNULLDSV );
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST), 0x00 );

                // Set the input layout
                pd3dImmediateContext->IASetInputLayout( NULL );

                // Set vertex buffer
                UINT stride = 0;
                UINT offset = 0;
                ID3D11Buffer* pBuffer[1] = { NULL };
                pd3dImmediateContext->IASetVertexBuffers( 0, 1, pBuffer, &stride, &offset );

                // Set primitive topology
                pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

                pd3dImmediateContext->VSSetShader( CommonUtil.GetFullScreenVS(), NULL, 0 );
                pd3dImmediateContext->PSSetShader( pDeferredLightingPS, NULL, 0 );
                pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );
                pd3dImmediateContext->PSSetShaderResources( 0, 1, LightUtil.GetPointLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 1, 1, LightUtil.GetSpotLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 2, 1, &pDepthSRV );
                pd3dImmediateContext->PSSetShaderResources( 4, 1, LightUtil.GetPointLightBufferColorSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 5, 1, LightUtil.GetSpotLightBufferColorSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 6, 1, LightUtil.GetSpotLightBufferSpotParamsSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 7, (unsigned)CurrentGuiState.m_nNumGBufferRenderTargets, &m_pGBufferSRV[0] );

                pd3dImmediateContext->PSSetShaderResources( 12, 1, CommonUtil.GetLightIndexBufferSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 13, 1, CommonUtil.GetSpotIndexBufferSRVParam() );

                // Draw fullscreen quad
                pd3dImmediateContext->Draw(3,0);

                // restore to default
                for( int i = 2; i <= 13; i++ )
                {
                    pd3dImmediateContext->PSSetShaderResources( i, 1, &pNULLSRV );
                }
            }
            TIMER_End(); // Cull and light

            TIMER_End(); // Core algorithm

            TIMER_Begin( 0, L"Blit to main RT" );
            {
                ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
                pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, pNULLDSV );
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST), 0x00 );

                // Set the input layout
                pd3dImmediateContext->IASetInputLayout( NULL );

                // Set vertex buffer
                UINT stride = 0;
                UINT offset = 0;
                ID3D11Buffer* pBuffer[1] = { NULL };
                pd3dImmediateContext->IASetVertexBuffers( 0, 1, pBuffer, &stride, &offset );

                // Set primitive topology
                pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

                pd3dImmediateContext->VSSetShader( CommonUtil.GetFullScreenVS(), NULL, 0 );
                pd3dImmediateContext->PSSetShader( CommonUtil.GetFullScreenPS(), NULL, 0 );
                pd3dImmediateContext->PSSetSamplers( 0, 1, CommonUtil.GetSamplerStateParam(SAMPLER_STATE_LINEAR) );
                pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 2, 1, &m_pOffScreenBufferSRV );

                // Draw fullscreen quad
                pd3dImmediateContext->Draw(3,0);

                // restore to default
                pd3dImmediateContext->PSSetShaderResources( 2, 1, &pNULLSRV );
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_GREATER), 0x00 );  // we are using inverted 32-bit float depth for better precision
            }
            TIMER_End(); // Blit to main RT

            TIMER_Begin( 0, L"Light debug drawing" );
            {
                ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
                pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, DepthStencilBuffer.m_pDepthStencilView );

                // Light debug drawing
                if( CurrentGuiState.m_bLightDrawingEnabled )
                {
                    LightUtil.RenderLights( fElapsedTime, CurrentGuiState.m_uNumPointLights, CurrentGuiState.m_uNumSpotLights, CommonUtil );
                }
            }
            TIMER_End(); // Light debug drawing
        }
    }

    //--------------------------------------------------------------------------------------
    // Add shaders to the shader cache
    //--------------------------------------------------------------------------------------
    void TiledDeferredUtil::AddShadersToCache( AMD::ShaderCache *pShaderCache )
    {
        // Ensure all shaders (and input layouts) are released
        SAFE_RELEASE(m_pSceneDeferredBuildGBufferVS);
        SAFE_RELEASE(m_pLayoutDeferredBuildGBuffer11);

        for( int i = 0; i < NUM_GBUFFER_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pSceneDeferredBuildGBufferPS[i]);
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_COMPUTE_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pLightCullAndShadeCS[i]);
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_COMPUTE_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDebugDrawNumLightsPerTileCS[i]);
        }

        for( int i = 0; i < NUM_DEFERRED_LIGHTING_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDeferredLightingPS[i]);
        }

        for( int i = 0; i < NUM_DEBUG_DRAW_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pDebugDrawNumLightsPerTilePS[i]);
        }

        AMD::ShaderCache::Macro ShaderMacroBuildGBufferPS[2];
        wcscpy_s( ShaderMacroBuildGBufferPS[0].m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"USE_ALPHA_TEST" );
        wcscpy_s( ShaderMacroBuildGBufferPS[1].m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"NUM_GBUFFER_RTS" );

        const D3D11_INPUT_ELEMENT_DESC Layout[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TANGENT",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pSceneDeferredBuildGBufferVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"RenderSceneToGBufferVS",
            L"Shaders\\Source\\Deferred.hlsl", 0, NULL, &m_pLayoutDeferredBuildGBuffer11, Layout, ARRAYSIZE( Layout ) );

        for( int i = 0; i < 2; i++ )
        {
            // USE_ALPHA_TEST false first time through, then true
            ShaderMacroBuildGBufferPS[0].m_iValue = i;

            for( int j = 2; j <= MAX_NUM_GBUFFER_RENDER_TARGETS; j++ )
            {
                // set NUM_GBUFFER_RTS
                ShaderMacroBuildGBufferPS[1].m_iValue = j;

                pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pSceneDeferredBuildGBufferPS[(MAX_NUM_GBUFFER_RENDER_TARGETS-1)*i+j-2], AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"RenderSceneToGBufferPS",
                    L"Shaders\\source\\Deferred.hlsl", 2, ShaderMacroBuildGBufferPS, NULL, NULL, 0 );
            }
        }

        AMD::ShaderCache::Macro ShaderMacroLightCullCS[2];
        wcscpy_s( ShaderMacroLightCullCS[0].m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"NUM_GBUFFER_RTS" );
        wcscpy_s( ShaderMacroLightCullCS[1].m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"LIGHTS_PER_TILE_MODE" );

        // Set LIGHTS_PER_TILE_MODE to 0 (lights per tile visualization disabled)
        ShaderMacroLightCullCS[1].m_iValue = 0;

        for( int i = 2; i <= MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            // set NUM_GBUFFER_RTS
            ShaderMacroLightCullCS[0].m_iValue = i;

            pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pLightCullAndShadeCS[i-2], AMD::ShaderCache::SHADER_TYPE_COMPUTE, L"cs_5_0", L"CullLightsAndDoLightingCS",
                L"Shaders\\source\\TilingDeferred.hlsl", 2, ShaderMacroLightCullCS, NULL, NULL, 0 );
        }

        // Set NUM_GBUFFER_RTS to 2
        ShaderMacroLightCullCS[0].m_iValue = 2;

        for( int i = 0; i < 2; i++ )
        {
            // LIGHTS_PER_TILE_MODE 1 first time through (grayscale), then 2 (radar colors)
            ShaderMacroLightCullCS[1].m_iValue = i+1;

            pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawNumLightsPerTileCS[i], AMD::ShaderCache::SHADER_TYPE_COMPUTE, L"cs_5_0", L"CullLightsAndDoLightingCS",
                L"Shaders\\source\\TilingDeferred.hlsl", 2, ShaderMacroLightCullCS, NULL, NULL, 0 );
        }

        AMD::ShaderCache::Macro ShaderMacroDeferredLightingPS[2];
        wcscpy_s( ShaderMacroDeferredLightingPS[0].m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"NUM_GBUFFER_RTS" );
        wcscpy_s( ShaderMacroDeferredLightingPS[1].m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"LIGHTS_PER_TILE_MODE" );

        // Set LIGHTS_PER_TILE_MODE to 0 (lights per tile visualization disabled)
        ShaderMacroDeferredLightingPS[1].m_iValue = 0;

        for( int i = 2; i <= MAX_NUM_GBUFFER_RENDER_TARGETS; i++ )
        {
            // set NUM_GBUFFER_RTS
            ShaderMacroDeferredLightingPS[0].m_iValue = i;

            pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDeferredLightingPS[i-2], AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DoLightingDeferredPS",
                L"Shaders\\source\\Deferred.hlsl", 2, ShaderMacroDeferredLightingPS, NULL, NULL, 0 );
        }

        // Set NUM_GBUFFER_RTS to 2
        ShaderMacroDeferredLightingPS[0].m_iValue = 2;

        for( int i = 0; i < 2; i++ )
        {
            // LIGHTS_PER_TILE_MODE 1 first time through (grayscale), then 2 (radar colors)
            ShaderMacroDeferredLightingPS[1].m_iValue = i+1;

            pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawNumLightsPerTilePS[i], AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DoLightingDeferredPS",
                L"Shaders\\source\\Deferred.hlsl", 2, ShaderMacroDeferredLightingPS, NULL, NULL, 0 );
        }
    }

    //--------------------------------------------------------------------------------------
    // Return one of the light culling and shading compute shaders, based on GUI settings
    //--------------------------------------------------------------------------------------
    ID3D11ComputeShader * TiledDeferredUtil::GetLightCullAndShadeCS( int nNumGBufferRenderTargets ) const
    {
        return m_pLightCullAndShadeCS[nNumGBufferRenderTargets-2];
    }

    //--------------------------------------------------------------------------------------
    // Return one of the lights-per-tile visualization compute shaders, based on GUI settings
    //--------------------------------------------------------------------------------------
    ID3D11ComputeShader * TiledDeferredUtil::GetDebugDrawNumLightsPerTileCS( int nDebugDrawType ) const
    {
        if ( ( nDebugDrawType != DEBUG_DRAW_RADAR_COLORS ) && ( nDebugDrawType != DEBUG_DRAW_GRAYSCALE ) )
        {
            return NULL;
        }

        const int nIndexDebugDrawType = ( nDebugDrawType == DEBUG_DRAW_RADAR_COLORS ) ? 1 : 0;
        return m_pDebugDrawNumLightsPerTileCS[nIndexDebugDrawType];
    }

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
