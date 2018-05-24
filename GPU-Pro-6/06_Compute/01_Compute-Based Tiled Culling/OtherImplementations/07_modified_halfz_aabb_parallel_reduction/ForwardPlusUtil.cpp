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
// File: ForwardPlusUtil.cpp
//
// Helper functions for the ComputeBasedTiledCulling sample.
//--------------------------------------------------------------------------------------

#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"
#include "..\\DXUT\\Optional\\DXUTCamera.h"

#include "..\\AMD_SDK\\AMD_SDK.h"

#include "ForwardPlusUtil.h"
#include "CommonUtil.h"
#include "LightUtil.h"

#pragma warning( disable : 4100 ) // disable unreference formal parameter warnings for /W4 builds

// there should only be one ForwardPlusUtil object
static int ForwardPlusUtilObjectCounter = 0;

namespace ComputeBasedTiledCulling
{

    //--------------------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------------------
    ForwardPlusUtil::ForwardPlusUtil()
        :m_pScenePositionOnlyVS(NULL)
        ,m_pScenePositionAndTexVS(NULL)
        ,m_pSceneForwardVS(NULL)
        ,m_pSceneAlphaTestOnlyPS(NULL)
        ,m_pLayoutPositionOnly11(NULL)
        ,m_pLayoutPositionAndTex11(NULL)
        ,m_pLayoutForward11(NULL)
        ,m_pBlendStateOpaque(NULL)
        ,m_pBlendStateOpaqueDepthOnly(NULL)
        ,m_pBlendStateAlphaToCoverageDepthOnly(NULL)
        ,m_pBlendStateAlpha(NULL)
    {
        assert( ForwardPlusUtilObjectCounter == 0 );
        ForwardPlusUtilObjectCounter++;

        for( int i = 0; i < NUM_FORWARD_PIXEL_SHADERS; i++ )
        {
            m_pSceneForwardPS[i] = NULL;
        }
    }


    //--------------------------------------------------------------------------------------
    // Destructor
    //--------------------------------------------------------------------------------------
    ForwardPlusUtil::~ForwardPlusUtil()
    {
        SAFE_RELEASE(m_pScenePositionOnlyVS);
        SAFE_RELEASE(m_pScenePositionAndTexVS);
        SAFE_RELEASE(m_pSceneForwardVS);
        SAFE_RELEASE(m_pSceneAlphaTestOnlyPS);
        SAFE_RELEASE(m_pLayoutPositionOnly11);
        SAFE_RELEASE(m_pLayoutPositionAndTex11);
        SAFE_RELEASE(m_pLayoutForward11);

        for( int i = 0; i < NUM_FORWARD_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pSceneForwardPS[i]);
        }

        SAFE_RELEASE(m_pBlendStateOpaque);
        SAFE_RELEASE(m_pBlendStateOpaqueDepthOnly);
        SAFE_RELEASE(m_pBlendStateAlphaToCoverageDepthOnly);
        SAFE_RELEASE(m_pBlendStateAlpha);

        assert( ForwardPlusUtilObjectCounter == 1 );
        ForwardPlusUtilObjectCounter--;
    }

    //--------------------------------------------------------------------------------------
    // Device creation hook function
    //--------------------------------------------------------------------------------------
    HRESULT ForwardPlusUtil::OnCreateDevice( ID3D11Device* pd3dDevice )
    {
        HRESULT hr;

        // Create blend states 
        D3D11_BLEND_DESC BlendStateDesc;
        ZeroMemory( &BlendStateDesc, sizeof( D3D11_BLEND_DESC ) );
        BlendStateDesc.AlphaToCoverageEnable = FALSE;
        BlendStateDesc.IndependentBlendEnable = FALSE;
        BlendStateDesc.RenderTarget[0].BlendEnable = FALSE;
        BlendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE; 
        BlendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ZERO; 
        BlendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE; 
        BlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ZERO; 
        BlendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateOpaque ) );
        BlendStateDesc.RenderTarget[0].RenderTargetWriteMask = 0;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateOpaqueDepthOnly ) );
        BlendStateDesc.AlphaToCoverageEnable = TRUE;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateAlphaToCoverageDepthOnly ) );
        BlendStateDesc.AlphaToCoverageEnable = FALSE;
        BlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
        BlendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
        BlendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        BlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
        BlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        BlendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateAlpha ) );

        return S_OK;
    }


    //--------------------------------------------------------------------------------------
    // Device destruction hook function
    //--------------------------------------------------------------------------------------
    void ForwardPlusUtil::OnDestroyDevice()
    {
        SAFE_RELEASE(m_pScenePositionOnlyVS);
        SAFE_RELEASE(m_pScenePositionAndTexVS);
        SAFE_RELEASE(m_pSceneForwardVS);
        SAFE_RELEASE(m_pSceneAlphaTestOnlyPS);
        SAFE_RELEASE(m_pLayoutPositionOnly11);
        SAFE_RELEASE(m_pLayoutPositionAndTex11);
        SAFE_RELEASE(m_pLayoutForward11);

        for( int i = 0; i < NUM_FORWARD_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pSceneForwardPS[i]);
        }

        SAFE_RELEASE(m_pBlendStateOpaque);
        SAFE_RELEASE(m_pBlendStateOpaqueDepthOnly);
        SAFE_RELEASE(m_pBlendStateAlphaToCoverageDepthOnly);
        SAFE_RELEASE(m_pBlendStateAlpha);
    }


    //--------------------------------------------------------------------------------------
    // Resized swap chain hook function
    //--------------------------------------------------------------------------------------
    HRESULT ForwardPlusUtil::OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc )
    {
        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Releasing swap chain hook function
    //--------------------------------------------------------------------------------------
    void ForwardPlusUtil::OnReleasingSwapChain()
    {
    }

    //--------------------------------------------------------------------------------------
    // Render hook function, to draw the scene using Forward+
    //--------------------------------------------------------------------------------------
    void ForwardPlusUtil::OnRender( float fElapsedTime, const GuiState& CurrentGuiState, const DepthStencilBuffer& DepthStencilBuffer, const Scene& Scene, const CommonUtil& CommonUtil, const LightUtil& LightUtil )
    {
        ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

        ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();

        // Clear the backbuffer and depth stencil
        float ClearColor[4] = { 0.0013f, 0.0015f, 0.0050f, 0.0f };
        pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );
        pd3dImmediateContext->ClearDepthStencilView( DepthStencilBuffer.m_pDepthStencilView, D3D11_CLEAR_DEPTH, 0.0f, 0 );  // we are using inverted depth, so clear to zero

        // Default pixel shader
        ID3D11PixelShader* pScenePS = GetScenePS(false);
        ID3D11PixelShader* pScenePSAlphaTest = GetScenePS(true);

        // See if we need to use one of the debug drawing shaders instead
        bool bDebugDrawingEnabled = ( CurrentGuiState.m_nDebugDrawType == DEBUG_DRAW_RADAR_COLORS ) || ( CurrentGuiState.m_nDebugDrawType == DEBUG_DRAW_GRAYSCALE );
        if( bDebugDrawingEnabled )
        {
            pScenePS = CommonUtil.GetDebugDrawNumLightsPerTilePS(CurrentGuiState.m_nDebugDrawType);
            pScenePSAlphaTest = CommonUtil.GetDebugDrawNumLightsPerTilePS(CurrentGuiState.m_nDebugDrawType);
        }

        // Depth bounds compute shader
        ID3D11ComputeShader* pDepthBoundsCS = CommonUtil.GetDepthBoundsCS();
        ID3D11ShaderResourceView* pDepthSRV = DepthStencilBuffer.m_pDepthStencilSRV;

        // Switch off alpha blending
        float BlendFactor[1] = { 0.0f };
        pd3dImmediateContext->OMSetBlendState( m_pBlendStateOpaque, BlendFactor, 0xffffffff );

        // Render objects here...
        {
            ID3D11RenderTargetView* pNULLRTV = NULL;
            ID3D11DepthStencilView* pNULLDSV = NULL;
            ID3D11ShaderResourceView* pNULLSRV = NULL;
            ID3D11UnorderedAccessView* pNULLUAV = NULL;
            ID3D11SamplerState* pNULLSampler = NULL;

            TIMER_Begin( 0, L"Core algorithm" );

            TIMER_Begin( 0, L"Depth pre-pass" );
            {
                // Depth pre-pass (to eliminate pixel overdraw during forward rendering)
                pd3dImmediateContext->OMSetRenderTargets( 1, &pNULLRTV, DepthStencilBuffer.m_pDepthStencilView );  // null color buffer
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_GREATER), 0x00 );  // we are using inverted 32-bit float depth for better precision
                pd3dImmediateContext->IASetInputLayout( m_pLayoutPositionOnly11 );
                pd3dImmediateContext->VSSetShader( m_pScenePositionOnlyVS, NULL, 0 );
                pd3dImmediateContext->PSSetShader( NULL, NULL, 0 );  // null pixel shader
                pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );

                // Draw the grid objects (i.e. the "lots of triangles" system)
                for( int i = 0; i < CurrentGuiState.m_nNumGridObjects; i++ )
                {
                    CommonUtil.DrawGrid(i, CurrentGuiState.m_nGridObjectTriangleDensity, false);
                }

                // Draw the main scene
                Scene.m_pSceneMesh->Render( pd3dImmediateContext );

                // Draw the alpha test geometry
                ID3D11BlendState* pBlendStateForAlphaTest = m_pBlendStateOpaqueDepthOnly;
                pd3dImmediateContext->RSSetState( CommonUtil.GetRasterizerState(RASTERIZER_STATE_DISABLE_CULLING) );
                pd3dImmediateContext->OMSetBlendState( pBlendStateForAlphaTest, BlendFactor, 0xffffffff );
                pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, DepthStencilBuffer.m_pDepthStencilView );  // bind color buffer to prevent D3D warning
                pd3dImmediateContext->IASetInputLayout( m_pLayoutPositionAndTex11 );
                pd3dImmediateContext->VSSetShader( m_pScenePositionAndTexVS, NULL, 0 );
                pd3dImmediateContext->PSSetShader( m_pSceneAlphaTestOnlyPS, NULL, 0 );
                pd3dImmediateContext->PSSetSamplers( 0, 1, CommonUtil.GetSamplerStateParam(SAMPLER_STATE_ANISO) );
                Scene.m_pAlphaMesh->Render( pd3dImmediateContext, 0 );

                // Restore to default
                pd3dImmediateContext->RSSetState( NULL );
                pd3dImmediateContext->OMSetBlendState( m_pBlendStateOpaque, BlendFactor, 0xffffffff );
            }
            TIMER_End(); // Depth pre-pass

            TIMER_Begin( 0, L"Light culling" );
            {
                pd3dImmediateContext->OMSetRenderTargets( 1, &pNULLRTV, pNULLDSV );  // null color buffer and depth-stencil
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST), 0x00 );
                pd3dImmediateContext->VSSetShader( NULL, NULL, 0 );  // null vertex shader
                pd3dImmediateContext->PSSetShader( NULL, NULL, 0 );  // null pixel shader
                pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );

                // Calculate per-tile depth bounds on the GPU, using a Compute Shader
                pd3dImmediateContext->CSSetShader( pDepthBoundsCS, NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, &pDepthSRV );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1,  CommonUtil.GetDepthBoundsUAVParam(), NULL );
                pd3dImmediateContext->Dispatch(CommonUtil.GetNumTilesX(),CommonUtil.GetNumTilesY(),1);
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &pNULLUAV, NULL );

                // Cull lights on the GPU, using a Compute Shader
                pd3dImmediateContext->CSSetShader( CommonUtil.GetLightCullCS(), NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, LightUtil.GetPointLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 1, 1, LightUtil.GetSpotLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->CSSetShaderResources( 2, 1, CommonUtil.GetDepthBoundsSRVParam() );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, CommonUtil.GetLightIndexBufferUAVParam(), NULL );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1,  CommonUtil.GetSpotIndexBufferUAVParam(), NULL );
                pd3dImmediateContext->Dispatch(CommonUtil.GetNumTilesX(),CommonUtil.GetNumTilesY(),1);

                pd3dImmediateContext->CSSetShader( NULL, NULL, 0 );
                pd3dImmediateContext->CSSetShaderResources( 0, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 1, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetShaderResources( 2, 1, &pNULLSRV );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 0, 1, &pNULLUAV, NULL );
                pd3dImmediateContext->CSSetUnorderedAccessViews( 1, 1, &pNULLUAV, NULL );
            }
            TIMER_End(); // Light culling

            TIMER_Begin( 0, L"Forward rendering" );
            {
                // Forward rendering
                pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, DepthStencilBuffer.m_pDepthStencilView );
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_EQUAL_AND_DISABLE_DEPTH_WRITE), 0x00 );
                pd3dImmediateContext->IASetInputLayout( m_pLayoutForward11 );
                pd3dImmediateContext->VSSetShader( m_pSceneForwardVS, NULL, 0 );
                pd3dImmediateContext->PSSetShader( pScenePS, NULL, 0 );
                pd3dImmediateContext->PSSetSamplers( 0, 1, CommonUtil.GetSamplerStateParam(SAMPLER_STATE_ANISO) );
                pd3dImmediateContext->PSSetShaderResources( 2, 1, LightUtil.GetPointLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 3, 1, LightUtil.GetPointLightBufferColorSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 4, 1, CommonUtil.GetLightIndexBufferSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 5, 1, LightUtil.GetSpotLightBufferCenterAndRadiusSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 6, 1, LightUtil.GetSpotLightBufferColorSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 7, 1, LightUtil.GetSpotLightBufferSpotParamsSRVParam() );
                pd3dImmediateContext->PSSetShaderResources( 8, 1, CommonUtil.GetSpotIndexBufferSRVParam() );

                // Draw the grid objects (i.e. the "lots of triangles" system)
                for( int i = 0; i < CurrentGuiState.m_nNumGridObjects; i++ )
                {
                    // uncomment these RSSetState calls to see the grid objects in wireframe (to see the triangle density)
                    //pd3dImmediateContext->RSSetState( CommonUtil.GetRasterizerState(RASTERIZER_STATE_WIREFRAME) );
                    CommonUtil.DrawGrid(i, CurrentGuiState.m_nGridObjectTriangleDensity);
                    //pd3dImmediateContext->RSSetState( NULL );
                }

                // Draw the main scene
                Scene.m_pSceneMesh->Render( pd3dImmediateContext, 0, 1 );

                // Draw the alpha test geometry
                pd3dImmediateContext->RSSetState( CommonUtil.GetRasterizerState(RASTERIZER_STATE_DISABLE_CULLING) );
                pd3dImmediateContext->PSSetShader( pScenePSAlphaTest, NULL, 0 );
                Scene.m_pAlphaMesh->Render( pd3dImmediateContext, 0, 1 );
                pd3dImmediateContext->RSSetState( NULL );

                // restore to default
                pd3dImmediateContext->PSSetShaderResources( 2, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 3, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 4, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 5, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 6, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 7, 1, &pNULLSRV );
                pd3dImmediateContext->PSSetShaderResources( 8, 1, &pNULLSRV );
                pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_GREATER), 0x00 );  // we are using inverted 32-bit float depth for better precision
            }
            TIMER_End(); // Forward rendering

            TIMER_End(); // Core algorithm

            TIMER_Begin( 0, L"Light debug drawing" );
            {
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
    void ForwardPlusUtil::AddShadersToCache( AMD::ShaderCache *pShaderCache )
    {
        // Ensure all shaders (and input layouts) are released
        SAFE_RELEASE(m_pScenePositionOnlyVS);
        SAFE_RELEASE(m_pScenePositionAndTexVS);
        SAFE_RELEASE(m_pSceneForwardVS);
        SAFE_RELEASE(m_pSceneAlphaTestOnlyPS);
        SAFE_RELEASE(m_pLayoutPositionOnly11);
        SAFE_RELEASE(m_pLayoutPositionAndTex11);
        SAFE_RELEASE(m_pLayoutForward11);

        for( int i = 0; i < NUM_FORWARD_PIXEL_SHADERS; i++ )
        {
            SAFE_RELEASE(m_pSceneForwardPS[i]);
        }

        AMD::ShaderCache::Macro ShaderMacroSceneForwardPS;
        wcscpy_s( ShaderMacroSceneForwardPS.m_wsName, AMD::ShaderCache::m_uMACRO_MAX_LENGTH, L"USE_ALPHA_TEST" );

        const D3D11_INPUT_ELEMENT_DESC Layout[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TANGENT",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 32, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pScenePositionOnlyVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"RenderScenePositionOnlyVS",
            L"Shaders\\Source\\Forward.hlsl", 0, NULL, &m_pLayoutPositionOnly11, Layout, ARRAYSIZE( Layout ) );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pScenePositionAndTexVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"RenderScenePositionAndTexVS",
            L"Shaders\\Source\\Forward.hlsl", 0, NULL, &m_pLayoutPositionAndTex11, Layout, ARRAYSIZE( Layout ) );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pSceneForwardVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"RenderSceneForwardVS",
            L"Shaders\\Source\\Forward.hlsl", 0, NULL, &m_pLayoutForward11, Layout, ARRAYSIZE( Layout ) );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pSceneAlphaTestOnlyPS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"RenderSceneAlphaTestOnlyPS",
            L"Shaders\\source\\Forward.hlsl", 0, NULL, NULL, NULL, 0 );

        // USE_ALPHA_TEST false
        ShaderMacroSceneForwardPS.m_iValue = 0;
        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pSceneForwardPS[0], AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"RenderSceneForwardPS",
            L"Shaders\\source\\Forward.hlsl", 1, &ShaderMacroSceneForwardPS, NULL, NULL, 0 );

        // USE_ALPHA_TEST true
        ShaderMacroSceneForwardPS.m_iValue = 1;
        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pSceneForwardPS[1], AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"RenderSceneForwardPS",
            L"Shaders\\source\\Forward.hlsl", 1, &ShaderMacroSceneForwardPS, NULL, NULL, 0 );
    }

    //--------------------------------------------------------------------------------------
    // Return one of the forward pixel shaders, based on settings for alpha test
    //--------------------------------------------------------------------------------------
    ID3D11PixelShader * ForwardPlusUtil::GetScenePS( bool bAlphaTestEnabled ) const
    {
        const int nIndexAlphaTest = bAlphaTestEnabled ? 1 : 0;
        return m_pSceneForwardPS[nIndexAlphaTest];
    }

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
