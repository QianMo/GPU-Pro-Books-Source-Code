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
// File: LightUtil.cpp
//
// Helper functions for the ComputeBasedTiledCulling sample.
//--------------------------------------------------------------------------------------

#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"

#include "..\\AMD_SDK\\AMD_SDK.h"

#include "LightUtil.h"
#include "CommonUtil.h"

#pragma warning( disable : 4100 ) // disable unreference formal parameter warnings for /W4 builds

using namespace DirectX;

// Packs integer RGB color values for into ABGR format (i.e. DXGI_FORMAT_R8G8B8A8_UNORM).
#define COLOR( r, g, b ) (DWORD)((255 << 24) | ((b) << 16) | ((g) << 8) | (r))


struct LightUtilSpriteVertex
{
    XMFLOAT3 v3Pos;
    XMFLOAT2 v2TexCoord;
};

// static array for sprite quad vertex data
static LightUtilSpriteVertex         g_QuadForLightsVertexData[6];

struct LightArrayData
{
    XMFLOAT4 v4PositionAndRadius;
    XMFLOAT4 v4Color;
};

// static array for the point light data
static LightArrayData       g_PointLightDataArray[ComputeBasedTiledCulling::MAX_NUM_LIGHTS];

struct LightUtilConeVertex
{
    XMFLOAT3 v3Pos;
    XMFLOAT3 v3Norm;
    XMFLOAT2 v2TexCoord;
};

// static arrays for cone vertex and index data (for visualizing spot lights)
static const int            g_nConeNumTris = 90;
static const int            g_nConeNumVertices = 2*g_nConeNumTris;
static const int            g_nConeNumIndices = 3*g_nConeNumTris;
static LightUtilConeVertex  g_ConeForSpotLightsVertexData[g_nConeNumVertices];
static unsigned short       g_ConeForSpotLightsIndexData[g_nConeNumIndices];

// these are half-precision (i.e. 16-bit) float values, 
// stored as unsigned shorts
struct LightUtilSpotParams
{
    unsigned short fLightDirX;
    unsigned short fLightDirY;
    unsigned short fCosineOfConeAngleAndLightDirZSign;
    unsigned short fFalloffRadius;
};

// static arrays for the spot light data
static XMFLOAT4             g_SpotLightDataArrayCenterAndRadius[ComputeBasedTiledCulling::MAX_NUM_LIGHTS];
static DWORD                g_SpotLightDataArrayColor[ComputeBasedTiledCulling::MAX_NUM_LIGHTS];
static LightUtilSpotParams  g_SpotLightDataArraySpotParams[ComputeBasedTiledCulling::MAX_NUM_LIGHTS];

// rotation matrices used when visualizing the spot lights
static XMMATRIX             g_SpotLightDataArraySpotMatrices[ComputeBasedTiledCulling::MAX_NUM_LIGHTS];

// miscellaneous constants
static const float TWO_PI = 6.28318530718f;

// there should only be one LightUtil object
static int LightUtilObjectCounter = 0;

static float GetRandFloat( float fRangeMin, float fRangeMax )
{
    // Generate random numbers in the half-closed interval
    // [rangeMin, rangeMax). In other words,
    // rangeMin <= random number < rangeMax
    return  (float)rand() / (RAND_MAX + 1) * (fRangeMax - fRangeMin) + fRangeMin;
}

static DWORD GetRandColor()
{
    static unsigned uCounter = 0;
    uCounter++;

    XMFLOAT4 Color;
    if( uCounter%2 == 0 )
    {
        // since green contributes the most to perceived brightness, 
        // cap it's min value to avoid overly dim lights
        Color = XMFLOAT4(GetRandFloat(0.0f,1.0f),GetRandFloat(0.27f,1.0f),GetRandFloat(0.0f,1.0f),1.0f);
    }
    else
    {
        // else ensure the red component has a large value, again 
        // to avoid overly dim lights
        Color = XMFLOAT4(GetRandFloat(0.9f,1.0f),GetRandFloat(0.0f,1.0f),GetRandFloat(0.0f,1.0f),1.0f);
    }

    DWORD dwR = (DWORD)(Color.x * 255.0f + 0.5f);
    DWORD dwG = (DWORD)(Color.y * 255.0f + 0.5f);
    DWORD dwB = (DWORD)(Color.z * 255.0f + 0.5f);

    return COLOR(dwR, dwG, dwB);
}

static XMFLOAT4 GetRandColorFloat4()
{
    static unsigned uCounter = 0;
    uCounter++;

    XMFLOAT4 Color;
    if (uCounter % 2 == 0)
    {
        // since green contributes the most to perceived brightness, 
        // cap it's min value to avoid overly dim lights
        Color = XMFLOAT4(GetRandFloat(0.0f, 1.0f), GetRandFloat(0.27f, 1.0f), GetRandFloat(0.0f, 1.0f), 1.0f);
    }
    else
    {
        // else ensure the red component has a large value, again 
        // to avoid overly dim lights
        Color = XMFLOAT4(GetRandFloat(0.9f, 1.0f), GetRandFloat(0.0f, 1.0f), GetRandFloat(0.0f, 1.0f), 1.0f);
    }

    return Color;
}

static XMFLOAT3 GetRandLightDirection()
{
    static unsigned uCounter = 0;
    uCounter++;

    XMFLOAT3 vLightDir;
    vLightDir.x = GetRandFloat(-1.0f,1.0f);
    vLightDir.y = GetRandFloat( 0.1f,1.0f);
    vLightDir.z = GetRandFloat(-1.0f,1.0f);

    if( uCounter%2 == 0 )
    {
        vLightDir.y = -vLightDir.y;
    }

    XMFLOAT3 vResult;
    XMVECTOR NormalizedLightDir = XMVector3Normalize( XMLoadFloat3( &vLightDir) );
    XMStoreFloat3( &vResult, NormalizedLightDir );

    return vResult;
}

static LightUtilSpotParams PackSpotParams(const XMFLOAT3& vLightDir, float fCosineOfConeAngle, float fFalloffRadius)
{
    assert( fCosineOfConeAngle > 0.0f );
    assert( fFalloffRadius > 0.0f );

    LightUtilSpotParams PackedParams;
    PackedParams.fLightDirX = AMD::ConvertF32ToF16( vLightDir.x );
    PackedParams.fLightDirY = AMD::ConvertF32ToF16( vLightDir.y );
    PackedParams.fCosineOfConeAngleAndLightDirZSign = AMD::ConvertF32ToF16( fCosineOfConeAngle );
    PackedParams.fFalloffRadius = AMD::ConvertF32ToF16( fFalloffRadius );

    // put the sign bit for light dir z in the sign bit for the cone angle
    // (we can do this because we know the cone angle is always positive)
    if( vLightDir.z < 0.0f )
    {
        PackedParams.fCosineOfConeAngleAndLightDirZSign |= 0x8000;
    }
    else
    {
        PackedParams.fCosineOfConeAngleAndLightDirZSign &= 0x7FFF;
    }

    return PackedParams;
}

namespace ComputeBasedTiledCulling
{

    //--------------------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------------------
    LightUtil::LightUtil()
        :m_pPointLightBuffer(NULL)
        ,m_pPointLightBufferSRV(NULL)
        ,m_pSpotLightBufferCenterAndRadius(NULL)
        ,m_pSpotLightBufferCenterAndRadiusSRV(NULL)
        ,m_pSpotLightBufferColor(NULL)
        ,m_pSpotLightBufferColorSRV(NULL)
        ,m_pSpotLightBufferSpotParams(NULL)
        ,m_pSpotLightBufferSpotParamsSRV(NULL)
        ,m_pSpotLightBufferSpotMatrices(NULL)
        ,m_pSpotLightBufferSpotMatricesSRV(NULL)
        ,m_pQuadForLightsVB(NULL)
        ,m_pConeForSpotLightsVB(NULL)
        ,m_pConeForSpotLightsIB(NULL)
        ,m_pDebugDrawPointLightsVS(NULL)
        ,m_pDebugDrawPointLightsPS(NULL)
        ,m_pDebugDrawPointLightsLayout11(NULL)
        ,m_pDebugDrawSpotLightsVS(NULL)
        ,m_pDebugDrawSpotLightsPS(NULL)
        ,m_pDebugDrawSpotLightsLayout11(NULL)
        ,m_pBlendStateAdditive(NULL)
    {
        assert( LightUtilObjectCounter == 0 );
        LightUtilObjectCounter++;
    }


    //--------------------------------------------------------------------------------------
    // Destructor
    //--------------------------------------------------------------------------------------
    LightUtil::~LightUtil()
    {
        SAFE_RELEASE(m_pPointLightBuffer);
        SAFE_RELEASE(m_pPointLightBufferSRV);
        SAFE_RELEASE(m_pSpotLightBufferCenterAndRadius);
        SAFE_RELEASE(m_pSpotLightBufferCenterAndRadiusSRV);
        SAFE_RELEASE(m_pSpotLightBufferColor);
        SAFE_RELEASE(m_pSpotLightBufferColorSRV);
        SAFE_RELEASE(m_pSpotLightBufferSpotParams);
        SAFE_RELEASE(m_pSpotLightBufferSpotParamsSRV);
        SAFE_RELEASE(m_pSpotLightBufferSpotMatrices);
        SAFE_RELEASE(m_pSpotLightBufferSpotMatricesSRV);
        SAFE_RELEASE(m_pQuadForLightsVB);
        SAFE_RELEASE(m_pConeForSpotLightsVB);
        SAFE_RELEASE(m_pConeForSpotLightsIB);
        SAFE_RELEASE(m_pDebugDrawPointLightsVS);
        SAFE_RELEASE(m_pDebugDrawPointLightsPS);
        SAFE_RELEASE(m_pDebugDrawPointLightsLayout11);
        SAFE_RELEASE(m_pDebugDrawSpotLightsVS);
        SAFE_RELEASE(m_pDebugDrawSpotLightsPS);
        SAFE_RELEASE(m_pDebugDrawSpotLightsLayout11);
        SAFE_RELEASE(m_pBlendStateAdditive);

        assert( LightUtilObjectCounter == 1 );
        LightUtilObjectCounter--;
    }

    //--------------------------------------------------------------------------------------
    // Device creation hook function
    //--------------------------------------------------------------------------------------
    HRESULT LightUtil::OnCreateDevice( ID3D11Device* pd3dDevice )
    {
        HRESULT hr;

        D3D11_SUBRESOURCE_DATA InitData;

        // Create the point light buffer
        D3D11_BUFFER_DESC LightBufferDesc;
        ZeroMemory( &LightBufferDesc, sizeof(LightBufferDesc) );
        LightBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
        LightBufferDesc.ByteWidth = sizeof( g_PointLightDataArray );
        LightBufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        LightBufferDesc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        LightBufferDesc.StructureByteStride = sizeof(LightArrayData);
        InitData.pSysMem = g_PointLightDataArray;
        V_RETURN( pd3dDevice->CreateBuffer( &LightBufferDesc, &InitData, &m_pPointLightBuffer ) );
        DXUT_SetDebugName( m_pPointLightBuffer, "PointLightBuffer" );

        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory( &SRVDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );
        SRVDesc.Format = DXGI_FORMAT_UNKNOWN;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.ElementOffset = 0;
        SRVDesc.Buffer.ElementWidth = MAX_NUM_LIGHTS;
        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pPointLightBuffer, &SRVDesc, &m_pPointLightBufferSRV ) );

        // Create the spot light buffer (center and radius)
        ZeroMemory( &LightBufferDesc, sizeof(LightBufferDesc) );
        LightBufferDesc.Usage = D3D11_USAGE_IMMUTABLE;
        LightBufferDesc.ByteWidth = sizeof( g_SpotLightDataArrayCenterAndRadius );
        LightBufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        InitData.pSysMem = g_SpotLightDataArrayCenterAndRadius;
        V_RETURN( pd3dDevice->CreateBuffer( &LightBufferDesc, &InitData, &m_pSpotLightBufferCenterAndRadius ) );
        DXUT_SetDebugName( m_pSpotLightBufferCenterAndRadius, "SpotLightBufferCenterAndRadius" );

        ZeroMemory( &SRVDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );
        SRVDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.ElementOffset = 0;
        SRVDesc.Buffer.ElementWidth = MAX_NUM_LIGHTS;
        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pSpotLightBufferCenterAndRadius, &SRVDesc, &m_pSpotLightBufferCenterAndRadiusSRV ) );

        // Create the spot light buffer (color)
        LightBufferDesc.ByteWidth = sizeof( g_SpotLightDataArrayColor );
        InitData.pSysMem = g_SpotLightDataArrayColor;
        V_RETURN( pd3dDevice->CreateBuffer( &LightBufferDesc, &InitData, &m_pSpotLightBufferColor ) );
        DXUT_SetDebugName( m_pSpotLightBufferColor, "SpotLightBufferColor" );

        SRVDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pSpotLightBufferColor, &SRVDesc, &m_pSpotLightBufferColorSRV ) );

        // Create the spot light buffer (spot light parameters)
        LightBufferDesc.ByteWidth = sizeof( g_SpotLightDataArraySpotParams );
        InitData.pSysMem = g_SpotLightDataArraySpotParams;
        V_RETURN( pd3dDevice->CreateBuffer( &LightBufferDesc, &InitData, &m_pSpotLightBufferSpotParams ) );
        DXUT_SetDebugName( m_pSpotLightBufferSpotParams, "SpotLightBufferSpotParams" );

        SRVDesc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pSpotLightBufferSpotParams, &SRVDesc, &m_pSpotLightBufferSpotParamsSRV ) );

        // Create the light buffer (spot light matrices, only used for debug drawing the spot lights)
        LightBufferDesc.ByteWidth = sizeof( g_SpotLightDataArraySpotMatrices );
        InitData.pSysMem = g_SpotLightDataArraySpotMatrices;
        V_RETURN( pd3dDevice->CreateBuffer( &LightBufferDesc, &InitData, &m_pSpotLightBufferSpotMatrices ) );
        DXUT_SetDebugName( m_pSpotLightBufferSpotMatrices, "SpotLightBufferSpotMatrices" );

        SRVDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        SRVDesc.Buffer.ElementWidth = 4*MAX_NUM_LIGHTS;
        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pSpotLightBufferSpotMatrices, &SRVDesc, &m_pSpotLightBufferSpotMatricesSRV ) );

        // Create the vertex buffer for the sprites (a single quad)
        D3D11_BUFFER_DESC VBDesc;
        ZeroMemory( &VBDesc, sizeof(VBDesc) );
        VBDesc.Usage = D3D11_USAGE_IMMUTABLE;
        VBDesc.ByteWidth = sizeof( g_QuadForLightsVertexData );
        VBDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        InitData.pSysMem = g_QuadForLightsVertexData;
        V_RETURN( pd3dDevice->CreateBuffer( &VBDesc, &InitData, &m_pQuadForLightsVB ) );
        DXUT_SetDebugName( m_pQuadForLightsVB, "QuadForLightsVB" );

        // Create the vertex buffer for the cone
        VBDesc.ByteWidth = sizeof( g_ConeForSpotLightsVertexData );
        InitData.pSysMem = g_ConeForSpotLightsVertexData;
        V_RETURN( pd3dDevice->CreateBuffer( &VBDesc, &InitData, &m_pConeForSpotLightsVB ) );
        DXUT_SetDebugName( m_pConeForSpotLightsVB, "ConeForSpotLightsVB" );

        // Create the index buffer for the cone
        D3D11_BUFFER_DESC IBDesc;
        ZeroMemory( &IBDesc, sizeof(IBDesc) );
        IBDesc.Usage = D3D11_USAGE_IMMUTABLE;
        IBDesc.ByteWidth = sizeof( g_ConeForSpotLightsIndexData );
        IBDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        InitData.pSysMem = g_ConeForSpotLightsIndexData;
        V_RETURN( pd3dDevice->CreateBuffer( &IBDesc, &InitData, &m_pConeForSpotLightsIB ) );
        DXUT_SetDebugName( m_pConeForSpotLightsIB, "ConeForSpotLightsIB" );

        // Create blend states 
        D3D11_BLEND_DESC BlendStateDesc;
        ZeroMemory( &BlendStateDesc, sizeof( D3D11_BLEND_DESC ) );
        BlendStateDesc.AlphaToCoverageEnable = FALSE;
        BlendStateDesc.IndependentBlendEnable = FALSE;
        BlendStateDesc.RenderTarget[0].BlendEnable = TRUE;
        BlendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE; 
        BlendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ONE; 
        BlendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ZERO; 
        BlendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ONE; 
        BlendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        BlendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        V_RETURN( pd3dDevice->CreateBlendState( &BlendStateDesc, &m_pBlendStateAdditive ) );

        return hr;
    }


    //--------------------------------------------------------------------------------------
    // Device destruction hook function
    //--------------------------------------------------------------------------------------
    void LightUtil::OnDestroyDevice()
    {
        SAFE_RELEASE( m_pPointLightBuffer );
        SAFE_RELEASE( m_pPointLightBufferSRV );

        SAFE_RELEASE( m_pSpotLightBufferCenterAndRadius );
        SAFE_RELEASE( m_pSpotLightBufferCenterAndRadiusSRV );
        SAFE_RELEASE( m_pSpotLightBufferColor );
        SAFE_RELEASE( m_pSpotLightBufferColorSRV );
        SAFE_RELEASE( m_pSpotLightBufferSpotParams );
        SAFE_RELEASE( m_pSpotLightBufferSpotParamsSRV );

        SAFE_RELEASE( m_pSpotLightBufferSpotMatrices );
        SAFE_RELEASE( m_pSpotLightBufferSpotMatricesSRV );

        SAFE_RELEASE( m_pQuadForLightsVB );

        SAFE_RELEASE( m_pConeForSpotLightsVB );
        SAFE_RELEASE( m_pConeForSpotLightsIB );

        SAFE_RELEASE( m_pDebugDrawPointLightsVS );
        SAFE_RELEASE( m_pDebugDrawPointLightsPS );
        SAFE_RELEASE( m_pDebugDrawPointLightsLayout11 );

        SAFE_RELEASE( m_pDebugDrawSpotLightsVS );
        SAFE_RELEASE( m_pDebugDrawSpotLightsPS );
        SAFE_RELEASE( m_pDebugDrawSpotLightsLayout11 );

        SAFE_RELEASE( m_pBlendStateAdditive );
    }


    //--------------------------------------------------------------------------------------
    // Resized swap chain hook function
    //--------------------------------------------------------------------------------------
    HRESULT LightUtil::OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc )
    {
        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Releasing swap chain hook function
    //--------------------------------------------------------------------------------------
    void LightUtil::OnReleasingSwapChain()
    {
    }

    //--------------------------------------------------------------------------------------
    // Render hook function, to draw the lights (as instanced quads)
    //--------------------------------------------------------------------------------------
    void LightUtil::RenderLights( float fElapsedTime, unsigned uNumPointLights, unsigned uNumSpotLights, const CommonUtil& CommonUtil ) const
    {
        ID3D11ShaderResourceView* pNULLSRV = NULL;
        ID3D11SamplerState* pNULLSampler = NULL;

        ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

        // save blend state (for later restore)
        ID3D11BlendState* pBlendStateStored11 = NULL;
        FLOAT afBlendFactorStored11[4];
        UINT uSampleMaskStored11;
        pd3dImmediateContext->OMGetBlendState( &pBlendStateStored11, afBlendFactorStored11, &uSampleMaskStored11 );
        FLOAT BlendFactor[4] = { 0,0,0,0 };

        // save depth state (for later restore)
        ID3D11DepthStencilState* pDepthStencilStateStored11 = NULL;
        UINT uStencilRefStored11;
        pd3dImmediateContext->OMGetDepthStencilState( &pDepthStencilStateStored11, &uStencilRefStored11 );

        // point lights
        if( uNumPointLights > 0 )
        {
            // additive blending, enable depth test but don't write depth, disable culling
            pd3dImmediateContext->OMSetBlendState( m_pBlendStateAdditive, BlendFactor, 0xFFFFFFFF );
            pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_WRITE), 0x00 );
            pd3dImmediateContext->RSSetState( CommonUtil.GetRasterizerState(RASTERIZER_STATE_DISABLE_CULLING) );

            // Set the input layout
            pd3dImmediateContext->IASetInputLayout( m_pDebugDrawPointLightsLayout11 );

            // Set vertex buffer
            UINT uStride = sizeof( LightUtilSpriteVertex );
            UINT uOffset = 0;
            pd3dImmediateContext->IASetVertexBuffers( 0, 1, &m_pQuadForLightsVB, &uStride, &uOffset );

            // Set primitive topology
            pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

            pd3dImmediateContext->VSSetShader( m_pDebugDrawPointLightsVS, NULL, 0 );
            pd3dImmediateContext->VSSetShaderResources( 2, 1, GetPointLightBufferSRVParam() );
            pd3dImmediateContext->PSSetShader( m_pDebugDrawPointLightsPS, NULL, 0 );
            pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
            pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
            pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );

            pd3dImmediateContext->DrawInstanced(6,uNumPointLights,0,0);

            // restore to default
            pd3dImmediateContext->VSSetShaderResources( 2, 1, &pNULLSRV );
        }

        // spot lights
        if( uNumSpotLights > 0 )
        {
            // render spot lights as ordinary opaque geometry
            pd3dImmediateContext->OMSetBlendState( m_pBlendStateAdditive, BlendFactor, 0xFFFFFFFF );
            pd3dImmediateContext->OMSetDepthStencilState( CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_GREATER), 0x00 );
            pd3dImmediateContext->RSSetState( CommonUtil.GetRasterizerState(RASTERIZER_STATE_DISABLE_CULLING) );

            // Set the input layout
            pd3dImmediateContext->IASetInputLayout( m_pDebugDrawSpotLightsLayout11 );

            // Set vertex buffer
            UINT uStride = sizeof( LightUtilConeVertex );
            UINT uOffset = 0;
            pd3dImmediateContext->IASetVertexBuffers( 0, 1, &m_pConeForSpotLightsVB, &uStride, &uOffset );
            pd3dImmediateContext->IASetIndexBuffer(m_pConeForSpotLightsIB, DXGI_FORMAT_R16_UINT, 0 );

            // Set primitive topology
            pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

            pd3dImmediateContext->VSSetShader( m_pDebugDrawSpotLightsVS, NULL, 0 );
            pd3dImmediateContext->VSSetShaderResources( 5, 1, GetSpotLightBufferCenterAndRadiusSRVParam() );
            pd3dImmediateContext->VSSetShaderResources( 6, 1, GetSpotLightBufferColorSRVParam() );
            pd3dImmediateContext->VSSetShaderResources( 7, 1, GetSpotLightBufferSpotParamsSRVParam() );
            pd3dImmediateContext->VSSetShaderResources( 9, 1, GetSpotLightBufferSpotMatricesSRVParam() );
            pd3dImmediateContext->PSSetShader( m_pDebugDrawSpotLightsPS, NULL, 0 );
            pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
            pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
            pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );

            pd3dImmediateContext->DrawIndexedInstanced(g_nConeNumIndices,uNumSpotLights,0,0,0);

            // restore to default
            pd3dImmediateContext->VSSetShaderResources( 5, 1, &pNULLSRV );
            pd3dImmediateContext->VSSetShaderResources( 6, 1, &pNULLSRV );
            pd3dImmediateContext->VSSetShaderResources( 7, 1, &pNULLSRV );
            pd3dImmediateContext->VSSetShaderResources( 9, 1, &pNULLSRV );
        }

        // restore to default
        pd3dImmediateContext->RSSetState( NULL );

        // restore to previous
        pd3dImmediateContext->OMSetDepthStencilState( pDepthStencilStateStored11, uStencilRefStored11 );
        pd3dImmediateContext->OMSetBlendState( pBlendStateStored11, afBlendFactorStored11, uSampleMaskStored11 );
        SAFE_RELEASE( pDepthStencilStateStored11 );
        SAFE_RELEASE( pBlendStateStored11 );

    }

    //--------------------------------------------------------------------------------------
    // Fill in the data for the lights (center, radius, and color).
    // Also fill in the vertex data for the sprite quad.
    //--------------------------------------------------------------------------------------
    void LightUtil::InitLights( const XMVECTOR &BBoxMin, const XMVECTOR &BBoxMax )
    {
        // init the random seed to 1, so that results are deterministic 
        // across different runs of the sample
        srand(1);

        // scale the size of the lights based on the size of the scene
        XMVECTOR BBoxExtents = 0.5f * (BBoxMax - BBoxMin);
        float fRadius = 0.075f * XMVectorGetX(XMVector3Length(BBoxExtents));

        // For point lights, the radius of the bounding sphere for the light (used for culling) 
        // and the falloff distance of the light (used for lighting) are the same. Not so for 
        // spot lights. A spot light is a right circular cone. The height of the cone is the 
        // falloff distance. We want to fit the cone of the spot light inside the bounding sphere. 
        // From calculus, we know the cone with maximum volume that can fit inside a sphere has height:
        // h_cone = (4/3)*r_sphere
        float fSpotLightFalloffRadius = 1.333333333333f * fRadius;

        XMFLOAT3 vBBoxMin, vBBoxMax;
        XMStoreFloat3( &vBBoxMin, BBoxMin );
        XMStoreFloat3( &vBBoxMax, BBoxMax );

        // initialize the point light data
        for (int i = 0; i < MAX_NUM_LIGHTS; i++)
        {
            g_PointLightDataArray[i].v4PositionAndRadius = XMFLOAT4(GetRandFloat(vBBoxMin.x, vBBoxMax.x), GetRandFloat(vBBoxMin.y, vBBoxMax.y), GetRandFloat(vBBoxMin.z, vBBoxMax.z), fRadius);
            g_PointLightDataArray[i].v4Color = GetRandColorFloat4();
        }

        // initialize the spot light data
        for (int i = 0; i < MAX_NUM_LIGHTS; i++)
        {
            g_SpotLightDataArrayCenterAndRadius[i] = XMFLOAT4(GetRandFloat(vBBoxMin.x,vBBoxMax.x), GetRandFloat(vBBoxMin.y,vBBoxMax.y), GetRandFloat(vBBoxMin.z,vBBoxMax.z), fRadius);
            g_SpotLightDataArrayColor[i] = GetRandColor();

            XMFLOAT3 vLightDir = GetRandLightDirection();

            // Okay, so we fit a max-volume cone inside our bounding sphere for the spot light. We need to find 
            // the cone angle for that cone. Google on "cone inside sphere" (without the quotes) to find info 
            // on how to derive these formulas for the height and radius of the max-volume cone inside a sphere.
            // h_cone = (4/3)*r_sphere
            // r_cone = sqrt(8/9)*r_sphere
            // tan(theta) = r_cone/h_cone = sqrt(2)/2 = 0.7071067811865475244
            // theta = 35.26438968 degrees
            // store the cosine of this angle: cosine(35.26438968 degrees) = 0.816496580927726

            // random direction, cosine of cone angle, falloff radius calcuated above
            g_SpotLightDataArraySpotParams[i] = PackSpotParams(vLightDir, 0.816496580927726f, fSpotLightFalloffRadius);

            // build a "rotate from one vector to another" matrix, to point the spot light 
            // cone along its light direction
            XMVECTOR s = XMVectorSet(0.0f,-1.0f,0.0f,0.0f);
            XMVECTOR t = XMLoadFloat3( &vLightDir );
            XMFLOAT3 v;
            XMStoreFloat3( &v, XMVector3Cross(s,t) );
            float e = XMVectorGetX(XMVector3Dot(s,t));
            float h = 1.0f / (1.0f + e);

            XMFLOAT4X4 f4x4Rotation;
            XMStoreFloat4x4( &f4x4Rotation, XMMatrixIdentity() );
            f4x4Rotation._11 = e + h*v.x*v.x;
            f4x4Rotation._12 = h*v.x*v.y - v.z;
            f4x4Rotation._13 = h*v.x*v.z + v.y;
            f4x4Rotation._21 = h*v.x*v.y + v.z;
            f4x4Rotation._22 = e + h*v.y*v.y;
            f4x4Rotation._23 = h*v.y*v.z - v.x;
            f4x4Rotation._31 = h*v.x*v.z - v.y;
            f4x4Rotation._32 = h*v.y*v.z + v.x;
            f4x4Rotation._33 = e + h*v.z*v.z;
            XMMATRIX mRotation = XMLoadFloat4x4( &f4x4Rotation );

            g_SpotLightDataArraySpotMatrices[i] = XMMatrixTranspose(mRotation);
        }

        // initialize the vertex buffer data for a quad (for drawing the lights)
        float fQuadHalfSize = 0.083f * fRadius;
        g_QuadForLightsVertexData[0].v3Pos = XMFLOAT3(-fQuadHalfSize, -fQuadHalfSize, 0.0f );
        g_QuadForLightsVertexData[0].v2TexCoord = XMFLOAT2( 0.0f, 0.0f );
        g_QuadForLightsVertexData[1].v3Pos = XMFLOAT3(-fQuadHalfSize,  fQuadHalfSize, 0.0f );
        g_QuadForLightsVertexData[1].v2TexCoord = XMFLOAT2( 0.0f, 1.0f );
        g_QuadForLightsVertexData[2].v3Pos = XMFLOAT3( fQuadHalfSize, -fQuadHalfSize, 0.0f );
        g_QuadForLightsVertexData[2].v2TexCoord = XMFLOAT2( 1.0f, 0.0f );
        g_QuadForLightsVertexData[3].v3Pos = XMFLOAT3(-fQuadHalfSize,  fQuadHalfSize, 0.0f );
        g_QuadForLightsVertexData[3].v2TexCoord = XMFLOAT2( 0.0f, 1.0f );
        g_QuadForLightsVertexData[4].v3Pos = XMFLOAT3( fQuadHalfSize,  fQuadHalfSize, 0.0f );
        g_QuadForLightsVertexData[4].v2TexCoord = XMFLOAT2( 1.0f, 1.0f );
        g_QuadForLightsVertexData[5].v3Pos = XMFLOAT3( fQuadHalfSize, -fQuadHalfSize, 0.0f );
        g_QuadForLightsVertexData[5].v2TexCoord = XMFLOAT2( 1.0f, 0.0f );

        // initialize the vertex and index buffer data for a cone (for drawing the spot lights)
        {
            // h_cone = (4/3)*r_sphere
            // r_cone = sqrt(8/9)*r_sphere
            float fConeSphereRadius = 0.033f * fRadius;
            float fConeHeight = 1.333333333333f * fConeSphereRadius;
            float fConeRadius = 0.942809041582f * fConeSphereRadius;

            for (int i = 0; i < g_nConeNumTris; i++)
            {
                // We want to calculate points along the circle at the end of the cone.
                // The parametric equations for this circle are:
                // x=r_cone*cosine(t)
                // z=r_cone*sine(t)
                float t = ((float)i / (float)g_nConeNumTris) * TWO_PI;
                g_ConeForSpotLightsVertexData[2*i+1].v3Pos = XMFLOAT3( fConeRadius*cos(t), -fConeHeight, fConeRadius*sin(t) );
                g_ConeForSpotLightsVertexData[2*i+1].v2TexCoord = XMFLOAT2( 0.0f, 1.0f );

                // normal = (h_cone*cosine(t), r_cone, h_cone*sine(t))
                XMFLOAT3 vNormal = XMFLOAT3( fConeHeight*cos(t), fConeRadius, fConeHeight*sin(t) );
                XMStoreFloat3( &vNormal, XMVector3Normalize( XMLoadFloat3( &vNormal ) ) );
                g_ConeForSpotLightsVertexData[2*i+1].v3Norm = vNormal;
#ifdef _DEBUG
                // check that the normal is actually perpendicular
                float dot = XMVectorGetX( XMVector3Dot( XMLoadFloat3( &g_ConeForSpotLightsVertexData[2*i+1].v3Pos ), XMLoadFloat3( &vNormal ) ) );
                assert( abs(dot) < 0.001f );
#endif
            }

            // create duplicate points for the top of the cone, each with its own normal
            for (int i = 0; i < g_nConeNumTris; i++)
            {
                g_ConeForSpotLightsVertexData[2*i].v3Pos = XMFLOAT3( 0.0f, 0.0f, 0.0f );
                g_ConeForSpotLightsVertexData[2*i].v2TexCoord = XMFLOAT2( 0.0f, 0.0f );

                XMFLOAT3 vNormal;
                XMVECTOR Normal = XMLoadFloat3(&g_ConeForSpotLightsVertexData[2*i+1].v3Norm) + XMLoadFloat3(&g_ConeForSpotLightsVertexData[2*i+3].v3Norm);
                XMStoreFloat3( &vNormal, XMVector3Normalize( Normal ) );
                g_ConeForSpotLightsVertexData[2*i].v3Norm = vNormal;
            }

            // fill in the index buffer for the cone
            for (int i = 0; i < g_nConeNumTris; i++)
            {
                g_ConeForSpotLightsIndexData[3*i+0] = (unsigned short)(2*i);
                g_ConeForSpotLightsIndexData[3*i+1] = (unsigned short)(2*i+3);
                g_ConeForSpotLightsIndexData[3*i+2] = (unsigned short)(2*i+1);
            }

            // fix up the last triangle
            g_ConeForSpotLightsIndexData[3*g_nConeNumTris-2] = 1;
        }
    }

    //--------------------------------------------------------------------------------------
    // Add shaders to the shader cache
    //--------------------------------------------------------------------------------------
    void LightUtil::AddShadersToCache( AMD::ShaderCache *pShaderCache )
    {
        // Ensure all shaders (and input layouts) are released
        SAFE_RELEASE( m_pDebugDrawPointLightsVS );
        SAFE_RELEASE( m_pDebugDrawPointLightsPS );
        SAFE_RELEASE( m_pDebugDrawPointLightsLayout11 );
        SAFE_RELEASE( m_pDebugDrawSpotLightsVS );
        SAFE_RELEASE( m_pDebugDrawSpotLightsPS );
        SAFE_RELEASE( m_pDebugDrawSpotLightsLayout11 );

        const D3D11_INPUT_ELEMENT_DESC LayoutForSprites[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        const D3D11_INPUT_ELEMENT_DESC LayoutForCone[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 24, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawPointLightsVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"DebugDrawPointLightsVS",
            L"Shaders\\Source\\DebugDraw.hlsl", 0, NULL, &m_pDebugDrawPointLightsLayout11, LayoutForSprites, ARRAYSIZE( LayoutForSprites ) );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawPointLightsPS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DebugDrawPointLightsPS",
            L"Shaders\\source\\DebugDraw.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawSpotLightsVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"DebugDrawSpotLightsVS",
            L"Shaders\\Source\\DebugDraw.hlsl", 0, NULL, &m_pDebugDrawSpotLightsLayout11, LayoutForCone, ARRAYSIZE( LayoutForCone ) );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawSpotLightsPS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DebugDrawSpotLightsPS",
            L"Shaders\\source\\DebugDraw.hlsl", 0, NULL, NULL, NULL, 0 );
    }

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
