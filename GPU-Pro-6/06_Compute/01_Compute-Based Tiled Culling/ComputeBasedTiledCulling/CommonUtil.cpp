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
// File: CommonUtil.cpp
//
// Helper functions for the ComputeBasedTiledCulling sample.
//--------------------------------------------------------------------------------------

#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Core\\DDSTextureLoader.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"

#include "..\\AMD_SDK\\AMD_SDK.h"

#include "CommonUtil.h"

using namespace DirectX;

struct CommonUtilGridVertex
{
    XMFLOAT3 v3Pos;
    XMFLOAT3 v3Norm;
    XMFLOAT2 v2TexCoord;
    XMFLOAT3 v3Tangent;
};

// Grid data (for throwing a lot of tris at the GPU)
// 30x30 cells, times 2 tris per cell, times 2 for front and back, 
// that's 3600 tris per grid (half front facing and half back facing), 
// times 280 grid objects equals 1,008,000 triangles (half front facing and half back facing)
static const int            g_nNumGridCells1DHigh = 30;
static const int            g_nNumGridVerticesHigh = 2 * (g_nNumGridCells1DHigh + 1) * (g_nNumGridCells1DHigh + 1);
static const int            g_nNumGridIndicesHigh = 2 * 6 * g_nNumGridCells1DHigh * g_nNumGridCells1DHigh;
static CommonUtilGridVertex g_GridVertexDataHigh[ComputeBasedTiledCulling::MAX_NUM_GRID_OBJECTS][g_nNumGridVerticesHigh];
static unsigned short       g_GridIndexDataHigh[g_nNumGridIndicesHigh];

// Grid data (for throwing a lot of tris at the GPU)
// 21x21 cells, times 2 tris per cell, times 2 for front and back, 
// that's 1764 tris per grid (half front facing and half back facing), 
// times 280 grid objects equals 493,920 triangles (half front facing and half back facing)
static const int            g_nNumGridCells1DMed = 21;
static const int            g_nNumGridVerticesMed = 2 * (g_nNumGridCells1DMed + 1) * (g_nNumGridCells1DMed + 1);
static const int            g_nNumGridIndicesMed = 2 * 6 * g_nNumGridCells1DMed * g_nNumGridCells1DMed;
static CommonUtilGridVertex g_GridVertexDataMed[ComputeBasedTiledCulling::MAX_NUM_GRID_OBJECTS][g_nNumGridVerticesMed];
static unsigned short       g_GridIndexDataMed[g_nNumGridIndicesMed];

// Grid data (for throwing a lot of tris at the GPU)
// 11x11 cells, times 2 tris per cell, times 2 for front and back, 
// that's 484 tris per grid (half front facing and half back facing), 
// times 280 grid objects equals 135,520 triangles (half front facing and half back facing)
static const int            g_nNumGridCells1DLow = 11;
static const int            g_nNumGridVerticesLow = 2 * (g_nNumGridCells1DLow + 1) * (g_nNumGridCells1DLow + 1);
static const int            g_nNumGridIndicesLow = 2 * 6 * g_nNumGridCells1DLow * g_nNumGridCells1DLow;
static CommonUtilGridVertex g_GridVertexDataLow[ComputeBasedTiledCulling::MAX_NUM_GRID_OBJECTS][g_nNumGridVerticesLow];
static unsigned short       g_GridIndexDataLow[g_nNumGridIndicesLow];

static const int            g_nNumGridIndices[ComputeBasedTiledCulling::TRIANGLE_DENSITY_NUM_TYPES] = { g_nNumGridIndicesLow, g_nNumGridIndicesMed, g_nNumGridIndicesHigh };

struct CommonUtilSpriteVertex
{
    XMFLOAT3 v3Pos;
    XMFLOAT2 v2TexCoord;
};

// static array for sprite quad vertex data
static CommonUtilSpriteVertex         g_QuadForLegendVertexData[6];

// constants for the legend for the lights-per-tile visualization
static const int g_nLegendNumLines = 17;
static const int g_nLegendTextureWidth = 32;
static const int g_nLegendPaddingLeft = 5;
static const int g_nLegendPaddingBottom = 2*AMD::HUD::iElementDelta;

// there should only be one CommonUtil object
static int CommonUtilObjectCounter = 0;

template <size_t nNumGridVertices, size_t nNumGridIndices>
static void InitGridObjectData(int nNumGridCells1D, CommonUtilGridVertex GridVertexData[ComputeBasedTiledCulling::MAX_NUM_GRID_OBJECTS][nNumGridVertices], unsigned short GridIndexData[nNumGridIndices])
{
    const float fGridSizeWorldSpace = 100.0f;
    const float fGridSizeWorldSpaceHalf = 0.5f * fGridSizeWorldSpace;

    const float fPosStep = fGridSizeWorldSpace / (float)(nNumGridCells1D);
    const float fTexStep = 1.0f / (float)(nNumGridCells1D);

    const float fPosX = 725.0f;
    const float fPosYStart = 1000.0f;
    const float fStepY = 1.05f * fGridSizeWorldSpace;
    const float fPosZStart = 1467.0f;
    const float fStepZ = 1.05f * fGridSizeWorldSpace;

    for( int nGrid = 0; nGrid < ComputeBasedTiledCulling::MAX_NUM_GRID_OBJECTS; nGrid++ )
    {
        const float fCurrentPosYOffset = fPosYStart - (float)((nGrid/28)%10)*fStepY;
        const float fCurrentPosZOffset = fPosZStart - (float)(nGrid%28)*fStepZ;

        // front side verts
        for( int i = 0; i < nNumGridCells1D+1; i++ )
        {
            const float fPosY = fCurrentPosYOffset + fGridSizeWorldSpaceHalf - (float)i*fPosStep;
            const float fV = (float)i*fTexStep;
            for( int j = 0; j < nNumGridCells1D+1; j++ )
            {
                const float fPosZ = fCurrentPosZOffset - fGridSizeWorldSpaceHalf + (float)j*fPosStep;
                const float fU = (float)j*fTexStep;
                const int idx = (nNumGridCells1D+1) * i + j;
                GridVertexData[nGrid][idx].v3Pos = XMFLOAT3(fPosX, fPosY, fPosZ);
                GridVertexData[nGrid][idx].v3Norm = XMFLOAT3(1,0,0);
                GridVertexData[nGrid][idx].v2TexCoord = XMFLOAT2(fU,fV);
                GridVertexData[nGrid][idx].v3Tangent = XMFLOAT3(0,0,-1);
            }
        }

        // back side verts
        for( int i = 0; i < nNumGridCells1D+1; i++ )
        {
            const float fPosY = fCurrentPosYOffset + fGridSizeWorldSpaceHalf - (float)i*fPosStep;
            const float fV = (float)i*fTexStep;
            for( int j = 0; j < nNumGridCells1D+1; j++ )
            {
                const float fPosZ = fCurrentPosZOffset + fGridSizeWorldSpaceHalf - (float)j*fPosStep;
                const float fU = (float)j*fTexStep;
                const int idx = (nNumGridCells1D+1) * (nNumGridCells1D+1) + (nNumGridCells1D+1) * i + j;
                GridVertexData[nGrid][idx].v3Pos = XMFLOAT3(fPosX, fPosY, fPosZ);
                GridVertexData[nGrid][idx].v3Norm = XMFLOAT3(-1,0,0);
                GridVertexData[nGrid][idx].v2TexCoord = XMFLOAT2(fU,fV);
                GridVertexData[nGrid][idx].v3Tangent = XMFLOAT3(0,0,1);
            }
        }
    }

    // front side tris
    for( int i = 0; i < nNumGridCells1D; i++ )
    {
        for( int j = 0; j < nNumGridCells1D; j++ )
        {
            const int vertexStartIndexThisRow = (nNumGridCells1D+1) * i + j;
            const int vertexStartIndexNextRow = (nNumGridCells1D+1) * (i+1) + j;
            const int idx = (6 * nNumGridCells1D * i) + (6*j);
            GridIndexData[idx+0] = (unsigned short)(vertexStartIndexThisRow);
            GridIndexData[idx+1] = (unsigned short)(vertexStartIndexThisRow+1);
            GridIndexData[idx+2] = (unsigned short)(vertexStartIndexNextRow);
            GridIndexData[idx+3] = (unsigned short)(vertexStartIndexThisRow+1);
            GridIndexData[idx+4] = (unsigned short)(vertexStartIndexNextRow+1);
            GridIndexData[idx+5] = (unsigned short)(vertexStartIndexNextRow);
        }
    }

    // back side tris
    for( int i = 0; i < nNumGridCells1D; i++ )
    {
        for( int j = 0; j < nNumGridCells1D; j++ )
        {
            const int vertexStartIndexThisRow = (nNumGridCells1D+1) * (nNumGridCells1D+1) + (nNumGridCells1D+1) * i + j;
            const int vertexStartIndexNextRow = (nNumGridCells1D+1) * (nNumGridCells1D+1) + (nNumGridCells1D+1) * (i+1) + j;
            const int idx = (6 * nNumGridCells1D * nNumGridCells1D) + (6 * nNumGridCells1D * i) + (6*j);
            GridIndexData[idx+0] = (unsigned short)(vertexStartIndexThisRow);
            GridIndexData[idx+1] = (unsigned short)(vertexStartIndexThisRow+1);
            GridIndexData[idx+2] = (unsigned short)(vertexStartIndexNextRow);
            GridIndexData[idx+3] = (unsigned short)(vertexStartIndexThisRow+1);
            GridIndexData[idx+4] = (unsigned short)(vertexStartIndexNextRow+1);
            GridIndexData[idx+5] = (unsigned short)(vertexStartIndexNextRow);
        }
    }
}

namespace ComputeBasedTiledCulling
{

    //--------------------------------------------------------------------------------------
    // Constructor
    //--------------------------------------------------------------------------------------
    CommonUtil::CommonUtil()
        :m_uWidth(0)
        ,m_uHeight(0)
        ,m_pDepthBoundsTexture(NULL)
        ,m_pDepthBoundsSRV(NULL)
        ,m_pDepthBoundsUAV(NULL)
        ,m_pLightIndexBuffer(NULL)
        ,m_pLightIndexBufferSRV(NULL)
        ,m_pLightIndexBufferUAV(NULL)
        ,m_pSpotIndexBuffer(NULL)
        ,m_pSpotIndexBufferSRV(NULL)
        ,m_pSpotIndexBufferUAV(NULL)
        ,m_pGridDiffuseTextureSRV(NULL)
        ,m_pGridNormalMapSRV(NULL)
        ,m_pQuadForLegendVB(NULL)
        ,m_pDepthBoundsCS(NULL)
        ,m_pLightCullCS(NULL)
        ,m_pDebugDrawNumLightsPerTileRadarColorsPS(NULL)
        ,m_pDebugDrawNumLightsPerTileGrayscalePS(NULL)
        ,m_pDebugDrawLegendForNumLightsPerTileVS(NULL)
        ,m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS(NULL)
        ,m_pDebugDrawLegendForNumLightsPerTileGrayscalePS(NULL)
        ,m_pDebugDrawLegendForNumLightsLayout11(NULL)
        ,m_pFullScreenVS(NULL)
        ,m_pFullScreenPS(NULL)
    {
        assert( CommonUtilObjectCounter == 0 );
        CommonUtilObjectCounter++;

        for( int i = 0; i < TRIANGLE_DENSITY_NUM_TYPES; i++ )
        {
            for( int j = 0; j < MAX_NUM_GRID_OBJECTS; j++ )
            {
                m_pGridVB[i][j] = NULL;
            }

            m_pGridIB[i] = NULL;
        }

        for( int i = 0; i < DEPTH_STENCIL_STATE_NUM_TYPES; i++ )
        {
            m_pDepthStencilState[i] = NULL;
        }

        for( int i = 0; i < RASTERIZER_STATE_NUM_TYPES; i++ )
        {
            m_pRasterizerState[i] = NULL;
        }

        for( int i = 0; i < SAMPLER_STATE_NUM_TYPES; i++ )
        {
            m_pSamplerState[i] = NULL;
        }
    }


    //--------------------------------------------------------------------------------------
    // Destructor
    //--------------------------------------------------------------------------------------
    CommonUtil::~CommonUtil()
    {
        assert( CommonUtilObjectCounter == 1 );
        CommonUtilObjectCounter--;

        SAFE_RELEASE(m_pDepthBoundsTexture);
        SAFE_RELEASE(m_pDepthBoundsSRV);
        SAFE_RELEASE(m_pDepthBoundsUAV);
        SAFE_RELEASE(m_pLightIndexBuffer);
        SAFE_RELEASE(m_pLightIndexBufferSRV);
        SAFE_RELEASE(m_pLightIndexBufferUAV);
        SAFE_RELEASE(m_pSpotIndexBuffer);
        SAFE_RELEASE(m_pSpotIndexBufferSRV);
        SAFE_RELEASE(m_pSpotIndexBufferUAV);
        SAFE_RELEASE(m_pGridDiffuseTextureSRV);
        SAFE_RELEASE(m_pGridNormalMapSRV);
        SAFE_RELEASE(m_pQuadForLegendVB);
        SAFE_RELEASE(m_pDepthBoundsCS);
        SAFE_RELEASE(m_pLightCullCS);
        SAFE_RELEASE(m_pDebugDrawNumLightsPerTileRadarColorsPS);
        SAFE_RELEASE(m_pDebugDrawNumLightsPerTileGrayscalePS);
        SAFE_RELEASE(m_pDebugDrawLegendForNumLightsPerTileVS);
        SAFE_RELEASE(m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS);
        SAFE_RELEASE(m_pDebugDrawLegendForNumLightsPerTileGrayscalePS);
        SAFE_RELEASE(m_pDebugDrawLegendForNumLightsLayout11);
        SAFE_RELEASE(m_pFullScreenVS);
        SAFE_RELEASE(m_pFullScreenPS);

        for( int i = 0; i < TRIANGLE_DENSITY_NUM_TYPES; i++ )
        {
            for( int j = 0; j < MAX_NUM_GRID_OBJECTS; j++ )
            {
                SAFE_RELEASE(m_pGridVB[i][j]);
            }

            SAFE_RELEASE(m_pGridIB[i]);
        }

        for( int i = 0; i < DEPTH_STENCIL_STATE_NUM_TYPES; i++ )
        {
            SAFE_RELEASE(m_pDepthStencilState[i]);
        }

        for( int i = 0; i < RASTERIZER_STATE_NUM_TYPES; i++ )
        {
            SAFE_RELEASE(m_pRasterizerState[i]);
        }

        for( int i = 0; i < SAMPLER_STATE_NUM_TYPES; i++ )
        {
            SAFE_RELEASE(m_pSamplerState[i]);
        }
    }

    //--------------------------------------------------------------------------------------
    // Device creation hook function
    //--------------------------------------------------------------------------------------
    HRESULT CommonUtil::OnCreateDevice( ID3D11Device* pd3dDevice )
    {
        HRESULT hr;

        D3D11_SUBRESOURCE_DATA InitData;

        // Create the vertex buffer for the grid objects
        D3D11_BUFFER_DESC VBDesc;
        ZeroMemory( &VBDesc, sizeof(VBDesc) );
        VBDesc.Usage = D3D11_USAGE_IMMUTABLE;
        VBDesc.ByteWidth = sizeof( g_GridVertexDataHigh[0] );
        VBDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        for( int i = 0; i < MAX_NUM_GRID_OBJECTS; i++ )
        {
            InitData.pSysMem = g_GridVertexDataHigh[i];
            V_RETURN( pd3dDevice->CreateBuffer( &VBDesc, &InitData, &m_pGridVB[TRIANGLE_DENSITY_HIGH][i] ) );
        }

        VBDesc.ByteWidth = sizeof( g_GridVertexDataMed[0] );
        for( int i = 0; i < MAX_NUM_GRID_OBJECTS; i++ )
        {
            InitData.pSysMem = g_GridVertexDataMed[i];
            V_RETURN( pd3dDevice->CreateBuffer( &VBDesc, &InitData, &m_pGridVB[TRIANGLE_DENSITY_MEDIUM][i] ) );
        }

        VBDesc.ByteWidth = sizeof( g_GridVertexDataLow[0] );
        for( int i = 0; i < MAX_NUM_GRID_OBJECTS; i++ )
        {
            InitData.pSysMem = g_GridVertexDataLow[i];
            V_RETURN( pd3dDevice->CreateBuffer( &VBDesc, &InitData, &m_pGridVB[TRIANGLE_DENSITY_LOW][i] ) );
        }

        // Create the index buffer for the grid objects
        D3D11_BUFFER_DESC IBDesc;
        ZeroMemory( &IBDesc, sizeof(IBDesc) );
        IBDesc.Usage = D3D11_USAGE_IMMUTABLE;
        IBDesc.ByteWidth = sizeof( g_GridIndexDataHigh );
        IBDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        InitData.pSysMem = g_GridIndexDataHigh;
        V_RETURN( pd3dDevice->CreateBuffer( &IBDesc, &InitData, &m_pGridIB[TRIANGLE_DENSITY_HIGH] ) );

        IBDesc.ByteWidth = sizeof( g_GridIndexDataMed );
        InitData.pSysMem = g_GridIndexDataMed;
        V_RETURN( pd3dDevice->CreateBuffer( &IBDesc, &InitData, &m_pGridIB[TRIANGLE_DENSITY_MEDIUM] ) );

        IBDesc.ByteWidth = sizeof( g_GridIndexDataLow );
        InitData.pSysMem = g_GridIndexDataLow;
        V_RETURN( pd3dDevice->CreateBuffer( &IBDesc, &InitData, &m_pGridIB[TRIANGLE_DENSITY_LOW] ) );

        // Load the diffuse and normal map for the grid
        {
            WCHAR path[MAX_PATH];
            DXUTFindDXSDKMediaFileCch( path, MAX_PATH, L"misc\\default_diff.dds" );

            // Create the shader resource view.
            CreateDDSTextureFromFile( pd3dDevice, path, NULL, &m_pGridDiffuseTextureSRV );

            DXUTFindDXSDKMediaFileCch( path, MAX_PATH, L"misc\\default_norm.dds" );

            // Create the shader resource view.
            CreateDDSTextureFromFile( pd3dDevice, path, NULL, &m_pGridNormalMapSRV );
        }

        // Default depth-stencil state, except with inverted DepthFunc 
        // (because we are using inverted 32-bit float depth for better precision)
        D3D11_DEPTH_STENCIL_DESC DepthStencilDesc;
        ZeroMemory( &DepthStencilDesc, sizeof( D3D11_DEPTH_STENCIL_DESC ) );
        DepthStencilDesc.DepthEnable = TRUE; 
        DepthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL; 
        DepthStencilDesc.DepthFunc = D3D11_COMPARISON_GREATER;  // we are using inverted 32-bit float depth for better precision
        DepthStencilDesc.StencilEnable = FALSE; 
        DepthStencilDesc.StencilReadMask = D3D11_DEFAULT_STENCIL_READ_MASK; 
        DepthStencilDesc.StencilWriteMask = D3D11_DEFAULT_STENCIL_WRITE_MASK; 
        V_RETURN( pd3dDevice->CreateDepthStencilState( &DepthStencilDesc, &m_pDepthStencilState[DEPTH_STENCIL_STATE_DEPTH_GREATER] ) );

        // Disable depth test write
        DepthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO; 
        V_RETURN( pd3dDevice->CreateDepthStencilState( &DepthStencilDesc, &m_pDepthStencilState[DEPTH_STENCIL_STATE_DISABLE_DEPTH_WRITE] ) );

        // Disable depth test
        DepthStencilDesc.DepthEnable = FALSE; 
        V_RETURN( pd3dDevice->CreateDepthStencilState( &DepthStencilDesc, &m_pDepthStencilState[DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST] ) );

        // Comparison greater with depth writes disabled
        DepthStencilDesc.DepthEnable = TRUE; 
        DepthStencilDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO; 
        DepthStencilDesc.DepthFunc = D3D11_COMPARISON_GREATER;  // we are using inverted 32-bit float depth for better precision
        V_RETURN( pd3dDevice->CreateDepthStencilState( &DepthStencilDesc, &m_pDepthStencilState[DEPTH_STENCIL_STATE_DEPTH_GREATER_AND_DISABLE_DEPTH_WRITE] ) );

        // Comparison equal with depth writes disabled
        DepthStencilDesc.DepthFunc = D3D11_COMPARISON_EQUAL; 
        V_RETURN( pd3dDevice->CreateDepthStencilState( &DepthStencilDesc, &m_pDepthStencilState[DEPTH_STENCIL_STATE_DEPTH_EQUAL_AND_DISABLE_DEPTH_WRITE] ) );

        // Disable culling
        D3D11_RASTERIZER_DESC RasterizerDesc;
        RasterizerDesc.FillMode = D3D11_FILL_SOLID;
        RasterizerDesc.CullMode = D3D11_CULL_NONE;       // disable culling
        RasterizerDesc.FrontCounterClockwise = FALSE;
        RasterizerDesc.DepthBias = 0;
        RasterizerDesc.DepthBiasClamp = 0.0f;
        RasterizerDesc.SlopeScaledDepthBias = 0.0f;
        RasterizerDesc.DepthClipEnable = TRUE;
        RasterizerDesc.ScissorEnable = FALSE;
        RasterizerDesc.MultisampleEnable = FALSE;
        RasterizerDesc.AntialiasedLineEnable = FALSE;
        V_RETURN( pd3dDevice->CreateRasterizerState( &RasterizerDesc, &m_pRasterizerState[RASTERIZER_STATE_DISABLE_CULLING] ) );

        RasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;  // wireframe
        RasterizerDesc.CullMode = D3D11_CULL_BACK;
        V_RETURN( pd3dDevice->CreateRasterizerState( &RasterizerDesc, &m_pRasterizerState[RASTERIZER_STATE_WIREFRAME] ) );

        RasterizerDesc.FillMode = D3D11_FILL_WIREFRAME;  // wireframe and ...
        RasterizerDesc.CullMode = D3D11_CULL_NONE;       // disable culling
        V_RETURN( pd3dDevice->CreateRasterizerState( &RasterizerDesc, &m_pRasterizerState[RASTERIZER_STATE_WIREFRAME_DISABLE_CULLING] ) );

        // Create state objects
        D3D11_SAMPLER_DESC SamplerDesc;
        ZeroMemory( &SamplerDesc, sizeof(SamplerDesc) );
        SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
        SamplerDesc.AddressU = SamplerDesc.AddressV = SamplerDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
        SamplerDesc.MaxAnisotropy = 16;
        SamplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        SamplerDesc.MinLOD = -D3D11_FLOAT32_MAX;
        SamplerDesc.MaxLOD =  D3D11_FLOAT32_MAX;
        V_RETURN( pd3dDevice->CreateSamplerState( &SamplerDesc, &m_pSamplerState[SAMPLER_STATE_POINT] ) );
        SamplerDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        V_RETURN( pd3dDevice->CreateSamplerState( &SamplerDesc, &m_pSamplerState[SAMPLER_STATE_LINEAR] ) );
        SamplerDesc.Filter = D3D11_FILTER_ANISOTROPIC;
        V_RETURN( pd3dDevice->CreateSamplerState( &SamplerDesc, &m_pSamplerState[SAMPLER_STATE_ANISO] ) );

        return hr;
    }


    //--------------------------------------------------------------------------------------
    // Device destruction hook function
    //--------------------------------------------------------------------------------------
    void CommonUtil::OnDestroyDevice()
    {
        for( int i = 0; i < TRIANGLE_DENSITY_NUM_TYPES; i++ )
        {
            for( int j = 0; j < MAX_NUM_GRID_OBJECTS; j++ )
            {
                SAFE_RELEASE(m_pGridVB[i][j]);
            }

            SAFE_RELEASE(m_pGridIB[i]);
        }

        SAFE_RELEASE(m_pGridDiffuseTextureSRV);
        SAFE_RELEASE(m_pGridNormalMapSRV);

        SAFE_RELEASE(m_pDepthBoundsCS);
        SAFE_RELEASE(m_pLightCullCS);

        SAFE_RELEASE( m_pDebugDrawNumLightsPerTileRadarColorsPS );
        SAFE_RELEASE( m_pDebugDrawNumLightsPerTileGrayscalePS );

        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsPerTileVS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsPerTileGrayscalePS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsLayout11 );

        SAFE_RELEASE(m_pFullScreenVS);
        SAFE_RELEASE(m_pFullScreenPS);

        for( int i = 0; i < DEPTH_STENCIL_STATE_NUM_TYPES; i++ )
        {
            SAFE_RELEASE(m_pDepthStencilState[i]);
        }

        for( int i = 0; i < RASTERIZER_STATE_NUM_TYPES; i++ )
        {
            SAFE_RELEASE(m_pRasterizerState[i]);
        }

        for( int i = 0; i < SAMPLER_STATE_NUM_TYPES; i++ )
        {
            SAFE_RELEASE(m_pSamplerState[i]);
        }
    }


    //--------------------------------------------------------------------------------------
    // Resized swap chain hook function
    //--------------------------------------------------------------------------------------
    HRESULT CommonUtil::OnResizedSwapChain( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, int nLineHeight )
    {
        HRESULT hr;

        m_uWidth = pBackBufferSurfaceDesc->Width;
        m_uHeight = pBackBufferSurfaceDesc->Height;

        // depends on m_uWidth and m_uHeight, so don't do this 
        // until you have updated them (see above)
        unsigned uNumTiles = GetNumTilesX()*GetNumTilesY();
        unsigned uMaxNumElementsPerTile = GetMaxNumElementsPerTile();

        V_RETURN( AMD::CreateSurface( &m_pDepthBoundsTexture, &m_pDepthBoundsSRV, NULL, &m_pDepthBoundsUAV, DXGI_FORMAT_R32G32B32A32_FLOAT, GetNumTilesX(), GetNumTilesY(), 1 ) );

        D3D11_BUFFER_DESC BufferDesc;
        ZeroMemory( &BufferDesc, sizeof(BufferDesc) );
        BufferDesc.Usage = D3D11_USAGE_DEFAULT;
        BufferDesc.ByteWidth = 2 * uMaxNumElementsPerTile * uNumTiles;
        BufferDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        V_RETURN( pd3dDevice->CreateBuffer( &BufferDesc, NULL, &m_pLightIndexBuffer ) );
        DXUT_SetDebugName( m_pLightIndexBuffer, "LightIndexBuffer" );

        V_RETURN( pd3dDevice->CreateBuffer( &BufferDesc, NULL, &m_pSpotIndexBuffer ) );
        DXUT_SetDebugName( m_pSpotIndexBuffer, "SpotIndexBuffer" );

        D3D11_SHADER_RESOURCE_VIEW_DESC SRVDesc;
        ZeroMemory( &SRVDesc, sizeof( D3D11_SHADER_RESOURCE_VIEW_DESC ) );
        SRVDesc.Format = DXGI_FORMAT_R16_UINT;
        SRVDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        SRVDesc.Buffer.ElementOffset = 0;
        SRVDesc.Buffer.ElementWidth = uMaxNumElementsPerTile * uNumTiles;
        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pLightIndexBuffer, &SRVDesc, &m_pLightIndexBufferSRV ) );

        V_RETURN( pd3dDevice->CreateShaderResourceView( m_pSpotIndexBuffer, &SRVDesc, &m_pSpotIndexBufferSRV ) );

        D3D11_UNORDERED_ACCESS_VIEW_DESC UAVDesc;
        ZeroMemory( &UAVDesc, sizeof( D3D11_UNORDERED_ACCESS_VIEW_DESC ) );
        UAVDesc.Format = DXGI_FORMAT_R16_UINT;
        UAVDesc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        UAVDesc.Buffer.FirstElement = 0;
        UAVDesc.Buffer.NumElements = uMaxNumElementsPerTile * uNumTiles;
        V_RETURN( pd3dDevice->CreateUnorderedAccessView( m_pLightIndexBuffer, &UAVDesc, &m_pLightIndexBufferUAV ) );

        V_RETURN( pd3dDevice->CreateUnorderedAccessView( m_pSpotIndexBuffer, &UAVDesc, &m_pSpotIndexBufferUAV ) );

        // initialize the vertex buffer data for a quad (for drawing the lights-per-tile legend)
        const float kTextureHeight = (float)g_nLegendNumLines * (float)nLineHeight;
        const float kTextureWidth = (float)g_nLegendTextureWidth;
        const float kPaddingLeft = (float)g_nLegendPaddingLeft;
        const float kPaddingBottom = (float)g_nLegendPaddingBottom;
        float fLeft = kPaddingLeft;
        float fRight = kPaddingLeft + kTextureWidth;
        float fTop = (float)m_uHeight - kPaddingBottom - kTextureHeight;
        float fBottom =(float)m_uHeight - kPaddingBottom;
        g_QuadForLegendVertexData[0].v3Pos = XMFLOAT3( fLeft,  fBottom, 0.0f );
        g_QuadForLegendVertexData[0].v2TexCoord = XMFLOAT2( 0.0f, 0.0f );
        g_QuadForLegendVertexData[1].v3Pos = XMFLOAT3( fLeft,  fTop, 0.0f );
        g_QuadForLegendVertexData[1].v2TexCoord = XMFLOAT2( 0.0f, 1.0f );
        g_QuadForLegendVertexData[2].v3Pos = XMFLOAT3( fRight, fBottom, 0.0f );
        g_QuadForLegendVertexData[2].v2TexCoord = XMFLOAT2( 1.0f, 0.0f );
        g_QuadForLegendVertexData[3].v3Pos = XMFLOAT3( fLeft,  fTop, 0.0f );
        g_QuadForLegendVertexData[3].v2TexCoord = XMFLOAT2( 0.0f, 1.0f );
        g_QuadForLegendVertexData[4].v3Pos = XMFLOAT3( fRight,  fTop, 0.0f );
        g_QuadForLegendVertexData[4].v2TexCoord = XMFLOAT2( 1.0f, 1.0f );
        g_QuadForLegendVertexData[5].v3Pos = XMFLOAT3( fRight, fBottom, 0.0f );
        g_QuadForLegendVertexData[5].v2TexCoord = XMFLOAT2( 1.0f, 0.0f );

        // Create the vertex buffer for the sprite (a single quad)
        D3D11_SUBRESOURCE_DATA InitData;
        D3D11_BUFFER_DESC VBDesc;
        ZeroMemory( &VBDesc, sizeof(VBDesc) );
        VBDesc.Usage = D3D11_USAGE_IMMUTABLE;
        VBDesc.ByteWidth = sizeof( g_QuadForLegendVertexData );
        VBDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        InitData.pSysMem = g_QuadForLegendVertexData;
        V_RETURN( pd3dDevice->CreateBuffer( &VBDesc, &InitData, &m_pQuadForLegendVB ) );
        DXUT_SetDebugName( m_pQuadForLegendVB, "QuadForLegendVB" );

        return S_OK;
    }

    //--------------------------------------------------------------------------------------
    // Releasing swap chain hook function
    //--------------------------------------------------------------------------------------
    void CommonUtil::OnReleasingSwapChain()
    {
        SAFE_RELEASE(m_pDepthBoundsTexture);
        SAFE_RELEASE(m_pDepthBoundsSRV);
        SAFE_RELEASE(m_pDepthBoundsUAV);
        SAFE_RELEASE(m_pLightIndexBuffer);
        SAFE_RELEASE(m_pLightIndexBufferSRV);
        SAFE_RELEASE(m_pLightIndexBufferUAV);
        SAFE_RELEASE(m_pSpotIndexBuffer);
        SAFE_RELEASE(m_pSpotIndexBufferSRV);
        SAFE_RELEASE(m_pSpotIndexBufferUAV);
        SAFE_RELEASE(m_pQuadForLegendVB);
    }

    void CommonUtil::InitStaticData()
    {
        // make sure our indices will actually fit in a R16_UINT
        assert( g_nNumGridVerticesHigh <= 65536 );
        assert( g_nNumGridVerticesMed  <= 65536 );
        assert( g_nNumGridVerticesLow  <= 65536 );

        InitGridObjectData<g_nNumGridVerticesHigh,g_nNumGridIndicesHigh>( g_nNumGridCells1DHigh, g_GridVertexDataHigh, g_GridIndexDataHigh );
        InitGridObjectData<g_nNumGridVerticesMed, g_nNumGridIndicesMed >( g_nNumGridCells1DMed,  g_GridVertexDataMed,  g_GridIndexDataMed  );
        InitGridObjectData<g_nNumGridVerticesLow, g_nNumGridIndicesLow >( g_nNumGridCells1DLow,  g_GridVertexDataLow,  g_GridIndexDataLow  );
    }

    //--------------------------------------------------------------------------------------
    // Calculate AABB around all meshes in the scene
    //--------------------------------------------------------------------------------------
    void CommonUtil::CalculateSceneMinMax( CDXUTSDKMesh &Mesh, XMVECTOR *pBBoxMinOut, XMVECTOR *pBBoxMaxOut )
    {
        *pBBoxMaxOut = Mesh.GetMeshBBoxCenter( 0 ) + Mesh.GetMeshBBoxExtents( 0 );
        *pBBoxMinOut = Mesh.GetMeshBBoxCenter( 0 ) - Mesh.GetMeshBBoxExtents( 0 );

        for( unsigned i = 1; i < Mesh.GetNumMeshes(); i++ )
        {
            XMVECTOR vNewMax = Mesh.GetMeshBBoxCenter( i ) + Mesh.GetMeshBBoxExtents( i );
            XMVECTOR vNewMin = Mesh.GetMeshBBoxCenter( i ) - Mesh.GetMeshBBoxExtents( i );

            *pBBoxMaxOut = XMVectorMax(*pBBoxMaxOut, vNewMax);
            *pBBoxMinOut = XMVectorMin(*pBBoxMinOut, vNewMin);
        }

    }

    //--------------------------------------------------------------------------------------
    // Add shaders to the shader cache
    //--------------------------------------------------------------------------------------
    void CommonUtil::AddShadersToCache( AMD::ShaderCache *pShaderCache )
    {
        // Ensure all shaders (and input layouts) are released

        SAFE_RELEASE(m_pDepthBoundsCS);
        SAFE_RELEASE(m_pLightCullCS);

        SAFE_RELEASE( m_pDebugDrawNumLightsPerTileRadarColorsPS );
        SAFE_RELEASE( m_pDebugDrawNumLightsPerTileGrayscalePS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsPerTileVS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsPerTileGrayscalePS );
        SAFE_RELEASE( m_pDebugDrawLegendForNumLightsLayout11 );

        SAFE_RELEASE(m_pFullScreenVS);
        SAFE_RELEASE(m_pFullScreenPS);

        const D3D11_INPUT_ELEMENT_DESC LayoutForSprites[] =
        {
            { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0,  0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
            { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        };

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawNumLightsPerTileRadarColorsPS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DebugDrawNumLightsPerTileRadarColorsPS",
            L"Shaders\\source\\DebugDraw.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawNumLightsPerTileGrayscalePS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DebugDrawNumLightsPerTileGrayscalePS",
            L"Shaders\\source\\DebugDraw.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawLegendForNumLightsPerTileVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"DebugDrawLegendForNumLightsPerTileVS",
            L"Shaders\\Source\\DebugDraw.hlsl", 0, NULL, &m_pDebugDrawLegendForNumLightsLayout11, LayoutForSprites, ARRAYSIZE( LayoutForSprites ) );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DebugDrawLegendForNumLightsPerTileRadarColorsPS",
            L"Shaders\\source\\DebugDraw.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDebugDrawLegendForNumLightsPerTileGrayscalePS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"DebugDrawLegendForNumLightsPerTileGrayscalePS",
            L"Shaders\\source\\DebugDraw.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pFullScreenVS, AMD::ShaderCache::SHADER_TYPE_VERTEX, L"vs_5_0", L"FullScreenQuadVS",
            L"Shaders\\Source\\Common.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pFullScreenPS, AMD::ShaderCache::SHADER_TYPE_PIXEL, L"ps_5_0", L"FullScreenBlitPS",
            L"Shaders\\source\\Common.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pDepthBoundsCS, AMD::ShaderCache::SHADER_TYPE_COMPUTE, L"cs_5_0", L"CalculateDepthBoundsCS",
            L"Shaders\\source\\ParallelReduction.hlsl", 0, NULL, NULL, NULL, 0 );

        pShaderCache->AddShader( (ID3D11DeviceChild**)&m_pLightCullCS, AMD::ShaderCache::SHADER_TYPE_COMPUTE, L"cs_5_0", L"CullLightsCS",
            L"Shaders\\source\\TilingForward.hlsl", 0, NULL, NULL, NULL, 0 );
    }

    //--------------------------------------------------------------------------------------
    // Draw the legend for the lights-per-tile visualization
    //--------------------------------------------------------------------------------------
    void CommonUtil::RenderLegend( CDXUTTextHelper *pTxtHelper, int nLineHeight, XMFLOAT4 Color, int nDebugDrawType ) const
    {
        // draw the legend texture for the lights-per-tile visualization
        {
            ID3D11ShaderResourceView* pNULLSRV = NULL;
            ID3D11SamplerState* pNULLSampler = NULL;

            // choose pixel shader based on radar vs. grayscale
            ID3D11PixelShader* pPixelShader = ( nDebugDrawType == DEBUG_DRAW_GRAYSCALE ) ? m_pDebugDrawLegendForNumLightsPerTileGrayscalePS : m_pDebugDrawLegendForNumLightsPerTileRadarColorsPS;

            ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

            // save depth state (for later restore)
            ID3D11DepthStencilState* pDepthStencilStateStored11 = NULL;
            UINT uStencilRefStored11;
            pd3dImmediateContext->OMGetDepthStencilState( &pDepthStencilStateStored11, &uStencilRefStored11 );

            // disable depth test
            pd3dImmediateContext->OMSetDepthStencilState( GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST), 0x00 );

            // Set the input layout
            pd3dImmediateContext->IASetInputLayout( m_pDebugDrawLegendForNumLightsLayout11 );

            // Set vertex buffer
            UINT uStride = sizeof( CommonUtilSpriteVertex );
            UINT uOffset = 0;
            pd3dImmediateContext->IASetVertexBuffers( 0, 1, &m_pQuadForLegendVB, &uStride, &uOffset );

            // Set primitive topology
            pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

            pd3dImmediateContext->VSSetShader( m_pDebugDrawLegendForNumLightsPerTileVS, NULL, 0 );
            pd3dImmediateContext->PSSetShader( pPixelShader, NULL, 0 );
            pd3dImmediateContext->PSSetShaderResources( 0, 1, &pNULLSRV );
            pd3dImmediateContext->PSSetShaderResources( 1, 1, &pNULLSRV );
            pd3dImmediateContext->PSSetSamplers( 0, 1, &pNULLSampler );

            pd3dImmediateContext->Draw(6,0);

            // restore to previous
            pd3dImmediateContext->OMSetDepthStencilState( pDepthStencilStateStored11, uStencilRefStored11 );
            SAFE_RELEASE( pDepthStencilStateStored11 );
        }

        // draw the legend text for the lights-per-tile visualization
        // (twice, to get a drop shadow)
        XMFLOAT4 Colors[2] = { XMFLOAT4(0,0,0,Color.w), Color };
        for( int loopCount = 0; loopCount < 2; loopCount++ )
        {
            // 17 lines times line height
            int nTextureHeight = g_nLegendNumLines*nLineHeight;

            // times 2 below, one for point lights and one for spot lights
            int nMaxNumLightsPerTile = (int)(2*GetMaxNumLightsPerTile());
            WCHAR szBuf[16];

            pTxtHelper->Begin();
            pTxtHelper->SetForegroundColor( Colors[loopCount] );
            pTxtHelper->SetInsertionPos( g_nLegendPaddingLeft+g_nLegendTextureWidth+4-loopCount, (int)m_uHeight-g_nLegendPaddingBottom-nTextureHeight-nLineHeight-3-loopCount );
            pTxtHelper->DrawTextLine( L"Num Lights" );

            int nStartVal, nEndVal;

            if( nDebugDrawType == DEBUG_DRAW_GRAYSCALE )
            {
                float fLightsPerBand = (float)(nMaxNumLightsPerTile) / (float)g_nLegendNumLines;

                for( int i = 0; i < g_nLegendNumLines; i++ )
                {
                    nStartVal = (int)((g_nLegendNumLines-1-i)*fLightsPerBand) + 1;
                    nEndVal = (int)((g_nLegendNumLines-i)*fLightsPerBand);
                    swprintf_s( szBuf, 16, L"[%d,%d]", nStartVal, nEndVal );
                    pTxtHelper->DrawTextLine( szBuf );
                }
            }
            else
            {
                // use a log scale to provide more detail when the number of lights is smaller

                // want to find the base b such that the logb of nMaxNumLightsPerTile is 14
                // (because we have 14 radar colors)
                float fLogBase = exp(0.07142857f*log((float)nMaxNumLightsPerTile));

                swprintf_s( szBuf, 16, L"> %d", nMaxNumLightsPerTile );
                pTxtHelper->DrawTextLine( szBuf );

                swprintf_s( szBuf, 16, L"%d", nMaxNumLightsPerTile );
                pTxtHelper->DrawTextLine( szBuf );

                nStartVal = (int)pow(fLogBase,g_nLegendNumLines-4) + 1;
                nEndVal = nMaxNumLightsPerTile-1;
                swprintf_s( szBuf, 16, L"[%d,%d]", nStartVal, nEndVal );
                pTxtHelper->DrawTextLine( szBuf );

                for( int i = 0; i < g_nLegendNumLines-5; i++ )
                {
                    nStartVal = (int)pow(fLogBase,g_nLegendNumLines-5-i) + 1;
                    nEndVal = (int)pow(fLogBase,g_nLegendNumLines-4-i);
                    if( nStartVal == nEndVal )
                    {
                        swprintf_s( szBuf, 16, L"%d", nStartVal );
                    }
                    else
                    {
                        swprintf_s( szBuf, 16, L"[%d,%d]", nStartVal, nEndVal );
                    }
                    pTxtHelper->DrawTextLine( szBuf );
                }

                pTxtHelper->DrawTextLine( L"1" );
                pTxtHelper->DrawTextLine( L"0" );
            }

            pTxtHelper->End();
        }
    }

    //--------------------------------------------------------------------------------------
    // Calculate the number of tiles in the horizontal direction
    //--------------------------------------------------------------------------------------
    unsigned CommonUtil::GetNumTilesX() const
    {
        return (unsigned)( ( m_uWidth + TILE_RES - 1 ) / (float)TILE_RES );
    }

    //--------------------------------------------------------------------------------------
    // Calculate the number of tiles in the vertical direction
    //--------------------------------------------------------------------------------------
    unsigned CommonUtil::GetNumTilesY() const
    {
        return (unsigned)( ( m_uHeight + TILE_RES - 1 ) / (float)TILE_RES );
    }

    //--------------------------------------------------------------------------------------
    // Adjust max number of lights per tile based on screen height.
    // This assumes that the demo has a constant vertical field of view (fovy).
    //
    // Note that the light culling tile size stays fixed as screen size changes.
    // With a constant fovy, reducing the screen height shrinks the projected 
    // view of the scene, and so more lights can fall into our fixed tile size.
    //
    // This function reduces the max lights per tile as screen height increases, 
    // to save memory. It was tuned for this particular demo and is not intended 
    // as a general solution for all scenes.
    //--------------------------------------------------------------------------------------
    unsigned CommonUtil::GetMaxNumLightsPerTile() const
    {
        const unsigned kAdjustmentMultipier = 16;

        // I haven't tested at greater than 1080p, so cap it
        unsigned uHeight = (m_uHeight > 1080) ? 1080 : m_uHeight;

        // adjust max lights per tile down as height increases
        return ( MAX_NUM_LIGHTS_PER_TILE - ( kAdjustmentMultipier * ( uHeight / 120 ) ) );
    }

    //--------------------------------------------------------------------------------------
    // Calculate the max number of list elements in the per-tile lists.
    // For example, per-tile lists can contain the light count and other non-light-index data.
    //--------------------------------------------------------------------------------------
    unsigned CommonUtil::GetMaxNumElementsPerTile() const
    {
        // max num lights times 2 (because the halfZ method has two lists per tile, list A and B),
        // plus two more to store the 32-bit halfZ, plus one more for the light count of list A,
        // plus one more for the light count of list B
        return (2*GetMaxNumLightsPerTile() + 4);
    }

    //--------------------------------------------------------------------------------------
    // Draw the "a lot of triangles" grid
    //--------------------------------------------------------------------------------------
    void CommonUtil::DrawGrid(int nGridNumber, int nTriangleDensity, bool bWithTextures) const
    {
        // clamp nGridNumber
        nGridNumber = (nGridNumber < 0) ? 0 : nGridNumber;
        nGridNumber = (nGridNumber > MAX_NUM_GRID_OBJECTS-1) ? MAX_NUM_GRID_OBJECTS-1 : nGridNumber;

        // clamp nTriangleDensity
        nTriangleDensity = (nTriangleDensity < 0) ? 0 : nTriangleDensity;
        nTriangleDensity = (nTriangleDensity > TRIANGLE_DENSITY_NUM_TYPES-1) ? TRIANGLE_DENSITY_NUM_TYPES-1 : nTriangleDensity;

        ID3D11Buffer* const * pGridVB = m_pGridVB[nTriangleDensity];
        ID3D11Buffer* pGridIB = m_pGridIB[nTriangleDensity];

        ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

        // Set vertex buffer
        UINT uStride = sizeof( CommonUtilGridVertex );
        UINT uOffset = 0;
        pd3dImmediateContext->IASetVertexBuffers( 0, 1, &pGridVB[nGridNumber], &uStride, &uOffset );
        pd3dImmediateContext->IASetIndexBuffer( pGridIB, DXGI_FORMAT_R16_UINT, 0 );

        // Set primitive topology
        pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );

        if( bWithTextures )
        {
            pd3dImmediateContext->PSSetShaderResources( 0, 1, &m_pGridDiffuseTextureSRV );
            pd3dImmediateContext->PSSetShaderResources( 1, 1, &m_pGridNormalMapSRV );
        }

        pd3dImmediateContext->DrawIndexed( g_nNumGridIndices[nTriangleDensity], 0, 0 );
    }

    //--------------------------------------------------------------------------------------
    // Return one of the lights-per-tile visualization shaders, based on nDebugDrawType
    //--------------------------------------------------------------------------------------
    ID3D11PixelShader * CommonUtil::GetDebugDrawNumLightsPerTilePS( int nDebugDrawType ) const
    {
        if ( ( nDebugDrawType != DEBUG_DRAW_RADAR_COLORS ) && ( nDebugDrawType != DEBUG_DRAW_GRAYSCALE ) )
        {
            return NULL;
        }

        if( nDebugDrawType == DEBUG_DRAW_RADAR_COLORS )
        {
            return m_pDebugDrawNumLightsPerTileRadarColorsPS;
        }
        else
        {
            // default
            return m_pDebugDrawNumLightsPerTileGrayscalePS;
        }
    }

} // namespace ComputeBasedTiledCulling

//--------------------------------------------------------------------------------------
// EOF
//--------------------------------------------------------------------------------------
