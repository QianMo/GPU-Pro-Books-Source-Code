// Copyright 2013 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#pragma once

#include "RenderTechnique.h"
#include "TerrainStructs.fxh"
#include "Visibility.h"

// Structure describing terrain rendering parameters
struct SRenderingParams
{
    STerrainAttribs m_TerrainAttribs;

    enum TEXTURING_MODE
    {
        TM_HEIGHT_BASED = 0,
		TM_MATERIAL_MASK = 1,
        TM_MATERIAL_MASK_NM = 2
    };
    
    // Patch rendering params
    TEXTURING_MODE m_TexturingMode;
    int m_iRingDimension;
    int m_iNumRings;

    int m_iNumShadowCascades;
    BOOL m_bBestCascadeSearch;
    BOOL m_bSmoothShadows;
    int m_iColOffset, m_iRowOffset;

    BOOL m_bEnableClouds;

    SRenderingParams() : 
		m_TexturingMode(TM_MATERIAL_MASK),
        m_iRingDimension(65),
        m_iNumRings(15),
        m_iNumShadowCascades(4),
        m_bBestCascadeSearch(TRUE),
        m_bSmoothShadows(TRUE),
        m_iColOffset(0), 
        m_iRowOffset(0),
        m_bEnableClouds(TRUE)
	{}
};

struct SRingSectorMesh
{
    CComPtr<ID3D11Buffer> pIndBuff;
    UINT uiNumIndices;
    SBoundingBox BoundBox;
    SRingSectorMesh() : uiNumIndices(0){}
};

// This class renders the adaptive model using DX11 API
class CEarthHemsiphere
{
public:
    CEarthHemsiphere(void) : m_uiNumStitchIndices(0){}

    // Renders the model
	void Render(ID3D11DeviceContext* pd3dImmediateContext,
                const D3DXMATRIX &CameraViewProjMatrix,
                ID3D11Buffer *pcbCameraAttribs,
                ID3D11Buffer *pcbLightAttribs,
                ID3D11Buffer *pcMediaScatteringParams,
                ID3D11ShaderResourceView *pShadowMapSRV,
                ID3D11ShaderResourceView *pLiSpCloudTransparencySRV,
                ID3D11ShaderResourceView *pPrecomputedNetDensitySRV,
                ID3D11ShaderResourceView *pAmbientSkylightSRV,
                bool bZOnlyPass);
    
    // Creates Direct3D11 device resources
    HRESULT OnD3D11CreateDevice( class CElevationDataSource *pDataSource,
                                 const SRenderingParams &Params,
                                 ID3D11Device* pd3dDevice,
                                 ID3D11DeviceContext* pd3dImmediateContext,
                                 LPCTSTR HeightMapPath,
                                 LPCTSTR MaterialMaskPath,
								 LPCTSTR *TileTexturePath,
                                 LPCTSTR *TileNormalMapPath );

    // Releases Direct3D11 device resources
    void OnD3D11DestroyDevice( );

    enum {NUM_TILE_TEXTURES = 1 + 4};// One base material + 4 masked materials

    void UpdateParams(const SRenderingParams &NewParams);

private:
    HRESULT CreateRenderStates(ID3D11Device* pd3dDevice);
    void RenderNormalMap(ID3D11Device* pd3dDevice,
                         ID3D11DeviceContext* pd3dImmediateContext,
                         const UINT16 *pHeightMap,
                         size_t HeightMapPitch,
                         int iHeightMapDim);

    CRenderTechnique m_RenderEarthHemisphereTech;
    CRenderTechnique m_RenderEarthHemisphereZOnlyTech;

    CComPtr<ID3D11Buffer> m_pVertBuff;
    CComPtr<ID3D11InputLayout> m_pInputLayout;
    CComPtr<ID3D11ShaderResourceView> m_ptex2DNormalMapSRV, m_ptex2DMtrlMaskSRV;
	CComPtr<ID3D11DepthStencilState> m_pEnableDepthTestDS;
    CComPtr<ID3D11DepthStencilState> m_pDisableDepthTestDS;
	CComPtr<ID3D11BlendState> m_pDefaultBS;
    CComPtr<ID3D11RasterizerState> m_pRSSolidFill, m_pRSSolidFillNoCull, m_pRSZOnlyPass, m_pRSWireframeFill;
    CComPtr<ID3D11SamplerState> m_psamPointClamp, m_psamLinearMirror, m_psamLinearWrap, m_psamComaprison, m_psamLinearClamp;
    CComPtr<ID3D11Buffer> m_pcbTerrainAttribs;

	CComPtr<ID3D11ShaderResourceView> m_ptex2DTilesSRV[NUM_TILE_TEXTURES];
    CComPtr<ID3D11ShaderResourceView> m_ptex2DTilNormalMapsSRV[NUM_TILE_TEXTURES];

    std::vector<SRingSectorMesh> m_SphereMeshes;
    
    CComPtr<ID3D11Buffer> m_pStitchIndBuff;
    UINT m_uiNumStitchIndices;

    SRenderingParams m_Params;

private:
    CEarthHemsiphere(const CEarthHemsiphere&);
    CEarthHemsiphere& operator = (const CEarthHemsiphere&);
};
