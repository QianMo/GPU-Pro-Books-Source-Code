//--------------------------------------------------------------------------------------
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
//--------------------------------------------------------------------------------------

#pragma once

#include <D3D11.h>
#include <atlcomcli.h>

class CRenderTechnique
{
public:
    CRenderTechnique(void);
    ~CRenderTechnique(void);

    void Release();
    void SetDeviceAndContext(ID3D11Device *pDevice, ID3D11DeviceContext *pCtx){m_pDevice = pDevice; m_pContext = pCtx;}
    void Apply();

    ID3D11VertexShader      * GetVS(){return m_pVS;}
    ID3D11GeometryShader    * GetGS(){return m_pGS;}
    ID3D11PixelShader       * GetPS(){return m_pPS;}
    ID3D11ComputeShader     * GetCS(){return m_pCS;}
    ID3D11RasterizerState   * GetRS(){return m_pRS;}
    ID3D11DepthStencilState * GetDS(){return m_pDS;}
    ID3D11BlendState        * GetBS(){return m_pBS;}
	ID3DBlob				* GetVSByteCode(){return m_pVSByteCode;}

    void SetVS(ID3D11VertexShader      *pVS ){m_pVS = pVS;}
    void SetGS(ID3D11GeometryShader    *pGS ){m_pGS = pGS;}
    void SetPS(ID3D11PixelShader       *pPS ){m_pPS = pPS;}
    void SetCS(ID3D11ComputeShader     *pCS ){m_pCS = pCS;}
    void SetRS(ID3D11RasterizerState   *pRS ){m_pRS = pRS;}
    void SetDS(ID3D11DepthStencilState *pDS, UINT uiSampleRef = 0 ){m_pDS = pDS; m_uiSampleRef = uiSampleRef;}
    void SetBS(ID3D11BlendState        *pBS ){m_pBS = pBS;}

    HRESULT CreateVertexShaderFromFile   (LPCTSTR strFilePath, LPCSTR strFunctionName, const D3D_SHADER_MACRO* pDefines);
    HRESULT CreateGeometryShaderFromFile (LPCTSTR strFilePath, LPCSTR strFunctionName, const D3D_SHADER_MACRO* pDefines);
    HRESULT CreateGSWithSOFromFile       (LPCTSTR strFilePath, LPCSTR strFunctionName, const D3D_SHADER_MACRO* pDefines, D3D11_SO_DECLARATION_ENTRY *pDecl, UINT NumDeclEntries, UINT *pStrides, UINT NumStrides, UINT RasterizedStream);
    HRESULT CreatePixelShaderFromFile    (LPCTSTR strFilePath, LPCSTR strFunctionName, const D3D_SHADER_MACRO* pDefines);
    HRESULT CreateComputeShaderFromFile  (LPCTSTR strFilePath, LPCSTR strFunctionName, const D3D_SHADER_MACRO* pDefines);
    
    HRESULT CreateVGPShadersFromFile     (LPCTSTR strFilePath, LPSTR strVSFunctionName, LPSTR strGSFunctionName, LPSTR strPSFunctionName, const D3D_SHADER_MACRO* pDefines);

    bool IsValid(){ return m_pDevice && m_pContext && (m_pPS || m_pCS || (m_pVS && m_pGS));}

    HRESULT CreateDefaultBlendState();
    HRESULT CreateDefaultDepthState(BOOL bEnableDepth = TRUE, D3D11_DEPTH_WRITE_MASK WriteMask = D3D11_DEPTH_WRITE_MASK_ALL);
    HRESULT CreateDefaultRasterizerState(D3D11_FILL_MODE FillMode = D3D11_FILL_SOLID,
                                         D3D11_CULL_MODE CullMode = D3D11_CULL_BACK,
                                         BOOL bIsFrontCCW = FALSE);

    HRESULT CreateSampler(ID3D11SamplerState **ppSamplerState,
                          D3D11_FILTER Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR,
                          D3D11_TEXTURE_ADDRESS_MODE AddressMode = D3D11_TEXTURE_ADDRESS_CLAMP);

private:
    CComPtr<ID3D11Device> m_pDevice;
    CComPtr<ID3D11DeviceContext> m_pContext;
    CComPtr<ID3D11VertexShader> m_pVS;
    CComPtr<ID3D11GeometryShader> m_pGS;
    CComPtr<ID3D11PixelShader> m_pPS;
    CComPtr<ID3D11ComputeShader> m_pCS;
    CComPtr<ID3D11RasterizerState> m_pRS;
    CComPtr<ID3D11DepthStencilState> m_pDS;
    CComPtr<ID3D11BlendState> m_pBS;
    UINT m_uiSampleRef;
	CComPtr<ID3DBlob> m_pVSByteCode;
};
