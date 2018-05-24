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

#include "stdafx.h"

#include "RenderTechnique.h"
#include <D3DX11.h>
#include <D3Dcompiler.h>
#include <fstream>

CRenderTechnique::CRenderTechnique(void)
{
}


CRenderTechnique::~CRenderTechnique(void)
{
}

void CRenderTechnique::Release()
{
    m_pVS.Release();
    m_pGS.Release();
    m_pPS.Release();
    m_pCS.Release();
    m_pRS.Release();
    m_pDS.Release();
    m_pBS.Release();
	m_pVSByteCode.Release();
    m_pContext.Release();
    m_pDevice.Release();
}

static
HRESULT CompileShaderFromFile(LPCTSTR strFilePath, 
                              LPCSTR strFunctionName,
                              const D3D_SHADER_MACRO* pDefines, 
                              LPCSTR profile, 
                              ID3DBlob **ppBlobOut)
{
    DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3D10_SHADER_DEBUG;
#else
    // Warning: do not use this flag as it causes shader compiler to fail the compilation and 
    // report strange errors:
    // dwShaderFlags |= D3D10_SHADER_OPTIMIZATION_LEVEL3;
#endif
	HRESULT hr;
	do
	{
		CComPtr<ID3DBlob> errors;
        hr = D3DX11CompileFromFile(strFilePath, pDefines, NULL, strFunctionName, profile, dwShaderFlags, 0, NULL, ppBlobOut, &errors, NULL);
		if( errors )
		{
			OutputDebugStringA((char*) errors->GetBufferPointer());
			if( FAILED(hr) && 
				IDRETRY != MessageBoxA(NULL, (char*) errors->GetBufferPointer(), "FX Error", MB_ICONERROR|MB_ABORTRETRYIGNORE) )
			{
				break;
			}
		}
	} while( FAILED(hr) );
	return hr;
}

HRESULT CRenderTechnique::CreateVertexShaderFromFile(LPCTSTR strFilePath, 
                                                      LPCSTR strFunctionName,
                                                      const D3D_SHADER_MACRO* pDefines)
{
	HRESULT hr;
    m_pVSByteCode.Release();
    hr = CompileShaderFromFile( strFilePath, strFunctionName, pDefines, "vs_5_0", &m_pVSByteCode );
    if(FAILED(hr))return hr;

    m_pVS.Release();
    if( m_pDevice )
        hr = m_pDevice->CreateVertexShader( m_pVSByteCode->GetBufferPointer(), m_pVSByteCode->GetBufferSize(), NULL, &m_pVS );
    else
        return E_FAIL;

    return hr;
}

HRESULT CRenderTechnique::CreateGeometryShaderFromFile(LPCTSTR strFilePath, 
                                                        LPCSTR strFunctionName,
                                                        const D3D_SHADER_MACRO* pDefines)
{
    CComPtr<ID3DBlob> pShaderByteCode;

    HRESULT hr;
    hr = CompileShaderFromFile( strFilePath, strFunctionName, pDefines, "gs_5_0", &pShaderByteCode );
    if(FAILED(hr))return hr;

    m_pGS.Release();
    if( m_pDevice )
        hr = m_pDevice->CreateGeometryShader( pShaderByteCode->GetBufferPointer(), pShaderByteCode->GetBufferSize(), NULL, &m_pGS );
    else
        return E_FAIL;

    return hr;
}

HRESULT CRenderTechnique::CreateGSWithSOFromFile(LPCTSTR strFilePath, 
                                                 LPCSTR strFunctionName,
                                                 const D3D_SHADER_MACRO* pDefines, 
                                                 D3D11_SO_DECLARATION_ENTRY *pDecl, 
                                                 UINT NumDeclEntries, 
                                                 UINT *pStrides, 
                                                 UINT NumStrides, 
                                                 UINT RasterizedStream)
{
    CComPtr<ID3DBlob> pShaderByteCode;
    
    HRESULT hr;
    hr = CompileShaderFromFile( strFilePath, strFunctionName, pDefines, "gs_5_0", &pShaderByteCode );
    if(FAILED(hr))return hr;
    
    m_pGS.Release();
    if( m_pDevice )
        hr = m_pDevice->CreateGeometryShaderWithStreamOutput( pShaderByteCode->GetBufferPointer(), pShaderByteCode->GetBufferSize(), pDecl, NumDeclEntries, pStrides, NumStrides, RasterizedStream, NULL, &m_pGS );
    else
        return E_FAIL;

    return hr;
}

HRESULT CRenderTechnique::CreatePixelShaderFromFile(LPCTSTR strFilePath, 
                                                     LPCSTR strFunctionName,
                                                     const D3D_SHADER_MACRO* pDefines)
{
    CComPtr<ID3DBlob> pShaderByteCode;

    HRESULT hr;
    hr = CompileShaderFromFile( strFilePath, strFunctionName, pDefines, "ps_5_0", &pShaderByteCode );
    if(FAILED(hr))return hr;

    m_pPS.Release();
    if( m_pDevice )
        hr = m_pDevice->CreatePixelShader( pShaderByteCode->GetBufferPointer(), pShaderByteCode->GetBufferSize(), NULL, &m_pPS );
    else
        return E_FAIL;

    return hr;
}

HRESULT CRenderTechnique::CreateComputeShaderFromFile(LPCTSTR strFilePath, 
                                                       LPCSTR strFunctionName,
                                                       const D3D_SHADER_MACRO* pDefines)
{
    CComPtr<ID3DBlob> pShaderByteCode;

    HRESULT hr;
    hr = CompileShaderFromFile( strFilePath, strFunctionName, pDefines, "cs_5_0", &pShaderByteCode );
    if(FAILED(hr))return hr;

    m_pCS.Release();
    if( m_pDevice )
        hr = m_pDevice->CreateComputeShader( pShaderByteCode->GetBufferPointer(), pShaderByteCode->GetBufferSize(), NULL, &m_pCS );
    else
        return E_FAIL;

    return hr;
}

HRESULT CRenderTechnique::CreateVGPShadersFromFile(LPCTSTR strFilePath, 
                                                    LPSTR strVSFunctionName, 
                                                    LPSTR strGSFunctionName, 
                                                    LPSTR strPSFunctionName, 
                                                    const D3D_SHADER_MACRO* pDefines)
{
    HRESULT hr = S_OK;
    if( strVSFunctionName )
    {
        hr = CreateVertexShaderFromFile(strFilePath, strVSFunctionName, pDefines);
        if( FAILED(hr) )return hr;
    }

    if( strPSFunctionName )
    {
        hr = CreatePixelShaderFromFile(strFilePath, strPSFunctionName, pDefines);
        if( FAILED(hr) )return hr;
    }

    if( strGSFunctionName )
    {
        hr = CreateGeometryShaderFromFile(strFilePath, strGSFunctionName, pDefines);
        if( FAILED(hr) )return hr;
    }

    return hr;
}

void CRenderTechnique::Apply()
{
    m_pContext->HSSetShader(NULL, NULL, 0);
    m_pContext->DSSetShader(NULL, NULL, 0);
    m_pContext->VSSetShader(m_pVS, NULL, 0);
    m_pContext->GSSetShader(m_pGS, NULL, 0);
    m_pContext->PSSetShader(m_pPS, NULL, 0);
    m_pContext->CSSetShader(m_pCS, NULL, 0);
    m_pContext->RSSetState(m_pRS);
    m_pContext->OMSetDepthStencilState(m_pDS, m_uiSampleRef);  
    float fBlendFactor[] = {0, 0, 0, 0};
    m_pContext->OMSetBlendState(m_pBS, fBlendFactor, 0xFFFFFFFF);
}

HRESULT CRenderTechnique::CreateDefaultBlendState()
{
    HRESULT hr;
    D3D11_BLEND_DESC DefaultBlendStateDesc;
    ZeroMemory(&DefaultBlendStateDesc, sizeof(DefaultBlendStateDesc));
    DefaultBlendStateDesc.IndependentBlendEnable = FALSE;
    for(int i=0; i< _countof(DefaultBlendStateDesc.RenderTarget); i++)
        DefaultBlendStateDesc.RenderTarget[i].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    V( m_pDevice->CreateBlendState( &DefaultBlendStateDesc, &m_pBS) );
    
    return hr;
}

HRESULT CRenderTechnique::CreateDefaultDepthState(BOOL bEnableDepth, 
                                                  D3D11_DEPTH_WRITE_MASK WriteMask)
{
    HRESULT hr;
    
    D3D11_DEPTH_STENCIL_DESC DSDesc;
    ZeroMemory(&DSDesc, sizeof(DSDesc));
    DSDesc.DepthEnable = bEnableDepth;
    DSDesc.DepthWriteMask = WriteMask;
    DSDesc.DepthFunc = D3D11_COMPARISON_GREATER;
    CComPtr<ID3D11DepthStencilState> pDisableDepthTestDS;
    V( m_pDevice->CreateDepthStencilState(  &DSDesc, &m_pDS) );

    return hr;
}

HRESULT CRenderTechnique::CreateDefaultRasterizerState(D3D11_FILL_MODE FillMode,
                                                       D3D11_CULL_MODE CullMode,
                                                       BOOL bIsFrontCCW)
{
    HRESULT hr;

    D3D11_RASTERIZER_DESC RSDesc;
    ZeroMemory(&RSDesc, sizeof(RSDesc));
    RSDesc.FillMode = FillMode;
    RSDesc.CullMode = CullMode;
    RSDesc.FrontCounterClockwise = bIsFrontCCW;
    CComPtr<ID3D11RasterizerState> pRSSolidFillNoCull;
    V( m_pDevice->CreateRasterizerState( &RSDesc, &m_pRS) );

    return hr;
}

HRESULT CRenderTechnique::CreateSampler(ID3D11SamplerState **ppSamplerState,
                                        D3D11_FILTER Filter, 
                                        D3D11_TEXTURE_ADDRESS_MODE AddressMode)
{
    
    D3D11_SAMPLER_DESC SamDesc = 
    {
        Filter,
        AddressMode,
        AddressMode,
        AddressMode,
        0, //FLOAT MipLODBias;
        0, //UINT MaxAnisotropy;
        D3D11_COMPARISON_NEVER, // D3D11_COMPARISON_FUNC ComparisonFunc;
        {0.f, 0.f, 0.f, 0.f}, //FLOAT BorderColor[ 4 ];
        -FLT_MAX, //FLOAT MinLOD;
        +FLT_MAX //FLOAT MaxLOD;
    };
    
    HRESULT hr = m_pDevice->CreateSamplerState(&SamDesc, ppSamplerState);
    return hr;
}
