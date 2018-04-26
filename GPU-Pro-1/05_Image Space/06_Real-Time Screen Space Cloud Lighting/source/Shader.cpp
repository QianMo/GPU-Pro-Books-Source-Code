//--------------------------------------------------------------------------------------
// Shader.cpp
// 
// Kaori Kubota
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "Shader.h"


//--------------------------------------------------------------------------------------
CShader::CShader()
: m_pVS(NULL)
, m_pPS(NULL)
, m_pVSConst(NULL)
, m_pPSConst(NULL)
, m_phVSConst(NULL)
, m_phPSConst(NULL)
, m_nVSConstNum(0)
, m_nPSConstNum(0)
{
}
	
//--------------------------------------------------------------------------------------
CShader::~CShader()
{
	Delete();
}

//--------------------------------------------------------------------------------------
VOID CShader::Delete()
{
	if (m_pVS != NULL) {
		m_pVS->Release();
		m_pVS = NULL;
	}
	if (m_pPS != NULL) {
		m_pPS->Release();
		m_pPS = NULL;
	}
	if (m_pVSConst != NULL) {
		m_pVSConst->Release();
		m_pVSConst = NULL;
	}
	if (m_pPSConst != NULL) {
		m_pPSConst->Release();
		m_pPSConst = NULL;
	}
	if (m_phVSConst != NULL) {
		delete [] m_phVSConst;
		m_phVSConst = NULL;
		m_phPSConst = NULL;
	}
	else if (m_phPSConst != NULL) {
		// if there is no vertex shader constant, 
		//   m_phPSConst indicates the allocated array.
		delete [] m_phPSConst;
		m_phPSConst = NULL;
	}
	m_nVSConstNum = m_nPSConstNum = 0;
}

//--------------------------------------------------------------------------------------
// CShader::Create
//    Create shaders and shader constant tables.
//    Allocate array of DXHANDLE for all paramter and store all handle from names.
//--------------------------------------------------------------------------------------
BOOL CShader::Create(LPDIRECT3DDEVICE9 pDev, const SShaderInitializeParameter& initParam)
{
	// Create vertex shader and its constant table
	HRESULT hr = pDev->CreateVertexShader(initParam.pVSData, &m_pVS);
	if ( FAILED(hr) ) {
		return FALSE;
	}
	hr = D3DXGetShaderConstantTable( initParam.pVSData, &m_pVSConst );
	if ( FAILED(hr) ) {
		return FALSE;
	}

	// Create pixel shader and its constant table.
	hr = pDev->CreatePixelShader(initParam.pPSData, &m_pPS);
	if ( FAILED(hr) ) {
		return FALSE;
	}
	hr = D3DXGetShaderConstantTable( initParam.pPSData, &m_pPSConst );
	if ( FAILED(hr) ) {
		return FALSE;
	}


	m_nVSConstNum = initParam.nVSParamNum;
	m_nPSConstNum = initParam.nPSParamNum;

	// allocate DXHANDLE array
	UINT nParamNum = initParam.nVSParamNum + initParam.nPSParamNum;
	if ( 0 < nParamNum ) {
		D3DXHANDLE* pHandle = new D3DXHANDLE[ nParamNum ];
		if ( 0 < initParam.nVSParamNum ) {
			m_phVSConst = pHandle;
			if ( 0 < initParam.nPSParamNum ) {
				m_phPSConst = &pHandle[initParam.nVSParamNum];
			}
		}
		else if ( 0 < initParam.nPSParamNum ) {
			m_phPSConst = pHandle;
		}
	}

	// Store handles from name.
	for (UINT i = 0; i < initParam.nVSParamNum; ++i) {
		m_phVSConst[i] = m_pVSConst->GetConstantByName( NULL, initParam.pVSParams[i] );
	}
	for (UINT i = 0; i < initParam.nPSParamNum; ++i) {
		m_phPSConst[i] = m_pPSConst->GetConstantByName( NULL, initParam.pPSParams[i] );
	}

	return TRUE;
}


