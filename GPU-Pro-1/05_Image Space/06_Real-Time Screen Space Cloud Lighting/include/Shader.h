//--------------------------------------------------------------------------------------
// Shader.h
// 
// Copyright (C) Tomohide Kano. All rights reserved.
//--------------------------------------------------------------------------------------

#if !defined(__INCLUDED_SHADER_H__)
#define __INCLUDED_SHADER_H__




//--------------------------------------------------------------------------------------
// SShaderInitializeParameter
// Initialize parameters for CShader 
//--------------------------------------------------------------------------------------
struct SShaderInitializeParameter {
	const DWORD* pVSData;    // vertex shader program 
	const DWORD* pPSData;    // pixel shader program
	const char** pVSParams;  // vertex shader constant names 
	const char** pPSParams;  // pixel shader constant names
	UINT        nVSParamNum; // the number of vertex shader constants
	UINT        nPSParamNum; // the number of pixel shader constants
};


//--------------------------------------------------------------------------------------
// CShader
//    This class creates shaders and shader constant tables and has arrays of parameter handles.
//--------------------------------------------------------------------------------------
class CShader {
public :
	CShader();
	~CShader();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SShaderInitializeParameter& initParam);
	VOID Delete();

	inline BOOL SetShaders(LPDIRECT3DDEVICE9 pDev);

	inline VOID SetVSValue(LPDIRECT3DDEVICE9 pDev, UINT nIndex, const VOID* pData, UINT nSize);
	inline VOID SetVSMatrix(LPDIRECT3DDEVICE9 pDev, UINT nIndex, const D3DXMATRIX* pMatrix);
	inline VOID SetPSValue(LPDIRECT3DDEVICE9 pDev, UINT nIndex, const VOID* pData, UINT nSize);

protected :
	LPDIRECT3DVERTEXSHADER9      m_pVS;             // VertexShader
	LPDIRECT3DPIXELSHADER9       m_pPS;             // PixelShader
	LPD3DXCONSTANTTABLE          m_pVSConst;        // vertex shader constant table
	LPD3DXCONSTANTTABLE          m_pPSConst;        // pixel shader constant table
	D3DXHANDLE*                  m_phVSConst;       // parameter handles of vertex shader 
	D3DXHANDLE*                  m_phPSConst;       // parameter handles of pixel shader
	UINT                         m_nVSConstNum;     // The number of parameter handles of pixel shader
	UINT                         m_nPSConstNum;     // The number of parameter handles of pixel shader
};


//--------------------------------------------------------------------------------------
// Set vertex and pixel shader to a graphics device
//--------------------------------------------------------------------------------------
BOOL CShader::SetShaders(LPDIRECT3DDEVICE9 pDev)
{
	if ( m_pVS == NULL || m_pPS == NULL ) {
		return FALSE;
	}
	pDev->SetVertexShader( m_pVS );
	pDev->SetPixelShader( m_pPS );

	return TRUE;
}


//--------------------------------------------------------------------------------------
VOID CShader::SetVSMatrix(LPDIRECT3DDEVICE9 pDev, UINT nIndex, const D3DXMATRIX* pMatrix)
{
	if (m_pVSConst != NULL && m_phVSConst != NULL && nIndex < m_nVSConstNum && m_phVSConst[nIndex] != NULL) {
		m_pVSConst->SetMatrix( pDev, m_phVSConst[nIndex], pMatrix );
	}
}

//--------------------------------------------------------------------------------------
VOID CShader::SetVSValue(LPDIRECT3DDEVICE9 pDev, UINT nIndex, const VOID* pData, UINT size)
{
	if (m_pVSConst != NULL && m_phVSConst != NULL && nIndex < m_nVSConstNum && m_phVSConst[nIndex] != NULL) {
		m_pVSConst->SetValue( pDev, m_phVSConst[nIndex], pData, size );
	}
}

//--------------------------------------------------------------------------------------
VOID CShader::SetPSValue(LPDIRECT3DDEVICE9 pDev, UINT nIndex, const VOID* pData, UINT size)
{
	if (m_pPSConst != NULL && m_phPSConst != NULL && nIndex < m_nPSConstNum && m_phPSConst[nIndex] != NULL) {
		m_pPSConst->SetValue( pDev, m_phPSConst[nIndex], pData, size );
	}
}


#endif

