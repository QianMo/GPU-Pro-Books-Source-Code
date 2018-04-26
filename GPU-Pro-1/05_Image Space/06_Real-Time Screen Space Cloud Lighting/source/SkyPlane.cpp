//--------------------------------------------------------------------------------------
// SkyPlane.cpp
//
// Implementation of a class to render sky with daylight scattering 
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "SkyPlane.h"



//--------------------------------------------------------------------------------------
CSkyPlane::CSkyPlane()
: m_pVB(NULL)
, m_pIB(NULL)
, m_pDecl(NULL)
, m_pSceneParam(NULL)
{
}


//--------------------------------------------------------------------------------------
CSkyPlane::~CSkyPlane()
{
	Delete();
}


//--------------------------------------------------------------------------------------
VOID CSkyPlane::Delete()
{
	if (m_pVB != NULL) {
		m_pVB->Release();
		m_pVB = NULL;
	}
	if (m_pIB != NULL) {
		m_pIB->Release();
		m_pIB = NULL;
	}
	if (m_pDecl != NULL) {
		m_pDecl->Release();
		m_pDecl = NULL;
	}

	CShader::Delete();
}

//--------------------------------------------------------------------------------------
BOOL CSkyPlane::Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam)
{
	Delete();
	
	m_pSceneParam = pSceneParam;

	// Create vertex declaraction 
	static const D3DVERTEXELEMENT9 s_elements[] = {
		{ 0,  0,  D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		D3DDECL_END()
	};
	HRESULT hr = pDev->CreateVertexDeclaration( s_elements, &m_pDecl );
	if ( FAILED(hr) ) {
		return FALSE;
	}

	if (!CreateBuffers(pDev)) {
		return FALSE;
	}

	if (!CreateShaders(pDev)) {
		return FALSE;
	}

	return TRUE;
}



//--------------------------------------------------------------------------------------
VOID CSkyPlane::Draw(LPDIRECT3DDEVICE9 pDev)
{
	if (m_pIB == NULL || m_pVB == NULL || m_pDecl == NULL) {
		return;
	}
	if ( !SetShaders( pDev ) ) {
		return;
	}

	// Set shader constants 
	if (m_pSceneParam != NULL) {

		if (m_pSceneParam->m_pCamera != NULL) {
			// transform screen position to world space
			D3DXMATRIX mC2W;
			D3DXMatrixInverse( &mC2W, NULL, m_pSceneParam->m_pCamera->GetWorld2ProjMatrix() );
			SetVSMatrix( pDev, VS_CONST_C2W, &mC2W );

			// view position
			SetVSValue( pDev, VS_CONST_EYE, m_pSceneParam->m_pCamera->GetEyePt(), sizeof(FLOAT)*3 );
			SetPSValue( pDev, PS_CONST_EYE, m_pSceneParam->m_pCamera->GetEyePt(), sizeof(FLOAT)*3 );
		}

		// Light 
		SetVSValue( pDev, VS_CONST_LITDIR, m_pSceneParam->m_vLightDir, sizeof(FLOAT)*3 );
		SetPSValue( pDev, PS_CONST_LITDIR, m_pSceneParam->m_vLightDir, sizeof(FLOAT)*3 );

		// Scattering parameter
		SScatteringShaderParameters param;
		m_pSceneParam->GetShaderParam( param );
		SetPSValue( pDev, PS_CONST_SCATTERING, &param, sizeof(SScatteringShaderParameters) );
	}

	pDev->SetIndices( m_pIB );
	pDev->SetStreamSource( 0, m_pVB, 0, sizeof(S_VERTEX) );
	pDev->SetVertexDeclaration( m_pDecl );

	pDev->DrawIndexedPrimitive(D3DPT_TRIANGLESTRIP, 0, 0, NUM_VERTICES, 0, NUM_TRIANGLES);
}



//--------------------------------------------------------------------------------------
// CSkyPlane::CreateBuffers
//     Create vertex and index buffer
//--------------------------------------------------------------------------------------
BOOL CSkyPlane::CreateBuffers(LPDIRECT3DDEVICE9 pDev)
{
	// create vertex buffer 
	// This sample uses a grid for rendering sky.   
	HRESULT hr = pDev->CreateVertexBuffer( sizeof(S_VERTEX)*NUM_VERTICES, 
		D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &m_pVB, NULL );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	S_VERTEX* pV = NULL;
	hr = m_pVB->Lock( 0, 0, (VOID**)&pV, 0 );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	for (UINT i = 0; i <= DIV_Y; ++i) {
		for (UINT j = 0; j <= DIV_X; ++j) {
			FLOAT fX = 1.0f - j/(FLOAT)(DIV_X);
			FLOAT fY = 1.0f - i/(FLOAT)(DIV_Y);
			pV->vPos = D3DXVECTOR4( fX*2.0f-1.0f, fY*2.0f-1.0f, 1.0f, 1.0f );
			++pV;
		}
	}
	m_pVB->Unlock();

	// create index buffer
	hr = pDev->CreateIndexBuffer( sizeof(USHORT)*NUM_INDICES, 
		D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &m_pIB, NULL );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	USHORT* pI = NULL;
	hr = m_pIB->Lock( 0, 0, (VOID**)&pI, 0 );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	for (USHORT i = 0; i < DIV_Y; ++i) {
		for (USHORT j = 0; j <= DIV_X; ++j) {
			(*pI) = i*(DIV_X+1) + j;
			++pI;
			(*pI) = (i+1)*(DIV_X+1) + j;
			++pI;
		}
		if (i+1 < DIV_Y) {
			(*pI) = (i+1)*(DIV_X+1) + DIV_X;
			++pI;
			(*pI) = (i+1)*(DIV_X+1);
			++pI;
		}
	}
	m_pIB->Unlock();

	return TRUE;
}



//--------------------------------------------------------------------------------------
// CSkyPlane::CreateShaders
//--------------------------------------------------------------------------------------
BOOL CSkyPlane::CreateShaders(LPDIRECT3DDEVICE9 pDev)
{
#include "../shader/header/SkyPlane.vsh"
#include "../shader/header/SkyPlane.psh"

	static const char* s_lpchVSConst[] = {
		"mC2W",
		"vEye",
		"litDir",
	};
	C_ASSERT( sizeof(s_lpchVSConst)/sizeof(s_lpchVSConst[0]) == VS_CONST_NUM);

	static const char* s_lpchPSConst[] = {
		"vEye",
		"litDir",
		"scat",
	};
	C_ASSERT( sizeof(s_lpchPSConst)/sizeof(s_lpchPSConst[0]) == PS_CONST_NUM);

	SShaderInitializeParameter initParams = {
		(const DWORD*)g_vsSkyPlane,
		(const DWORD*)g_psSkyPlane,
		s_lpchVSConst,
		s_lpchPSConst,
		VS_CONST_NUM,
		PS_CONST_NUM,
	};

	return CShader::Create( pDev, initParams );
}


