//--------------------------------------------------------------------------------------
// CloudGrid.cpp
// 
// Implementation of the classes to render cloud grid to density map and shadow map.
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "CloudGrid.h"




//--------------------------------------------------------------------------------------
CCloudGrid::CCloudGrid()
: m_pVB(NULL)
, m_pIB(NULL)
, m_pDecl(NULL)
, m_pCloudTex(NULL)
{
	m_fCloudCover = 0.5f;
	m_vVelocity = D3DXVECTOR2( 0.01f, 0.01f );
	m_vOffset = D3DXVECTOR2( 0.0f, 0.0f );
}

//--------------------------------------------------------------------------------------
CCloudGrid::~CCloudGrid()
{
	Delete();
}




//--------------------------------------------------------------------------------------
VOID CCloudGrid::Delete()
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

	if (m_pCloudTex != NULL) {
		m_pCloudTex->Release();
		m_pCloudTex = NULL;
	}
}


//--------------------------------------------------------------------------------------
BOOL CCloudGrid::Create(LPDIRECT3DDEVICE9 pDev)
{
	USHORT nCellNumX = 16;
	USHORT nCellNumZ = 16;

	// Create Buffers
	if (!CreateVertexBuffer(pDev, nCellNumX, nCellNumZ)) {
		return FALSE;
	}
	if (!CreateIndexBuffer( pDev, nCellNumX, nCellNumZ )) {
		return FALSE;
	}

	// Create vertex declaration
	static const D3DVERTEXELEMENT9 s_elements[] = {
		{ 0,  0,  D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		D3DDECL_END()
	};
	HRESULT hr = pDev->CreateVertexDeclaration( s_elements, &m_pDecl );
	if (FAILED(hr)) {
		return FALSE;
	}

	// create texture
	hr = D3DXCreateTextureFromFile( pDev, L"res/Cloud.bmp", &m_pCloudTex );
	if ( FAILED(hr) ) {
		return FALSE;
	}

	return TRUE;
}
	

//--------------------------------------------------------------------------------------
BOOL CCloudGrid::CreateVertexBuffer(LPDIRECT3DDEVICE9 pDev, USHORT nCellNumX, USHORT nCellNumZ)
{
	m_vStartXZ = D3DXVECTOR2( -20000.0f, -20000.0f );
	m_vCellSizeXZ = D3DXVECTOR2( (80000.0f)/(FLOAT)nCellNumX, (80000.0f)/(FLOAT)nCellNumZ );

	m_nVertices = (nCellNumX + 1)*(nCellNumZ + 1);

	HRESULT hr;
	hr = pDev->CreateVertexBuffer(sizeof(S_VERTEX)*m_nVertices, D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &m_pVB, NULL);
	if ( FAILED(hr) ) {
		return FALSE;
	}
	S_VERTEX* pVertices = NULL;
	hr = m_pVB->Lock( 0, 0, (VOID**)&pVertices, 0 );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	// The vertex buffer includes only x and z index in the grid and they are scaled in the vertex shader.
	// The height y is computed in the vertex shader using horizontal distance from view point.
	FLOAT fScaleX = 1.0f/(FLOAT)(nCellNumX+1);
	FLOAT fScaleZ = 1.0f/(FLOAT)(nCellNumZ+1);
	for (UINT z = 0; z < nCellNumZ+1; ++z) {
		for (UINT x = 0; x < nCellNumX+1; ++x) {
			pVertices->x = (FLOAT)x;     // x index
			pVertices->z = (FLOAT)z;     // z index
			pVertices->u = x * fScaleX;  // texture coordinate u
			pVertices->v = z * fScaleZ;  // texture coordinate v
			++pVertices;
		}
	}
	m_pVB->Unlock();

	// Initialize x and z components of the bounding box 
	// MaxY is changed at every frame according to the eye height.
	m_bound.vMin = D3DXVECTOR3( m_vStartXZ.x, 0.0f, m_vStartXZ.y );
	D3DXVECTOR2 vEndXZ( m_vCellSizeXZ.x * nCellNumX, m_vCellSizeXZ.y * nCellNumZ );
	D3DXVec2Add( &vEndXZ, &vEndXZ, &m_vStartXZ );
	m_bound.vMax = D3DXVECTOR3( vEndXZ.x, 0.0f, vEndXZ.y  );

	return TRUE;
}

//--------------------------------------------------------------------------------------
BOOL CCloudGrid::CreateIndexBuffer(LPDIRECT3DDEVICE9 pDev, USHORT nCellNumX, USHORT nCellNumZ)
{
	UINT nNumIndex = (nCellNumX+2) * 2 * nCellNumZ - 2;
	
	HRESULT hr = pDev->CreateIndexBuffer( sizeof(USHORT)*nNumIndex, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &m_pIB, NULL);
	if (FAILED(hr)) {
		return FALSE;
	}

	USHORT* pIndex = NULL;
	hr = m_pIB->Lock( 0, 0, (VOID**)&pIndex, 0 );
	if (FAILED(hr)) {
		return FALSE;
	}

	USHORT nVertexNumX = (USHORT)(nCellNumX+1);
	for ( SHORT x = (SHORT)nCellNumX; 0 <= x; --x ) {
		*pIndex++ = x;
		*pIndex++ = nVertexNumX + x;
	}
	for ( USHORT z = 1; z < (SHORT)nCellNumZ; ++z ) {
		*pIndex++ = z*nVertexNumX;
		*pIndex++ = z*nVertexNumX + nCellNumX;
		for ( SHORT x = nCellNumX; 0 <= x; --x ) {
			*pIndex++ = z*nVertexNumX + x;
			*pIndex++ = (z+1)*nVertexNumX + x;
		}
	}


	m_pIB->Unlock();
	m_nTriangles = nNumIndex-2;

	return TRUE;
}


//--------------------------------------------------------------------------------------
// Update cloud position.
//  The cloud is animated by scrolling uv 
//--------------------------------------------------------------------------------------
VOID CCloudGrid::Update(FLOAT dt, const CSceneCamera* pCamera)
{
	// increment uv scrolling parameters 
	D3DXVECTOR2 vec;
	D3DXVec2Scale( &vec, &m_vVelocity, dt ); 
	D3DXVec2Add( &m_vOffset, &m_vOffset, &vec );

	// Adjust the height so that clouds are always above.
	// cloud height = m_fDefaultHeight + m_fFallOffHeight * squaredistance_in_horizontal
	FLOAT fRange = 0.5f * pCamera->GetFarClip();
	FLOAT fHeight = fRange * 0.12f;
	m_fDefaultHeight = fHeight + pCamera->GetEyePt()->y;
	m_fFallOffHeight  = - ( 0.1f / fRange ) * (  pCamera->GetEyePt()->y / fHeight + 1.0f );

	// Update Bounding Box
	m_bound.vMax.y = m_fDefaultHeight;

}


//--------------------------------------------------------------------------------------
VOID CCloudGrid::Draw(LPDIRECT3DDEVICE9 pDev)
{
	if (m_pVB == NULL || m_pIB == NULL || m_pDecl == NULL) {
		return;
	}

	pDev->SetTexture( 0, m_pCloudTex );
	pDev->SetSamplerState( 0, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP );
	pDev->SetSamplerState( 0, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP );

	pDev->SetVertexDeclaration( m_pDecl );
	pDev->SetStreamSource( 0, m_pVB, 0, sizeof(S_VERTEX) );
	pDev->SetIndices( m_pIB );

	pDev->DrawIndexedPrimitive(D3DPT_TRIANGLESTRIP, 0, 0, m_nVertices, 0, m_nTriangles);
}




//--------------------------------------------------------------------------------------
// CRenderDensityShader
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
CRenderDensityShader::CRenderDensityShader()
: m_pSceneParam(NULL)
{
}

//--------------------------------------------------------------------------------------
CRenderDensityShader::~CRenderDensityShader()
{
	Delete();
}



//--------------------------------------------------------------------------------------
BOOL CRenderDensityShader::Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam)
{
	m_pSceneParam = pSceneParam;
	if ( pSceneParam == NULL ) {
		return FALSE;
	}

#include "../shader/header/CloudGrid.vsh"
#include "../shader/header/CloudGrid.psh"

	static const char* s_lpchVSConst[] = {
		"vXZParam",
		"vHeight",
		"mW2C",
		"vEye",
		"vUVParam",
	};
	C_ASSERT( sizeof(s_lpchVSConst)/sizeof(s_lpchVSConst[0]) == VS_CONST_NUM);

	static const char* s_lpchPSConst[] = {
		"fCloudCover",
	};
	C_ASSERT( sizeof(s_lpchPSConst)/sizeof(s_lpchPSConst[0]) == PS_CONST_NUM);

	SShaderInitializeParameter param = {
		(const DWORD*)g_vsCloudGrid, 
		(const DWORD*)g_psCloudGrid,
		s_lpchVSConst,
		s_lpchPSConst,
		VS_CONST_NUM,
		PS_CONST_NUM,
	};

	return CShader::Create( pDev, param );
}



//--------------------------------------------------------------------------------------
BOOL CRenderDensityShader::Begin(LPDIRECT3DDEVICE9 pDev, CCloudGrid* pCloud)
{
	if (!SetShaders( pDev )) {
		return FALSE;
	}

	if (m_pSceneParam != NULL && m_pSceneParam->m_pCamera != NULL) {
		// world to projection transform 
		SetVSMatrix( pDev, VS_CONST_W2C, m_pSceneParam->m_pCamera->GetWorld2ProjMatrix() );
		// view position
		SetVSValue( pDev, VS_CONST_EYE, m_pSceneParam->m_pCamera->GetEyePt(), sizeof(FLOAT)*3 );
	}
	if (pCloud != NULL) {
		CCloudGrid::SVSParam param;
		pCloud->GetVSParam( param );
		// uv scale and offset parameter
		SetVSValue( pDev, VS_CONST_UVPARAM, &param.vUVParam, sizeof(D3DXVECTOR4) );
		// xz position scale and offset parameter
		SetVSValue( pDev, VS_CONST_XZPARAM, &param.vXZParam, sizeof(D3DXVECTOR4) );
		// height parameters
		SetVSValue( pDev, VS_CONST_HEIGHT, &param.vHeight, sizeof(D3DXVECTOR2) );

		// cloud cover
		FLOAT fCloudCover = pCloud->GetCurrentCloudCover();
		SetPSValue( pDev, PS_CONST_COVER, &fCloudCover, sizeof(FLOAT)*1 );
	}

	return TRUE;
}

//--------------------------------------------------------------------------------------
VOID CRenderDensityShader::End()
{
}



//--------------------------------------------------------------------------------------
// CRenderShadowShader
//   Shader for rendering cloud grid to shadowmap
//--------------------------------------------------------------------------------------


//--------------------------------------------------------------------------------------
CRenderShadowShader::CRenderShadowShader()
: m_pSceneParam(NULL)
{
}

//--------------------------------------------------------------------------------------
CRenderShadowShader::~CRenderShadowShader()
{
	Delete();
}


//--------------------------------------------------------------------------------------
BOOL CRenderShadowShader::Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam)
{
	m_pSceneParam = pSceneParam;
	if ( pSceneParam == NULL ) {
		return FALSE;
	}

#include "../shader/header/CloudShadow.vsh"
#include "../shader/header/CloudShadow.psh"

	static const char* s_lpchVSConst[] = {
		"vXZParam",
		"vHeight",
		"mW2C",
		"vEye",
		"vUVParam",
	};
	C_ASSERT( sizeof(s_lpchVSConst)/sizeof(s_lpchVSConst[0]) == VS_CONST_NUM);

	static const char* s_lpchPSConst[] = {
		"fCloudCover",
	};
	C_ASSERT( sizeof(s_lpchPSConst)/sizeof(s_lpchPSConst[0]) == PS_CONST_NUM);

	SShaderInitializeParameter param = {
		(const DWORD*)g_vsCloudShadow, 
		(const DWORD*)g_psCloudShadow,
		s_lpchVSConst,
		s_lpchPSConst,
		VS_CONST_NUM,
		PS_CONST_NUM,
	};

	return CShader::Create( pDev, param );
}


//--------------------------------------------------------------------------------------
// CRenderShadowShader::Update
//   Compute volume of the shadow
//--------------------------------------------------------------------------------------
VOID CRenderShadowShader::Update(const CSceneCamera* pCamera, const SBoundingBox* pGround, const SBoundingBox* pCloud)
{
	if (pCloud == NULL || pGround == NULL || pCamera == NULL || m_pSceneParam == NULL) {
		return;
	}

	// look at the scene from light
	D3DXVECTOR3 vLight( 0.0f, 0.0f, 0.0f );
	D3DXVECTOR3 vUp( 0.0f, 1.0f, 0.0f );
	D3DXVECTOR3 vAt;
	D3DXVec3Add( &vAt, &vLight, &m_pSceneParam->m_vLightDir );
	D3DXMATRIX mW2Light;
	D3DXMatrixLookAtLH( &mW2Light, &vLight, &vAt, &vUp );

	// transform ground and cloud bounding box to the light coordinate
	SBoundingBox bbGroundInView;
	SBoundingBox bbCloudInView;
	pGround->Transform( bbGroundInView, &mW2Light );
	pCloud->Transform( bbCloudInView, &mW2Light );

	// minimize bounding box 
	// The view frustom should be take into account as well.
	SBoundingBox bound;
	D3DXVec3Minimize( &bound.vMax, &bbGroundInView.vMax, &bbCloudInView.vMax );
	D3DXVec3Maximize( &bound.vMin, &bbGroundInView.vMin, &bbCloudInView.vMin );
	bound.vMin.z = bbCloudInView.vMin.z;

	// if there is a valid volume 
	if (bound.vMin.x < bound.vMax.x && bound.vMin.y < bound.vMax.y && bound.vMin.z < bound.vMax.z) {
		D3DXVECTOR3 vCenter;
		D3DXVECTOR3 vDiag;
		bound.Centroid( &vCenter );
		D3DXVec3Subtract( &vDiag, &bound.vMax, &bound.vMin );

		// Move the view position to the center of the bounding box.
		// z is located behined the volume.
		D3DXVECTOR3 vEye( vCenter );
		vEye.z = vCenter.z - 0.5f * vDiag.z;
		D3DXVECTOR3 vMove;
		D3DXVec3Subtract( &vMove, &vLight, &vEye );
		D3DXMATRIX mTrans;
		D3DXMatrixTranslation( &mTrans, vMove.x, vMove.y, vMove.z );

		// Orthogonal projection matrix
		D3DXMATRIX mProj;
		D3DXMatrixOrthoLH( &mProj, vDiag.x, vDiag.y, 0.0f, vDiag.z );

		// Compute world to shadow map projection matrix
		D3DXMatrixMultiply( &m_mW2SProj, &mW2Light, &mTrans );
		D3DXMatrixMultiply( &m_mW2SProj, &m_mW2SProj, &mProj );

		// Compute world to shadowmap texture coordinate matrix
		D3DXMATRIX mProj2Tex(
			0.5f,  0.0f, 0.0f, 0.0f,
			0.0f, -0.5f, 0.0f, 0.0f,
			0.0f,  0.0f, 1.0f, 0.0f,
			0.5f,  0.5f, 0.0f, 1.0f );
		D3DXMatrixMultiply( &m_mW2S, &m_mW2SProj, &mProj2Tex );
	}
}


//--------------------------------------------------------------------------------------
// Setup shaders and shader constants.
//--------------------------------------------------------------------------------------
BOOL CRenderShadowShader::Begin(LPDIRECT3DDEVICE9 pDev, CCloudGrid* pCloud)
{
	if (!SetShaders( pDev )) {
		return FALSE;
	}

	if (m_pSceneParam != NULL && m_pSceneParam->m_pCamera != NULL) {
		// world to projection transform 
		SetVSMatrix( pDev, VS_CONST_W2C, &m_mW2SProj );
		// view position
		SetVSValue( pDev, VS_CONST_EYE, m_pSceneParam->m_pCamera->GetEyePt(), sizeof(FLOAT)*3 );
	}
	if (pCloud != NULL) {
		CCloudGrid::SVSParam param;
		pCloud->GetVSParam( param );		
		// uv scale and offset parameter
		SetVSValue( pDev, VS_CONST_UVPARAM, &param.vUVParam, sizeof(D3DXVECTOR4) );
		// xz position scale and offset parameter
		SetVSValue( pDev, VS_CONST_XZPARAM, &param.vXZParam, sizeof(D3DXVECTOR4) );
		// height parameters
		SetVSValue( pDev, VS_CONST_HEIGHT, &param.vHeight, sizeof(D3DXVECTOR2) );

		// cloud cover
		FLOAT fCloudCover = pCloud->GetCurrentCloudCover();
		SetPSValue( pDev, PS_CONST_COVER, &fCloudCover, sizeof(FLOAT)*1 );
	}

	return TRUE;
}

VOID CRenderShadowShader::End()
{
}


