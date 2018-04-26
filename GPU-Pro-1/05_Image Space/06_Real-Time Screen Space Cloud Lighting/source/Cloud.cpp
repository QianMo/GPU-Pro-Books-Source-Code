//--------------------------------------------------------------------------------------
// Cloud.cpp
//
// Implementation of the classes to light and render clouds.
// This file is the core of the technique.
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "Cloud.h"




//--------------------------------------------------------------------------------------
CCloudBlur::CCloudBlur()
: m_pVB(NULL)
, m_pDecl(NULL)
, m_pSceneParam(NULL)
{
}

//--------------------------------------------------------------------------------------
CCloudBlur::~CCloudBlur()
{
	Delete();
}

//--------------------------------------------------------------------------------------
VOID CCloudBlur::Delete()
{
	if (m_pVB != NULL) {
		m_pVB->Release();
		m_pVB = NULL;
	}
	if (m_pDecl != NULL) {
		m_pDecl->Release();
		m_pDecl = NULL;
	}

	CShader::Delete();
}
//--------------------------------------------------------------------------------------
BOOL CCloudBlur::Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam)
{
	Delete();

	HRESULT hr = pDev->CreateVertexBuffer( sizeof(S_VERTEX)*4, D3DUSAGE_WRITEONLY, 
		0, D3DPOOL_DEFAULT, &m_pVB, NULL );
	if ( FAILED(hr) ) {
		return FALSE;
	}


	S_VERTEX* pV = NULL;
	hr = m_pVB->Lock( 0, 0, (VOID**)&pV, 0 );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	static const S_VERTEX s_vertices[] = {
		{ {  1.0f,  1.0f, 1.0f, 1.0f }, { 1.0f, 0.0f } },
		{ {  1.0f, -1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f } },
		{ { -1.0f,  1.0f, 1.0f, 1.0f }, { 0.0f, 0.0f } },
		{ { -1.0f, -1.0f, 1.0f, 1.0f }, { 0.0f, 1.0f } },
	};
	memcpy( pV, s_vertices, sizeof(s_vertices) );
	m_pVB->Unlock();


	static const D3DVERTEXELEMENT9 s_elements[] = {
		{ 0,  0,  D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, 16,  D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		D3DDECL_END()
	};
	hr = pDev->CreateVertexDeclaration( s_elements, &m_pDecl );
	if ( FAILED(hr) ) {
		return FALSE;
	}

	if (!CreateShaders( pDev )) {
		return FALSE;
	}

	m_pSceneParam = pSceneParam;
	return TRUE;
}





//--------------------------------------------------------------------------------------
VOID CCloudBlur::SetShaderConstant(LPDIRECT3DDEVICE9 pDev, LPDIRECT3DTEXTURE9 pTex)
{
	if ( pTex != NULL ) {
		// offset parameter to sample center of texels.
		D3DSURFACE_DESC desc;
		pTex->GetLevelDesc( 0, &desc );
		D3DXVECTOR2 v( 0.5f / (FLOAT)desc.Width, 0.5f / (FLOAT)desc.Height );
		SetVSValue( pDev, VS_CONST_PIXELSIZE, &v, sizeof(D3DXVECTOR2) );
	}

	if ( m_pSceneParam != NULL && m_pSceneParam->m_pCamera != NULL ) {
		// view position
		SetPSValue( pDev, PS_CONST_EYE, m_pSceneParam->m_pCamera->GetEyePt(), sizeof(FLOAT)*3 );

		// transform screen position to world space
		D3DXMATRIX mC2W;
		D3DXMatrixInverse( &mC2W, NULL, m_pSceneParam->m_pCamera->GetWorld2ProjMatrix() );
		SetVSMatrix( pDev, VS_CONST_C2W, &mC2W );

		// Directional Light in projection space.
		D3DXVECTOR4 vLit( m_pSceneParam->m_vLightDir.x, m_pSceneParam->m_vLightDir.y, m_pSceneParam->m_vLightDir.z, 0.0f );
		if ( vLit.y > 0.0f ) {
			// assuming light direction is horizontal when sunset or sunrise.
			// otherwise, shadow of clouds converges at a point on the screen opposite to the light.
			vLit.y = 0.0f;
			D3DXVec4Normalize( &vLit, &vLit );
		}
		D3DXVECTOR4 vProjPos;
		D3DXVec4Transform( &vProjPos, &vLit, m_pSceneParam->m_pCamera->GetWorld2ProjMatrix() );

		// blur vector = vBlurDir.xy * uv + vBlurDir.zw 
		D3DXVECTOR4 vBlurDir;
		static const FLOAT EPSIRON = 0.000001f;
		if ( ( fabsf(vProjPos.w) < EPSIRON )|| ( fabsf(vProjPos.z) < EPSIRON ) ) {
			// if dot( litdir, ray ) == 0.0f : directional.
			// light is stil directional in projection space.
			vProjPos.w = vProjPos.z = 0.0f;
			D3DXVec4Normalize( &vProjPos, &vProjPos );
			vProjPos.y = -vProjPos.y;
			// directional blur
			vBlurDir = D3DXVECTOR4( 0.0f, 0.0f, -vProjPos.x, -vProjPos.y );			
		}
		else {
			// otherwise : point blur.
			// light direction is a position in projection space.
			if ( 0.0f < vProjPos.w ) {
				// transform screen position to texture coordinate
				D3DXVec4Scale( &vProjPos, &vProjPos, 1.0f/vProjPos.w );
				vProjPos.x =  0.5f * vProjPos.x + 0.5f; //  
				vProjPos.y = -0.5f * vProjPos.y + 0.5f; // upside down 
				vBlurDir = D3DXVECTOR4( 1.0f, 1.0f, -vProjPos.x, -vProjPos.y );
			}
			else {
				// transform screen position to texture coordinate
				D3DXVec4Scale( &vProjPos, &vProjPos, 1.0f/vProjPos.w );
				vProjPos.x =  0.5f * vProjPos.x + 0.5f; //  
				vProjPos.y = -0.5f * vProjPos.y + 0.5f; // upside down 
				// invert vector if light comes from behind the camera.
				vBlurDir = D3DXVECTOR4( -1.0f, -1.0f, vProjPos.x, vProjPos.y );
			}
		}
		SetVSValue( pDev, VS_CONST_OFFSET, &vBlurDir, sizeof(D3DXVECTOR4) );
	}

	if (m_pSceneParam != NULL) {
		// parameter to scale down blur vector acoording to the distance from the view position.
		SScatteringShaderParameters param;
		m_pSceneParam->GetShaderParam( param );

		D3DXVECTOR3 v( param.vESun.w, param.vSum.w, m_pSceneParam->m_fAtomosHeight );
		SetPSValue( pDev, PS_CONST_DISTANCE, &v, sizeof(D3DXVECTOR3) );
	}

	// maximum length of blur vector in texture space.
	FLOAT fMaxMove = 0.1f/(FLOAT)16;
	D3DXVECTOR2 vInvMax( 1.0f/fMaxMove, 1.0f/fMaxMove );
	SetPSValue( pDev, PS_CONST_MAX, &vInvMax, sizeof(D3DXVECTOR2) );

	// fall off parameter of weights.
	D3DXVECTOR4 vFallOff( -5000.0f, -1.5f, -1.5f, -1000.0f );
	SetPSValue( pDev, PS_CONST_FALLOFF, &vFallOff, sizeof(D3DXVECTOR4) );
}


//--------------------------------------------------------------------------------------
// CCloudBlur::Blur
//  Blur the indicated texture pTex 
//--------------------------------------------------------------------------------------
VOID CCloudBlur::Blur(LPDIRECT3DDEVICE9 pDev, LPDIRECT3DTEXTURE9 pTex)
{
	if (m_pVB == NULL || m_pDecl == NULL) {
		return;
	}
	if ( !SetShaders( pDev ) ) {
		return;
	}

	pDev->SetTexture( 0, pTex );
	pDev->SetSamplerState( 0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP );
	pDev->SetSamplerState( 0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP );
	SetShaderConstant( pDev, pTex );

	pDev->SetVertexDeclaration( m_pDecl );
	pDev->SetStreamSource( 0, m_pVB, 0, sizeof(S_VERTEX) );

	pDev->DrawPrimitive(D3DPT_TRIANGLESTRIP, 0, 2 );
}

//--------------------------------------------------------------------------------------
BOOL CCloudBlur::CreateShaders(LPDIRECT3DDEVICE9 pDev)
{
#include "../shader/header/CloudBlur.vsh"
#include "../shader/header/CloudBlur.psh"

	static const char* s_lpchVSConst[] = {
		"mC2W",
		"vPix",
		"vOff",
	};
	C_ASSERT( sizeof(s_lpchVSConst)/sizeof(s_lpchVSConst[0]) == VS_CONST_NUM);

	static const char* s_lpchPSConst[] = {
		"vParam",
		"vFallOff",
		"invMax",
		"vEye",
	};
	C_ASSERT( sizeof(s_lpchPSConst)/sizeof(s_lpchPSConst[0]) == PS_CONST_NUM);

	SShaderInitializeParameter param = {
		(const DWORD*)g_vsCloudBlur, 
		(const DWORD*)g_psCloudBlur,
		s_lpchVSConst,
		s_lpchPSConst,
		VS_CONST_NUM,
		PS_CONST_NUM,
	};

	return CShader::Create( pDev, param );
}



//--------------------------------------------------------------------------------------
// CCloudPlane 
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CCloudPlane::CCloudPlane()
: m_pVB(NULL)
, m_pIB(NULL)
, m_pDecl(NULL)
, m_pSceneParam(NULL)
, m_pDensityMap(NULL)
, m_pBlurredMap(NULL)
{
}

//--------------------------------------------------------------------------------------
CCloudPlane::~CCloudPlane()
{
	Delete();
}


//--------------------------------------------------------------------------------------
VOID CCloudPlane::Delete()
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
	if (m_pDensityMap != NULL) {
		m_pDensityMap->Release();
		m_pDensityMap = NULL;
	}
	if (m_pBlurredMap != NULL) {
		m_pBlurredMap->Release();
		m_pBlurredMap = NULL;
	}

	CShader::Delete();
}

//--------------------------------------------------------------------------------------
BOOL CCloudPlane::Create(LPDIRECT3DDEVICE9 pDev, 
						const SSceneParamter* pSceneParam, 
						LPDIRECT3DTEXTURE9 pDensityMap, 
						LPDIRECT3DTEXTURE9 pBlurredMap)
{
	Delete();
	
	m_pSceneParam = pSceneParam;

	// create index and vertex buffer.
	if (!CreateBuffers(pDev)) {
		return FALSE;
	}

	// Create vertex declaration
	static const D3DVERTEXELEMENT9 s_elements[] = {
		{ 0,  0,  D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, 16,  D3DDECLTYPE_FLOAT2, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		D3DDECL_END()
	};
	HRESULT hr = pDev->CreateVertexDeclaration( s_elements, &m_pDecl );
	if ( FAILED(hr) ) {
		return FALSE;
	}

	// Create shaders
	if (!CreateShaders(pDev)) {
		return FALSE;
	}

	// Set textures 
	if (pDensityMap != NULL) {
		pDensityMap->AddRef();
	}
	if (m_pDensityMap != NULL) {
		m_pDensityMap->Release();
	}
	m_pDensityMap = pDensityMap;

	if (pBlurredMap != NULL) {
		pBlurredMap->AddRef();
	}
	if (m_pBlurredMap != NULL) {
		m_pBlurredMap->Release();
	}
	m_pBlurredMap = pBlurredMap;

	return TRUE;
}

//--------------------------------------------------------------------------------------
BOOL CCloudPlane::CreateBuffers(LPDIRECT3DDEVICE9 pDev)
{
	// create vertex buffer
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
	FLOAT fDepth = 0.99999f;
	for (UINT i = 0; i <= DIV_Y; ++i) {
		for (UINT j = 0; j <= DIV_X; ++j) {
			FLOAT fX = 1.0f - j/(FLOAT)(DIV_X);
			FLOAT fY = i/(FLOAT)(DIV_Y);
			pV->vPos = D3DXVECTOR4( fX*2.0f-1.0f, -(fY*2.0f-1.0f), fDepth, 1.0f );
			pV->vTex[0] = fX;
			pV->vTex[1] = fY;
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
BOOL CCloudPlane::CreateShaders(LPDIRECT3DDEVICE9 pDev)
{
#include "../shader/header/CloudPlane.vsh"
#include "../shader/header/CloudPlane.psh"

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
		"vDistance",
		"cLit",
		"cAmb",
	};
	C_ASSERT( sizeof(s_lpchPSConst)/sizeof(s_lpchPSConst[0]) == PS_CONST_NUM);

	SShaderInitializeParameter initParams = {
		(const DWORD*)g_vsCloudPlane,
		(const DWORD*)g_psCloudPlane,
		s_lpchVSConst,
		s_lpchPSConst,
		VS_CONST_NUM,
		PS_CONST_NUM,
	};

	return CShader::Create( pDev, initParams );
}


//--------------------------------------------------------------------------------------
VOID CCloudPlane::Draw(LPDIRECT3DDEVICE9 pDev)
{
	if (m_pIB == NULL || m_pVB == NULL || m_pDecl == NULL || m_pDensityMap == NULL || m_pBlurredMap == NULL) {
		return;
	}
	if ( !SetShaders( pDev ) ) {
		return;
	}

	SetShaderConstant( pDev );

	// set textures
	pDev->SetTexture( 0, m_pDensityMap );
	pDev->SetSamplerState( 0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP );
	pDev->SetSamplerState( 0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP );
	pDev->SetTexture( 1, m_pBlurredMap );
	pDev->SetSamplerState( 1, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP );
	pDev->SetSamplerState( 1, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP );

	pDev->SetIndices( m_pIB );
	pDev->SetStreamSource( 0, m_pVB, 0, sizeof(S_VERTEX) );
	pDev->SetVertexDeclaration( m_pDecl );

	pDev->DrawIndexedPrimitive(D3DPT_TRIANGLESTRIP, 0, 0, NUM_VERTICES, 0, NUM_TRIANGLES);

}


//--------------------------------------------------------------------------------------
VOID CCloudPlane::SetShaderConstant(LPDIRECT3DDEVICE9 pDev)
{
	if (m_pSceneParam != NULL) {
		// Camera
		if (m_pSceneParam->m_pCamera != NULL) {
			// transform screen position to world
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
		SetPSValue( pDev, PS_CONST_LIGHT, &m_pSceneParam->m_vLightColor, sizeof(FLOAT)*3 );
		SetPSValue( pDev, PS_CONST_AMBIENT, &m_pSceneParam->m_vAmbientLight, sizeof(FLOAT)*3 );

		// scattering parameter
		SScatteringShaderParameters param;
		m_pSceneParam->GetShaderParam( param );
		SetPSValue( pDev, PS_CONST_SCATTERING, &param, sizeof(SScatteringShaderParameters) );

		// parameter to compute distance of cloud.
		D3DXVECTOR2 v;
		m_pSceneParam->GetCloudDistance( v );
		SetPSValue( pDev, PS_CONST_DISTANCE, &v, sizeof(FLOAT)*2 );
	}


}





//--------------------------------------------------------------------------------------
// CCloud
//  Interface object of clouds.
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
CCloud::CCloud()
: m_pSceneParam(NULL)
, m_pDensityMap(NULL)
, m_pBlurredMap(NULL)
, m_pShadowMap(NULL)
{
}


//--------------------------------------------------------------------------------------
CCloud::~CCloud()
{
	Delete();
}


//--------------------------------------------------------------------------------------
VOID CCloud::Delete()
{
	if (m_pDensityMap != NULL) {
		m_pDensityMap->Release();
		m_pDensityMap = NULL;
	}
	if (m_pBlurredMap != NULL) {
		m_pBlurredMap->Release();
		m_pBlurredMap = NULL;
	}
	if (m_pShadowMap != NULL) {
		m_pShadowMap->Release();
		m_pShadowMap = NULL;
	}

	m_finalCloud.Delete();
	m_grid.Delete();
	m_blur.Delete();
	m_densityShader.Delete();
	m_shadowShader.Delete();
}

//--------------------------------------------------------------------------------------
BOOL CCloud::Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam)
{
	m_pSceneParam = pSceneParam;
	
	// create render targets of density map and blurred density map.
	D3DVIEWPORT9 viewport;
	HRESULT hr = pDev->GetViewport(&viewport);
	if (FAILED(hr)) {
		return FALSE;
	}
	hr = pDev->CreateTexture( viewport.Width/2, viewport.Height/2, 1, 
		D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &m_pDensityMap, NULL );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	hr = pDev->CreateTexture( viewport.Width/2, viewport.Height/2, 1, 
		D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &m_pBlurredMap, NULL );
	if ( FAILED(hr) ) {
		return FALSE;
	}

	// create render targets of cloud shadowmap 
	UINT nWidth = max( 256, viewport.Width );
	UINT nHeight = max( 256, viewport.Height );
	hr = pDev->CreateTexture( nWidth, nHeight, 1, 
		D3DUSAGE_RENDERTARGET, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT, &m_pShadowMap, NULL );
	if ( FAILED(hr) ) {
		return FALSE;
	}


	if (!m_grid.Create(pDev)) {
		return FALSE;
	}
	if (!m_densityShader.Create(pDev, pSceneParam)) {
		return FALSE;
	}
	if (!m_shadowShader.Create(pDev, pSceneParam)) {
		return FALSE;
	}
	if (!m_blur.Create(pDev, pSceneParam)) {
		return FALSE;
	}
	if (!m_finalCloud.Create(pDev, pSceneParam, m_pDensityMap, m_pBlurredMap)) {
		return FALSE;
	}


	return TRUE;
}



//--------------------------------------------------------------------------------------
// Animate clouds and compute shadowmap volume.
//--------------------------------------------------------------------------------------
VOID CCloud::Update(FLOAT dt, const CSceneCamera* pCamera, const SBoundingBox& bbGround)
{
	// animate uv 
	m_grid.Update( dt, pCamera );

	// compute transform matrix for shadowmap
	m_shadowShader.Update( pCamera, &bbGround, &m_grid.GetBoundingBox() );
}


//--------------------------------------------------------------------------------------
// Render clouds into redertargets before scene rendering 
//  Cloud shadowmap, densitymap are rendered and then the density map is blurred.
//--------------------------------------------------------------------------------------
VOID CCloud::PrepareCloudTextures(LPDIRECT3DDEVICE9 pDev)
{
	// preserve current render target 
	HRESULT hr;
	LPDIRECT3DSURFACE9 pCurrentSurface;
	hr = pDev->GetRenderTarget( 0, &pCurrentSurface );
	if ( FAILED(hr) ) {
		return;
	}

	// Setup render states.
	// All passes in this function do not require a depth buffer and alpha blending 
	//  because there is no multiple clouds in this demo.
	pDev->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
	pDev->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
	pDev->SetRenderState( D3DRS_ZWRITEENABLE, FALSE );
	pDev->SetRenderState( D3DRS_ZENABLE, FALSE );

	// Pass 1 : Render clouds to a shadow map 
	if ( SetRenderTarget( pDev, m_pShadowMap ) ) {
		// Clouds are always far away so shadowmap of clouds does not have to have depth.
		// Only transparency is stored to the shadowmap. 
		pDev->Clear( 0, NULL, D3DCLEAR_TARGET, 0xFFFFFF, 1.0f, 0 );
		if (m_shadowShader.Begin(pDev, &m_grid)) {
			// Since the cloud grid is viewed from outside, reverse cullmode.
			pDev->SetRenderState( D3DRS_CULLMODE, D3DCULL_CW );
			m_grid.Draw( pDev );
			m_shadowShader.End();
			// restore
			pDev->SetRenderState( D3DRS_CULLMODE, D3DCULL_CCW );
		}

		// Pass 2 : Render cloud density 
		if ( SetRenderTarget( pDev, m_pDensityMap ) ) {
			pDev->Clear( 0, NULL, D3DCLEAR_TARGET, 0, 1.0f, 0 );
			if (m_densityShader.Begin(pDev, &m_grid)) {
				m_grid.Draw( pDev );
				m_densityShader.End();
			}

			// Pass 3 : Blur the density map
			if ( SetRenderTarget( pDev, m_pBlurredMap ) ) {
				m_blur.Blur( pDev, m_pDensityMap );
			}
		}
	}

	// restore render target and render states.
	pDev->SetRenderState( D3DRS_ZWRITEENABLE, TRUE );
	pDev->SetRenderState( D3DRS_ZENABLE, TRUE );
	pDev->SetRenderTarget( 0, pCurrentSurface );

	pCurrentSurface->Release();
}

//--------------------------------------------------------------------------------------
// Render final clouds with daylight scattering 
//--------------------------------------------------------------------------------------
VOID CCloud::DrawFinalQuad(LPDIRECT3DDEVICE9 pDev)
{
	m_finalCloud.Draw( pDev );
}


//--------------------------------------------------------------------------------------
BOOL CCloud::SetRenderTarget(LPDIRECT3DDEVICE9 pDev, LPDIRECT3DTEXTURE9 pTex) 
{
	if ( pTex == NULL ) {
		return FALSE;
	}
	LPDIRECT3DSURFACE9 pSurface;
	HRESULT hr = pTex->GetSurfaceLevel( 0, &pSurface );
	if ( FAILED(hr) ) {
		return FALSE;
	}
	hr = pDev->SetRenderTarget( 0, pSurface );
	pSurface->Release();
	return SUCCEEDED(hr);
}



