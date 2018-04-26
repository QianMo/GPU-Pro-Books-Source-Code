//--------------------------------------------------------------------------------------
// Ground.cpp
//
// Implementation of a height map ground class.
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------

#include "DXUT.h"
#include "Ground.h"
#include <wchar.h>
#include <stdio.h>



//--------------------------------------------------------------------------------------
CGround::CGround()
: m_pIB(NULL)
, m_pVB(NULL)
, m_pDecl(NULL)
, m_pfHeight(NULL)
{
	for (UINT i = 0; i < TEX_NUM; ++i) {
		m_pTex[i] = NULL;
	}
}

//--------------------------------------------------------------------------------------
CGround::~CGround()
{
	Delete();
}

//--------------------------------------------------------------------------------------
VOID CGround::Delete()
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

	if (m_pfHeight != NULL) {
		delete [] m_pfHeight;
		m_pfHeight = NULL;
	}

	for (UINT i = 0; i < TEX_NUM; ++i) {
		if (m_pTex[i] != NULL) {
			m_pTex[i]->Release();
			m_pTex[i] = NULL;
		}
	}

	if (m_pCloudShadow != NULL) {
		m_pCloudShadow->Release();
		m_pCloudShadow = NULL;
	}

	m_shader.Delete();
}

//--------------------------------------------------------------------------------------
BOOL CGround::Create(LPDIRECT3DDEVICE9 pDev, 
					 const SSceneParamter* pSceneParam, 
					 const char* ptcHeightmap, 
					 LPDIRECT3DTEXTURE9 pCloudShadowMap, 
					 const D3DXMATRIX* pShadowMatrix)
{
	Delete();


	// Loading heightmap (BITMAP);
	if (!LoadHeightmap( pDev, ptcHeightmap )) {
		return FALSE;
	}

	HRESULT hr;

	// Cerate vertex declaration 
	static const D3DVERTEXELEMENT9 s_elements[] = {
		{ 0,  0,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
		{ 0, 12,  D3DDECLTYPE_FLOAT3, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_NORMAL  , 0 },
		{ 0, 24,  D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
		D3DDECL_END()
	};
	hr = pDev->CreateVertexDeclaration( s_elements, &m_pDecl );
	if (FAILED(hr)) {
		return FALSE;
	}

	// Create textures
	static const TCHAR* ptchTex[] = {
		L"res/GroundBlend.bmp",
		L"res/Ground0.dds",
		L"res/Ground1.dds",
		L"res/Ground2.dds",
		L"res/Ground3.dds",
		L"res/Ground4.dds",
	};
	for (UINT i = 0; i < TEX_NUM; ++i) {
		hr = D3DXCreateTextureFromFile( pDev, ptchTex[i], &m_pTex[i]);
		if ( FAILED(hr) ) {
			return FALSE;
		}
	}

	// Set pointers for shadowmap matrix and texture 	
	m_pW2Shadow = pShadowMatrix;
	if (pCloudShadowMap != NULL) {
		pCloudShadowMap->AddRef();
	}
	if (m_pCloudShadow != NULL) {
		m_pCloudShadow->Release();
	}
	m_pCloudShadow = pCloudShadowMap;

	// Copy pointer of scene parameter.
	m_pSceneParam = pSceneParam;

	// Create shaders
	return CreateShaders( pDev );
}


//--------------------------------------------------------------------------------------
VOID CGround::Draw(LPDIRECT3DDEVICE9 pDev)
{
	if (!m_shader.SetShaders( pDev )) {
		return;
	}
	if (m_pVB == NULL || m_pIB == NULL || m_pDecl == NULL) {
		return;
	}

	SetShaderConstants( pDev );

	pDev->SetVertexDeclaration( m_pDecl );
	pDev->SetStreamSource( 0, m_pVB, 0, sizeof(S_VERTEX) );
	pDev->SetIndices( m_pIB );

	// Set texture and sampler states
	for (UINT i = 0; i < TEX_NUM; ++i) {
		pDev->SetTexture( i, m_pTex[i] );
		pDev->SetSamplerState( i, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP );
		pDev->SetSamplerState( i, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP );
	}
	pDev->SetTexture( TEX_NUM, m_pCloudShadow );
	pDev->SetSamplerState( TEX_NUM, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP );
	pDev->SetSamplerState( TEX_NUM, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP );

	pDev->DrawIndexedPrimitive(D3DPT_TRIANGLESTRIP, 0, 0, m_nVertices, 0, m_nTriangles);
}


//--------------------------------------------------------------------------------------
// CGround::SetShaderConstants
//  set vertex and pixel shader constants
//--------------------------------------------------------------------------------------
VOID CGround::SetShaderConstants(LPDIRECT3DDEVICE9 pDev)
{
	// local to world transform 
	D3DXMATRIX mL2W;
	D3DXMatrixIdentity( &mL2W );
	m_shader.SetVSValue( pDev, VS_CONST_L2W, &mL2W, sizeof(D3DXMATRIX) );

	if (m_pW2Shadow != NULL) {
		// local to shdaowmap texture coodinate
		D3DXMATRIX mL2S;
		D3DXMatrixMultiply( &mL2S, &mL2W, m_pW2Shadow );
		m_shader.SetVSValue( pDev, VS_CONST_L2S, &mL2S, sizeof(D3DXMATRIX) );
	}

	if (m_pSceneParam != NULL) {

		if (m_pSceneParam->m_pCamera != NULL) {
			// transform local coordinate to projection.
			D3DXMATRIX mL2C;
			D3DXMatrixMultiply( &mL2C, &mL2W, m_pSceneParam->m_pCamera->GetWorld2ProjMatrix() );
			m_shader.SetVSValue( pDev, VS_CONST_L2C, &mL2C, sizeof(D3DXMATRIX) );

			// view position
			m_shader.SetPSValue( pDev, PS_CONST_EYE, m_pSceneParam->m_pCamera->GetEyePt(), sizeof(FLOAT)*3 );
		}

		// Set light 
		m_shader.SetPSValue( pDev, PS_CONST_LITDIR, m_pSceneParam->m_vLightDir, sizeof(FLOAT)*3 );
		m_shader.SetPSValue( pDev, PS_CONST_LITCOL, m_pSceneParam->m_vLightColor, sizeof(FLOAT)*3 );		
		m_shader.SetPSValue( pDev, PS_CONST_LITAMB, m_pSceneParam->m_vAmbientLight, sizeof(FLOAT)*3 );

		// Set scattering parameters
		SScatteringShaderParameters param;
		m_pSceneParam->GetShaderParam( param );
		m_shader.SetPSValue( pDev, PS_CONST_SCATTERING, &param, sizeof(SScatteringShaderParameters) );
	}


	// Set material colors
	D3DXVECTOR4 vDiffuse( 1.0f, 1.0f, 1.0f, 1.0f );
	D3DXVECTOR4 vSpecular( 1.0f, 1.0f, 1.0f, 32.0f );
	m_shader.SetPSValue( pDev, PS_CONST_MATERIAL_DIFFUSE, &vDiffuse, sizeof(FLOAT)*4 );
	m_shader.SetPSValue( pDev, PS_CONST_MATERIAL_SPECULAR, &vSpecular, sizeof(FLOAT)*4 );
}

//--------------------------------------------------------------------------------------
// Compute height of the ground by x ans z.
//--------------------------------------------------------------------------------------
FLOAT CGround::GetHeight(FLOAT x, FLOAT z) const
{
	if (x < m_vStartXZ.x || z < m_vStartXZ.y) {
		return 0.0f;
	}
	if (m_vStartXZ.x + m_nCellNumX * m_vCellSizeXZ.x < x || 
		m_vStartXZ.y + m_nCellNumZ * m_vCellSizeXZ.y < z) {
		return 0.0f;
	}

	// compute x,z index and ratio in the cell
	FLOAT fX = (x - m_vStartXZ.x) / m_vCellSizeXZ.x;
	FLOAT fZ = (z - m_vStartXZ.y) / m_vCellSizeXZ.y;
	INT nX = (INT)fX;
	INT nZ = (INT)fZ;
	fX -= nX;
	fZ -= nZ;
	UINT nVertexX = m_nCellNumX+1;
	UINT n = nZ*nVertexX + nX;

	if (fX + fZ <= 1.0f) {
		// upper left triangle in a cell
		return m_pfHeight[n] + (m_pfHeight[n+1] - m_pfHeight[n]) * fX + (m_pfHeight[n+nVertexX] - m_pfHeight[n]) * fZ;
	}
	else {
		// bottom right triangle in a cell
		return m_pfHeight[n+nVertexX+1] + (m_pfHeight[n+1] - m_pfHeight[n+nVertexX+1]) * (1.0f-fZ) + (m_pfHeight[n+nVertexX] - m_pfHeight[n+nVertexX+1]) * (1.0f-fX);
	}

}



//--------------------------------------------------------------------------------------
BOOL CGround::CreateShaders(LPDIRECT3DDEVICE9 pDev)
{
#include "../shader/header/Ground.vsh"
#include "../shader/header/Ground.psh"

	static const char* s_lpchVSConst[] = {
		"mL2C",
		"mL2W",
		"mL2S",
	};
	C_ASSERT( sizeof(s_lpchVSConst)/sizeof(s_lpchVSConst[0]) == VS_CONST_NUM);

	static const char* s_lpchPSConst[] = {
		"vEye",
		"litDir",
		"litCol",
		"litAmb",
		"scat",
		"mDif",
		"mSpc",
	};
	SShaderInitializeParameter param = {
		(const DWORD*)g_vsGround, 
		(const DWORD*)g_psGround,
		s_lpchVSConst,
		s_lpchPSConst,
		VS_CONST_NUM,
		PS_CONST_NUM,
	};

	return m_shader.Create( pDev, param );
}


//--------------------------------------------------------------------------------------
BOOL CGround::LoadHeightmap(LPDIRECT3DDEVICE9 pDev, const char* ptchHeightmap)
{
	USHORT    bfType;              
	struct SFileHeader 
	{
		UINT      bfSize;              
		USHORT    bfReserved1;         
		USHORT    bfReserved2;         
		UINT      bfOffBits;           
	};
	struct SInfoHeader
	{
		UINT      biSize;              
		INT       biWidth;			   
		INT       biHeight;			   
		USHORT    biPlanes;			   
		USHORT    biBitCount;		   
		UINT      biCompression;	   
		UINT      biSizeImage;		   
		INT       biXPelsPerMeter;	   
		INT       biYPelsPerMeter;	   
		UINT      biClrUsed;		   
		UINT      biClrImportant;	   
	};

	SFileHeader fileHeader;
	SInfoHeader infoHeader;

	FILE* pFile = NULL;
	errno_t ret = fopen_s( &pFile, ptchHeightmap, "rb");
	if (ret != 0) {
		return FALSE;
	}
	size_t szRead;
	szRead = fread(&bfType, 1, sizeof(USHORT), pFile);
	if (szRead != sizeof(USHORT)) {
		fclose(pFile);
		return FALSE;
	}
	if (bfType != *((WORD*)"BM")) {
		fclose(pFile);
		return FALSE;
	}
	szRead = fread(&fileHeader.bfSize, 1, sizeof(UINT)*3, pFile);
	if (szRead != sizeof(UINT)*3) {
		fclose(pFile);
		return FALSE;
	}
	szRead = fread(&infoHeader, 1, sizeof(SInfoHeader), pFile);
	if (szRead != sizeof(SInfoHeader)) {
		fclose(pFile);
		return FALSE;
	}
	if (infoHeader.biBitCount != 8) {
		// unsupported.
		fclose(pFile);
		return FALSE;
	}

	// skip palette assuming grey scale bitmap.
	size_t nHeaderSize = sizeof(SFileHeader) + sizeof(USHORT) + sizeof(SInfoHeader);
	size_t szSkip = fileHeader.bfOffBits - nHeaderSize;
	if (0 < szSkip) {
		fseek( pFile, (long)szSkip, SEEK_CUR );
	}

	UINT nImageSize = infoHeader.biBitCount/8*infoHeader.biWidth*infoHeader.biHeight;
	BYTE* pData = new BYTE[nImageSize];
	if (pData == NULL) {
		fclose(pFile);
		return FALSE;
	}
	// load pixel data
	UINT nHeight = infoHeader.biHeight;
	UINT nWidth  = infoHeader.biWidth;
	UINT nPitch = 4*((nWidth+3)/4);
	for ( INT i = nHeight-1; 0 <= i; --i ) {
		szRead = fread( &pData[ i*nWidth ], 1, nWidth, pFile );
		if (szRead != nWidth) {
			assert(FALSE);
		}
		if ( 0 < nPitch - nWidth ) {
			fseek( pFile, nPitch - nWidth, SEEK_CUR );
		}
	}
	fclose(pFile);

	// Create vertex buffer.
	if ( !CreateVertexBuffer(pDev, infoHeader.biWidth, infoHeader.biHeight, pData) ) {
		delete [] pData;
		return FALSE;
	}
	delete [] pData;

	return CreateIndexBuffer(pDev, (USHORT)infoHeader.biWidth, (USHORT)infoHeader.biHeight);
}

//--------------------------------------------------------------------------------------
BOOL CGround::CreateIndexBuffer(LPDIRECT3DDEVICE9 pDev, USHORT nWidth, USHORT nHeight)
{
	UINT nNumIndex = (2*nWidth)*(nHeight-1) + 2*(nHeight-2);
	
	HRESULT hr = pDev->CreateIndexBuffer( sizeof(USHORT)*nNumIndex, D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &m_pIB, NULL);
	if (FAILED(hr)) {
		return FALSE;
	}

	USHORT* pIndex = NULL;
	hr = m_pIB->Lock( 0, 0, (VOID**)&pIndex, 0 );
	if (FAILED(hr)) {
		return FALSE;
	}

	for (USHORT i = 0; i+1 < nHeight; ++i) {
		for (USHORT j = 0; j < nWidth; ++j) {
			*pIndex++ = i*nWidth+j;
			*pIndex++ = (i+1)*nWidth+j;
		}
		if (i+2 < nHeight) {
			*pIndex++ = (i+1)*nWidth+(nWidth-1);
			*pIndex++ = (i+1)*nWidth;
		}
	}


	m_pIB->Unlock();
	m_nTriangles = nNumIndex-2;

	return TRUE;
}

//--------------------------------------------------------------------------------------
BOOL CGround::CreateVertexBuffer(LPDIRECT3DDEVICE9 pDev, UINT nWidth, UINT nHeight, const BYTE* pData)
{
	m_nCellNumX = nWidth-1;
	m_nCellNumZ = nHeight-1;

	UINT nVertices = nWidth*nHeight;
	HRESULT hr = pDev->CreateVertexBuffer(sizeof(S_VERTEX)*nVertices, 
		D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &m_pVB, NULL);
	if (FAILED(hr)) {
		return FALSE;
	}
	S_VERTEX* pV;
	hr = m_pVB->Lock( 0, 0, (VOID**)&pV, 0 );
	if (FAILED(hr)) {
		return FALSE;
	}

	// store height of vertices 
	FLOAT fBottom = 0.0f;
	FLOAT fHeightScale = 10.0f;
	m_pfHeight = new FLOAT[nHeight*nWidth];
	for (UINT i = 0; i < nHeight*nWidth; ++i) {
		m_pfHeight[i] = fBottom + pData[i]*fHeightScale;
	}

	// compute vertex position, texture coordinates and normal vector.
	D3DXVECTOR2 vStart( 0.0f, 0.0f );
	m_vCellSizeXZ = D3DXVECTOR2( 100.0f, 100.0f );

	for (UINT i = 0; i < nHeight; ++i) {
		for (UINT j = 0; j < nWidth; ++j) {
			// grid position
			pV->fPos[0] = vStart.x + m_vCellSizeXZ.x * j;
			pV->fPos[1] = m_pfHeight[i*nWidth+j];
			pV->fPos[2] = vStart.y + m_vCellSizeXZ.y* i;
			// texture coordinate
			pV->fTex[0] = (FLOAT)j;                    // texcoord u for ground textures
			pV->fTex[1] = (FLOAT)i;                    // texcoord v for ground textures
			pV->fTex[2] = (FLOAT)j/(FLOAT)(nWidth-1);  // texcoord u for a blend texture
			pV->fTex[3] = (FLOAT)i/(FLOAT)(nHeight-1); // texcoord v for a blend texture

			// compute normal vector 
			// x
			FLOAT subX0 = 0.0f;
			FLOAT subX1 = 0.0f;
			if (0 < j) {
				subX0 = (m_pfHeight[i*nWidth+j] - m_pfHeight[i*nWidth+j-1]);
			}
			if (j+1 < nWidth) {
				subX1 = (m_pfHeight[i*nWidth+j+1] - m_pfHeight[i*nWidth+j]);
			}
			FLOAT lenX0 = sqrtf( m_vCellSizeXZ.x*m_vCellSizeXZ.x + subX0*subX0 );
			FLOAT lenX1 = sqrtf( m_vCellSizeXZ.x*m_vCellSizeXZ.x + subX1*subX1 );
			FLOAT dxdy = (lenX1 * subX0 + lenX0 * subX1) / (m_vCellSizeXZ.x*(lenX0 + lenX1));
			// z
			FLOAT subZ0 = 0.0f;
			FLOAT subZ1 = 0.0f;
			if (0 < i) {
				subZ0 = (m_pfHeight[i*nWidth+j] - m_pfHeight[(i-1)*nWidth+j]);
			}
			if (i+1 < nHeight) {
				subZ1 = (m_pfHeight[(i+1)*nWidth+j] - m_pfHeight[i*nWidth+j]);
			}
			FLOAT lenZ0 = sqrtf( m_vCellSizeXZ.y*m_vCellSizeXZ.y + subZ0*subZ0 );
			FLOAT lenZ1 = sqrtf( m_vCellSizeXZ.y*m_vCellSizeXZ.y + subZ1*subZ1 );
			FLOAT dzdy = (lenZ1 * subZ0 + lenZ0 * subZ1) / (m_vCellSizeXZ.y*(lenZ0 + lenZ1));

			D3DXVECTOR3 vNormal( -dxdy, 1.0, -dzdy );
			D3DXVec3Normalize( &vNormal, &vNormal );
			pV->fNormal[0] = vNormal.x;
			pV->fNormal[1] = vNormal.y;
			pV->fNormal[2] = vNormal.z;

			++pV;
		}
	}


	m_pVB->Unlock();
	m_nVertices = nVertices;


	// Initialize the bounding box
	m_bound.vMin = D3DXVECTOR3( m_vStartXZ.x, fBottom, m_vStartXZ.y );
	m_bound.vMax = D3DXVECTOR3( m_vStartXZ.x + m_vCellSizeXZ.x * m_nCellNumX, fBottom + 255.0f*fHeightScale, m_vStartXZ.y + m_vCellSizeXZ.y * m_nCellNumZ ); 

	return TRUE;
}

