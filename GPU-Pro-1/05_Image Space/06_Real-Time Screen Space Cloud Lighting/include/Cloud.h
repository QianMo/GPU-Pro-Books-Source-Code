//--------------------------------------------------------------------------------------
// Cloud.h
//
// Declaration of the classes to light and render clouds
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------

#if !defined(__INCLUDED_CLOUD_H__)
#define __INCLUDED_CLOUD_H__


#include "CloudGrid.h"



//--------------------------------------------------------------------------------------
// CCloudBlur
//    Shader for blur density.
//--------------------------------------------------------------------------------------
class CCloudBlur : public CShader {
public :
	struct S_VERTEX {
		FLOAT afPos[4];
		FLOAT afTex[2];
	};
	enum {
		VS_CONST_C2W = 0,    // Inverse Matrix of World to Projection transform
		VS_CONST_PIXELSIZE,  // half pixel size in uv space
		VS_CONST_OFFSET,     // paramter for computing blur vector 
		VS_CONST_NUM,
	};
	enum {
		PS_CONST_DISTANCE = 0, // parameter for computing distance from the view position 
		PS_CONST_FALLOFF,      // fall off parameter of the blur
		PS_CONST_MAX,          // inverse of the maximum length of the blur vector
		PS_CONST_EYE,          // view position
		PS_CONST_NUM,
	};
public :
	CCloudBlur();
	~CCloudBlur();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam);
	VOID Delete();

	VOID Blur(LPDIRECT3DDEVICE9 pDev, LPDIRECT3DTEXTURE9 pTex);

protected :
	BOOL CreateShaders(LPDIRECT3DDEVICE9 pDev);
	VOID SetShaderConstant(LPDIRECT3DDEVICE9 pDev, LPDIRECT3DTEXTURE9 pTex);
protected :
	LPDIRECT3DVERTEXBUFFER9      m_pVB;
	LPDIRECT3DVERTEXDECLARATION9 m_pDecl;
	const SSceneParamter*        m_pSceneParam;
};



//--------------------------------------------------------------------------------------
// CCloudPlane
//    render final cloud as a screen quad.
//--------------------------------------------------------------------------------------
class CCloudPlane : public CShader {
public :
	struct S_VERTEX {
		D3DXVECTOR4 vPos;
		FLOAT vTex[2];
	};
	enum {
		DIV_X = 4,
		DIV_Y = 4,
		NUM_VERTICES = (DIV_X+1) * (DIV_Y+1),
		NUM_INDICES  = 2*DIV_Y * (DIV_X+1) + (DIV_Y-1)*2,
		NUM_TRIANGLES = NUM_INDICES-2,
	};
	// vertex shader constants
	enum {
		VS_CONST_C2W = 0, // inverse matrix of world to projection
		VS_CONST_EYE,     // view position
		VS_CONST_LITDIR,  // light direction
		VS_CONST_NUM,
	};
	// pixel shader constants
	enum {
		PS_CONST_EYE = 0,    // view position
		PS_CONST_LITDIR,     // light direction 
		PS_CONST_SCATTERING, // scattering parameters
		PS_CONST_DISTANCE,   // parameter to compute distance to the cloud
		PS_CONST_LIGHT,      // light color
		PS_CONST_AMBIENT,    // ambient light
		PS_CONST_NUM, 
	};
public :
	CCloudPlane();
	~CCloudPlane();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam, LPDIRECT3DTEXTURE9 pDensityMap, LPDIRECT3DTEXTURE9 pBlurredMap);
	VOID SetTexture(LPDIRECT3DTEXTURE9 pDensityMap, LPDIRECT3DTEXTURE9 pBlurredMap);
	VOID Delete();

	VOID Draw(LPDIRECT3DDEVICE9 pDev);
protected :
	BOOL CreateBuffers(LPDIRECT3DDEVICE9 pDev);
	BOOL CreateShaders(LPDIRECT3DDEVICE9 pDev);
	VOID SetShaderConstant(LPDIRECT3DDEVICE9 pDev);
public :
	LPDIRECT3DVERTEXBUFFER9      m_pVB;            // vertex buffer
	LPDIRECT3DINDEXBUFFER9       m_pIB;            // index buffer
	LPDIRECT3DVERTEXDECLARATION9 m_pDecl;          // vertex declaration
	const SSceneParamter*        m_pSceneParam;    // scene parameter

	LPDIRECT3DTEXTURE9           m_pDensityMap;    // density map
	LPDIRECT3DTEXTURE9           m_pBlurredMap;    // blurred density map

};



//--------------------------------------------------------------------------------------
// CCloud
//    Interface object to light and render clouds.
//--------------------------------------------------------------------------------------
class CCloud {
public :
	CCloud();
	~CCloud();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam);
	VOID Delete();

	VOID Update(FLOAT dt, const CSceneCamera* pCamera, const SBoundingBox& bbGround);

	VOID PrepareCloudTextures(LPDIRECT3DDEVICE9 pDev);
	VOID DrawFinalQuad(LPDIRECT3DDEVICE9 pDev);

	inline VOID SetCloudCover(FLOAT fCloudCover);
	inline FLOAT GetCurrentCloudCover() const;
	inline const D3DXMATRIX* GetWorld2ShadowMatrix() const;
	inline LPDIRECT3DTEXTURE9 GetShadowMap();

protected :
	BOOL SetRenderTarget(LPDIRECT3DDEVICE9 pDev, LPDIRECT3DTEXTURE9 pTex);

protected :
	// render targets
	LPDIRECT3DTEXTURE9     m_pDensityMap;        // density map
	LPDIRECT3DTEXTURE9     m_pBlurredMap;        // blurred density map
	LPDIRECT3DTEXTURE9     m_pShadowMap;         // shadow map

	const SSceneParamter*  m_pSceneParam;        // scene parameters
	CCloudGrid             m_grid;               // cloud grid
	CRenderDensityShader   m_densityShader;      // shader to render density 
	CRenderShadowShader    m_shadowShader;       // shader to render shadow
	CCloudBlur       m_blur;               // blur shader
	CCloudPlane            m_finalCloud;         // object to render a screen cloud in the final pass 
};


//--------------------------------------------------------------------------------------
// fCloudCover = [0.0f 1.0f]
//--------------------------------------------------------------------------------------
VOID CCloud::SetCloudCover(FLOAT fCloudCover)
{
	m_grid.SetCloudCover( fCloudCover );
}

//--------------------------------------------------------------------------------------
FLOAT CCloud::GetCurrentCloudCover() const
{
	return m_grid.GetCurrentCloudCover();
}

//--------------------------------------------------------------------------------------
const D3DXMATRIX* CCloud::GetWorld2ShadowMatrix() const
{
	return m_shadowShader.GetW2ShadowMapMatrix();
}

//--------------------------------------------------------------------------------------
LPDIRECT3DTEXTURE9 CCloud::GetShadowMap()
{
	return m_pShadowMap;
}

#endif 
