//--------------------------------------------------------------------------------------
// CloudGrid.h
//
// Declaration of the classes to render cloud grid to density map and shadow map.
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------


#if !defined(__INCLUDED_CLOUD_PLANE_H__)
#define __INCLUDED_CLOUD_PLANE_H__


#include "SceneParameter.h"
#include "Shader.h"



//--------------------------------------------------------------------------------------
// CCloudGrid
//     Grid for rendering cloud density and shadow.
//--------------------------------------------------------------------------------------
class CCloudGrid {
public :
	struct SVSParam {
		D3DXVECTOR4 vUVParam;
		D3DXVECTOR4 vXZParam;
		D3DXVECTOR2 vHeight;
	};
	struct S_VERTEX{
		FLOAT x; // x index
		FLOAT z; // z index
		FLOAT u; // texture coordinate u 
		FLOAT v; // texture coordinate v 
	};

	CCloudGrid();
	~CCloudGrid();

	BOOL Create(LPDIRECT3DDEVICE9 pDev);
	VOID Delete();

	VOID Update(FLOAT dt, const CSceneCamera* pCamera);
	VOID Draw(LPDIRECT3DDEVICE9 pDev);

	inline VOID SetCloudCover(FLOAT fCloudCover);
	inline FLOAT GetCurrentCloudCover() const;
	inline const SBoundingBox& GetBoundingBox() const;
	inline VOID GetVSParam(SVSParam& param) const;

protected :
	BOOL CreateVertexBuffer(LPDIRECT3DDEVICE9 pDev, USHORT nCellNumX, USHORT nCellNumZ);
	BOOL CreateIndexBuffer(LPDIRECT3DDEVICE9 pDev, USHORT nWidth, USHORT nHeight);

protected :
	LPDIRECT3DVERTEXBUFFER9      m_pVB;       // vertex buffer of the grid
	LPDIRECT3DINDEXBUFFER9       m_pIB;       // index buffer of the grid
	LPDIRECT3DVERTEXDECLARATION9 m_pDecl;     // vertex declaration
	LPDIRECT3DTEXTURE9           m_pCloudTex; // density texture

	UINT m_nVertices;            // the number of vertices
	UINT m_nTriangles;           // the number of triangles

	D3DXVECTOR2 m_vStartXZ;      // minimum x,z position 
	D3DXVECTOR2 m_vCellSizeXZ;   // cell width in x and z axis.
	FLOAT m_fDefaultHeight;      // cloud height above the view position
	FLOAT m_fFallOffHeight;      // delta height.

	FLOAT m_fCloudCover;         // cloud cover
	D3DXVECTOR2 m_vVelocity;     // wind velocity
	D3DXVECTOR2 m_vOffset;       // current uv offset
							    
	SBoundingBox m_bound;        // bounding box of the grid
};


//--------------------------------------------------------------------------------------
// cloud cover = [0.0 1.0]
//--------------------------------------------------------------------------------------
VOID CCloudGrid::SetCloudCover(FLOAT fCloudCover)
{
	m_fCloudCover = fCloudCover;
}

//--------------------------------------------------------------------------------------
FLOAT CCloudGrid::GetCurrentCloudCover() const
{
	return m_fCloudCover;
}


//--------------------------------------------------------------------------------------
// Return a bounding box of the grid
//--------------------------------------------------------------------------------------
const SBoundingBox& CCloudGrid::GetBoundingBox() const
{
	return m_bound;
}

//--------------------------------------------------------------------------------------
// Return parameters for vertex shader
//--------------------------------------------------------------------------------------
VOID CCloudGrid::GetVSParam(SVSParam& param) const
{
	param.vUVParam = D3DXVECTOR4( 5.0f, 5.0f, m_vOffset.x, m_vOffset.y );
	param.vXZParam = D3DXVECTOR4( m_vCellSizeXZ.x, m_vCellSizeXZ.y, m_vStartXZ.x, m_vStartXZ.y );
	param.vHeight  = D3DXVECTOR2( m_fFallOffHeight, m_fDefaultHeight );
}



//--------------------------------------------------------------------------------------
// Shader for rendering cloud density
//--------------------------------------------------------------------------------------
class CRenderDensityShader : public CShader {
public :
	// vertex shader constants
	enum {
		VS_CONST_XZPARAM = 0, // xz scale and offset of the position
		VS_CONST_HEIGHT,    // parameter to compute height
		VS_CONST_W2C,       // transform world to projection space
		VS_CONST_EYE,       // view position
		VS_CONST_UVPARAM,   // uv scale and offset
		VS_CONST_NUM,
	};
	// pixel shader constants
	enum {
		PS_CONST_COVER = 0, // cloud cover
		PS_CONST_NUM,
	};

public :
	CRenderDensityShader();
	~CRenderDensityShader();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam);
	BOOL Begin(LPDIRECT3DDEVICE9 pDev, CCloudGrid* pCloud);
	VOID End();
protected :
	VOID SetShaderConstant(LPDIRECT3DDEVICE9 pDev, CCloudGrid* pCloud);

	const SSceneParamter* m_pSceneParam;
};


//--------------------------------------------------------------------------------------
// Shader for rendering clouds to shadow map
//--------------------------------------------------------------------------------------
class CRenderShadowShader : public CShader {
public :
	// vertex shader constants
	enum {
		VS_CONST_XZPARAM = 0, // xz scale and offset of the position
		VS_CONST_HEIGHT,    // parameter to compute height
		VS_CONST_W2C,	    // transform world to projection space
		VS_CONST_EYE,	    // view position
		VS_CONST_UVPARAM,   // uv scale and offset
		VS_CONST_NUM,
	};
	// pixel shader constants
	enum {
		PS_CONST_COVER = 0, // cloud cover
		PS_CONST_NUM,
	};

public :
	CRenderShadowShader();
	~CRenderShadowShader();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam);
	VOID Update(const CSceneCamera* pCamera, const SBoundingBox* pGround, const SBoundingBox* pCloud);
	BOOL Begin(LPDIRECT3DDEVICE9 pDev, CCloudGrid* pCloud);
	VOID End();

	inline const D3DXMATRIX* GetW2ShadowMapMatrix() const;
protected :
	VOID SetShaderConstant(LPDIRECT3DDEVICE9 pDev, CCloudGrid* pCloud);

	const SSceneParamter* m_pSceneParam;
	D3DXMATRIX            m_mW2SProj;        // Transform world to Projection matrix to render to the shadowmap
	D3DXMATRIX            m_mW2S;            // Transform world to shadowmap texture coordinate
};

//--------------------------------------------------------------------------------------
// Return transform world space to shadowmap texture coordinate
//--------------------------------------------------------------------------------------
const D3DXMATRIX* CRenderShadowShader::GetW2ShadowMapMatrix() const
{
	return &m_mW2S;
}




#endif

