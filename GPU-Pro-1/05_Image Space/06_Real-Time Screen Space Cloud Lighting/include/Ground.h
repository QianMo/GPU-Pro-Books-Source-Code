//--------------------------------------------------------------------------------------
// Ground.h
//
// Declaration of a height map ground class.
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//--------------------------------------------------------------------------------------

#if !defined(__INCLUDED_GROUND_H__)
#define __INCLUDED_GROUND_H__


#include "SceneParameter.h"
#include "Shader.h"



//--------------------------------------------------------------------------------------
// Ground object
//  A uniform grid terrain.
//--------------------------------------------------------------------------------------

class CGround {
public :
	// vertex shader constants
	enum {
		VS_CONST_L2C = 0,   // local to projection matrix
		VS_CONST_L2W,       // local to world matrix
		VS_CONST_L2S,       // local to shadowmap matrix
		VS_CONST_NUM,
	};
	// pixel shader constants
	enum {
		PS_CONST_EYE = 0,           // view position
		PS_CONST_LITDIR,            // light direction
		PS_CONST_LITCOL,            // light color
		PS_CONST_LITAMB,            // light ambient
		PS_CONST_SCATTERING,        // scattering 
		PS_CONST_MATERIAL_DIFFUSE,  // diffuse reflection color
		PS_CONST_MATERIAL_SPECULAR, // specular reflection color
		PS_CONST_NUM,
	};
	enum {
		TEX_BLEND = 0,
		TEX_GROUND0,
		TEX_GROUND1,
		TEX_GROUND2,
		TEX_GROUND3,
		TEX_GROUND4,
		TEX_NUM,
	};
	struct S_VERTEX {
		FLOAT fPos[3];
		FLOAT fNormal[3];
		FLOAT fTex[4];
	};

public :
	CGround();
	~CGround();

	BOOL Create(
		LPDIRECT3DDEVICE9 pDev, 
		const SSceneParamter* pSceneParam, 
		const char* ptcHeightmap, 
		LPDIRECT3DTEXTURE9 pCloudShadowMap, 
		const D3DXMATRIX* pShadowMatrix);
	VOID Delete();

	VOID Draw(LPDIRECT3DDEVICE9 pDev);

	FLOAT GetHeight(FLOAT x, FLOAT z) const;

	inline VOID GetCenterPosition(D3DXVECTOR3& vCenter) const;
	inline const SBoundingBox& GetBoundingBox() const;

protected :
	VOID SetShaderConstants(LPDIRECT3DDEVICE9 pDev);
	BOOL CreateShaders(LPDIRECT3DDEVICE9 pDev);
	BOOL CreateIndexBuffer(LPDIRECT3DDEVICE9 pDev, USHORT nWidth, USHORT nHeight);
	BOOL CreateVertexBuffer(LPDIRECT3DDEVICE9 pDev, UINT nWidth, UINT nHeight, const BYTE* pData);
	BOOL LoadHeightmap(LPDIRECT3DDEVICE9 pDev, const char* ptchHeightmap);

public :
	LPDIRECT3DINDEXBUFFER9       m_pIB;            // index buffer
	LPDIRECT3DVERTEXBUFFER9      m_pVB;            // vertex buffer
	LPDIRECT3DVERTEXDECLARATION9 m_pDecl;          // vertex declaration
	LPDIRECT3DTEXTURE9           m_pTex[TEX_NUM];  // ground textures

	LPDIRECT3DTEXTURE9           m_pCloudShadow;   // shadowmap
	const D3DXMATRIX*            m_pW2Shadow;      // world to shadow map matrix
	const SSceneParamter*        m_pSceneParam;    // scene parameters
	FLOAT*                       m_pfHeight;       // heightmap
	CShader                      m_shader;         // ground shader

	UINT                         m_nVertices;      // the number of vertices
	UINT                         m_nTriangles;     // the number of triangles

	UINT                         m_nCellNumX;      // the number of cells in X direction
	UINT                         m_nCellNumZ;      // the number of cells in Z direction
	D3DXVECTOR2                  m_vCellSizeXZ;    // cell width in x and z direction
	D3DXVECTOR2                  m_vStartXZ;       // minimum x z position

	SBoundingBox                 m_bound;          // bounding box
};


//--------------------------------------------------------------------------------------
// Return the center position (x,z) of the terrain 
//--------------------------------------------------------------------------------------
VOID CGround::GetCenterPosition(D3DXVECTOR3& vCenter) const
{
	vCenter.x = m_vStartXZ.x + 0.5f * m_nCellNumX * m_vCellSizeXZ.x;
	vCenter.y = 0.0f;
	vCenter.z = m_vStartXZ.y + 0.5f * m_nCellNumZ * m_vCellSizeXZ.y;
}

//--------------------------------------------------------------------------------------
// Return the bounding box of the terrain.
//--------------------------------------------------------------------------------------
const SBoundingBox& CGround::GetBoundingBox() const
{
	return m_bound;
}



#endif

