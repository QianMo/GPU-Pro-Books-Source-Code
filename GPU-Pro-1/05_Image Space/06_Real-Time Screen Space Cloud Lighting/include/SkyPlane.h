//--------------------------------------------------------------------------------------
// SkyPlane.h
// 
// Implementation of a class to render sky with daylight scattering 
// 
// Copyright (C) Kaori Kubota. All rights reserved.
//
// 
//--------------------------------------------------------------------------------------

#if !defined(__INCLUDED_SKY_PLANE_H__)
#define __INCLUDED_SKY_PLANE_H__

#include "SceneParameter.h"
#include "Shader.h"



//--------------------------------------------------------------------------------------
// CSkyPlane 
//   A class to render sky with daylight scattering as a screen quad.
//--------------------------------------------------------------------------------------
class CSkyPlane : public CShader {
public :
	struct S_VERTEX {
		D3DXVECTOR4 vPos;
	};
	enum {
		DIV_X = 4,
		DIV_Y = 4,
		NUM_VERTICES = (DIV_X+1) * (DIV_Y+1),
		NUM_INDICES  = 2*DIV_Y * (DIV_X+1) + (DIV_Y-1)*2,
		NUM_TRIANGLES = NUM_INDICES-2,
	};
	enum {
		VS_CONST_C2W = 0,   // transform screen position to world space
		VS_CONST_EYE,       // view position
		VS_CONST_LITDIR,    // light direction
		VS_CONST_NUM,
	};
	enum {
		PS_CONST_EYE = 0,   // view position
		PS_CONST_LITDIR,    // light direction
		PS_CONST_SCATTERING,// scattering parameter
		PS_CONST_NUM,
	};
public :
	CSkyPlane();
	~CSkyPlane();

	BOOL Create(LPDIRECT3DDEVICE9 pDev, const SSceneParamter* pSceneParam);
	VOID Delete();

	VOID Draw(LPDIRECT3DDEVICE9 pDev);

protected :
	BOOL CreateBuffers(LPDIRECT3DDEVICE9 pDev);
	BOOL CreateShaders(LPDIRECT3DDEVICE9 pDev);

public :
	LPDIRECT3DVERTEXBUFFER9      m_pVB;
	LPDIRECT3DINDEXBUFFER9       m_pIB;
	LPDIRECT3DVERTEXDECLARATION9 m_pDecl;

	const SSceneParamter*         m_pSceneParam;

};

#endif // #if !defined(__INCLUDED_SKY_PLANE_H__)

