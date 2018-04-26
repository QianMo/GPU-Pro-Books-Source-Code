#pragma once
#pragma warning(disable: 4995)

#include <windows.h>
#include <strsafe.h>
#include <DXUT.h>
#include <DXUTgui.h>
#include <SDKMesh.h>
#include <SDKmisc.h>
#include <DxErr.h>
#include <queue>
#define LENGTHOF( x ) sizeof( x ) / sizeof( x[0] )

typedef D3DXVECTOR3 float3;
typedef D3DXVECTOR4 float4;
typedef D3DXMATRIX	float4x4;

#define EFFECT_PATH L"..\\effects"
#define ASSET_PATH L"..\\assets"

typedef HRESULT (CALLBACK* RENDERCALLBACK)(ID3D10Device* d3dDevice);

struct RENDER_PATH_DESC {
	RENDERCALLBACK renderFunc;
	TCHAR funcName[ 128 ];
};


class Camera {
public:
	void SetLookAt( const D3DXVECTOR3& eye, const D3DXVECTOR3& at, const D3DXVECTOR3& up );
	void SetPerspective( float aspect, float fovy, float znear, float zfar );
	void SetViewport( UINT width, UINT height );
	D3DXMATRIX* GetViewProjMat( D3DXMATRIX* pOut );
	D3DXVECTOR4* GetEyePos( D3DXVECTOR4* pOut );

	D3DXVECTOR3	eye;
	D3DXVECTOR3 at;
	D3DXVECTOR3 up;
	float		aspect;
	float		fovy;
	float		znear;
	float		zfar;

	D3DXMATRIX	viewMat;
	D3DXMATRIX	projMat;
	D3DXMATRIX	viewProjMat;

	D3D10_VIEWPORT	viewport;
};

struct Particle {
	float radius;
	float angle;
	float angularAccel;
	float height;
	float radiusAccel;
	float heightAccel;
};


void CreateTransformationsFromParticle( D3DXMATRIX* o2w, D3DXMATRIX* n2w, Particle* p );
