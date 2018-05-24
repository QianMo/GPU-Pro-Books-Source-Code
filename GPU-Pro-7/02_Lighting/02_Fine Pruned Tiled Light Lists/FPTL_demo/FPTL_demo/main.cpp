// This sample was made by Morten S. Mikkelsen
// Though the demo does run faster with fine pruning enabled you will experience much greater gains
// in real world production when using asynchronous compute interleaved with rendering of shadow maps.
// The main purpose of this demo is to serve as a reference implementation.


#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
//#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
//#include "SDKMesh.h"
#include <d3d11_2.h>
#include "strsafe.h"
#include <stdlib.h>
#include <math.h>


#define RECOMPILE_SCRBOUND_CS_SHADER

#define DISABLE_QUAT

#include <geommath/geommath.h>
#include "shader.h"
#include "shaderpipeline.h"
#include "std_cbuffer.h"
#include "shaderutils.h"
#include "texture_rt.h"

#include <wchar.h>

CDXUTDialogResourceManager g_DialogResourceManager;
CDXUTTextHelper *		g_pTxtHelper = NULL;

CFirstPersonCamera                  g_Camera;


CShader scrbound_shader;
CShader lightlist_coarse_shader;
CShader lightlist_exact_shader;
CShader vert_shader, pix_shader;
CShader vert_shader_basic;

CShaderPipeline shader_dpthfill_pipeline;
CShaderPipeline shader_pipeline;


ID3D11Buffer * g_pMeshInstanceCB = NULL;

//

ID3D11ShaderResourceView * g_pLightDataBufferSRV = NULL;
ID3D11Buffer * g_pLightDataBuffer = NULL;
ID3D11Buffer * g_pLightDataBuffer_staged = NULL;
ID3D11Buffer * g_pLightClipInfo = NULL;

ID3D11Buffer * g_pScrSpaceAABounds = NULL;
ID3D11Buffer * g_pScrSpaceAABounds_staged = NULL;
ID3D11UnorderedAccessView* g_pScrBoundUAV = NULL;
ID3D11ShaderResourceView * g_pScrBoundSRV = NULL;


ID3D11Buffer * g_pLightListBuffer = NULL;
ID3D11Buffer * g_pLightListBuffer_staged = NULL;
ID3D11UnorderedAccessView * g_pLightListBufferUAV = NULL;
ID3D11ShaderResourceView * g_pLightListBufferSRV = NULL;


ID3D11Buffer * g_pOrientedBounds = NULL;
ID3D11Buffer * g_pOrientedBounds_staged = NULL;
ID3D11ShaderResourceView * g_pOrientedBoundsSRV = NULL;


#define WIDEN2(x)		L ## x
#define WIDEN(x)		WIDEN2(x)


#define MODEL_NAME		"ground2_reduced.fil"
#define MODEL_PATH		".\\"



#define MODEL_PATH_W	WIDEN(MODEL_PATH)


#include "mesh_fil.h"


ID3D11InputLayout * g_pVertexLayout = NULL;
ID3D11InputLayout * g_pVertexSimpleLayout = NULL;



#define MAX_LEN			64
#define NR_TEXTURES		1


const WCHAR tex_names[NR_TEXTURES][MAX_LEN] = {L"normals.png"};
const char stex_names[NR_TEXTURES][MAX_LEN] = {"g_norm_tex"};


ID3D11ShaderResourceView * g_pTexturesHandler[NR_TEXTURES];

#ifndef M_PI
	#define M_PI 3.1415926535897932384626433832795
#endif

bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void* pUserContext );

void myFrustum(float * pMat, const float fLeft, const float fRight, const float fBottom, const float fTop, const float fNear, const float fFar);


CMeshFil g_cMesh;


static int g_iCullMethod = 1;
static int g_iVisualMode = 0;
static int g_iMenuVisib = 1;

#include "light_definitions.h"
#include "LightTiling.h"



int g_iSqrtNrLights = 0;
int g_iNrLights = MAX_NR_LIGHTS_PER_CAMERA;

SFiniteLightData g_sLgtData[MAX_NR_LIGHTS_PER_CAMERA];
SFiniteLightBound g_sLgtColiData[MAX_NR_LIGHTS_PER_CAMERA];

CLightTiler g_cLightTiler;

static float frnd() { return (float) (((double) (rand() % (RAND_MAX+1))) / RAND_MAX); }

CTextureObject g_tex_depth;

void InitApp()
{
    
	g_Camera.SetRotateButtons( true, false, false );
    g_Camera.SetEnablePositionMovement( true );

	
    g_Camera.SetScalers( 0.2f*0.005f, 3*100.0f );
    DirectX::XMVECTOR vEyePt, vEyeTo;

	Vec3 cam_pos = 75.68f*Normalize(Vec3(16,0,40));	// normal

	vEyePt = DirectX::XMVectorSet(cam_pos.x, cam_pos.y, -cam_pos.z, 1.0f);
	//vEyeTo = DirectX::XMVectorSet(0.0f, 2.0f, 0.0f, 1.0f);
	vEyeTo = DirectX::XMVectorSet(10.0f, 2.0f, 0.0f, 1.0f);

	
    g_Camera.SetViewParams( vEyePt, vEyeTo );

	g_iSqrtNrLights = (int) sqrt((double) g_iNrLights);
	assert((g_iSqrtNrLights*g_iSqrtNrLights)<=g_iNrLights);
	g_iNrLights = g_iSqrtNrLights*g_iSqrtNrLights;
}

void RenderText()
{
	g_pTxtHelper->Begin();
	g_pTxtHelper->SetInsertionPos( 2, 0 );
	g_pTxtHelper->SetForegroundColor(DirectX::XMFLOAT4(1.0f, 1.0f, 0.0f, 1.0f));
	g_pTxtHelper->DrawTextLine( DXUTGetFrameStats(true) );

	if(g_iMenuVisib!=0)
	{
		g_pTxtHelper->DrawTextLine(L"This scene is forward lit by 1024 lights of different shapes. High performance is achieved using FPTL.\n");
		g_pTxtHelper->DrawTextLine(L"Rotate the camera by using the mouse while pressing and holding the left mouse button.\n");
		g_pTxtHelper->DrawTextLine(L"Move the camera by using the arrow keys or: w, a, s, d\n");
		g_pTxtHelper->DrawTextLine(L"Hide menu using the h key.\n");

		if (g_iCullMethod == 0)
			g_pTxtHelper->DrawTextLine(L"Fine pruning disabled (toggle using m)\n");
		else
			g_pTxtHelper->DrawTextLine(L"Fine pruning enabled (toggle using m)\n");

		if (g_iVisualMode == 0)
			g_pTxtHelper->DrawTextLine(L"Show Overlaps enabled (toggle using v)\n");
		else
			g_pTxtHelper->DrawTextLine(L"Show Overlaps disabled (toggle using v)\n");
	}

	g_pTxtHelper->End();
}


float Lerp(const float fA, const float fB, const float fT) { return fA*(1-fT) + fB*fT; }


// carpet bomb terrain with 1024 lights of various types
void BuildLightsBuffer()
{
	static bool bBufferMade = false;

	if(!bBufferMade)
	{
		bBufferMade = true;

		// build light lists
		int iNrLgts = 0;
		int iRealCounter = 0;
		while(iNrLgts<g_iNrLights)
		{
			// 5 light types define in 
			unsigned int uFlag = rand()%MAX_TYPES;

			const int iX = iNrLgts % g_iSqrtNrLights;
			const int iZ = iNrLgts / g_iSqrtNrLights;
			const float fX = 4000*(2*((iX+0.5f)/g_iSqrtNrLights)-1);
			const float fZ = 4000*(2*((iZ+0.5f)/g_iSqrtNrLights)-1);
			const float fY = g_cMesh.QueryTopY(fX, fZ)+12*5 + (uFlag==WEDGE_LIGHT || uFlag==CAPSULE_LIGHT ? 2*50 : 0);
			const Vec3 vCen = Vec3(fX, fY, fZ);

			const float fT = frnd();

			const float fRad = (uFlag==SPOT_CIRCULAR_LIGHT ? 3.0 : 2.0)*(100*fT + 80*(1-fT)+1);

			{
				SFiniteLightData &lgtData = g_sLgtData[iNrLgts];
				SFiniteLightBound &lgtColiData = g_sLgtColiData[iNrLgts];

				float fFar = fRad;
				float fNear = 0.80*fRad;
				lgtData.vLpos = vCen;
				lgtData.fInvRange = -1/(fFar-fNear);
				lgtData.fNearRadiusOverRange_LP0 = fFar/(fFar-fNear);
				lgtData.fSphRadiusSq = fFar*fFar;
				

				float fFov = 1.0*frnd()+0.2;		// full fov from left to right
				float fSeg = Lerp(fRad, 0.5*fRad, frnd());
				lgtData.fSegLength = (uFlag==WEDGE_LIGHT || uFlag==CAPSULE_LIGHT) ? fSeg : 0.0f;

				lgtData.uLightType = uFlag;



				// default coli settings
				lgtColiData.vBoxAxisX = Vec3(1,0,0);
				lgtColiData.vBoxAxisY = Vec3(0,1,0);
				lgtColiData.vBoxAxisZ = Vec3(0,0,1);
				lgtColiData.vScaleXZ = Vec2(1.0f, 1.0f);

				// build colision info for each light type
				if(uFlag==CAPSULE_LIGHT)
				{
					lgtData.vLdir = lgtColiData.vBoxAxisX;
					lgtColiData.fRadius = fRad + 0.5*fSeg;
					lgtColiData.vCen = vCen + (0.5*fSeg)*lgtColiData.vBoxAxisX;
					lgtColiData.vBoxAxisX *= (fRad+0.5*fSeg); lgtColiData.vBoxAxisY *= fRad; lgtColiData.vBoxAxisZ *= fRad;

					lgtData.vCol = Vec3(1.0, 0.1, 1.0);
				}
				else if(uFlag==SPHERE_LIGHT)
				{
					lgtColiData.vBoxAxisX *= fRad; lgtColiData.vBoxAxisY *= fRad; lgtColiData.vBoxAxisZ *= fRad;
					lgtColiData.fRadius = fRad;
					lgtColiData.vCen = vCen;

					lgtData.vCol = Vec3(1,1,1);
				}
				else if(uFlag==SPOT_CIRCULAR_LIGHT || uFlag==WEDGE_LIGHT)
				{
					if(uFlag==SPOT_CIRCULAR_LIGHT)
						lgtData.vCol = Vec3(0*0.7,0.6,1);	
					else
						lgtData.vCol = Vec3(1,0.6,0*0.7);	
					fFov *= 2;

					float fQ = uFlag==WEDGE_LIGHT ? 0.1 : 1;
					Vec3 vDir = Normalize( Vec3(fQ*0.5*(2*frnd()-1),  -1, fQ*0.5*(2*frnd()-1)) );
					//lgtData.vBoxAxisX = vDir;		// Spot Dir
					lgtData.fPenumbra = cosf(fFov*0.5);
					lgtData.fInvUmbraDelta = 1/( lgtData.fPenumbra - cosf(0.02*(fFov*0.5)) );

					lgtColiData.vBoxAxisY = -vDir;

					Vec3 vY = lgtColiData.vBoxAxisY;
					Vec3 vTmpAxis = (fabsf(vY.x)<=fabsf(vY.y) && fabsf(vY.x)<=fabsf(vY.z)) ? Vec3(1,0,0) : ( fabsf(vY.y)<=fabsf(vY.z) ? Vec3(0,1,0) : Vec3(0,0,1) );
					Vec3 vX = Normalize( Cross(vY,vTmpAxis ) );
					lgtColiData.vBoxAxisZ = Cross(vX, vY);
					lgtColiData.vBoxAxisX = vX;

					// this is silly but nevertheless where this is passed in engine (note the coliData is setup with vBoxAxisY==-vDir).
					lgtData.vBoxAxisX = vDir;
					lgtData.vLdir = lgtColiData.vBoxAxisX;

					// apply nonuniform scale to OBB of spot light
					bool bSqueeze = uFlag==SPOT_CIRCULAR_LIGHT && fFov<0.7*(M_PI*0.5f);

					float fS = bSqueeze ? tan(0.5*fFov) : sin(0.5*fFov);

					lgtColiData.vCen += (vCen + ((0.5f*fRad)*vDir) + ((0.5f*lgtData.fSegLength)*vX));

					lgtColiData.vBoxAxisX *= (fS*fRad + 0.5*lgtData.fSegLength);
					lgtColiData.vBoxAxisY *= (0.5f*fRad);
					lgtColiData.vBoxAxisZ *= (fS*fRad);

					

					float fAltDx = sin(0.5*fFov);
					float fAltDy = cos(0.5*fFov);
					fAltDy = fAltDy-0.5;
					if(fAltDy<0) fAltDy=-fAltDy;

					fAltDx *= fRad; fAltDy *= fRad;
					fAltDx += (0.5f*lgtData.fSegLength);

					float fAltDist = sqrt(fAltDy*fAltDy+fAltDx*fAltDx);
					lgtColiData.fRadius = fAltDist>(0.5*fRad) ? fAltDist : (0.5*fRad);

					if(bSqueeze)
						lgtColiData.vScaleXZ = Vec2(0.01f, 0.01f);

				}
				else if(uFlag==BOX_LIGHT)
				{
					Mat33 rot; LoadRotation(&rot, 2*M_PI*frnd(), 2*M_PI*frnd(), 2*M_PI*frnd());
					float fSx = 5*2*(10*frnd()+4);
					float fSy = 5*2*(10*frnd()+4);
					float fSz = 5*2*(10*frnd()+4);

					float fSx2 = 0.1f*fSx;
					float fSy2 = 0.1f*fSy;
					float fSz2 = 0.1f*fSz;

					lgtData.vBoxAxisX = GetColumn(rot, 0);
					lgtData.vLdir = GetColumn(rot, 1);
					lgtData.vBoxAxisZ = GetColumn(rot, 2);

					lgtColiData.vBoxAxisX = fSx*lgtData.vBoxAxisX;
					lgtColiData.vBoxAxisY = fSy*lgtData.vLdir;
					lgtColiData.vBoxAxisZ = fSz*lgtData.vBoxAxisZ;

					lgtColiData.vCen = vCen;
					lgtColiData.fRadius = sqrtf(fSx*fSx+fSy*fSy+fSz*fSz);

					lgtData.vCol = Vec3(0.1,1,0.16);
					lgtData.fSphRadiusSq = lgtColiData.fRadius*lgtColiData.fRadius;

					lgtData.vBoxInnerDist = Vec3(fSx2, fSy2, fSz2);
					lgtData.vBoxInvRange = Vec3( 1/(fSx-fSx2), 1/(fSy-fSy2), 1/(fSz-fSz2) );
				}

				

				++iNrLgts;
			}
		}
	

		g_cLightTiler.InitTiler();
	}
}

void render_surface(ID3D11DeviceContext* pd3dImmediateContext, CShaderPipeline &shader_pipe, bool bSimpleLayout)
{
	shader_pipe.PrepPipelineForRendering(pd3dImmediateContext);
	

	// set streams and layout
	UINT stride = sizeof(SFilVert), offset = 0;
	pd3dImmediateContext->IASetVertexBuffers( 0, 1, g_cMesh.GetVertexBuffer(), &stride, &offset );
	pd3dImmediateContext->IASetIndexBuffer( g_cMesh.GetIndexBuffer(), DXGI_FORMAT_R32_UINT, 0 );
	pd3dImmediateContext->IASetInputLayout( bSimpleLayout ? g_pVertexSimpleLayout : g_pVertexLayout );

	// Set primitive topology
	pd3dImmediateContext->IASetPrimitiveTopology( D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST );
	pd3dImmediateContext->DrawIndexed( 3*g_cMesh.GetNrTriangles(), 0, 0 );
	shader_pipe.FlushResources(pd3dImmediateContext);
}


Mat44 g_m44Proj, g_m44InvProj, g_mViewToScr, g_mScrToView;


Vec3 XMVToVec3(const DirectX::XMVECTOR vec)
{
	return Vec3(DirectX::XMVectorGetX(vec), DirectX::XMVectorGetY(vec), DirectX::XMVectorGetZ(vec));
}

Vec4 XMVToVec4(const DirectX::XMVECTOR vec)
{
	return Vec4(DirectX::XMVectorGetX(vec), DirectX::XMVectorGetY(vec), DirectX::XMVectorGetZ(vec), DirectX::XMVectorGetW(vec));
}

Mat44 ToMat44(const DirectX::XMMATRIX &dxmat)
{
	Mat44 res;

	for (int c = 0; c < 4; c++)
		SetColumn(&res, c, XMVToVec4(dxmat.r[c]));

	return res;
}


void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, 
                                  double fTime, float fElapsedTime, void* pUserContext )
{
	HRESULT hr;

	//const float fTimeDiff = DXUTGetElapsedTime();

	// clear screen
    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
	ID3D11DepthStencilView* pDSV = g_tex_depth.GetDSV();//DXUTGetD3D11DepthStencilView();
	//DXUTGetD3D11DepthStencil();

	
	Vec3 vToPoint = XMVToVec3(g_Camera.GetLookAtPt());

	Vec3 cam_pos = XMVToVec3(g_Camera.GetEyePt());
    Mat44 world_to_view = ToMat44(g_Camera.GetViewMatrix() );	// get world to view projection

	Mat44 mZflip; LoadIdentity(&mZflip);
	SetColumn(&mZflip, 2, Vec4(0,0,-1,0));
#ifndef LEFT_HAND_COORDINATES
	world_to_view = mZflip * world_to_view * mZflip;
#else
	world_to_view = world_to_view * mZflip;
#endif
	
	Mat44 m44LocalToWorld; LoadIdentity(&m44LocalToWorld);
	Mat44 m44LocalToView = world_to_view * m44LocalToWorld;
	Mat44 m44ViewToLocal = ~m44LocalToView;
	Mat44 Trans = g_m44Proj * m44LocalToView;

	// fill constant buffers
	D3D11_MAPPED_SUBRESOURCE MappedSubResource;
	V( pd3dImmediateContext->Map( g_pMeshInstanceCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource ) );
	((cbMeshInstance *)MappedSubResource.pData)->g_mLocToView = Transpose(m44LocalToView);
	((cbMeshInstance *)MappedSubResource.pData)->g_mWorldViewProjection = Transpose(Trans);
	((cbMeshInstance *)MappedSubResource.pData)->g_mScrToView = Transpose(g_mScrToView);
	((cbMeshInstance *)MappedSubResource.pData)->g_vCamPos = m44ViewToLocal * Vec3(0,0,0);
	((cbMeshInstance *)MappedSubResource.pData)->g_iWidth = DXUTGetDXGIBackBufferSurfaceDesc()->Width;
	((cbMeshInstance *)MappedSubResource.pData)->g_iHeight = DXUTGetDXGIBackBufferSurfaceDesc()->Height;
	((cbMeshInstance *)MappedSubResource.pData)->g_iMode = g_iVisualMode;
    pd3dImmediateContext->Unmap( g_pMeshInstanceCB, 0 );

	V( pd3dImmediateContext->Map( g_pLightClipInfo, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedSubResource ) );
	((cbBoundsInfo *) MappedSubResource.pData)->g_mProjection = Transpose(g_m44Proj);
	((cbBoundsInfo *) MappedSubResource.pData)->g_mInvProjection = Transpose(g_m44InvProj);
	((cbBoundsInfo *) MappedSubResource.pData)->g_mScrProjection = Transpose(g_mViewToScr);
	((cbBoundsInfo *) MappedSubResource.pData)->g_mInvScrProjection = Transpose(g_mScrToView);
	((cbBoundsInfo *) MappedSubResource.pData)->g_iNrVisibLights = g_iNrLights;
	((cbBoundsInfo *) MappedSubResource.pData)->g_vStuff = Vec3(0,0,0);
	pd3dImmediateContext->Unmap( g_pLightClipInfo, 0 );

	// build light list
	g_cLightTiler.InitFrame(world_to_view, g_m44Proj);
	for(int l=0; l<g_iNrLights; l++)
	{
		g_cLightTiler.AddLight(g_sLgtData[l], g_sLgtColiData[l]);
	}
	g_cLightTiler.CompileLightList();


	// transfer light bounds
	V( pd3dImmediateContext->Map( g_pOrientedBounds_staged, 0, D3D11_MAP_WRITE, 0, &MappedSubResource ) );
	for(int l=0; l<g_iNrLights; l++)
	{
		((SFiniteLightBound *) MappedSubResource.pData)[l] = g_cLightTiler.GetOrderedBoundsList()[l];
	}
	pd3dImmediateContext->Unmap( g_pOrientedBounds_staged, 0 );

	V( pd3dImmediateContext->Map( g_pLightDataBuffer_staged, 0, D3D11_MAP_WRITE, 0, &MappedSubResource ) );
	for(int l=0; l<g_iNrLights; l++)
	{
		((SFiniteLightData *) MappedSubResource.pData)[l] = g_cLightTiler.GetLightsDataList()[l];
		if(g_iVisualMode==0) ((SFiniteLightData *) MappedSubResource.pData)[l].vCol = Vec3(1,1,1);
	}
	pd3dImmediateContext->Unmap( g_pLightDataBuffer_staged, 0 );

	pd3dImmediateContext->CopyResource(g_pLightDataBuffer, g_pLightDataBuffer_staged);


	// Convert OBBs of lights into screen space AABBs (incl. depth)
	const int nrGroups = (g_iNrLights*8 + 63)/64;
	ID3D11UnorderedAccessView* g_pNullUAV = NULL;

	ID3D11Buffer * pDatas[] = {g_pLightClipInfo};
	pd3dImmediateContext->CSSetConstantBuffers(0, 1, pDatas);		// 

	pd3dImmediateContext->CopyResource(g_pOrientedBounds, g_pOrientedBounds_staged);
	ID3D11ShaderResourceView * pSRVbounds[] = {g_pOrientedBoundsSRV};
	pd3dImmediateContext->CSSetShaderResources(0, 1, pSRVbounds);

	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &g_pScrBoundUAV, 0);
	ID3D11ComputeShader * pShaderCS = (ID3D11ComputeShader *) scrbound_shader.GetDeviceChild();
	pd3dImmediateContext->CSSetShader( pShaderCS, NULL, 0 );
	pd3dImmediateContext->Dispatch(nrGroups, 1, 1);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &g_pNullUAV, 0);

	ID3D11ShaderResourceView * pNullSRV_bound[] = {NULL};
	pd3dImmediateContext->CSSetShaderResources(0, 1, pNullSRV_bound);

	// debugging code...
	/*
	pd3dImmediateContext->CopyResource(g_pScrSpaceAABounds_staged, g_pScrSpaceAABounds);
	V( pd3dImmediateContext->Map( g_pScrSpaceAABounds_staged, 0, D3D11_MAP_READ, 0, &MappedSubResource ) );
	const Vec3 * pData0 = ((Vec3 *) MappedSubResource.pData);
	const Vec3 * pData1 = g_cLightTiler.GetScrBoundsList();
	pd3dImmediateContext->Unmap( g_pScrSpaceAABounds_staged, 0 );*/
	

	// prefill depth
	const bool bRenderFront = true;
	float ClearColor[4] = { 0.03f, 0.05f, 0.1f, 0.0f };

	pd3dImmediateContext->OMSetRenderTargets( 0, NULL, pDSV );
    pd3dImmediateContext->ClearDepthStencilView( pDSV, D3D11_CLEAR_DEPTH, 1.0f, 0 );


	pd3dImmediateContext->RSSetState( GetDefaultRasterSolidCullBack()  );
	pd3dImmediateContext->OMSetDepthStencilState( GetDefaultDepthStencilState(), 0 );

	render_surface(pd3dImmediateContext, shader_dpthfill_pipeline, true);

	// switch to back-buffer
	pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, g_tex_depth.GetReadOnlyDSV() );
	pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );


	// build a light list per 16x16 tile
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &g_pLightListBufferUAV, 0);
	ID3D11ShaderResourceView * pSRV[] = {g_tex_depth.GetSRV(),g_pScrBoundSRV, g_pLightDataBufferSRV};
	pd3dImmediateContext->CSSetShaderResources(0, 3, pSRV);
	const int iNrTilesX = (DXUTGetDXGIBackBufferSurfaceDesc()->Width+15)/16;
	const int iNrTilesY = (DXUTGetDXGIBackBufferSurfaceDesc()->Height+15)/16;
	pShaderCS = (ID3D11ComputeShader *) (g_iCullMethod==0 ? lightlist_coarse_shader.GetDeviceChild() : lightlist_exact_shader.GetDeviceChild());
	pd3dImmediateContext->CSSetShader( pShaderCS, NULL, 0 );
	pd3dImmediateContext->Dispatch(iNrTilesX, iNrTilesY, 1);
	//pd3dImmediateContext->CSSetShaderResources(0, 0, NULL);
	pd3dImmediateContext->CSSetUnorderedAccessViews(0, 1, &g_pNullUAV, 0);
	ID3D11ShaderResourceView * pNullSRV[] = {NULL, NULL, NULL};
	pd3dImmediateContext->CSSetShaderResources(0, 3, pNullSRV);

	// debugging code...
	/*
	pd3dImmediateContext->CopyResource(g_pLightListBuffer_staged, g_pLightListBuffer);
	V( pd3dImmediateContext->Map( g_pLightListBuffer_staged, 0, D3D11_MAP_READ, 0, &MappedSubResource ) );
	const unsigned int * pData2 = ((const unsigned int *) MappedSubResource.pData);
	pd3dImmediateContext->Unmap( g_pLightListBuffer_staged, 0 );	
	*/

	
	// Do tiled forward rendering
	render_surface(pd3dImmediateContext, shader_pipeline, false);



	// fire off menu text
	RenderText();
}


int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

	 
    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackFrameMove( OnFrameMove );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

    DXUTSetCallbackKeyboard( OnKeyboard );

    InitApp();
    DXUTInit( true, true );
    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"Fine Pruned Tiled Lighting Demo" );
	int dimX = 1280, dimY = 960;
	DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, dimX, dimY);
    //DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, 1024, 768);
    DXUTMainLoop(); // Enter into the DXUT render loop

    return DXUTGetExitCode();
}



//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------


HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;


    // Setup the camera's projection parameters
	int w = pBackBufferSurfaceDesc->Width;
	int h = pBackBufferSurfaceDesc->Height;

	const float fFov = 30;
	const float fNear = 10;
	const float fFar = 10000;
	//const float fNear = 45;//275;
	//const float fFar = 65;//500;
	const float fHalfWidthAtMinusNear = fNear * tanf((fFov*((float) M_PI))/360);
	const float fHalfHeightAtMinusNear = fHalfWidthAtMinusNear * (((float) 3)/4.0);//(3.0f/4.0f);

	const float fS = 1.0;// 1280.0f / 960.0f;

	//glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
	myFrustum(g_m44Proj.m_fMat, -fS*fHalfWidthAtMinusNear, fS*fHalfWidthAtMinusNear, -fHalfHeightAtMinusNear, fHalfHeightAtMinusNear, fNear, fFar);


	{
		float fAspectRatio = fS;
		g_Camera.SetProjParams( (fFov*M_PI)/360, fAspectRatio, fNear, fFar );
	}


	Mat44 mToScr;
	SetRow(&mToScr, 0, Vec4(0.5*w, 0,     0,  0.5*w));
	SetRow(&mToScr, 1, Vec4(0,     -0.5*h, 0,  0.5*h));
	SetRow(&mToScr, 2, Vec4(0,     0,     1,  0));
	SetRow(&mToScr, 3, Vec4(0,     0,     0,  1));

	g_mViewToScr = mToScr * g_m44Proj;
	g_mScrToView = ~g_mViewToScr;
	g_m44InvProj = ~g_m44Proj;

	



    // Set GUI size and locations
    /*g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
    g_HUD.SetSize( 170, 170 );
    g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 245, pBackBufferSurfaceDesc->Height - 520 );
    g_SampleUI.SetSize( 245, 520 );*/

	// create render targets
	const bool bEnableReadBySampling = true;
	const bool bEnableWriteTo = true;
	const bool bAllocateMipMaps = false;
	const bool bAllowStandardMipMapGeneration = false;
	const void * pInitData = NULL;

	g_tex_depth.CleanUp();

	g_tex_depth.CreateTexture(pd3dDevice,w,h, DXGI_FORMAT_R24G8_TYPELESS, bAllocateMipMaps, false, NULL,
								bEnableReadBySampling, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, bEnableWriteTo, DXGI_FORMAT_D24_UNORM_S8_UINT,
								true);


	////////////////////////////////////////////////
	SAFE_RELEASE(g_pLightListBufferUAV);
	SAFE_RELEASE(g_pLightListBufferSRV);
	SAFE_RELEASE(g_pLightListBuffer);
	SAFE_RELEASE(g_pLightListBuffer_staged);
	



	const int iNrLightsPerDWord = 3;	// 10_10_10_2
	const int iNrLightsPerTile = 24;
	const int iNrDWordsRequred = (iNrLightsPerTile+2)/iNrLightsPerDWord;

	const int nrTiles = ((w+15)/16)*((h+15)/16);
	D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(bd));
	bd.Usage = D3D11_USAGE_DEFAULT;
    bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    bd.CPUAccessFlags = 0;
    bd.MiscFlags = 0;
	bd.ByteWidth = iNrDWordsRequred*sizeof( unsigned int ) * nrTiles;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, NULL, &g_pLightListBuffer ) );

	bd.Usage = D3D11_USAGE_STAGING;
    bd.BindFlags = 0;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    bd.MiscFlags = 0;
	V_RETURN( pd3dDevice->CreateBuffer( &bd, NULL, &g_pLightListBuffer_staged ) );




	g_pLightListBufferUAV = NULL; 

	D3D11_UNORDERED_ACCESS_VIEW_DESC uavbuffer_desc;
	ZeroMemory( &uavbuffer_desc, sizeof(uavbuffer_desc) );

	uavbuffer_desc.Format = DXGI_FORMAT_R10G10B10A2_UINT;
	uavbuffer_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
	uavbuffer_desc.Buffer.NumElements = iNrDWordsRequred*nrTiles;
	
	hr = pd3dDevice->CreateUnorderedAccessView( g_pLightListBuffer, &uavbuffer_desc, &g_pLightListBufferUAV );

	g_pLightListBufferSRV = NULL;

	D3D11_SHADER_RESOURCE_VIEW_DESC srvbuffer_desc;
	ZeroMemory( &srvbuffer_desc, sizeof(srvbuffer_desc) );
	srvbuffer_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	//srvbuffer_desc.Buffer.ElementWidth = desc.ByteWidth / elemSize;

	assert((iNrDWordsRequred&0x3)==0);		// must be a multiple of 4 (read lists back in as raw 32 bit dwords)
	srvbuffer_desc.Buffer.NumElements = (iNrDWordsRequred/4)*nrTiles;
	srvbuffer_desc.Format = DXGI_FORMAT_R32G32B32A32_UINT;
	hr = pd3dDevice->CreateShaderResourceView( g_pLightListBuffer, &srvbuffer_desc, &g_pLightListBufferSRV );




	shader_pipeline.RegisterResourceView("g_vLightList", g_pLightListBufferSRV);


	////////////////////////////////////////////////
	V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
	//V_RETURN( g_D3DSettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
#if 0
    switch( uMsg )
    {
        case WM_KEYDOWN:    // Prevent the camera class to use some prefefined keys that we're already using    
		{
            switch( (UINT)wParam )
            {
                case VK_CONTROL:    
                case VK_LEFT:
				{
					g_fL0ay -= 0.05f;
					return 0;
				}
				break;
                case VK_RIGHT:         
				{
					g_fL0ay += 0.05f;
					return 0;
				}
                case VK_UP:
				{
					g_fL0ax += 0.05f;
					if(g_fL0ax>(M_PI/2.5)) g_fL0ax=(M_PI/2.5);
					return 0;
				}
				break;
                case VK_DOWN:
				{
					g_fL0ax -= 0.05f;
					if(g_fL0ax<-(M_PI/2.5)) g_fL0ax=-(M_PI/2.5);
					return 0;
				}
				break;
				case 'F':
				{
					int iTing;
					iTing = 0;
				}
				default:
					;
            }
		}
		break;
    }
#endif
    // Pass all remaining windows messages to camera so it can respond to user input
    g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );


    return 0;
}

void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
	if(bKeyDown)
	{
		if(nChar=='M')
		{
			g_iCullMethod = 1-g_iCullMethod;
		}

		if(nChar=='V')
		{
			g_iVisualMode = 1-g_iVisualMode;
		}

		if (nChar == 'H')
		{
			g_iMenuVisib = 1 - g_iMenuVisib;
		}
	}
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D9 or D3D10 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
		s_bFirstTime = false;
		pDeviceSettings->d3d11.AutoCreateDepthStencil = false;

		/*
		s_bFirstTime = false;
        if( ( DXUT_D3D11_DEVICE == pDeviceSettings->ver &&
              pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE ) )
        {
            DXUTDisplaySwitchingToREFWarning( pDeviceSettings->ver );
        }

        // Enable 4xMSAA by default
        DXGI_SAMPLE_DESC MSAA4xSampleDesc = { 4, 0 };
        pDeviceSettings->d3d11.sd.SampleDesc = MSAA4xSampleDesc;*/
    }

    return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    // Update the camera's position based on user input 
    g_Camera.FrameMove( fElapsedTime );
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
    return true;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependant on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext )
{
    HRESULT hr;

    // Get device context
    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();


	// create text helper
	V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice(pd3dDevice, pd3dImmediateContext) );
	g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );

	InitUtils(pd3dDevice);

	// set compiler flag
	DWORD dwShaderFlags = D3D10_SHADER_ENABLE_STRICTNESS;
#if defined( DEBUG ) || defined( _DEBUG )
    // Set the D3D10_SHADER_DEBUG flag to embed debug information in the shaders.
    // Setting this flag improves the shader debugging experience, but still allows 
    // the shaders to be optimized and to run exactly the way they will run in 
    // the release configuration of this program.
    dwShaderFlags |= D3D10_SHADER_DEBUG;
#endif
	//dwShaderFlags |= D3DCOMPILE_OPTIMIZATION_LEVEL0;

	// create constant buffers
    D3D11_BUFFER_DESC bd;
	memset(&bd, 0, sizeof(bd));
    bd.Usage = D3D11_USAGE_DYNAMIC;
    bd.ByteWidth = (sizeof( cbMeshInstance )+0xf)&(~0xf);
    bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    bd.MiscFlags = 0;
    V_RETURN( pd3dDevice->CreateBuffer( &bd, NULL, &g_pMeshInstanceCB ) );

	CONST D3D10_SHADER_MACRO* pDefines = NULL;

	// compile per tile light list generation compute shader (with and without fine pruning)
	CONST D3D10_SHADER_MACRO sDefineExact[] = {{"FINE_PRUNING_ENABLED", NULL}, {NULL, NULL}};
	lightlist_exact_shader.CompileShaderFunction(pd3dDevice, L"lightlist_cs.hlsl", sDefineExact, "main", "cs_5_0", dwShaderFlags );
	lightlist_coarse_shader.CompileShaderFunction(pd3dDevice, L"lightlist_cs.hlsl", pDefines, "main", "cs_5_0", dwShaderFlags );

	// compile compute shader for screen-space AABB generation
#ifdef RECOMPILE_SCRBOUND_CS_SHADER
	scrbound_shader.CompileShaderFunction(pd3dDevice, L"scrbound_cs.hlsl", pDefines, "main", "cs_5_0", dwShaderFlags );
	FILE * fptr_out = fopen("scrbound_cs.bsh", "wb");
	fwrite(scrbound_shader.GetBufferPointer(), 1, scrbound_shader.GetBufferSize(), fptr_out);
	fclose(fptr_out);
#else
	scrbound_shader.CreateComputeShaderFromBinary(pd3dDevice, "scrbound_cs.bsh");
#endif


	// compile tiled forward lighting shader
	vert_shader.CompileShaderFunction(pd3dDevice, L"shader_lighting.hlsl", pDefines, "RenderSceneVS", "vs_5_0", dwShaderFlags );
	pix_shader.CompileShaderFunction(pd3dDevice, L"shader_lighting.hlsl", pDefines, "RenderScenePS", "ps_5_0", dwShaderFlags );


	// prepare shader pipeline
	shader_pipeline.SetVertexShader(&vert_shader);
	shader_pipeline.SetPixelShader(&pix_shader);

	// register constant buffers
	shader_pipeline.RegisterConstBuffer("cbMeshInstance", g_pMeshInstanceCB);

	
	// register samplers
	shader_pipeline.RegisterSampler("g_samWrap", GetDefaultSamplerWrap() );
	shader_pipeline.RegisterSampler("g_samClamp", GetDefaultSamplerClamp() );
	shader_pipeline.RegisterSampler("g_samShadow", GetDefaultShadowSampler() );

	// depth only pre-pass
	vert_shader_basic.CompileShaderFunction(pd3dDevice, L"shader_basic.hlsl", pDefines, "RenderSceneVS", "vs_5_0", dwShaderFlags );
	
	shader_dpthfill_pipeline.SetVertexShader(&vert_shader_basic);
	shader_dpthfill_pipeline.RegisterConstBuffer("cbMeshInstance", g_pMeshInstanceCB);
	
	
	// create all textures
	WCHAR dest_str[256];
	for(int t=0; t<NR_TEXTURES; t++)
	{
		wcscpy(dest_str, MODEL_PATH_W);
		wcscat(dest_str, tex_names[t]);

		V_RETURN(DXUTCreateShaderResourceViewFromFile(pd3dDevice, dest_str, &g_pTexturesHandler[t]));


		shader_pipeline.RegisterResourceView(stex_names[t], g_pTexturesHandler[t]);
		if(t==1) shader_dpthfill_pipeline.RegisterResourceView(stex_names[t], g_pTexturesHandler[t]);
	}
	

	// create vertex decleration
	const D3D11_INPUT_ELEMENT_DESC vertexlayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,	0, ATTR_OFFS(SFilVert, vert),  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,		0, ATTR_OFFS(SFilVert, s), D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,	0, ATTR_OFFS(SFilVert, norm), D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32G32B32A32_FLOAT,0, ATTR_OFFS(SFilVert, vOs), D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 2, DXGI_FORMAT_R32G32B32A32_FLOAT,0, ATTR_OFFS(SFilVert, vOt), D3D11_INPUT_PER_VERTEX_DATA, 0 }
    };
    V_RETURN( pd3dDevice->CreateInputLayout( vertexlayout, ARRAYSIZE( vertexlayout ), 
                                             vert_shader.GetBufferPointer(), vert_shader.GetBufferSize(), 
                                             &g_pVertexLayout ) );


	g_cMesh.ReadMeshFil(pd3dDevice, MODEL_PATH  MODEL_NAME, 4000.0f, true, true);
	


	bd.ByteWidth = (sizeof( cbBoundsInfo )+0xf)&(~0xf);
	V_RETURN( pd3dDevice->CreateBuffer( &bd, NULL, &g_pLightClipInfo ) );


	D3D11_SHADER_RESOURCE_VIEW_DESC srvbuffer_desc;

	// attribute data for lights such as attenuation, color, etc.
	{
		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		bd.CPUAccessFlags = 0;
		bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bd.ByteWidth = sizeof( SFiniteLightData ) * MAX_NR_LIGHTS_PER_CAMERA;
		bd.StructureByteStride = sizeof( SFiniteLightData );
		V_RETURN( pd3dDevice->CreateBuffer( &bd, NULL, &g_pLightDataBuffer ) );

		bd.Usage = D3D11_USAGE_STAGING;
		bd.BindFlags = 0;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		bd.MiscFlags = 0;
		bd.ByteWidth = sizeof( SFiniteLightData ) * MAX_NR_LIGHTS_PER_CAMERA;
		bd.StructureByteStride = 0;
		V_RETURN( pd3dDevice->CreateBuffer( &bd, NULL, &g_pLightDataBuffer_staged ) );

		ZeroMemory( &srvbuffer_desc, sizeof(srvbuffer_desc) );
		srvbuffer_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;//D3D11_SRV_DIMENSION_BUFFEREX
		srvbuffer_desc.Buffer.NumElements = MAX_NR_LIGHTS_PER_CAMERA;
		srvbuffer_desc.Format = DXGI_FORMAT_UNKNOWN;
		hr = pd3dDevice->CreateShaderResourceView( g_pLightDataBuffer, &srvbuffer_desc, &g_pLightDataBufferSRV );

		shader_pipeline.RegisterResourceView("g_vLightData", g_pLightDataBufferSRV);
	}


	// buffer for GPU generated screen-space AABB per light
	{
		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		bd.CPUAccessFlags = 0;
		bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bd.ByteWidth = 2 * sizeof(Vec3) * MAX_NR_LIGHTS_PER_CAMERA;
		bd.StructureByteStride = sizeof(Vec3);
		V_RETURN(pd3dDevice->CreateBuffer(&bd, NULL, &g_pScrSpaceAABounds));


		// ability to make a staged copy for debugging purposes
		bd.Usage = D3D11_USAGE_STAGING;
		bd.BindFlags = 0;//D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
		bd.MiscFlags = 0;
		bd.StructureByteStride = 0;
		bd.ByteWidth = 2 * sizeof(Vec3) * MAX_NR_LIGHTS_PER_CAMERA;
		V_RETURN(pd3dDevice->CreateBuffer(&bd, NULL, &g_pScrSpaceAABounds_staged));


		D3D11_UNORDERED_ACCESS_VIEW_DESC uavbuffer_desc;
		ZeroMemory(&uavbuffer_desc, sizeof(uavbuffer_desc));

		uavbuffer_desc.Format = DXGI_FORMAT_UNKNOWN;
		uavbuffer_desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
		uavbuffer_desc.Buffer.NumElements = MAX_NR_LIGHTS_PER_CAMERA * 2;

		hr = pd3dDevice->CreateUnorderedAccessView(g_pScrSpaceAABounds, &uavbuffer_desc, &g_pScrBoundUAV);



		ZeroMemory(&srvbuffer_desc, sizeof(srvbuffer_desc));
		srvbuffer_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;//D3D11_SRV_DIMENSION_BUFFEREX
		//srvbuffer_desc.Buffer.ElementWidth = desc.ByteWidth / elemSize;
		srvbuffer_desc.Buffer.NumElements = MAX_NR_LIGHTS_PER_CAMERA * 2;
		//srvbuffer_desc.Buffer.ElementWidth = sizeof( Vec3 );
		srvbuffer_desc.Format = DXGI_FORMAT_UNKNOWN;
		hr = pd3dDevice->CreateShaderResourceView(g_pScrSpaceAABounds, &srvbuffer_desc, &g_pScrBoundSRV);
	}


	// a nonuniformly scaled OBB per light
	{
		bd.Usage = D3D11_USAGE_DEFAULT;
		bd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
		bd.CPUAccessFlags = 0;
		bd.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
		bd.ByteWidth = sizeof(SFiniteLightBound) * MAX_NR_LIGHTS_PER_CAMERA;
		bd.StructureByteStride = sizeof(SFiniteLightBound);
		V_RETURN(pd3dDevice->CreateBuffer(&bd, NULL, &g_pOrientedBounds));

		bd.Usage = D3D11_USAGE_STAGING;
		bd.BindFlags = 0;
		bd.MiscFlags = 0;
		bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
		bd.StructureByteStride = 0;
		V_RETURN(pd3dDevice->CreateBuffer(&bd, NULL, &g_pOrientedBounds_staged));

		ZeroMemory(&srvbuffer_desc, sizeof(srvbuffer_desc));
		srvbuffer_desc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;//D3D11_SRV_DIMENSION_BUFFEREX
		//srvbuffer_desc.Buffer.ElementWidth = desc.ByteWidth / elemSize;
		srvbuffer_desc.Buffer.NumElements = MAX_NR_LIGHTS_PER_CAMERA;
		//srvbuffer_desc.Buffer.ElementWidth = sizeof( Vec3 );
		srvbuffer_desc.Format = DXGI_FORMAT_UNKNOWN;
		hr = pd3dDevice->CreateShaderResourceView(g_pOrientedBounds, &srvbuffer_desc, &g_pOrientedBoundsSRV);
	}




	const D3D11_INPUT_ELEMENT_DESC simplevertexlayout[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,	0, ATTR_OFFS(SFilVert, vert),  D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

	V_RETURN( pd3dDevice->CreateInputLayout( simplevertexlayout, ARRAYSIZE( simplevertexlayout ), 
                                             vert_shader_basic.GetBufferPointer(), vert_shader_basic.GetBufferSize(), 
                                             &g_pVertexSimpleLayout ) );


	BuildLightsBuffer();

	return S_OK;
}

//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
	g_DialogResourceManager.OnD3D11DestroyDevice();
	SAFE_DELETE( g_pTxtHelper );

	g_tex_depth.CleanUp();
	

	SAFE_RELEASE( g_pLightDataBufferSRV );
	SAFE_RELEASE( g_pLightDataBuffer );
	SAFE_RELEASE( g_pLightDataBuffer_staged );

	SAFE_RELEASE( g_pMeshInstanceCB );
	SAFE_RELEASE( g_pLightClipInfo );

	SAFE_RELEASE( g_pScrBoundUAV );
	SAFE_RELEASE( g_pScrBoundSRV );
	SAFE_RELEASE(g_pScrSpaceAABounds);
	SAFE_RELEASE(g_pScrSpaceAABounds_staged);

	SAFE_RELEASE(g_pLightListBufferUAV);
	SAFE_RELEASE(g_pLightListBufferSRV);
	SAFE_RELEASE( g_pLightListBuffer );
	SAFE_RELEASE( g_pLightListBuffer_staged );

	SAFE_RELEASE(g_pOrientedBoundsSRV);
	SAFE_RELEASE( g_pOrientedBounds );
	SAFE_RELEASE( g_pOrientedBounds_staged );
//





	for(int t=0; t<NR_TEXTURES; t++)
		SAFE_RELEASE( g_pTexturesHandler[t] );

	SAFE_RELEASE( g_pVertexLayout );
	SAFE_RELEASE( g_pVertexSimpleLayout );
	g_cMesh.CleanUp();

	vert_shader.CleanUp();
	pix_shader.CleanUp();
	vert_shader_basic.CleanUp();
	
	scrbound_shader.CleanUp();
	lightlist_coarse_shader.CleanUp();
	lightlist_exact_shader.CleanUp();


	DeinitUtils();
}


// [0;1] but right hand coordinate system
void myFrustum(float * pMat, const float fLeft, const float fRight, const float fBottom, const float fTop, const float fNear, const float fFar)
{
	// first column
	pMat[0*4 + 0] = (2 * fNear) / (fRight - fLeft); pMat[0*4 + 1] = 0; pMat[0*4 + 2] = 0; pMat[0*4 + 3] = 0;

	// second column
	pMat[1*4 + 0] = 0; pMat[1*4 + 1] = (2 * fNear) / (fTop - fBottom); pMat[1*4 + 2] = 0; pMat[1*4 + 3] = 0;

	// fourth column
	pMat[3*4 + 0] = 0; pMat[3*4 + 1] = 0; pMat[3*4 + 2] = -(fFar * fNear) / (fFar - fNear); pMat[3*4 + 3] = 0;

	// third column
	pMat[2*4 + 0] = (fRight + fLeft) / (fRight - fLeft);
	pMat[2*4 + 1] = (fTop + fBottom) / (fTop - fBottom);
	pMat[2*4 + 2] = -fFar / (fFar - fNear);
	pMat[2*4 + 3] = -1;

#ifdef LEFT_HAND_COORDINATES
	for(int r=0; r<4; r++) pMat[2*4 + r] = -pMat[2*4 + r];
#endif
}