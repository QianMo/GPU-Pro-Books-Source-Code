//#define PIX_DEBUG
#define _CRT_SECURE_NO_WARNINGS
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#define NOMINMAX
#include <DXUT.h>
#include <DXUTcamera.h>
#include <DXUTgui.h>
#include <DXUTsettingsDlg.h>
#include <SDKmisc.h>
#include <SDKMesh.h>
#include "ASM/AdaptiveShadowMap.h"
#define TW_NO_LIB_PRAGMA
#define TW_STATIC
#include "Misc/AtlasQuads.h"
#include "AntTweakBar/include/AntTweakBar.h"
#include "../Core/Math/Math.h"
#include "../Core/Math/IO.h"
#include "../Core/Util/DebugRenderer.h"
#include "../Core/Util/Random.h"
#include "Lighting/LightBuffer.h"
#include "Lighting/Light.h"
#include "Scene/Scene.h"
#include "Renderer/Renderer.h"
#include "ShaderCache/SimpleShader.h"
#include "Platform11/IABuffer11.h"
#include "Platform11/GPUTimer11.h"
#include "Platform11/StructuredBuffer11.h"
#include "TextureLoader/TextureLoader.h"
#include "PushBuffer/PreallocatedPushBuffer.h"
#include "Mesh/Mesh.h"
#include "Scene/Scene.h"
#include "Scene/Camera.h"
#include <algorithm>
#include "../Core/Util/XMLWriter.h"
#include "../Core/Rendering/Mesh/Attribute.h"
#include "_Shaders/HLSL2C.inc"
#include <time.h>
#include "resource.h"

Scene*                     g_Scene;
Camera                     g_Camera;

float g_fAspectRatio;
float g_fFOV = D3DX_PI / 4;

enum EScene
{
    Scene_PowerPlant,
    Scene_Columns,
    //Scene_Town,
} g_CurrentScene = Scene_Columns;//Scene_PowerPlant;

LightBuffer                g_LightBuffer;
RenderTarget2D             g_NormalBuffer;
RenderTarget2D             g_GeomNormalBuffer;
RenderTarget2D             g_ColorBuffer;
RenderTarget2D             g_HDRColorBuffer;

RenderTarget2D             g_DepthBufferCopy;

CDXUTDialogResourceManager g_DialogResourceManager;
CDXUTDialog                g_HUD;
CD3DSettingsDlg            g_SettingsDlg;
CDXUTTextHelper*           g_pTxtHelper = NULL;
CFirstPersonCamera         g_DXUTCamera;

Vec3  g_SunDirection;
Vec3  g_SunDir0;
Vec3  g_SunDir1;

unsigned int g_UpdateDeltaTime = 4500;
unsigned int g_currentTick;
unsigned int g_lastTick = GetTickCount();

bool g_ShadowsParallax = true;
bool g_ShadowsCrossfade = true;
bool g_MRF = true;
bool g_MovingSun = true;
bool g_ShowASMDebug = false;

bool g_PreRenderDebug = false;
bool g_PreRenderEnabled = true;

SceneRenderer g_Renderer;

DebugRenderer g_DebugRenderer;

unsigned int g_FrameCounter;

extern float gfx_asm_K;

class MyAdaptiveShadowMap : public CAdaptiveShadowMap
{
public:
    void RenderTile( const Vec4i& viewport, const Camera& renderCamera, bool isLayer, DeviceContext11& dc ) override;
    const Vec3 GetLightDirection( unsigned t ) override;
} * g_ASM;

RenderTarget2D g_WorkBufferDepth;
RenderTarget2D g_WorkBufferColor;

enum
{
    ShadowSetting_None,
    ShadowSetting_PCF,
    ShadowSetting_PCF9,
} g_ShadowSetting = ShadowSetting_PCF9;

bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void*)
{
    if((DXUT_D3D11_DEVICE==pDeviceSettings->ver && pDeviceSettings->d3d11.DriverType==D3D_DRIVER_TYPE_REFERENCE))
        DXUTDisplaySwitchingToREFWarning(pDeviceSettings->ver);
    //pDeviceSettings->d3d11.SyncInterval = 1;
    return true;
}

bool CALLBACK IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo* AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo* DeviceInfo, DXGI_FORMAT BackBufferFormat, bool bWindowed, void*)
{               	
    if(!DeviceInfo->ComputeShaders_Plus_RawAndStructuredBuffers_Via_Shader_4_x)
        return false;
    return true;
}

static void __stdcall ReloadShaders(void*)
{
    Platform::InvokeOnShutdown(Platform::Object_Shader);
    Platform::InvokeOnInit(Platform::Object_Shader);
}

static void __stdcall RebuildCaches(void*)
{
    g_ASM->Reset();
}

template<size_t N>
inline void UISetParam(CTwBar* pBar, const char* name, const int (&data)[N])
{
    if(pBar!=NULL)
        TwSetParam(pBar, NULL, name, TW_PARAM_INT32, N, data);
}

static void __stdcall GetFrameTime(void* value, void*)
{
    *static_cast<unsigned*>(value) = Vec4::Round<x>(1000.0f/DXUTGetFPS());
}

static void DeleteScene()
{
    SAFE_DELETE(g_Scene);
}

static void __stdcall SetScene( const void *value, void* )
{
    DeleteScene();

    static const struct
    {
        const char* m_fileName;
        D3DXVECTOR3 m_cameraPos;
        D3DXVECTOR3 m_lookDirection;
        Vec3 m_sunDir0;
        Vec3 m_sunDir1;
    } s_sceneDesc[] =
    {
        { "powerplant.xml", D3DXVECTOR3( -118.495811f, 56.9409981f, 41.1740074f ), D3DXVECTOR3( 0.811743617f, -0.531859636f, -0.241241157f ), Vec3::Normalize( Vec3( -0.67f, -0.67f, 0.33f ) ), Vec3::Normalize( Vec3( 0.67f, -0.67f, 0.33f ) ) },
        { "ShadowColumns.xml", D3DXVECTOR3( -39.8466988f, 9.26408482f, 47.5949821f ), D3DXVECTOR3( -0.159736812f, -0.218718752f, -0.962624550f ), Vec3::Normalize( Vec3( -0.245890126f, -0.311993349f, 0.823042572f ) ), Vec3::Normalize( Vec3( 0.370005995f, -0.393109643f, 0.787361503f ) ) },
        { "town.xml", D3DXVECTOR3( 218.976273f, 50.5728073f, 257.667328f ), D3DXVECTOR3( -0.212247059f, -0.497315764f, 0.841206372f ), Vec3( 0.67f, -0.67f, 0.33f ), Vec3( -0.67f, -0.67f, 0.33f ) },
    };

    g_CurrentScene = (EScene)*static_cast<const int*>( value );
    D3DXVECTOR3 cameraPos = s_sceneDesc[ g_CurrentScene ].m_cameraPos;
    D3DXVECTOR3 lookAtPoint = cameraPos + s_sceneDesc[ g_CurrentScene ].m_lookDirection;
    g_DXUTCamera.SetViewParams( &cameraPos, &lookAtPoint );

    g_SunDir0 = s_sceneDesc[ g_CurrentScene ].m_sunDir0;
    g_SunDir1 = s_sceneDesc[ g_CurrentScene ].m_sunDir1;

    g_Scene = new Scene();
    if( g_Scene->LoadXML( s_sceneDesc[ g_CurrentScene ].m_fileName ) )
    {
        CubeShadowMapPointLight::UpdateAll( &g_Renderer, g_Scene );
        g_ASM->Reset();
        return;
    }
    DeleteScene();
}

static void __stdcall GetScene( void *value, void* )
{
  *static_cast<int*>(value) = g_CurrentScene;
}

void ProcessNode(SceneObject* node, std::vector<Light*>& lightList)
{
    while (node != nullptr)
    {
        Light* light = dynamic_cast<Light*>(node);
        if (light != nullptr)
            lightList.push_back(light);
        
        ProcessNode(node->GetFirstChild(), lightList);
        node = node->NextSibling();
    }
}

void FillLightList(std::vector<Light*>& lightList)
{
    SceneObject* root = g_Scene->GetSceneRoot();
    ProcessNode(root, lightList);
}

static void InitUI()
{    
    TwBar* GeneralUIBar = TwNewBar("Options");
    { int a[] = { 700, 10 }; UISetParam(GeneralUIBar, "position", a); }
    { int a[] = { 270, 250 }; UISetParam(GeneralUIBar, "size", a); }

    LPCWSTR pSrcText = DXUTGetDeviceStats();
    size_t srcTextSize = wcslen(pSrcText) + 1;
    size_t convertedChars = 0;
    char deviceDesc[128], twDeviceText[128];
    wcstombs_s( &convertedChars, deviceDesc, srcTextSize, pSrcText, _TRUNCATE );
    sprintf_s( twDeviceText, " label='%s' ", deviceDesc );
    //TwAddButton( GeneralUIBar, "DT1", NULL, NULL, twDeviceText );
    //TwAddButton( GeneralUIBar, "Reload Shaders", &ReloadShaders, NULL, " key=Ctrl+R " );

    //TwAddVarCB(GeneralUIBar, "Frame Time, ms", TW_TYPE_UINT32, NULL, &GetFrameTime, NULL, " group = 'Performance' ");

    static const TwEnumVal c_SceneEnum[] =
    {
        { Scene_PowerPlant, "Power Plant" },
        { Scene_Columns, "Columns" },
        //{ Scene_Town, "Town" },
    };

    TwType sceneEnum = TwDefineEnum("SceneEnum", c_SceneEnum, ARRAYSIZE(c_SceneEnum));
    TwAddVarCB( GeneralUIBar, "Scene", sceneEnum, &SetScene, &GetScene, nullptr, " key=F1 " );

    static const TwEnumVal c_ShadowSettings[] =
    {
        { ShadowSetting_None, "None" },
        { ShadowSetting_PCF,  "PCF" },
        { ShadowSetting_PCF9, "PCF9" },
    };

    //TwType shadowSettings = TwDefineEnum("ShadowSettings", c_ShadowSettings, ARRAYSIZE(c_ShadowSettings));
    //TwAddVarRW(GeneralUIBar, "Enable Sun Shadows",    shadowSettings, &g_ShadowSetting,    " group = 'Sun Shadow Map' ");
    TwAddVarRW(GeneralUIBar, "Parallax correction",  TW_TYPE_BOOLCPP, &g_ShadowsParallax,  " group = 'Shadow Map Settings' ");
    TwAddVarRW(GeneralUIBar, "Crossfade",            TW_TYPE_BOOLCPP, &g_ShadowsCrossfade, " group = 'Shadow Map Settings' ");
//    TwAddVarRW(GeneralUIBar, "Sun Direction",        TW_TYPE_DIR3F,   &g_SunDirection,     " group = 'Shadow Map Settings' ");
    TwAddVarRW(GeneralUIBar, "Moving Sun",           TW_TYPE_BOOLCPP, &g_MovingSun,        " group = 'Shadow Map Settings' ");
    //TwAddVarRW(GeneralUIBar, "Update Delta Time",           TW_TYPE_UINT32, &g_UpdateDeltaTime,        " group = 'Shadow Map Settings' ");

    TwAddButton(GeneralUIBar, "Rebuild Caches", &RebuildCaches, NULL, " group = 'ASM' " );
    TwAddVarRW(GeneralUIBar, "MRF", TW_TYPE_BOOLCPP, &g_MRF, " group = 'ASM' " );
    TwAddVarRW(GeneralUIBar, "Show Debug", TW_TYPE_BOOLCPP, &g_ShowASMDebug, " group = 'ASM' " );
    //TwAddVarRW(GeneralUIBar, "gfx_asm_K",           TW_TYPE_FLOAT, &gfx_asm_K,        " group = 'ASM' ");
}
            
HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* Device11, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext)
{
    HRESULT hr = S_OK;
    V_RETURN(Platform::Init(Device11, DXUTGetD3D11DeviceContext()));

    V_RETURN(g_DialogResourceManager.OnD3D11CreateDevice(Device11, DXUTGetD3D11DeviceContext()));
    V_RETURN(g_SettingsDlg.OnD3D11CreateDevice(Device11));
    V_RETURN(g_DebugRenderer.Init());

    g_pTxtHelper = new CDXUTTextHelper(Device11, DXUTGetD3D11DeviceContext(), &g_DialogResourceManager, 15);

    TwInit(TW_DIRECT3D11, Device11);

    g_ASM = new MyAdaptiveShadowMap();

    SetScene( &g_CurrentScene, nullptr );

    InitUI();

    g_WorkBufferDepth.Init( 512, 256, DXGI_FORMAT_R16_TYPELESS, 1, nullptr, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL );
    g_WorkBufferColor.Init( 512, 256, DXGI_FORMAT_R16_FLOAT );

    return hr;
}

void CALLBACK OnD3D11DestroyDevice(void*)
{
    g_WorkBufferDepth.Clear();
    g_WorkBufferColor.Clear();

    delete g_ASM;

    g_DebugRenderer.Clear();

    g_LightBuffer.Clear();
    g_NormalBuffer.Clear();
    g_GeomNormalBuffer.Clear();
    g_ColorBuffer.Clear();
    g_HDRColorBuffer.Clear();
    g_DepthBufferCopy.Clear();
    
    DeleteScene();

    TwTerminate();

    SAFE_DELETE(g_pTxtHelper);
    
    g_DialogResourceManager.OnD3D11DestroyDevice();
    g_SettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();

    Platform::Shutdown();
}

Mat4x4 GetScreenToClip(unsigned long width, unsigned long height)
{
    Vec2 scale = Vec2(2.0f) / Vec2(float(width), float(height));
    return Mat4x4::Transpose(Mat4x4(
        Vec4(scale.x, 0.0f, 0.0f, -1.0f),
        Vec4(0.0f, -scale.y, 0.0f, 1.0f),
        c_ZAxis,
        c_WAxis));
}

void RenderScene( Camera& camera )
{
    DeviceContext11& dc = Platform::GetImmediateContext();
    dc.PushRC();

    dc.BindRT(0, &g_ColorBuffer);
    dc.BindRT(1, &g_GeomNormalBuffer);
    dc.BindRT(2, &g_NormalBuffer);

    dc.ClearRenderTarget(0, Vec4(0,0,0,0));
    dc.ClearDepthStencil(1.0f, 0);

    g_Renderer.DrawPrePass(g_Scene, &camera);

    dc.RestoreRC();
        
    bool lightBufferRendered = g_LightBuffer.Render(&g_NormalBuffer, Platform::GetBackBufferDS(), &g_GeomNormalBuffer, &camera);
    dc.RestoreRC();

    Platform::GetBackBufferDS()->CopyTo(g_DepthBufferCopy);

    dc.ClearRenderTarget(0, Vec4(0,0,1,0));

    static const Vec4 c_ScreenQuadData[] =
    {
        Vec4(+1, -1, 1, 1),
        Vec4(-1, -1, 0, 1),
        Vec4(+1, +1, 1, 0),
        Vec4(-1, +1, 0, 0),
    };

    static StaticIABuffer<ARRAYSIZE(c_ScreenQuadData), sizeof(Vec4)> s_QuadVB(c_ScreenQuadData);
    dc.BindVertexBuffer(0, &s_QuadVB, 0);
    dc.SetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);

    dc.BindPS(0, &g_LightBuffer);
    dc.BindPS(1, &g_ColorBuffer);
    dc.BindPS(2, &g_NormalBuffer);
    dc.BindPS(3, &g_GeomNormalBuffer);

    if( g_ShadowSetting != ShadowSetting_None )
    {
        dc.BindPS(4, &g_DepthBufferCopy);
    }

    static const D3D11_INPUT_ELEMENT_DESC c_InputDesc[] = { { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, D3D11_APPEND_ALIGNED_ELEMENT, D3D11_INPUT_PER_VERTEX_DATA, 0 } };
    static SimpleShaderDesc s_ColorPassShaderDesc("ColorPass.shader", nullptr, "ColorPass.shader", nullptr, nullptr, nullptr, c_InputDesc, ARRAYSIZE(c_InputDesc));
    const char* shaderFlags[] = 
    { 
        lightBufferRendered                    ? "USE_LIGHTBUFFER"         : nullptr,
        g_ShadowSetting != ShadowSetting_None  ? "SUN_SHADOW"              : nullptr,
        g_ShadowSetting == ShadowSetting_PCF9  ? "ASM_PCF9"                : nullptr,
        g_ShadowsCrossfade                     ? "SHADOWS_CROSSFADE"       : nullptr,
        g_ShadowsParallax                      ? "SHADOWS_PARALLAX"        : nullptr,
        g_MRF                                  ? "ASM_MRF"                 : nullptr,
        g_ASM->PreRenderAvailable()            ? "ASM_PRERENDER_AVAILABLE" : nullptr,
    };
    s_ColorPassShaderDesc.SetFlags(ARRAYSIZE(shaderFlags), shaderFlags);
    g_SimpleShaderCache.Get(s_ColorPassShaderDesc).Bind();

    static size_t s_depthStencilBlockIndex = Platform::GetDepthStencilCache().ConcurrentGetIndex( DepthStencilDesc11( true, D3D11_DEPTH_WRITE_MASK_ZERO, D3D11_COMPARISON_NOT_EQUAL ) );
    dc.SetDepthStencilState( &Platform::GetDepthStencilCache().ConcurrentGetByIndex( s_depthStencilBlockIndex ) );

    auto desc = Platform::GetBackBufferRT()->GetDesc();
    Mat4x4 screenToWorld = GetScreenToClip(desc.Width, desc.Height) * g_Camera.GetViewProjectionInverse();

    ASM_ResolveShaderData asmShaderData;
    g_ASM->GetResolveShaderData( asmShaderData );

    g_ASM->SetResolveTextures( dc );

    PreallocatedPushBuffer<> pb;
    pb.PushConstantPS( screenToWorld, 0 );
    pb.PushConstantPS<UVec3>( g_ASM->GetLightDirection( g_currentTick ) );
    pb.PushConstantPS( float( g_LightBuffer.GetWidthInQuads() ) );
    pb.PushConstantPS( asmShaderData, 3 );

    pb.Draw(4, 0);
    pb.Execute();
    
    dc.PopRC();
    dc.InvalidateContextCache();
}

void DrawText(float fTime)
{
    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 10, 10 );
    g_pTxtHelper->SetForegroundColor( D3DXCOLOR( 1.0f, 0.75f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats(DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );
    g_pTxtHelper->End();
}

static const Mat4x4 c_DebugVertexTransform = Mat4x4::ScalingTranslationD3D( 100.0f, Vec2( 300.0f, 400.0f ) );
static const Mat4x4 c_InvDebugVertexTransform = Mat4x4::Inverse( c_DebugVertexTransform );

finline Vec2 DebugVertex(const Vec2& a)
{
  return a * c_DebugVertexTransform;
}

void CALLBACK OnD3D11FrameRender(ID3D11Device* Device11, ID3D11DeviceContext* Context11, double, float fElapsedTime, void*)
{
    unsigned int t = GetTickCount();
    unsigned int dt = t - g_lastTick;
    g_lastTick = t;

    g_Camera.SetViewMatrix( Mat4x4( (float*)g_DXUTCamera.GetViewMatrix() ) );

    DeviceContext11& dc = Platform::GetImmediateContext();
    dc.ClearRenderTarget(0, Vec4(0,0,1,0));
    dc.ClearDepthStencil(1.0f, 0);

    const DXGI_SURFACE_DESC* pDesc = DXUTGetDXGIBackBufferSurfaceDesc();
    g_DebugRenderer.SetViewportTransform(pDesc->Width, pDesc->Height);

    if( g_MovingSun )
        g_currentTick += dt;

    g_ASM->Tick( g_currentTick, dt, false, false, g_UpdateDeltaTime );
    
//    g_ASM->Update( CAABBox( -1000.0f, 1000.0f ) );

    g_ASM->PrepareRender( g_Camera, !g_PreRenderEnabled );

    if( g_PreRenderDebug && g_ASM->PreRenderAvailable() )
    {
        g_PreRenderEnabled = false;
        g_MovingSun = false;
    }

    g_ASM->Render( g_WorkBufferDepth, g_WorkBufferColor, dc );

    if( g_ShowASMDebug )
        g_ASM->DrawDebug( g_DebugRenderer );

    RenderScene(g_Camera);

    g_DebugRenderer.Render();

    Platform::GetImmediateContext().FlushToDevice();
    g_HUD.OnRender(fElapsedTime);
    TwDraw();

    dc.InvalidateContextCache();

    ++g_FrameCounter;
}

void MyAdaptiveShadowMap::RenderTile(
    const Vec4i& viewport,
    const Camera& renderCamera,
    bool isLayer,
    DeviceContext11& dc )
{
    printf( "(%.5lu) rendering %s: %gm x %gm\r\n",
        g_FrameCounter % 100000,
        isLayer ? "layer" : "tile",
        2.0f / renderCamera.GetProjection().e11,
        2.0f / renderCamera.GetProjection().e22 );

    D3D11_VIEWPORT vp = { };
    vp.TopLeftX = float( viewport.x );
    vp.TopLeftY = float( viewport.y );
    vp.Width = float( viewport.z );
    vp.Height = float( viewport.w );
    vp.MaxDepth = 1.0f;
    dc.SetViewport( vp );

    dc.SetRenderStateF<RS_SLOPE_SCALED_DEPTH_BIAS>( 2.0f );

    Camera camera( renderCamera );
    if( isLayer )
    {
        g_Renderer.DrawASMLayerShadowMap( g_Scene, &camera );
    }
    else
    {
        g_Renderer.DrawShadowMap( g_Scene, &camera );
    }

    dc.SetRenderStateF<RS_SLOPE_SCALED_DEPTH_BIAS>( 0.0f );
}

const Vec3 MyAdaptiveShadowMap::GetLightDirection( unsigned t )
{
    float f = float( ( t >> 5 ) & 0xfff ) / 8096.0f;
    return -Vec3::Normalize( Vec3::Lerp( g_SunDir0, g_SunDir1, fabsf( f * 2.0f - 1.0f ) ) );
}

void CALLBACK OnFrameMove(double fTime, float fElapsedTime, void*)
{
    g_DXUTCamera.FrameMove(fElapsedTime);
}

LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void*)
{
    if(*pbNoFurtherProcessing) return 0;

    if(TwEventWin(hWnd, uMsg, wParam, lParam))
         return 0;

    switch(uMsg)
    {
    case WM_KEYDOWN:
        if( wParam == VK_SHIFT ) g_DXUTCamera.SetScalers( 0.01f, 100.0f );
        break;
    case WM_KEYUP:
        if( wParam == VK_SHIFT ) g_DXUTCamera.SetScalers();
        break;
    case WM_CHAR:
        if( wParam == VK_SPACE )
        {
            g_PreRenderEnabled = true;
        }
        else if( wParam == VK_RETURN )
        {
            g_PreRenderDebug = !g_PreRenderDebug;
        }
        break;
    }
    if( *pbNoFurtherProcessing ) return 0;

    *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing ) return 0;

    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing ) return 0;

    g_DXUTCamera.HandleMessages(hWnd, uMsg, wParam, lParam);
    return 0;
}

HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* Device11, IDXGISwapChain* pSwapChain, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* )
{
    HRESULT hr;
    V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( Device11, pBackBufferSurfaceDesc ) );
    V_RETURN( g_SettingsDlg.OnD3D11ResizedSwapChain( Device11, pBackBufferSurfaceDesc ) );
    g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 200, 0 );

    RenderTarget2D colorRT, depthRT;
    V_RETURN( colorRT.Init( DXUTGetD3D11RenderTargetView() ) );
    V_RETURN( depthRT.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, DXGI_FORMAT_R24G8_TYPELESS, 1, nullptr, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL ) );
    V_RETURN( Platform::SetFrameBuffer( &colorRT, &depthRT ) );
    colorRT.Clear();
    depthRT.Clear();

    g_LightBuffer.Clear();
    g_NormalBuffer.Clear();
    g_ColorBuffer.Clear();
    g_GeomNormalBuffer.Clear();
    g_HDRColorBuffer.Clear();
    g_DepthBufferCopy.Clear();

    V_RETURN( g_LightBuffer.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height ) );
    V_RETURN( g_NormalBuffer.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, DXGI_FORMAT_R8G8B8A8_UNORM ) );
    V_RETURN( g_ColorBuffer.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ) );
    V_RETURN( g_GeomNormalBuffer.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, DXGI_FORMAT_R8G8B8A8_UNORM ) );
    V_RETURN( g_HDRColorBuffer.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, DXGI_FORMAT_R16G16B16A16_FLOAT ) );
    V_RETURN( g_DepthBufferCopy.Init( pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, DXGI_FORMAT_R24G8_TYPELESS, 1, nullptr, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_DEPTH_STENCIL ) );

    g_fAspectRatio = (float)pBackBufferSurfaceDesc->Width / (float)pBackBufferSurfaceDesc->Height;
    g_DXUTCamera.SetProjParams( g_fFOV, g_fAspectRatio, 0.1f, 1000.0f );
    
    g_Camera.SetProjection( Mat4x4( reinterpret_cast<const float*>( g_DXUTCamera.GetProjMatrix() ) ) );

    TwWindowSize( Platform::GetBackBufferRT()->GetDesc().Width, Platform::GetBackBufferRT()->GetDesc().Height );
    return S_OK;
}

void CALLBACK OnD3D11ReleasingSwapChain(void*)
{
    Platform::SetFrameBuffer(nullptr, nullptr);
}

void CALLBACK KeyboardProc(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext)
{
}

INT_PTR CALLBACK SceneDlgProc( HWND hDlg, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
    switch(uMsg)
    {
    case WM_INITDIALOG:
        {
            HWND hWndScenesList = GetDlgItem( hDlg, IDC_SCENES_COMBO );
            SendMessage( hWndScenesList, CB_ADDSTRING, (WPARAM)0, (LPARAM)L"Power Plant" ); 
            SendMessage( hWndScenesList, CB_ADDSTRING, (WPARAM)0, (LPARAM)L"Columns" ); 
            SendMessage( hWndScenesList, CB_SETCURSEL, (WPARAM)g_CurrentScene, (LPARAM)0 ); 
        }
        break;
    case WM_COMMAND:
        switch(LOWORD(wParam))
        {
        case IDCANCEL:
            EndDialog( hDlg, 1 );
            break;
        case IDOK:
            g_CurrentScene = (EScene)SendMessage( GetDlgItem( hDlg, IDC_SCENES_COMBO ), CB_GETCURSEL, 0, 0 );
            EndDialog( hDlg, 0 );
            break;
        }
        break;
    }
    return 0;
}

int main(int argc,char* argv[])
{
    if( DialogBox( GetModuleHandle(0), MAKEINTRESOURCE(IDD_SCENE_DIALOG), 0, SceneDlgProc ) == 1 ) 
    {
        return 1;
    }

    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

    Math::Init();

    Platform::SetDir(Platform::File_Mesh, "Data/");
    Platform::SetDir(Platform::File_Texture, "Data/");

    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackKeyboard( KeyboardProc );
    DXUTSetIsInGammaCorrectMode( true );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

    g_SettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.Init( &g_DialogResourceManager );

    DXUTInit( true, true, NULL );
    DXUTSetCursorSettings( true, true );
    DXUTCreateWindow( L"Test" );
    DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, 1000, 1000 );

    DXUTMainLoop();

    Platform::RemoveAllCallbacks();

    return DXUTGetExitCode();
}
