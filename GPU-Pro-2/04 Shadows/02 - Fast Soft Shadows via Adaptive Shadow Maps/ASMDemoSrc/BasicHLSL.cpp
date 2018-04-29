#include "DXUT/Core/DXUT.h"
#include "DXUT/Optional/DXUTgui.h"
#include "DXUT/Optional/DXUTsettingsdlg.h"
#include "DXUT/Optional/DXUTres.h"
#include "DXUT/Optional/SDKmisc.h"
#include "DXUT/Optional/DXUTcamera.h"
#include "Util/ShaderObject9.h"
#include "Util/DebugRenderer9.h"
#include "ASM/ASM.h"
#include "Scene.h"
#include <direct.h>

bool CALLBACK IsDeviceAcceptable(D3DCAPS9* pCaps, D3DFORMAT AdapterFormat, D3DFORMAT BackBufferFormat, bool bWindowed, void*)
{
  return pCaps->PixelShaderVersion>=D3DPS_VERSION(3,0);
}

bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* pDeviceSettings, void*)
{
  pDeviceSettings->d3d9.pp.PresentationInterval = D3DPRESENT_INTERVAL_IMMEDIATE;
  return true;
}

ID3DXFont*                 g_pFont;
ID3DXSprite*               g_pSprite;
CDXUTDialogResourceManager g_DialogResourceManager;
CDXUTDialog                g_HUD;
CD3DSettingsDlg            g_SettingsDlg;
CFirstPersonCamera         g_Camera;

static float g_LightAngle = 0.22f;
static Vec4  g_PenumbraSizeParams(220.0f, -0.05f, 0, 0);
static bool  g_MRF = true;
static bool  g_Layer = true;
static bool  g_ShowShaderCost = false;
static bool  g_ShowTiles = false;
static bool  g_bShowGUI = true;
static bool  g_bUseCameraPath = true;

static Scene* g_Scene = NULL;
static DebugRenderer9 g_Debug;
static unsigned g_StartTime = GetTickCount();

static class AdaptiveShadowMap: public ASM
{
  virtual void GetSceneAABB(Vec3* AABBMin, Vec3* AABBMax)
  {
    *AABBMin = Vec3(-1000, 0,  -1000);
    *AABBMax = Vec3( 1000, 100, 1000);
  }
  virtual void RenderScene(const Mat4x4& ViewMat, const Mat4x4& ProjMat)
  {
    g_Scene->RenderShadowMap(ViewMat*ProjMat);
  }
  virtual void RenderScene(const Mat4x4& ViewMat, const Mat4x4& ProjMat, const Vec4& VSParam, const Vec4& PSParam)
  {
    g_Scene->RenderShadowMapLayer(ViewMat*ProjMat, VSParam, PSParam);
  }
} g_AdaptiveShadowMap;

inline Vec3 GetLightDir()
{
  const float c_PI = 3.14159f;
  return Vec3(0, sinf(g_LightAngle*c_PI), cosf(g_LightAngle*c_PI));
}

#define IDC_TOGGLEFULLSCREEN    1
#define IDC_CHANGEDEVICE        4
#define IDC_MRF                 5
#define IDC_LAYER               6
#define IDC_SHOW_SHADER_COST    7
#define IDC_SHOW_TILES          8
#define IDC_PENUMBRA_C0_STATIC  9
#define IDC_PENUMBRA_C0         10
#define IDC_PENUMBRA_C1         11
#define IDC_LIGHT_ANGLE_STATIC  12
#define IDC_LIGHT_ANGLE         13

void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
  switch(nControlID)
  {
  case IDC_TOGGLEFULLSCREEN: DXUTToggleFullScreen(); break;
  case IDC_CHANGEDEVICE:     g_SettingsDlg.SetActive(!g_SettingsDlg.IsActive()); break;
  case IDC_SHOW_TILES:       g_ShowTiles = g_HUD.GetCheckBox(IDC_SHOW_TILES)->GetChecked(); break;
  case IDC_MRF:              g_MRF = g_HUD.GetCheckBox(IDC_MRF)->GetChecked(); break;
  case IDC_LAYER:            g_Layer = g_HUD.GetCheckBox(IDC_LAYER)->GetChecked(); break;
  case IDC_SHOW_SHADER_COST: g_ShowShaderCost = g_HUD.GetCheckBox(IDC_SHOW_SHADER_COST)->GetChecked(); break;
  case IDC_PENUMBRA_C0:      g_PenumbraSizeParams.x = (float)g_HUD.GetSlider(IDC_PENUMBRA_C0)->GetValue(); break;
  case IDC_PENUMBRA_C1:      g_PenumbraSizeParams.y = (float)g_HUD.GetSlider(IDC_PENUMBRA_C1)->GetValue()/50.0f - 1.0f; break;
  case IDC_LIGHT_ANGLE:
    g_LightAngle = (float)g_HUD.GetSlider(IDC_LIGHT_ANGLE)->GetValue()/100.0f;
    const float EPS = 1e-2f;
    float f = g_LightAngle - 0.5f;
    g_LightAngle = fabs(f)<1e-2f ? g_LightAngle + (float)_copysign(EPS, f) : g_LightAngle;
    g_AdaptiveShadowMap.SetLightDir(GetLightDir());
    break;
  }
}

HRESULT CALLBACK OnResetDevice(IDirect3DDevice9* Device9, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void*)
{
  const int c_ElementWidth = 145;
  const int c_ElementHeight = 24;
  int iY = 10; int iX = DXUTGetD3D9BackBufferSurfaceDesc()->Width - c_ElementWidth - 10;
  g_HUD.RemoveAllControls();
  g_HUD.AddCheckBox(IDC_MRF, L"Multi-Resolution Filtering", iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, g_MRF);
  g_HUD.AddCheckBox(IDC_LAYER, L"Layer", iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, g_Layer);
  g_HUD.AddCheckBox(IDC_SHOW_SHADER_COST, L"Show Shader Cost", iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, g_ShowShaderCost);
  g_HUD.AddCheckBox(IDC_SHOW_TILES, L"Show Tiles", iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, g_ShowTiles);
  iY += c_ElementHeight;
  g_HUD.AddStatic(IDC_LIGHT_ANGLE_STATIC, L"Light Direction", iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight);
  g_HUD.AddSlider(IDC_LIGHT_ANGLE, iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, 1, 100, int(g_LightAngle*100.0f));
  iY += c_ElementHeight;
  g_HUD.AddStatic(IDC_PENUMBRA_C0_STATIC, L"Light Area", iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight);
  g_HUD.AddSlider(IDC_PENUMBRA_C0, iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, 1, 500, int(g_PenumbraSizeParams.x));
  //g_HUD.AddSlider(IDC_PENUMBRA_C1, iX, iY += c_ElementHeight, c_ElementWidth, c_ElementHeight, 1, 100, int(g_PenumbraSizeParams.y*50.0f + 50.0f));

  HRESULT hr = S_OK;

  if(g_pFont) V_RETURN(g_pFont->OnResetDevice());
  V_RETURN(D3DXCreateSprite(Device9, &g_pSprite));
  V_RETURN(g_DialogResourceManager.OnD3D9ResetDevice());
  V_RETURN(g_SettingsDlg.OnD3D9ResetDevice());

  V_RETURN(g_Scene->Init(Device9));

  V_RETURN(g_Debug.Init(Device9));
  g_Debug.SetViewportTransform(pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height);

  V_RETURN(g_AdaptiveShadowMap.Init(Device9));
  g_AdaptiveShadowMap.SetLightDir(GetLightDir());

  float fAspectRatio = (float)pBackBufferSurfaceDesc->Width/(float)pBackBufferSurfaceDesc->Height;
  g_Camera.SetProjParams(D3DX_PI/3, fAspectRatio, 0.1f, 200.0f);

  return hr;
}

void CALLBACK OnLostDevice(void*)
{
  if(g_pFont) g_pFont->OnLostDevice();
  SAFE_RELEASE(g_pSprite);
  g_DialogResourceManager.OnD3D9LostDevice();
  g_SettingsDlg.OnD3D9LostDevice();

  g_Debug.Release();
  g_AdaptiveShadowMap.Release();
  g_Scene->Release();
}

HRESULT CALLBACK OnCreateDevice(IDirect3DDevice9* Device9, const D3DSURFACE_DESC* pBackBufferSurfaceDesc, void*)
{
  HRESULT hr;
  g_pFont = NULL;
  V_RETURN(D3DXCreateFont(Device9, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH|FF_DONTCARE, L"Arial", &g_pFont));
  V_RETURN(g_DialogResourceManager.OnD3D9CreateDevice(Device9));
  V_RETURN(g_SettingsDlg.OnD3D9CreateDevice(Device9));
  g_Scene = new Scene("Data\\Hut\\scene.txt", "Data\\Hut\\camera.txt");
  return S_OK;
}

void CALLBACK OnDestroyDevice(void*)
{
  OnLostDevice(0);
  SAFE_RELEASE(g_pFont);
  g_DialogResourceManager.OnD3D9DestroyDevice();
  g_SettingsDlg.OnD3D9DestroyDevice();
  SAFE_DELETE(g_Scene);
}

void RenderText(double fTime)
{
  CDXUTTextHelper txtHelper(g_pFont, g_pSprite, 15);
  txtHelper.Begin();
  txtHelper.SetInsertionPos(10, 10);
  txtHelper.SetForegroundColor(D3DXCOLOR(1.0f, 0.75f, 0.0f, 1.0f));
  txtHelper.DrawTextLine(DXUTGetFrameStats( DXUTIsVsyncEnabled()));
  if(g_bShowGUI)
  {
    const D3DSURFACE_DESC* pd3dsdBackBuffer = DXUTGetD3D9BackBufferSurfaceDesc();
    txtHelper.SetInsertionPos(10, pd3dsdBackBuffer->Height - 100);
    txtHelper.DrawTextLine( L"Controls:\n"
                            L"  F1 to toggle GUI\n"
                            L"  Move: A,W,S,D or Arrow Keys\n"
                            L"  Move up/down: Q,E or PgUp,PgDn\n"
                            L"  Reset camera: Home\n");
    if(g_ShowShaderCost && g_MRF)
    {
      txtHelper.SetInsertionPos(pd3dsdBackBuffer->Width - 600, 10);
      txtHelper.SetForegroundColor(D3DXCOLOR(0, 0, 0, 1.0f));
      txtHelper.DrawTextLine( L"Shader cost legend:\n"
                              L"  blue: no shadow map samples + 2 samples from helper textures\n"
                              L"  green: 18 shadow map samples + 6 samples from helper textures\n"
                              L"  red: 36 shadow map samples + 8 samples from helper textures\n");
    }
  }
  txtHelper.End();
}

void CALLBACK OnFrameRender(IDirect3DDevice9* Device9, double fTime, float fElapsedTime, void*)
{
  if(g_SettingsDlg.IsActive())
  {
    g_SettingsDlg.OnRender(fElapsedTime);
    return;
  }

  Device9->BeginScene();

  Mat4x4 ViewMat = g_bUseCameraPath ? g_Scene->GetCameraPath().GetTransform(GetTickCount() - g_StartTime) : Mat4x4((float*)g_Camera.GetViewMatrix());
  Mat4x4 ProjMat((float*)g_Camera.GetProjMatrix());
  Mat4x4 ViewProj = ViewMat*ProjMat;

  g_AdaptiveShadowMap.Render(ViewProj);

  Device9->Clear(0, NULL, D3DCLEAR_TARGET|D3DCLEAR_ZBUFFER, 0xffffffff, 1.0f, 0);

  Device9->SetRenderState(D3DRS_COLORWRITEENABLE, 0);
  g_Scene->RenderDepthPass(ViewProj);

  Device9->SetRenderState(D3DRS_COLORWRITEENABLE, ~0UL);
  Device9->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
  g_AdaptiveShadowMap.SetTextures(1, 2, 3, 4);
  g_Scene->RenderColorPass(ViewProj, GetLightDir(), Mat4x4::Inverse(ViewMat).r[3], g_AdaptiveShadowMap.GetViewProj(), g_AdaptiveShadowMap.GetAtlasSize(), g_PenumbraSizeParams, g_MRF, g_Layer, g_ShowShaderCost);
  Device9->SetTexture(1, NULL);
  Device9->SetTexture(2, NULL);
  Device9->SetTexture(3, NULL);
  Device9->SetTexture(4, NULL);
  Device9->SetRenderState(D3DRS_ZWRITEENABLE, TRUE);

  if(g_ShowTiles)
   g_AdaptiveShadowMap.DebugDraw(g_Debug);
  g_Debug.Render();

  RenderText(fTime);
  if(g_bShowGUI)
    g_HUD.OnRender(fElapsedTime);

  Device9->EndScene();
}

LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing, void*)
{
  *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc(hWnd, uMsg, wParam, lParam);

  if(*pbNoFurtherProcessing)
  {
    return 0;
  }
  if(g_SettingsDlg.IsActive())
  {
    g_SettingsDlg.MsgProc(hWnd, uMsg, wParam, lParam);
    return 0;
  }
  *pbNoFurtherProcessing = g_HUD.MsgProc(hWnd, uMsg, wParam, lParam);
  if(*pbNoFurtherProcessing)
  {
    return 0;
  }
  g_Camera.HandleMessages(hWnd, uMsg, wParam, lParam);
  return 0;
}

void CALLBACK KeyboardProc(UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext)
{
  if(bKeyDown)
  {
    switch(nChar)
    {
    case VK_F1:
      g_bShowGUI = !g_bShowGUI; break;
    default:
      if(g_bUseCameraPath)
      {
        Mat4x4 m = Mat4x4::Inverse(g_Scene->GetCameraPath().GetTransform(GetTickCount() - g_StartTime));
        Vec3 v = Vec3(c_ZAxis)*m;
        D3DXVECTOR3 a = D3DXVECTOR3(&m.e41), b = D3DXVECTOR3(&v.x);
        g_Camera.SetViewParams(&a, &b);
        g_bUseCameraPath = false;
      }
      break;
    }
  }
}

void CALLBACK OnFrameMove(double, float fElapsedTime, void*)
{
  g_Camera.FrameMove(fElapsedTime);
}

#ifdef NDEBUG
INT WINAPI wWinMain( HINSTANCE, HINSTANCE, LPWSTR, int )
#else
int main()
#endif
{
  Math::Init();

  WCHAR pszFileName[MAX_PATH] = { };
  GetModuleFileName(NULL, pszFileName, sizeof(pszFileName));
  std::wstring FileName = pszFileName;
  _wchdir(FileName.substr(0, FileName.rfind(L"\\")).c_str());

  DXUTSetCallbackD3D9DeviceAcceptable(IsDeviceAcceptable);
  DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
  DXUTSetCallbackD3D9DeviceCreated(OnCreateDevice);
  DXUTSetCallbackD3D9DeviceDestroyed(OnDestroyDevice);
  DXUTSetCallbackD3D9DeviceReset(OnResetDevice);
  DXUTSetCallbackD3D9DeviceLost(OnLostDevice);
  DXUTSetCallbackD3D9FrameRender(OnFrameRender);
  DXUTSetCallbackMsgProc(MsgProc);
  DXUTSetCallbackKeyboard(KeyboardProc);
  DXUTSetCallbackFrameMove(OnFrameMove);

  g_SettingsDlg.Init(&g_DialogResourceManager);
  g_HUD.Init(&g_DialogResourceManager);
  g_HUD.SetCallback(OnGUIEvent);

  DXUTSetCursorSettings(true, true);
  DXUTInit(true, true);
  DXUTCreateWindow(L"Fast Soft Shadows");
  DXUTCreateDevice(true, 1000, 1000);

  DXUTMainLoop();

  return DXUTGetExitCode();
}
