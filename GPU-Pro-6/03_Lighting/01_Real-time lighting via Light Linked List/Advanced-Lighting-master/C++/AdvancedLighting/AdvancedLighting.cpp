//--------------------------------------------------------------------------------------
// File: AdvancedLighting.cpp
//
// This sample demonstrates lighting via a GPU linked list.
// This sample is based off Microsoft DirectX SDK sample CascadedShadowMap11
//
// Copyright (c) Microsoft Corporation. All rights reserved.
//--------------------------------------------------------------------------------------
#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "SDKMesh.h" 
#include "SceneManager.h" 
 

using namespace DirectX;

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
SceneManager                g_Scene;

CDXUTDialogResourceManager  g_DialogResourceManager; // manager for shared resources of dialogs
CFirstPersonCamera          g_ViewerCamera;          
CFirstPersonCamera          g_LightCamera;          

CDXUTSDKMesh                g_MeshPowerPlant; 
CDXUTSDKMesh                g_TeapotMesh;

// DXUT GUI stuff     
CD3DSettingsDlg             g_D3DSettingsDlg;       // Device settings dialog
CDXUTDialog                 g_HUD;                  // manages the 3D   
CDXUTDialog                 g_SampleUI;             // dialog for sample specific controls
CDXUTTextHelper*            g_pTxtHelper = nullptr;
CDXUTComboBox*              g_DebugRenderingCombo;

bool                        g_bShowHelp           = false;    // If true, it renders the UI control text
bool                        g_bVisualizeCascades  = FALSE;
FLOAT                       g_fAspectRatio        = 1.0f;
 
//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
#define IDC_TOGGLEFULLSCREEN         1
#define IDC_TOGGLEWARP               2
#define IDC_CHANGEDEVICE             3
 
#define IDC_DEBUGRENDERING           4
#define IDC_TOGGLEDYNAMICLIGHTS      5

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool    CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );
void    CALLBACK OnFrameMove( double fTime, FLOAT fElapsedTime, void* pUserContext );
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, INT nControlID, CDXUTControl* pControl, void* pUserContext );
bool CALLBACK IsD3D11DeviceAcceptable(const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  FLOAT fElapsedTime, void* pUserContext );

void    InitApp();
void    DeinitApp();
void    RenderText();
HRESULT DestroyD3DComponents();
HRESULT CreateD3DComponents( ID3D11Device* pd3dDevice);  

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( _In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow )
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(hInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    UNREFERENCED_PARAMETER(nCmdShow );

    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( OnKeyboard );
    DXUTSetCallbackFrameMove( OnFrameMove );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );

    InitApp();

    DXUTInit( true, true, nullptr ); // Parse the command line, show msgboxes on error, no extra command line params

    DXUTSetCursorSettings( true, true ); // Show the cursor and clip it when in full screen
    DXUTCreateWindow( L"Advanced Lighting" );

    DXUTCreateDevice (D3D_FEATURE_LEVEL_11_0, true, 1280, 720 );

    DXUTMainLoop();                     // Enter into the DXUT render loop

    DeinitApp();

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{ 
  // Initialize dialogs
  g_D3DSettingsDlg.Init( &g_DialogResourceManager );
  g_SampleUI.Init( &g_DialogResourceManager );
  g_HUD.Init( &g_DialogResourceManager );
  
  g_HUD.SetCallback( OnGUIEvent ); INT iY = 10;
  
  // Add tons of GUI stuff
  g_HUD.AddButton( IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 0, iY, 170, 23 );
  g_HUD.AddButton( IDC_CHANGEDEVICE, L"Change device (F2)", 0, iY += 26, 170, 23, VK_F2 );
  g_HUD.AddButton( IDC_TOGGLEWARP, L"Toggle WARP (F4)", 0, iY += 26, 170, 23, VK_F4 );
  
  g_HUD.AddCheckBox( IDC_TOGGLEDYNAMICLIGHTS, L"Dynamic Lights", 0, iY+=26, 170, 23, true, VK_F8 );

  g_HUD.AddComboBox( IDC_DEBUGRENDERING, 0, iY +=26,  170, 23, VK_F9, false, &g_DebugRenderingCombo );

  g_DebugRenderingCombo->AddItem( L"Scene",     ULongToPtr( DEBUG_RENDERING_NONE     ) );
  g_DebugRenderingCombo->AddItem( L"Normals",   ULongToPtr( DEBUG_RENDERING_NORMALS  ) );
  g_DebugRenderingCombo->AddItem( L"Colors",    ULongToPtr( DEBUG_RENDERING_COLORS   ) );
  g_DebugRenderingCombo->AddItem( L"LLL",       ULongToPtr( DEBUG_RENDERING_LLL      ) );

  g_SampleUI.SetCallback( OnGUIEvent ); iY = 10;
}

//--------------------------------------------------------------------------------------
// Destroy the app 
//--------------------------------------------------------------------------------------
void DeinitApp()
{
}

//--------------------------------------------------------------------------------------
// Called right before creating a D3D device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);
  
  // Don't create the depth stencil buffer, we'll use our own
  pDeviceSettings->d3d11.AutoCreateDepthStencil = false;
  
  #if defined(DEBUG) || defined(_DEBUG)
    pDeviceSettings->d3d11.CreateFlags |= D3D11_CREATE_DEVICE_DEBUG;
  #endif
  
  // Done
  return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, FLOAT fElapsedTime, void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);
  UNREFERENCED_PARAMETER(fTime);

  // Update the camera's position based on user input 
  g_LightCamera.FrameMove( fElapsedTime );
  g_ViewerCamera.FrameMove( fElapsedTime );
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text
//--------------------------------------------------------------------------------------
void RenderText()
{
    UINT nBackBufferHeight = DXUTGetDXGIBackBufferSurfaceDesc()->Height;

    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 2, 0 );
    g_pTxtHelper->SetForegroundColor( Colors::Yellow );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );

    // Draw help
    if( g_bShowHelp )
    {
        g_pTxtHelper->SetInsertionPos( 2, nBackBufferHeight - 20 * 6 );
        g_pTxtHelper->SetForegroundColor( Colors::Orange );
        g_pTxtHelper->DrawTextLine( L"Controls:" );

        g_pTxtHelper->SetInsertionPos( 20, nBackBufferHeight - 20 * 5 );
        g_pTxtHelper->DrawTextLine( L"Move forward and backward with 'E' and 'D'\n"
                                    L"Move left and right with 'S' and 'D' \n"
                                    L"Click the mouse button to roate the camera\n");

        g_pTxtHelper->SetInsertionPos( 350, nBackBufferHeight - 20 * 5 );
        g_pTxtHelper->DrawTextLine( L"Hide help: F1\n"
                                    L"Quit: ESC\n" );
    }
    else
    {
        g_pTxtHelper->SetForegroundColor( Colors::White );
        g_pTxtHelper->DrawTextLine( L"Press F1 for help" );
    }

    g_pTxtHelper->End();
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);

  // Pass messages to dialog resource manager calls so GUI state is updated correctly
  *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
  if( *pbNoFurtherProcessing )
      return 0;
  
  // Pass messages to settings dialog if its active
  if( g_D3DSettingsDlg.IsActive() )
  {
      g_D3DSettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
      return 0;
  }
  
  // Give the dialogs a chance to handle the message first
  *pbNoFurtherProcessing = g_HUD.MsgProc( hWnd, uMsg, wParam, lParam );
  if( *pbNoFurtherProcessing )
      return 0;
  *pbNoFurtherProcessing = g_SampleUI.MsgProc( hWnd, uMsg, wParam, lParam );
  if( *pbNoFurtherProcessing )
      return 0;
  
  // Pass all remaining windows messages to camera so it can respond to user input
  g_ViewerCamera.HandleMessages( hWnd, uMsg, wParam, lParam );
  
  return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);
  UNREFERENCED_PARAMETER(bAltDown);

  if( bKeyDown )
  {
      switch( nChar )
      {
          case VK_F1:
              g_bShowHelp = !g_bShowHelp; break;
  
      }
  }
  else 
  {
    switch( nChar )
    {
      case 82:
        g_Scene.ReloadShaders(); break;
    }
  }
}


//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, INT nControlID, CDXUTControl* pControl, void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);
  UNREFERENCED_PARAMETER(pControl);
  UNREFERENCED_PARAMETER(nEvent);

  switch( nControlID )
  {
    case IDC_TOGGLEFULLSCREEN:
        DXUTToggleFullScreen(); break;
    case IDC_TOGGLEWARP:
        DXUTToggleWARP(); break;
    case IDC_CHANGEDEVICE:
        g_D3DSettingsDlg.SetActive( !g_D3DSettingsDlg.IsActive() ); break;
    break;

    case IDC_TOGGLEDYNAMICLIGHTS:
    {
      g_Scene.m_DynamicLights = !g_Scene.m_DynamicLights;

      break;  
    } 
    case IDC_DEBUGRENDERING:
    {
      g_Scene.m_DebugRendering = (DebugRendering)PtrToUlong( g_DebugRenderingCombo->GetSelectedData() );

      break;  
    }  
  }
}

//--------------------------------------------------------------------------------------
// Reject any D3D11 devices that aren't acceptable by returning false
//--------------------------------------------------------------------------------------
bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo* AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo* DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext )
{
  UNREFERENCED_PARAMETER(AdapterInfo);

  UNREFERENCED_PARAMETER(DeviceInfo);
  UNREFERENCED_PARAMETER(Output);

  UNREFERENCED_PARAMETER(BackBufferFormat);
  UNREFERENCED_PARAMETER(pUserContext);
  UNREFERENCED_PARAMETER(bWindowed);

  return true;
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{  
  UNREFERENCED_PARAMETER(pUserContext);

  g_MeshPowerPlant.Destroy();
  g_TeapotMesh.Destroy();
  DestroyD3DComponents();
}

//--------------------------------------------------------------------------------------
// Create any D3D11 resources that aren't dependent on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                      void* pUserContext )
{
  UNREFERENCED_PARAMETER(pBackBufferSurfaceDesc);
  UNREFERENCED_PARAMETER(pUserContext);

  HRESULT hr = S_OK;
  V_RETURN( g_MeshPowerPlant.Create( pd3dDevice, L"powerplant\\powerplant.sdkmesh" ) );
  V_RETURN( g_TeapotMesh.Create(     pd3dDevice, L"teapot\\teapot.sdkmesh" ) );

  return CreateD3DComponents( pd3dDevice );
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                          const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
  UNREFERENCED_PARAMETER(pSwapChain);
  UNREFERENCED_PARAMETER(pUserContext);

  HRESULT hr;
  XMVECTOR vMeshExtents = g_Scene.GetSceneAABBMax() - g_Scene.GetSceneAABBMin();
  XMVECTOR vMeshLength = XMVector3Length( vMeshExtents );
  FLOAT    fMeshLength = XMVectorGetByIndex( vMeshLength, 0);
  
  V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
  V_RETURN( g_D3DSettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
  
  g_fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT ) pBackBufferSurfaceDesc->Height;
  
  g_ViewerCamera.SetProjParams( XM_PI / 4, g_fAspectRatio, 0.05f, fMeshLength);
  
  g_Scene.OnResize(pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height);
     
  g_HUD.SetLocation( pBackBufferSurfaceDesc->Width - 170, 0 );
  g_HUD.SetSize( 170, 170 );
  g_SampleUI.SetLocation( pBackBufferSurfaceDesc->Width - 170, pBackBufferSurfaceDesc->Height - 300 );
  g_SampleUI.SetSize( 170, 300 );
  
  return S_OK;
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);

  g_DialogResourceManager.OnD3D11ReleasingSwapChain();
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                  FLOAT fElapsedTime, void* pUserContext )
{
  UNREFERENCED_PARAMETER(pUserContext);
  UNREFERENCED_PARAMETER(pd3dDevice);
  UNREFERENCED_PARAMETER(fTime);

  if( g_D3DSettingsDlg.IsActive() )
  {
    g_D3DSettingsDlg.OnRender( fElapsedTime );
    return;
  }
  
  auto pRTV = DXUTGetD3D11RenderTargetView();
  
  // Clear color
  pd3dImmediateContext->ClearRenderTargetView( pRTV, Colors::DimGray );

   // Start the frame
  g_Scene.InitFrame(&g_ViewerCamera, &g_LightCamera);
  
  // Draw Shadows
  g_Scene.RenderShadowCascades( &g_MeshPowerPlant );
  
  // Render the GBuffer
  g_Scene.RenderGBuffer(&g_MeshPowerPlant);
  
  // Process the light linked list
  g_Scene.ProcessLinkedList();
  
  // Composite the scene
  g_Scene.CompositeScene(pRTV);
  
  // Blended elements
  g_Scene.DrawAlpha(&g_TeapotMesh);

  // End the frame
  g_Scene.EndFrame(pRTV);
  
  // Hud
  {
     D3D11_VIEWPORT vp;
     vp.Width    = (FLOAT)DXUTGetDXGIBackBufferSurfaceDesc()->Width;
     vp.Height   = (FLOAT)DXUTGetDXGIBackBufferSurfaceDesc()->Height;
     vp.MinDepth = 0;
     vp.MaxDepth = 1;
     vp.TopLeftX = 0;
     vp.TopLeftY = 0;
  
     pd3dImmediateContext->RSSetViewports( 1, &vp);            
     pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, nullptr );
  
     DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );
  
     g_HUD.OnRender( fElapsedTime );
     g_SampleUI.OnRender( fElapsedTime );
     RenderText();
     DXUT_EndPerfEvent();
  }
}


//--------------------------------------------------------------------------------------
// When the user changes scene, recreate these components as they are scene 
// dependent.
//--------------------------------------------------------------------------------------
HRESULT CreateD3DComponents( ID3D11Device* pd3dDevice)
{
  HRESULT hr;
  
  auto pd3dImmediateContext = DXUTGetD3D11DeviceContext();
  V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
  V_RETURN( g_D3DSettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
  g_pTxtHelper  = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, 15 );
  
  static const XMVECTORF32 s_vecEye    = { 105.0f,  14.0f, -3.0f, 0.f };
  static const XMVECTORF32 s_vecLookAt = {   0.0f,  -7.5f,  0.0f, 0.f };
  XMFLOAT3                 vMin        = XMFLOAT3( -1000.0f, -1000.0f, -1000.0f );
  XMFLOAT3                 vMax        = XMFLOAT3( 1000.0f, 1000.0f, 1000.0f );
  
  g_ViewerCamera.SetViewParams( s_vecEye, s_vecLookAt );
  g_ViewerCamera.SetRotateButtons( true, false, false);
  g_ViewerCamera.SetScalers( 0.01f, 10.0f );
  g_ViewerCamera.SetDrag( true );
  g_ViewerCamera.SetEnableYAxisMovement( true );
  g_ViewerCamera.SetClipToBoundary( true, &vMin, &vMax );
  g_ViewerCamera.FrameMove( 0 );
  
  static const XMVECTORF32 s_lightEye = { -320.0f, 300.0f, -220.3f, 0.f };
  g_LightCamera.SetViewParams( s_lightEye, g_XMZero );
  g_LightCamera.SetRotateButtons( true, false, false );
  g_LightCamera.SetScalers( 0.01f, 50.0f );
  g_LightCamera.SetDrag( true );
  g_LightCamera.SetEnableYAxisMovement( true );
  g_LightCamera.SetClipToBoundary( true, &vMin, &vMax );
  g_LightCamera.SetProjParams( XM_PI / 4, 1.0f, 0.1f , 1000.0f);

  g_LightCamera.FrameMove( 0 );
         
  // Get the final sizes
  uint32_t width  = DXUTGetDXGIBackBufferSurfaceDesc()->Width; 
  uint32_t height = DXUTGetDXGIBackBufferSurfaceDesc()->Height;

  g_Scene.Init( pd3dDevice, pd3dImmediateContext, &g_MeshPowerPlant, width, height);
  
  return S_OK;
}

//--------------------------------------------------------------------------------------
HRESULT DestroyD3DComponents() 
{
  g_DialogResourceManager.OnD3D11DestroyDevice();
  g_D3DSettingsDlg.OnD3D11DestroyDevice();
  DXUTGetGlobalResourceCache().OnDestroyDevice();
  SAFE_DELETE( g_pTxtHelper );

  g_Scene.ReleaseResources();

  return S_OK;
}

