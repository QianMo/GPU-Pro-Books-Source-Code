//
// Copyright 2014 ADVANCED MICRO DEVICES, INC.  All Rights Reserved.
//
// AMD is granting you permission to use this software and documentation (if
// any) (collectively, the "Materials") pursuant to the terms and conditions
// of the Software License Agreement included with the Materials.  If you do
// not have a copy of the Software License Agreement, contact your AMD
// representative for a copy.
// You agree that you will not reverse engineer or decompile the Materials,
// in whole or in part, except as allowed by applicable law.
//
// WARRANTY DISCLAIMER: THE SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF
// ANY KIND.  AMD DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE, TITLE, NON-INFRINGEMENT, THAT THE SOFTWARE
// WILL RUN UNINTERRUPTED OR ERROR-FREE OR WARRANTIES ARISING FROM CUSTOM OF
// TRADE OR COURSE OF USAGE.  THE ENTIRE RISK ASSOCIATED WITH THE USE OF THE
// SOFTWARE IS ASSUMED BY YOU.
// Some jurisdictions do not allow the exclusion of implied warranties, so
// the above exclusion may not apply to You. 
// 
// LIMITATION OF LIABILITY AND INDEMNIFICATION:  AMD AND ITS LICENSORS WILL
// NOT, UNDER ANY CIRCUMSTANCES BE LIABLE TO YOU FOR ANY PUNITIVE, DIRECT,
// INCIDENTAL, INDIRECT, SPECIAL OR CONSEQUENTIAL DAMAGES ARISING FROM USE OF
// THE SOFTWARE OR THIS AGREEMENT EVEN IF AMD AND ITS LICENSORS HAVE BEEN
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.  
// In no event shall AMD's total liability to You for all damages, losses,
// and causes of action (whether in contract, tort (including negligence) or
// otherwise) exceed the amount of $100 USD.  You agree to defend, indemnify
// and hold harmless AMD and its licensors, and any of their directors,
// officers, employees, affiliates or agents from and against any and all
// loss, damage, liability and other expenses (including reasonable attorneys'
// fees), resulting from Your use of the Software or violation of the terms and
// conditions of this Agreement.  
//
// U.S. GOVERNMENT RESTRICTED RIGHTS: The Materials are provided with "RESTRICTED
// RIGHTS." Use, duplication, or disclosure by the Government is subject to the
// restrictions as set forth in FAR 52.227-14 and DFAR252.227-7013, et seq., or
// its successor.  Use of the Materials by the Government constitutes
// acknowledgement of AMD's proprietary rights in them.
// 
// EXPORT RESTRICTIONS: The Materials may be subject to export restrictions as
// stated in the Software License Agreement.
//

//--------------------------------------------------------------------------------------
// File: ComputeBasedTiledCulling.cpp
//
// Implements the Forward+ algorithm and the tiled deferred algorithm, for comparison.
//--------------------------------------------------------------------------------------

// DXUT now sits one directory up
#include "..\\DXUT\\Core\\DXUT.h"
#include "..\\DXUT\\Core\\DXUTmisc.h"
#include "..\\DXUT\\Optional\\DXUTgui.h"
#include "..\\DXUT\\Optional\\DXUTCamera.h"
#include "..\\DXUT\\Optional\\DXUTSettingsDlg.h"
#include "..\\DXUT\\Optional\\SDKmisc.h"
#include "..\\DXUT\\Optional\\SDKmesh.h"

// AMD SDK also sits one directory up
#include "..\\AMD_SDK\\AMD_SDK.h"
#include "..\\AMD_SDK\\ShaderCacheSampleHelper.h"

// Project includes
#include "resource.h"

#include "CommonUtil.h"
#include "ForwardPlusUtil.h"
#include "LightUtil.h"
#include "TiledDeferredUtil.h"

#pragma warning( disable : 4100 ) // disable unreference formal parameter warnings for /W4 builds

using namespace DirectX;
using namespace ComputeBasedTiledCulling;

//-----------------------------------------------------------------------------------------
// Constants
//-----------------------------------------------------------------------------------------
static const int TEXT_LINE_HEIGHT = 15;

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
CFirstPersonCamera          g_Camera;                // A first-person camera
CDXUTDialogResourceManager  g_DialogResourceManager; // manager for shared resources of dialogs
CD3DSettingsDlg             g_SettingsDlg;           // Device settings dialog
CDXUTTextHelper*            g_pTxtHelper = NULL;

// Direct3D 11 resources
CDXUTSDKMesh                g_SceneMesh;
CDXUTSDKMesh                g_AlphaMesh;
Scene                       g_Scene;

// depth buffer data
DepthStencilBuffer          g_DepthStencilBuffer;

// GUI state
GuiState                    g_CurrentGuiState;

// Number of currently active point lights
static AMD::Slider*         g_NumPointLightsSlider = NULL;
static int                  g_iNumActivePointLights = MAX_NUM_LIGHTS;

// Number of currently active spot lights
static AMD::Slider*         g_NumSpotLightsSlider = NULL;
static int                  g_iNumActiveSpotLights = 0;

// Current triangle density (i.e. the "lots of triangles" system)
static int                  g_iTriangleDensity = TRIANGLE_DENSITY_HIGH;
static const WCHAR*         g_szTriangleDensityLabel[TRIANGLE_DENSITY_NUM_TYPES] = { L"Low", L"Med", L"High" };

// Number of currently active grid objects (i.e. the "lots of triangles" system)
static AMD::Slider*         g_NumGridObjectsSlider = NULL;
static int                  g_iNumActiveGridObjects = MAX_NUM_GRID_OBJECTS;

// Number of currently active G-Buffer render targets
static AMD::Slider*         g_NumGBufferRTsSlider = NULL;
static int                  g_iNumActiveGBufferRTs = 3;

// The max distance the camera can travel
static float                g_fMaxDistance = 500.0f;

static int g_iPerfDataCounter = 0;
static int g_iAutoPerfTestFrameCounter = 0;
static bool g_bEnableAutoPerfTest = false;
static float g_fGpuPerfStatAccumulator[3] = {0,0,0};
static float g_fGpuPerfStat0[MAX_NUM_LIGHTS];
static float g_fGpuPerfStat1[MAX_NUM_LIGHTS];
static float g_fGpuPerfStat2[MAX_NUM_LIGHTS];
static int g_iGpuNumLights[MAX_NUM_LIGHTS];

//--------------------------------------------------------------------------------------
// Constant buffers
//--------------------------------------------------------------------------------------
#pragma pack(push,1)
struct CB_PER_OBJECT
{
    XMMATRIX  m_mWorld;
};

struct CB_PER_CAMERA
{
    XMMATRIX  m_mViewProjection;
};

struct CB_PER_FRAME
{
    XMMATRIX m_mView;
    XMMATRIX m_mProjection;
    XMMATRIX m_mProjectionInv;
    XMMATRIX m_mViewProjectionInvViewport;
    XMVECTOR m_AmbientColorUp;
    XMVECTOR m_AmbientColorDown;
    XMVECTOR m_vCameraPosAndAlphaTest;
    unsigned m_uNumLights;
    unsigned m_uNumSpotLights;
    unsigned m_uWindowWidth;
    unsigned m_uWindowHeight;
    unsigned m_uMaxNumLightsPerTile;
    unsigned m_uMaxNumElementsPerTile;
    unsigned m_uNumTilesX;
    unsigned m_uNumTilesY;
};

#pragma pack(pop)

ID3D11Buffer*                       g_pcbPerObject11 = NULL;
ID3D11Buffer*                       g_pcbPerCamera11 = NULL;
ID3D11Buffer*                       g_pcbPerFrame11 = NULL;

//--------------------------------------------------------------------------------------
// Set up AMD shader cache here
//--------------------------------------------------------------------------------------
AMD::ShaderCache            g_ShaderCache(AMD::ShaderCache::SHADER_AUTO_RECOMPILE_ENABLED, AMD::ShaderCache::ERROR_DISPLAY_ON_SCREEN);

//--------------------------------------------------------------------------------------
// AMD helper classes defined here
//--------------------------------------------------------------------------------------
static AMD::HUD             g_HUD;

// Global boolean for HUD rendering
bool                        g_bRenderHUD = true;

static CommonUtil        g_CommonUtil;
static ForwardPlusUtil   g_ForwardPlusUtil;
static LightUtil         g_LightUtil;
static TiledDeferredUtil g_TiledDeferredUtil;

//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
enum
{
    IDC_BUTTON_TOGGLEFULLSCREEN = 1,
    IDC_BUTTON_CHANGEDEVICE,
    IDC_RADIOBUTTON_FORWARD_PLUS,
    IDC_RADIOBUTTON_TILED_DEFERRED,
    IDC_CHECKBOX_SEPARATE_CULLING,
    IDC_CHECKBOX_ENABLE_LIGHT_DRAWING,
    IDC_SLIDER_NUM_POINT_LIGHTS,
    IDC_SLIDER_NUM_SPOT_LIGHTS,
    IDC_CHECKBOX_ENABLE_DEBUG_DRAWING,
    IDC_RADIOBUTTON_DEBUG_DRAWING_ONE,
    IDC_RADIOBUTTON_DEBUG_DRAWING_TWO,
    IDC_STATIC_TRIANGLE_DENSITY,
    IDC_SLIDER_TRIANGLE_DENSITY,
    IDC_SLIDER_NUM_GRID_OBJECTS,
    IDC_SLIDER_NUM_GBUFFER_RTS,
    IDC_RENDERING_METHOD_GROUP,
    IDC_TILE_DRAWING_GROUP,
    IDC_NUM_CONTROL_IDS
};

const int AMD::g_MaxApplicationControlID = IDC_NUM_CONTROL_IDS;

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext );
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext );
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext );
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext );
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext );

bool CALLBACK IsD3D11DeviceAcceptable( const CD3D11EnumAdapterInfo *AdapterInfo, UINT Output, const CD3D11EnumDeviceInfo *DeviceInfo,
                                       DXGI_FORMAT BackBufferFormat, bool bWindowed, void* pUserContext );
HRESULT CALLBACK OnD3D11CreateDevice( ID3D11Device* pd3dDevice, const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc,
                                     void* pUserContext );
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                         const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext );
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext );
void CALLBACK OnD3D11DestroyDevice( void* pUserContext );
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                 float fElapsedTime, void* pUserContext );

void InitApp();
void RenderText();

HRESULT AddShadersToCache();
void ClearD3D11DeviceContext();
void UpdateCameraConstantBuffer( const XMMATRIX& mViewProjAlreadyTransposed );
void UpdateCameraConstantBufferWithTranspose( const XMMATRIX& mViewProj );
void UpdateUI();
void WriteTestFile();
void EnableAutoPerfTest();

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, int nCmdShow )
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) || defined(_DEBUG)
    _CrtSetDbgFlag( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
#endif

    // Set DXUT callbacks
    DXUTSetCallbackMsgProc( MsgProc );
    DXUTSetCallbackKeyboard( OnKeyboard );
    DXUTSetCallbackFrameMove( OnFrameMove );
    DXUTSetCallbackDeviceChanging( ModifyDeviceSettings );

    DXUTSetCallbackD3D11DeviceAcceptable( IsD3D11DeviceAcceptable );
    DXUTSetCallbackD3D11DeviceCreated( OnD3D11CreateDevice );
    DXUTSetCallbackD3D11SwapChainResized( OnD3D11ResizedSwapChain );
    DXUTSetCallbackD3D11SwapChainReleasing( OnD3D11ReleasingSwapChain );
    DXUTSetCallbackD3D11DeviceDestroyed( OnD3D11DestroyDevice );
    DXUTSetCallbackD3D11FrameRender( OnD3D11FrameRender );

    InitApp();
    DXUTInit( true, true, NULL ); // Parse the command line, show msgboxes on error, no extra command line params
    DXUTSetCursorSettings( true, true );
    DXUTCreateWindow( L"ComputeBasedTiledCulling v1.0" );

    // Require D3D_FEATURE_LEVEL_11_0
    DXUTCreateDevice( D3D_FEATURE_LEVEL_11_0, true, 1920, 1080 );

    DXUTMainLoop(); // Enter into the DXUT render loop

    // Ensure the ShaderCache aborts if in a lengthy generation process
    g_ShaderCache.Abort();

    return DXUTGetExitCode();
}


//--------------------------------------------------------------------------------------
// Initialize the app 
//--------------------------------------------------------------------------------------
void InitApp()
{
    WCHAR szTemp[256];

    D3DCOLOR DlgColor = 0x88888888; // Semi-transparent background for the dialog

    g_SettingsDlg.Init( &g_DialogResourceManager );
    g_HUD.m_GUI.Init( &g_DialogResourceManager );
    g_HUD.m_GUI.SetBackgroundColors( DlgColor );
    g_HUD.m_GUI.SetCallback( OnGUIEvent );

    int iY = AMD::HUD::iElementDelta;

    g_HUD.m_GUI.AddButton( IDC_BUTTON_TOGGLEFULLSCREEN, L"Toggle full screen", AMD::HUD::iElementOffset, iY, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight );
    g_HUD.m_GUI.AddButton( IDC_BUTTON_CHANGEDEVICE, L"Change device (F2)", AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, VK_F2 );

    AMD::InitApp( g_ShaderCache, g_HUD, iY );

    iY += AMD::HUD::iGroupDelta;

    g_HUD.m_GUI.AddRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS, IDC_RENDERING_METHOD_GROUP, L"Forward+", AMD::HUD::iElementOffset, iY, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, true );
    g_HUD.m_GUI.AddRadioButton( IDC_RADIOBUTTON_TILED_DEFERRED, IDC_RENDERING_METHOD_GROUP, L"Tiled Deferred", AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, false );
    g_HUD.m_GUI.AddCheckBox( IDC_CHECKBOX_SEPARATE_CULLING, L"Separate Culling", 2 * AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth - AMD::HUD::iElementOffset, AMD::HUD::iElementHeight, false );
    g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_SEPARATE_CULLING )->SetChecked(false);
    g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_SEPARATE_CULLING )->SetEnabled(false);

    iY += AMD::HUD::iGroupDelta;

    g_HUD.m_GUI.AddCheckBox( IDC_CHECKBOX_ENABLE_LIGHT_DRAWING, L"Show Lights", AMD::HUD::iElementOffset, iY, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, true );

    // CDXUTDialog g_HUD.m_GUI will clean up these allocations in the CDXUTDialog destructor (see CDXUTDialog::RemoveAllControls)
    g_NumPointLightsSlider = new AMD::Slider( g_HUD.m_GUI, IDC_SLIDER_NUM_POINT_LIGHTS, iY, L"Active Point Lights", 0, MAX_NUM_LIGHTS, g_iNumActivePointLights );
    g_NumSpotLightsSlider = new AMD::Slider( g_HUD.m_GUI, IDC_SLIDER_NUM_SPOT_LIGHTS, iY, L"Active Spot Lights", 0, MAX_NUM_LIGHTS, g_iNumActiveSpotLights );

    iY += AMD::HUD::iGroupDelta;

    g_HUD.m_GUI.AddCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING, L"Show Lights Per Tile", AMD::HUD::iElementOffset, iY, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, false );
    g_HUD.m_GUI.AddRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE, IDC_TILE_DRAWING_GROUP, L"Radar Colors", 2 * AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth - AMD::HUD::iElementOffset, AMD::HUD::iElementHeight, true );
    g_HUD.m_GUI.AddRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_TWO, IDC_TILE_DRAWING_GROUP, L"Grayscale", 2 * AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth - AMD::HUD::iElementOffset, AMD::HUD::iElementHeight, false );
    g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE )->SetEnabled(false);
    g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_TWO )->SetEnabled(false);

    iY += AMD::HUD::iGroupDelta;

    // Use a standard DXUT slider here (not AMD::Slider), since we display a word for the slider value ("Low", "Med", "High") instead of a number
    wcscpy_s( szTemp, 256, L"Triangle Density: " );
    wcscat_s( szTemp, 256, g_szTriangleDensityLabel[g_iTriangleDensity] );
    g_HUD.m_GUI.AddStatic( IDC_STATIC_TRIANGLE_DENSITY, szTemp, AMD::HUD::iElementOffset, iY, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight );
    g_HUD.m_GUI.AddSlider( IDC_SLIDER_TRIANGLE_DENSITY, AMD::HUD::iElementOffset, iY += AMD::HUD::iElementDelta, AMD::HUD::iElementWidth, AMD::HUD::iElementHeight, 0, TRIANGLE_DENSITY_NUM_TYPES-1, g_iTriangleDensity );

    // CDXUTDialog g_HUD.m_GUI will clean up these allocations in the CDXUTDialog destructor (see CDXUTDialog::RemoveAllControls)
    g_NumGridObjectsSlider = new AMD::Slider( g_HUD.m_GUI, IDC_SLIDER_NUM_GRID_OBJECTS, iY, L"Active Grid Objects", 0, MAX_NUM_GRID_OBJECTS, g_iNumActiveGridObjects );
    g_NumGBufferRTsSlider = new AMD::Slider( g_HUD.m_GUI, IDC_SLIDER_NUM_GBUFFER_RTS, iY, L"Active G-Buffer RTs", 2, MAX_NUM_GBUFFER_RENDER_TARGETS, g_iNumActiveGBufferRTs );
    g_NumGBufferRTsSlider->SetEnabled(false);

    // Initialize the static data in CommonUtil
    g_CommonUtil.InitStaticData();

    UpdateUI();
}


//--------------------------------------------------------------------------------------
// Render the help and statistics text. This function uses the ID3DXFont interface for 
// efficient text rendering.
//--------------------------------------------------------------------------------------
void RenderText()
{
    bool bForwardPlus = g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetEnabled() &&
        g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetChecked();

    g_pTxtHelper->Begin();
    g_pTxtHelper->SetInsertionPos( 5, 5 );
    g_pTxtHelper->SetForegroundColor( XMVectorSet( 1.0f, 1.0f, 0.0f, 1.0f ) );
    g_pTxtHelper->DrawTextLine( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
    g_pTxtHelper->DrawTextLine( DXUTGetDeviceStats() );

    float fGpuTime = (float)TIMER_GetTime( Gpu, L"Render|Core algorithm" ) * 1000.0f;

    // count digits in the total time
    int iIntegerPart = (int)fGpuTime;
    int iNumDigits = 0;
    while( iIntegerPart > 0 )
    {
        iIntegerPart /= 10;
        iNumDigits++;
    }
    iNumDigits = ( iNumDigits == 0 ) ? 1 : iNumDigits;
    // three digits after decimal, 
    // plus the decimal point itself
    int iNumChars = iNumDigits + 4;

    // dynamic formatting for swprintf_s
    WCHAR szPrecision[16];
    swprintf_s( szPrecision, 16, L"%%%d.3f", iNumChars );

    WCHAR szBuf[256];
    WCHAR szFormat[256];
    swprintf_s( szFormat, 256, L"Total:           %s", szPrecision );
    swprintf_s( szBuf, 256, szFormat, fGpuTime );
    g_pTxtHelper->DrawTextLine( szBuf );

    float fGpuPerfStat0, fGpuPerfStat1, fGpuPerfStat2;
    if( bForwardPlus )
    {
        const float fGpuTimeDepthPrePass = (float)TIMER_GetTime( Gpu, L"Render|Core algorithm|Depth pre-pass" ) * 1000.0f;
        swprintf_s( szFormat, 256, L"+--------Z Pass: %s", szPrecision );
        swprintf_s( szBuf, 256, szFormat, fGpuTimeDepthPrePass );
        g_pTxtHelper->DrawTextLine( szBuf );

        const float fGpuTimeLightCulling = (float)TIMER_GetTime( Gpu, L"Render|Core algorithm|Light culling" ) * 1000.0f;
        swprintf_s( szFormat, 256, L"+----------Cull: %s", szPrecision );
        swprintf_s( szBuf, 256, szFormat, fGpuTimeLightCulling );
        g_pTxtHelper->DrawTextLine( szBuf );

        const float fGpuTimeForwardRendering = (float)TIMER_GetTime( Gpu, L"Render|Core algorithm|Forward rendering" ) * 1000.0f;
        swprintf_s( szFormat, 256, L"\\-------Forward: %s", szPrecision );
        swprintf_s( szBuf, 256, szFormat, fGpuTimeForwardRendering );
        g_pTxtHelper->DrawTextLine( szBuf );

        fGpuPerfStat0 = fGpuTimeDepthPrePass;
        fGpuPerfStat1 = fGpuTimeLightCulling;
        fGpuPerfStat2 = fGpuTimeForwardRendering;
    }
    else
    {
        const float fGpuTimeBuildGBuffer = (float)TIMER_GetTime( Gpu, L"Render|Core algorithm|G-Buffer" ) * 1000.0f;
        swprintf_s( szFormat, 256, L"+------G-Buffer: %s", szPrecision );
        swprintf_s( szBuf, 256, szFormat, fGpuTimeBuildGBuffer );
        g_pTxtHelper->DrawTextLine( szBuf );

        const float fGpuTimeLightCullingAndShading = (float)TIMER_GetTime( Gpu, L"Render|Core algorithm|Cull and light" ) * 1000.0f;
        swprintf_s( szFormat, 256, L"\\--Cull & Light: %s", szPrecision );
        swprintf_s( szBuf, 256, szFormat, fGpuTimeLightCullingAndShading );
        g_pTxtHelper->DrawTextLine( szBuf );

        fGpuPerfStat0 = fGpuTimeBuildGBuffer;
        fGpuPerfStat1 = fGpuTimeLightCullingAndShading;
        fGpuPerfStat2 = 0.0f;
    }

    g_pTxtHelper->SetInsertionPos( 5, DXUTGetDXGIBackBufferSurfaceDesc()->Height - AMD::HUD::iElementDelta );
    g_pTxtHelper->DrawTextLine( L"Toggle GUI    : F1" );

    g_pTxtHelper->End();

    bool bDebugDrawingEnabled = g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING )->GetEnabled() &&
            g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING )->GetChecked();
    if( bDebugDrawingEnabled )
    {
        // method 1 is radar colors, method 2 is grayscale
        bool bDebugDrawMethodOne = g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE )->GetEnabled() &&
            g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE )->GetChecked();
        int nDebugDrawType = bDebugDrawMethodOne ? DEBUG_DRAW_RADAR_COLORS : DEBUG_DRAW_GRAYSCALE;
        g_CommonUtil.RenderLegend( g_pTxtHelper, TEXT_LINE_HEIGHT, XMFLOAT4( 1.0f, 1.0f, 1.0f, 0.75f ), nDebugDrawType );
    }

    if( g_bEnableAutoPerfTest )
    {
        g_iAutoPerfTestFrameCounter++;
        if( g_iAutoPerfTestFrameCounter == 4 )
        {
            g_fGpuPerfStatAccumulator[0] = 0.0f;
            g_fGpuPerfStatAccumulator[1] = 0.0f;
            g_fGpuPerfStatAccumulator[2] = 0.0f;
        }
        else if( g_iAutoPerfTestFrameCounter > 4 && g_iAutoPerfTestFrameCounter <= 9 )
        {
            g_fGpuPerfStatAccumulator[0] += fGpuPerfStat0;
            g_fGpuPerfStatAccumulator[1] += fGpuPerfStat1;
            g_fGpuPerfStatAccumulator[2] += fGpuPerfStat2;
        }
        else if( g_iAutoPerfTestFrameCounter == 10 )
        {
            g_fGpuPerfStatAccumulator[0] /= 5.0f;
            g_fGpuPerfStatAccumulator[1] /= 5.0f;
            g_fGpuPerfStatAccumulator[2] /= 5.0f;
            g_fGpuPerfStat0[g_iPerfDataCounter] = g_fGpuPerfStatAccumulator[0];
            g_fGpuPerfStat1[g_iPerfDataCounter] = g_fGpuPerfStatAccumulator[1];
            g_fGpuPerfStat2[g_iPerfDataCounter] = g_fGpuPerfStatAccumulator[2];
            g_iGpuNumLights[g_iPerfDataCounter] = g_iNumActivePointLights;
            g_iPerfDataCounter++;
            if( g_iNumActivePointLights > MAX_NUM_LIGHTS-8 )
            {
                g_bEnableAutoPerfTest = false;
                WriteTestFile();
            }
            else
            {
                g_iAutoPerfTestFrameCounter = 0;
                g_iNumActivePointLights += 8;
                g_NumPointLightsSlider->SetValue(g_iNumActivePointLights);
            }
        }
    }
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

    XMVECTOR SceneMin, SceneMax;

    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();
    V_RETURN( g_DialogResourceManager.OnD3D11CreateDevice( pd3dDevice, pd3dImmediateContext ) );
    V_RETURN( g_SettingsDlg.OnD3D11CreateDevice( pd3dDevice ) );
    g_pTxtHelper = new CDXUTTextHelper( pd3dDevice, pd3dImmediateContext, &g_DialogResourceManager, TEXT_LINE_HEIGHT );

    // Load the scene mesh
    g_SceneMesh.Create( pd3dDevice, L"sponza\\sponza.sdkmesh", false );
    g_CommonUtil.CalculateSceneMinMax( g_SceneMesh, &SceneMin, &SceneMax );

    // Load the alpha-test mesh
    g_AlphaMesh.Create( pd3dDevice, L"sponza\\sponza_alpha.sdkmesh", false );

    // Put the mesh pointers in the wrapper struct that gets passed around
    g_Scene.m_pSceneMesh = &g_SceneMesh;
    g_Scene.m_pAlphaMesh = &g_AlphaMesh;

    // And the camera
    g_Scene.m_pCamera = &g_Camera;

    // Create constant buffers
    D3D11_BUFFER_DESC CBDesc;
    ZeroMemory( &CBDesc, sizeof(CBDesc) );
    CBDesc.Usage = D3D11_USAGE_DYNAMIC;
    CBDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    CBDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;

    CBDesc.ByteWidth = sizeof( CB_PER_OBJECT );
    V_RETURN( pd3dDevice->CreateBuffer( &CBDesc, NULL, &g_pcbPerObject11 ) );
    DXUT_SetDebugName( g_pcbPerObject11, "CB_PER_OBJECT" );

    CBDesc.ByteWidth = sizeof( CB_PER_CAMERA );
    V_RETURN( pd3dDevice->CreateBuffer( &CBDesc, NULL, &g_pcbPerCamera11 ) );
    DXUT_SetDebugName( g_pcbPerCamera11, "CB_PER_CAMERA" );

    CBDesc.ByteWidth = sizeof( CB_PER_FRAME );
    V_RETURN( pd3dDevice->CreateBuffer( &CBDesc, NULL, &g_pcbPerFrame11 ) );
    DXUT_SetDebugName( g_pcbPerFrame11, "CB_PER_FRAME" );

    // Create AMD_SDK resources here
    g_HUD.OnCreateDevice( pd3dDevice );
    TIMER_Init( pd3dDevice );

    static bool bFirstPass = true;

    // One-time setup
    if( bFirstPass )
    {
        // Setup the camera's view parameters
        XMVECTOR SceneCenter  = 0.5f * (SceneMax + SceneMin);
        XMVECTOR SceneExtents = 0.5f * (SceneMax - SceneMin);
        XMVECTOR BoundaryMin  = SceneCenter - 2.0f*SceneExtents;
        XMVECTOR BoundaryMax  = SceneCenter + 2.0f*SceneExtents;
        XMVECTOR BoundaryDiff = 4.0f*SceneExtents;  // BoundaryMax - BoundaryMin
        g_fMaxDistance = XMVectorGetX(XMVector3Length(BoundaryDiff));
        //XMVECTOR vEye = SceneCenter - XMVectorSet(0.45f*XMVectorGetX(SceneExtents), 0.35f*XMVectorGetY(SceneExtents), 0.0f, 0.0f);
        //XMVECTOR vAt  = SceneCenter - XMVectorSet(0.0f, 0.35f*XMVectorGetY(SceneExtents), 0.0f, 0.0f);
        XMVECTOR vEye = SceneCenter - XMVectorSet(0.67f*XMVectorGetX(SceneExtents), -0.05f*XMVectorGetY(SceneExtents), 0.0f, 0.0f);
        XMVECTOR vAt  = SceneCenter - XMVectorSet(0.0f, -0.05f*XMVectorGetY(SceneExtents), 0.0f, 0.0f);
        g_Camera.SetRotateButtons( true, false, false );
        g_Camera.SetEnablePositionMovement( true );
        g_Camera.SetViewParams( vEye, vAt );
        g_Camera.SetScalers( 0.005f, 0.1f*g_fMaxDistance );

        XMFLOAT3 vBoundaryMin, vBoundaryMax;
        XMStoreFloat3( &vBoundaryMin, BoundaryMin );
        XMStoreFloat3( &vBoundaryMax, BoundaryMax );
        g_Camera.SetClipToBoundary( true, &vBoundaryMin, &vBoundaryMax );

        // Init light buffer data
        LightUtil::InitLights( SceneMin, SceneMax );
    }

    // Create helper resources here
    g_CommonUtil.OnCreateDevice( pd3dDevice );
    g_ForwardPlusUtil.OnCreateDevice( pd3dDevice );
    g_LightUtil.OnCreateDevice( pd3dDevice );
    g_TiledDeferredUtil.OnCreateDevice( pd3dDevice );

    // Generate shaders ( this is an async operation - call AMD::ShaderCache::ShadersReady() to find out if they are complete ) 
    if( bFirstPass )
    {
        // Add the applications shaders to the cache
        AddShadersToCache();
        g_ShaderCache.GenerateShaders( AMD::ShaderCache::CREATE_TYPE_COMPILE_CHANGES );    // Only compile shaders that have changed (development mode)
        bFirstPass = false;
    }
    
    return S_OK;
}


//--------------------------------------------------------------------------------------
// Create any D3D11 resources that depend on the back buffer
//--------------------------------------------------------------------------------------
HRESULT CALLBACK OnD3D11ResizedSwapChain( ID3D11Device* pd3dDevice, IDXGISwapChain* pSwapChain,
                                         const DXGI_SURFACE_DESC* pBackBufferSurfaceDesc, void* pUserContext )
{
    HRESULT hr;

    V_RETURN( g_DialogResourceManager.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );
    V_RETURN( g_SettingsDlg.OnD3D11ResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc ) );

    // Setup the camera's projection parameters
    // Note, we are using inverted 32-bit float depth for better precision, 
    // so reverse near and far below
    float fAspectRatio = pBackBufferSurfaceDesc->Width / ( FLOAT )pBackBufferSurfaceDesc->Height;
    g_Camera.SetProjParams( XM_PI / 4, fAspectRatio, g_fMaxDistance, 0.1f );

    // Set the location and size of the AMD standard HUD
    g_HUD.m_GUI.SetLocation( pBackBufferSurfaceDesc->Width - AMD::HUD::iDialogWidth, 0 );
    g_HUD.m_GUI.SetSize( AMD::HUD::iDialogWidth, pBackBufferSurfaceDesc->Height );
    g_HUD.OnResizedSwapChain( pBackBufferSurfaceDesc );

    // Create our own depth stencil surface that's bindable as a shader resource
    V_RETURN( AMD::CreateDepthStencilSurface( &g_DepthStencilBuffer.m_pDepthStencilTexture, &g_DepthStencilBuffer.m_pDepthStencilSRV, &g_DepthStencilBuffer.m_pDepthStencilView, 
        DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R32_FLOAT, pBackBufferSurfaceDesc->Width, pBackBufferSurfaceDesc->Height, pBackBufferSurfaceDesc->SampleDesc.Count ) );

    g_CommonUtil.OnResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc, TEXT_LINE_HEIGHT );
    g_ForwardPlusUtil.OnResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc );
    g_LightUtil.OnResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc );
    g_TiledDeferredUtil.OnResizedSwapChain( pd3dDevice, pBackBufferSurfaceDesc );

    return S_OK;
}


//--------------------------------------------------------------------------------------
// Render the scene using the D3D11 device
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11FrameRender( ID3D11Device* pd3dDevice, ID3D11DeviceContext* pd3dImmediateContext, double fTime,
                                 float fElapsedTime, void* pUserContext )
{
    // Reset the timer at start of frame
    TIMER_Reset();

    // If the settings dialog is being shown, then render it instead of rendering the app's scene
    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.OnRender( fElapsedTime );
        return;
    }       

    ClearD3D11DeviceContext();

    const DXGI_SURFACE_DESC * BackBufferDesc = DXUTGetDXGIBackBufferSurfaceDesc();

    g_CurrentGuiState.m_uNumPointLights = (unsigned)g_iNumActivePointLights;
    g_CurrentGuiState.m_uNumSpotLights = (unsigned)g_iNumActiveSpotLights;

    // Check GUI state for debug drawing
    bool bDebugDrawingEnabled = g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING )->GetEnabled() &&
            g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING )->GetChecked();
    bool bDebugDrawMethodOne = g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE )->GetEnabled() &&
            g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE )->GetChecked();

    g_CurrentGuiState.m_nDebugDrawType = DEBUG_DRAW_NONE;
    if( bDebugDrawingEnabled )
    {
        g_CurrentGuiState.m_nDebugDrawType = bDebugDrawMethodOne ? DEBUG_DRAW_RADAR_COLORS : DEBUG_DRAW_GRAYSCALE;
    }

    // Check GUI state for whether to do culling in a separate pass for tiled deferred
    g_CurrentGuiState.m_bDoTiledDeferredWithSeparateCulling = g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_SEPARATE_CULLING )->GetEnabled() &&
        g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_SEPARATE_CULLING )->GetChecked();

    // Check GUI state for light drawing enabled
    g_CurrentGuiState.m_bLightDrawingEnabled = g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_LIGHT_DRAWING )->GetEnabled() &&
        g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_LIGHT_DRAWING )->GetChecked();

    g_CurrentGuiState.m_nGridObjectTriangleDensity = g_iTriangleDensity;
    g_CurrentGuiState.m_nNumGridObjects = g_iNumActiveGridObjects;
    g_CurrentGuiState.m_nNumGBufferRenderTargets = g_iNumActiveGBufferRTs;

    XMMATRIX mWorld = XMMatrixIdentity();

    // Get the projection & view matrix from the camera class
    XMMATRIX mView = g_Camera.GetViewMatrix();
    XMMATRIX mProj = g_Camera.GetProjMatrix();
    XMMATRIX mViewProjection = mView * mProj;

    // we need the inverse proj matrix in the per-tile light culling 
    // compute shader
    XMFLOAT4X4 f4x4Proj, f4x4InvProj;
    XMStoreFloat4x4( &f4x4Proj, mProj );
    XMStoreFloat4x4( &f4x4InvProj, XMMatrixIdentity() );
    f4x4InvProj._11 = 1.0f / f4x4Proj._11;
    f4x4InvProj._22 = 1.0f / f4x4Proj._22;
    f4x4InvProj._33 = 0.0f;
    f4x4InvProj._34 = 1.0f / f4x4Proj._43;
    f4x4InvProj._43 = 1.0f;
    f4x4InvProj._44 = -f4x4Proj._33 / f4x4Proj._43;
    XMMATRIX mInvProj = XMLoadFloat4x4( &f4x4InvProj );

    // we need the inverse viewproj matrix with viewport mapping, 
    // for converting from depth back to world-space position
    XMMATRIX mInvViewProj = XMMatrixInverse( NULL, mViewProjection );
    XMFLOAT4X4 f4x4Viewport ( 2.0f / (float)BackBufferDesc->Width, 0.0f,                                 0.0f, 0.0f,
                              0.0f,                               -2.0f / (float)BackBufferDesc->Height, 0.0f, 0.0f,
                              0.0f,                                0.0f,                                 1.0f, 0.0f,
                             -1.0f,                                1.0f,                                 0.0f, 1.0f  );
    XMMATRIX mInvViewProjViewport = XMLoadFloat4x4(&f4x4Viewport) * mInvViewProj;

    XMFLOAT4 CameraPosAndAlphaTest;
    XMStoreFloat4( &CameraPosAndAlphaTest, g_Camera.GetEyePt() );
    CameraPosAndAlphaTest.w = 0.5f;

    // Set the constant buffers
    HRESULT hr;
    D3D11_MAPPED_SUBRESOURCE MappedResource;

    // per-camera constants
    V( pd3dImmediateContext->Map( g_pcbPerCamera11, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
    CB_PER_CAMERA* pPerCamera = ( CB_PER_CAMERA* )MappedResource.pData;
    pPerCamera->m_mViewProjection = XMMatrixTranspose( mViewProjection );
    pd3dImmediateContext->Unmap( g_pcbPerCamera11, 0 );
    pd3dImmediateContext->VSSetConstantBuffers( 1, 1, &g_pcbPerCamera11 );
    pd3dImmediateContext->PSSetConstantBuffers( 1, 1, &g_pcbPerCamera11 );
    pd3dImmediateContext->CSSetConstantBuffers( 1, 1, &g_pcbPerCamera11 );

    // per-frame constants
    V( pd3dImmediateContext->Map( g_pcbPerFrame11, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
    CB_PER_FRAME* pPerFrame = ( CB_PER_FRAME* )MappedResource.pData;
    pPerFrame->m_mView = XMMatrixTranspose( mView );
    pPerFrame->m_mProjection = XMMatrixTranspose( mProj );
    pPerFrame->m_mProjectionInv = XMMatrixTranspose( mInvProj );
    pPerFrame->m_mViewProjectionInvViewport = XMMatrixTranspose( mInvViewProjViewport );
    pPerFrame->m_AmbientColorUp = XMVectorSet( 0.013f, 0.015f, 0.050f, 1.0f );
    pPerFrame->m_AmbientColorDown = XMVectorSet( 0.0013f, 0.0015f, 0.0050f, 1.0f );
    pPerFrame->m_vCameraPosAndAlphaTest = XMLoadFloat4( &CameraPosAndAlphaTest );
    pPerFrame->m_uNumLights = (unsigned)g_iNumActivePointLights;
    pPerFrame->m_uNumSpotLights = (unsigned)g_iNumActiveSpotLights;
    pPerFrame->m_uWindowWidth = BackBufferDesc->Width;
    pPerFrame->m_uWindowHeight = BackBufferDesc->Height;
    pPerFrame->m_uMaxNumLightsPerTile = g_CommonUtil.GetMaxNumLightsPerTile();
    pPerFrame->m_uMaxNumElementsPerTile = g_CommonUtil.GetMaxNumElementsPerTile();
    pPerFrame->m_uNumTilesX = g_CommonUtil.GetNumTilesX();
    pPerFrame->m_uNumTilesY = g_CommonUtil.GetNumTilesY();
    pd3dImmediateContext->Unmap( g_pcbPerFrame11, 0 );
    pd3dImmediateContext->VSSetConstantBuffers( 2, 1, &g_pcbPerFrame11 );
    pd3dImmediateContext->PSSetConstantBuffers( 2, 1, &g_pcbPerFrame11 );
    pd3dImmediateContext->CSSetConstantBuffers( 2, 1, &g_pcbPerFrame11 );

    // per-object constants
    V( pd3dImmediateContext->Map( g_pcbPerObject11, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
    CB_PER_OBJECT* pPerObject = ( CB_PER_OBJECT* )MappedResource.pData;
    pPerObject->m_mWorld = XMMatrixTranspose( mWorld );
    pd3dImmediateContext->Unmap( g_pcbPerObject11, 0 );
    pd3dImmediateContext->VSSetConstantBuffers( 0, 1, &g_pcbPerObject11 );
    pd3dImmediateContext->PSSetConstantBuffers( 0, 1, &g_pcbPerObject11 );
    pd3dImmediateContext->CSSetConstantBuffers( 0, 1, &g_pcbPerObject11 );

    bool bForwardPlus = g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetEnabled() &&
        g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetChecked();

    // Render objects here...
    if( g_ShaderCache.ShadersReady() )
    {
        TIMER_Begin( 0, L"Render" );

        if( bForwardPlus )
        {
            g_ForwardPlusUtil.OnRender( fElapsedTime, g_CurrentGuiState, g_DepthStencilBuffer, g_Scene, g_CommonUtil, g_LightUtil );
        }
        else
        {
            g_TiledDeferredUtil.OnRender( fElapsedTime, g_CurrentGuiState, g_DepthStencilBuffer, g_Scene, g_CommonUtil, g_LightUtil );
        }

        TIMER_End(); // Render
    }

    DXUT_BeginPerfEvent( DXUT_PERFEVENTCOLOR, L"HUD / Stats" );

    AMD::ProcessUIChanges();

    if (g_ShaderCache.ShadersReady())
    {

        // Render the HUD
        if (g_bRenderHUD)
        {
            g_HUD.OnRender(fElapsedTime);
        }

        RenderText();

        AMD::RenderHUDUpdates(g_pTxtHelper);
    }
    else
    {
        float ClearColor[4] = { 0.0013f, 0.0015f, 0.0050f, 0.0f };
        ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
        ID3D11DepthStencilView* pNULLDSV = NULL;
        pd3dImmediateContext->ClearRenderTargetView( pRTV, ClearColor );

        // Render shader cache progress if still processing
        pd3dImmediateContext->OMSetRenderTargets( 1, &pRTV, pNULLDSV );
        pd3dImmediateContext->OMSetDepthStencilState( g_CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DISABLE_DEPTH_TEST), 0x00 );
        g_ShaderCache.RenderProgress( g_pTxtHelper, TEXT_LINE_HEIGHT, XMVectorSet( 1.0f, 1.0f, 0.0f, 1.0f ) );
    }
    
    DXUT_EndPerfEvent();

    static DWORD dwTimefirst = GetTickCount();
    if ( GetTickCount() - dwTimefirst > 5000 )
    {    
        OutputDebugString( DXUTGetFrameStats( DXUTIsVsyncEnabled() ) );
        OutputDebugString( L"\n" );
        dwTimefirst = GetTickCount();
    }
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11ResizedSwapChain 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11ReleasingSwapChain( void* pUserContext )
{
    g_CommonUtil.OnReleasingSwapChain();
    g_ForwardPlusUtil.OnReleasingSwapChain();
    g_LightUtil.OnReleasingSwapChain();
    g_TiledDeferredUtil.OnReleasingSwapChain();

    g_DialogResourceManager.OnD3D11ReleasingSwapChain();

    SAFE_RELEASE( g_DepthStencilBuffer.m_pDepthStencilTexture );
    SAFE_RELEASE( g_DepthStencilBuffer.m_pDepthStencilView );
    SAFE_RELEASE( g_DepthStencilBuffer.m_pDepthStencilSRV );
}


//--------------------------------------------------------------------------------------
// Release D3D11 resources created in OnD3D11CreateDevice 
//--------------------------------------------------------------------------------------
void CALLBACK OnD3D11DestroyDevice( void* pUserContext )
{
    g_DialogResourceManager.OnD3D11DestroyDevice();
    g_SettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE( g_pTxtHelper );

    SAFE_RELEASE( g_DepthStencilBuffer.m_pDepthStencilTexture );
    SAFE_RELEASE( g_DepthStencilBuffer.m_pDepthStencilView );
    SAFE_RELEASE( g_DepthStencilBuffer.m_pDepthStencilSRV );

    // Delete additional render resources here...
    g_SceneMesh.Destroy();
    g_AlphaMesh.Destroy();

    SAFE_RELEASE( g_pcbPerObject11 );
    SAFE_RELEASE( g_pcbPerCamera11 );
    SAFE_RELEASE( g_pcbPerFrame11 );

    g_CommonUtil.OnDestroyDevice();
    g_ForwardPlusUtil.OnDestroyDevice();
    g_LightUtil.OnDestroyDevice();
    g_TiledDeferredUtil.OnDestroyDevice();

    // Destroy AMD_SDK resources here
    g_ShaderCache.OnDestroyDevice();
    g_HUD.OnDestroyDevice();
    TIMER_Destroy();
}


//--------------------------------------------------------------------------------------
// Called right before creating a D3D11 device, allowing the app to modify the device settings as needed
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings( DXUTDeviceSettings* pDeviceSettings, void* pUserContext )
{
    static bool s_bFirstTime = true;
    if( s_bFirstTime )
    {
        s_bFirstTime = false;

        // For the first device created if its a REF device, optionally display a warning dialog box
        if( pDeviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE )
        {
            DXUTDisplaySwitchingToREFWarning();
        }

        // Start with vsync disabled
        pDeviceSettings->d3d11.SyncInterval = 0;
    }

    // Sample quality is always zero
    pDeviceSettings->d3d11.sd.SampleDesc.Quality = 0;

    // This sample currently does not support MSAA
    pDeviceSettings->d3d11.sd.SampleDesc.Count = 1;

    // Don't auto create a depth buffer, as this sample requires a depth buffer 
    // be created such that it's bindable as a shader resource
    pDeviceSettings->d3d11.AutoCreateDepthStencil = false;

    return true;
}


//--------------------------------------------------------------------------------------
// Handle updates to the scene.  This is called regardless of which D3D API is used
//--------------------------------------------------------------------------------------
void CALLBACK OnFrameMove( double fTime, float fElapsedTime, void* pUserContext )
{
    // Update the camera's position based on user input 
    g_Camera.FrameMove( fElapsedTime );
}


//--------------------------------------------------------------------------------------
// Handle messages to the application
//--------------------------------------------------------------------------------------
LRESULT CALLBACK MsgProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* pbNoFurtherProcessing,
                          void* pUserContext )
{
    switch( uMsg )
    {
        case WM_GETMINMAXINFO:
            // override DXUT_MIN_WINDOW_SIZE_X and DXUT_MIN_WINDOW_SIZE_Y
            // to prevent windows that are too small
            ( ( MINMAXINFO* )lParam )->ptMinTrackSize.x = 512;
            ( ( MINMAXINFO* )lParam )->ptMinTrackSize.y = 512;
            *pbNoFurtherProcessing = true;
            break;
    }
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *pbNoFurtherProcessing = g_DialogResourceManager.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;

    // Pass messages to settings dialog if its active
    if( g_SettingsDlg.IsActive() )
    {
        g_SettingsDlg.MsgProc( hWnd, uMsg, wParam, lParam );
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    *pbNoFurtherProcessing = g_HUD.m_GUI.MsgProc( hWnd, uMsg, wParam, lParam );
    if( *pbNoFurtherProcessing )
        return 0;
    
    // Pass all remaining windows messages to camera so it can respond to user input
    g_Camera.HandleMessages( hWnd, uMsg, wParam, lParam );

    return 0;
}


//--------------------------------------------------------------------------------------
// Handle key presses
//--------------------------------------------------------------------------------------
void CALLBACK OnKeyboard( UINT nChar, bool bKeyDown, bool bAltDown, void* pUserContext )
{
    if( bKeyDown )
    {
        switch( nChar )
        {
        case VK_F1:
            g_bRenderHUD = !g_bRenderHUD;
            break;
        case VK_F4:
            if( !bAltDown ) DXUTSnapD3D11Screenshot( L"screenshot.bmp", false );
            break;
        case VK_F10:
            EnableAutoPerfTest();
            break;
        }
    }
}


//--------------------------------------------------------------------------------------
// Handles the GUI events
//--------------------------------------------------------------------------------------
void CALLBACK OnGUIEvent( UINT nEvent, int nControlID, CDXUTControl* pControl, void* pUserContext )
{
    WCHAR szTemp[256];

    switch( nControlID )
    {
        case IDC_BUTTON_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen();
            break;
        case IDC_BUTTON_CHANGEDEVICE:
            g_SettingsDlg.SetActive( !g_SettingsDlg.IsActive() );
            break;
        case IDC_RADIOBUTTON_FORWARD_PLUS:
        case IDC_RADIOBUTTON_TILED_DEFERRED:
            {
                bool bForwardPlus = g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetEnabled() &&
                    g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetChecked();
                g_NumGBufferRTsSlider->SetEnabled(!bForwardPlus);
                g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_SEPARATE_CULLING )->SetEnabled(!bForwardPlus);
            }
            break;
        case IDC_SLIDER_NUM_POINT_LIGHTS:
            {
                g_NumPointLightsSlider->OnGuiEvent();
            }
            break;
        case IDC_SLIDER_NUM_SPOT_LIGHTS:
            {
                g_NumSpotLightsSlider->OnGuiEvent();
            }
            break;
        case IDC_CHECKBOX_ENABLE_DEBUG_DRAWING:
            {
                bool bTileDrawingEnabled = g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING )->GetEnabled() &&
                    g_HUD.m_GUI.GetCheckBox( IDC_CHECKBOX_ENABLE_DEBUG_DRAWING )->GetChecked();
                g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_ONE )->SetEnabled(bTileDrawingEnabled);
                g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_DEBUG_DRAWING_TWO )->SetEnabled(bTileDrawingEnabled);
            }
            break;
        case IDC_SLIDER_TRIANGLE_DENSITY:
            {
                // update
                g_iTriangleDensity = ((CDXUTSlider*)pControl)->GetValue();
                wcscpy_s( szTemp, 256, L"Triangle Density: " );
                wcscat_s( szTemp, 256, g_szTriangleDensityLabel[g_iTriangleDensity] );
                g_HUD.m_GUI.GetStatic( IDC_STATIC_TRIANGLE_DENSITY )->SetText( szTemp );
            }
            break;
        case IDC_SLIDER_NUM_GRID_OBJECTS:
            {
                // update slider
                g_NumGridObjectsSlider->OnGuiEvent();
            }
            break;
        case IDC_SLIDER_NUM_GBUFFER_RTS:
            g_NumGBufferRTsSlider->OnGuiEvent();
            break;

        default:
            AMD::OnGUIEvent(nEvent, nControlID, pControl, pUserContext);
            break;
    }
}


//--------------------------------------------------------------------------------------
// Adds all shaders to the shader cache
//--------------------------------------------------------------------------------------
HRESULT AddShadersToCache()
{
    g_CommonUtil.AddShadersToCache(&g_ShaderCache);
    g_ForwardPlusUtil.AddShadersToCache(&g_ShaderCache);
    g_LightUtil.AddShadersToCache(&g_ShaderCache);
    g_TiledDeferredUtil.AddShadersToCache(&g_ShaderCache);

    return S_OK;
}

//--------------------------------------------------------------------------------------
// Stripped down version of DXUT ClearD3D11DeviceContext.
// For this sample, the HS, DS, and GS are not used. And it 
// is assumed that drawing code will always call VSSetShader, 
// PSSetShader, IASetVertexBuffers, IASetIndexBuffer (if applicable), 
// and IASetInputLayout.
//--------------------------------------------------------------------------------------
void ClearD3D11DeviceContext()
{
    ID3D11DeviceContext* pd3dDeviceContext = DXUTGetD3D11DeviceContext();

    ID3D11ShaderResourceView* pSRVs[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    ID3D11RenderTargetView* pRTVs[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    ID3D11DepthStencilView* pDSV = NULL;
    ID3D11Buffer* pBuffers[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
    ID3D11SamplerState* pSamplers[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };


    // Constant buffers
    pd3dDeviceContext->VSSetConstantBuffers( 0, 14, pBuffers );
    pd3dDeviceContext->PSSetConstantBuffers( 0, 14, pBuffers );
    pd3dDeviceContext->CSSetConstantBuffers( 0, 14, pBuffers );

    // Resources
    pd3dDeviceContext->VSSetShaderResources( 0, 16, pSRVs );
    pd3dDeviceContext->PSSetShaderResources( 0, 16, pSRVs );
    pd3dDeviceContext->CSSetShaderResources( 0, 16, pSRVs );

    // Samplers
    pd3dDeviceContext->VSSetSamplers( 0, 16, pSamplers );
    pd3dDeviceContext->PSSetSamplers( 0, 16, pSamplers );
    pd3dDeviceContext->CSSetSamplers( 0, 16, pSamplers );

    // Render targets
    pd3dDeviceContext->OMSetRenderTargets( 8, pRTVs, pDSV );

    // States
    FLOAT BlendFactor[4] = { 0,0,0,0 };
    pd3dDeviceContext->OMSetBlendState( NULL, BlendFactor, 0xFFFFFFFF );
    pd3dDeviceContext->OMSetDepthStencilState( g_CommonUtil.GetDepthStencilState(DEPTH_STENCIL_STATE_DEPTH_GREATER), 0x00 );  // we are using inverted 32-bit float depth for better precision
    pd3dDeviceContext->RSSetState( NULL );
}

void UpdateCameraConstantBuffer( const XMMATRIX& mViewProjAlreadyTransposed )
{
    HRESULT hr;
    D3D11_MAPPED_SUBRESOURCE MappedResource;
    ID3D11DeviceContext* pd3dImmediateContext = DXUTGetD3D11DeviceContext();

    V( pd3dImmediateContext->Map( g_pcbPerCamera11, 0, D3D11_MAP_WRITE_DISCARD, 0, &MappedResource ) );
    CB_PER_CAMERA* pPerCamera = ( CB_PER_CAMERA* )MappedResource.pData;
    memcpy( &pPerCamera->m_mViewProjection, &mViewProjAlreadyTransposed, sizeof(XMMATRIX) );
    pd3dImmediateContext->Unmap( g_pcbPerCamera11, 0 );
}


void UpdateCameraConstantBufferWithTranspose( const XMMATRIX& mViewProj )
{
    XMMATRIX mViewProjTransposed = XMMatrixTranspose( mViewProj );
    UpdateCameraConstantBuffer( mViewProjTransposed );
}


void UpdateUI()
{
    g_iNumActivePointLights = MAX_NUM_LIGHTS;
    g_HUD.m_GUI.GetSlider( IDC_SLIDER_NUM_POINT_LIGHTS )->SetRange( 0, MAX_NUM_LIGHTS );
    g_HUD.m_GUI.GetSlider( IDC_SLIDER_NUM_POINT_LIGHTS )->SetValue( g_iNumActivePointLights );

    g_iNumActiveSpotLights = 0;
    g_HUD.m_GUI.GetSlider( IDC_SLIDER_NUM_SPOT_LIGHTS )->SetRange( 0, MAX_NUM_LIGHTS );
    g_HUD.m_GUI.GetSlider( IDC_SLIDER_NUM_SPOT_LIGHTS )->SetValue( g_iNumActiveSpotLights );

    g_NumPointLightsSlider->OnGuiEvent();
    g_NumSpotLightsSlider->OnGuiEvent();
}

void WriteTestFile()
{
    HRESULT hr;
    IDXGIDevice * pDXGIDevice;
    V( DXUTGetD3D11Device()->QueryInterface(__uuidof(IDXGIDevice), (void **)&pDXGIDevice) );

    IDXGIAdapter * pDXGIAdapter;
    pDXGIDevice->GetAdapter(&pDXGIAdapter);

    DXGI_ADAPTER_DESC adapterDesc;
    pDXGIAdapter->GetDesc(&adapterDesc);

    WCHAR filename[256];
    wcscpy_s( filename, 256, adapterDesc.Description );

    bool bForwardPlus = g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetEnabled() &&
        g_HUD.m_GUI.GetRadioButton( IDC_RADIOBUTTON_FORWARD_PLUS )->GetChecked();

    if( bForwardPlus )
    {
        wcscat_s( filename, 256, L" fp_" );
    }
    else
    {
        wcscat_s( filename, 256, L" td_" );
    }

    const DXGI_SURFACE_DESC * BackBufferDesc = DXUTGetDXGIBackBufferSurfaceDesc();

    static const WCHAR* szTriangleDensityLabelForFilename[TRIANGLE_DENSITY_NUM_TYPES] = { L"low", L"med", L"high" };
    WCHAR sz[256-DXGI_MAX_DEVICE_IDENTIFIER_STRING];
    swprintf_s( sz, 256-DXGI_MAX_DEVICE_IDENTIFIER_STRING, L"%dx%d_ms%d_grd%03d", BackBufferDesc->Width, BackBufferDesc->Height, BackBufferDesc->SampleDesc.Count, g_iNumActiveGridObjects );
    wcscat_s( sz, 256-DXGI_MAX_DEVICE_IDENTIFIER_STRING, szTriangleDensityLabelForFilename[g_iTriangleDensity] );
    wcscat_s( filename, 256, sz );

    if( !bForwardPlus )
    {
        swprintf_s( sz, 256-DXGI_MAX_DEVICE_IDENTIFIER_STRING, L"_rt%d", g_iNumActiveGBufferRTs );
        wcscat_s( filename, 256, sz );
    }

    wcscat_s( filename, 256, L".csv" );

    FILE* pFile = NULL;

    _wfopen_s( &pFile, filename, L"wt" );

    if( pFile )
    {
        CHAR line[1024];
        if( bForwardPlus )
        {
            sprintf_s( line, 1024, "Num Point Lights,Depth Pre-Pass,Tiled Culling,Forward Rendering\n" );
            fwrite( line, strlen(line), 1, pFile );
            for( int i = 0; i < g_iPerfDataCounter; i++ )
            {
                sprintf_s( line, 1024, "%d,%.6f,%.6f,%.6f\n", g_iGpuNumLights[i], g_fGpuPerfStat0[i], g_fGpuPerfStat1[i], g_fGpuPerfStat2[i] );
                fwrite( line, strlen(line), 1, pFile );
            }
        }
        else
        {
            sprintf_s( line, 1024, "Num Point Lights,G-Buffer Pass,Tiled Culling and Lighting\n" );
            fwrite( line, strlen(line), 1, pFile );
            for( int i = 0; i < g_iPerfDataCounter; i++ )
            {
                sprintf_s( line, 1024, "%d,%.6f,%.6f\n", g_iGpuNumLights[i], g_fGpuPerfStat0[i], g_fGpuPerfStat1[i] );
                fwrite( line, strlen(line), 1, pFile );
            }
        }

        fclose( pFile );
    }

    SAFE_RELEASE(pDXGIAdapter);
    SAFE_RELEASE(pDXGIDevice);
}

void EnableAutoPerfTest()
{
    g_iNumActivePointLights = 0;
    g_iPerfDataCounter = 0;
    g_iAutoPerfTestFrameCounter = 0;
    g_bEnableAutoPerfTest = true;
    g_fGpuPerfStatAccumulator[0] = 0;
    g_fGpuPerfStatAccumulator[1] = 0;
    g_fGpuPerfStatAccumulator[2] = 0;
}

//--------------------------------------------------------------------------------------
// EOF.
//--------------------------------------------------------------------------------------
