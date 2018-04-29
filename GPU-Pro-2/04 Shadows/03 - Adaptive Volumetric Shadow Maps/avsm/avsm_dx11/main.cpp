// Copyright 2010 Intel Corporation
// All Rights Reserved
//
// Permission is granted to use, copy, distribute and prepare derivative works of this
// software for any purpose and without fee, provided, that the above copyright notice
// and this statement appear in all copies.  Intel makes no representations about the
// suitability of this software for any purpose.  THIS SOFTWARE IS PROVIDED "AS IS."
// INTEL SPECIFICALLY DISCLAIMS ALL WARRANTIES, EXPRESS OR IMPLIED, AND ALL LIABILITY,
// INCLUDING CONSEQUENTIAL AND OTHER INDIRECT DAMAGES, FOR THE USE OF THIS SOFTWARE,
// INCLUDING LIABILITY FOR INFRINGEMENT OF ANY PROPRIETARY RIGHTS, AND INCLUDING THE
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  Intel does not
// assume any responsibility for any errors which may appear in this software nor any
// responsibility to update it.

#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "SDKMesh.h"
#include "App.h"
#include "ParticleSystem.h"

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>

//--------------------------------------------------------------------------------------
// Defines
//--------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------
// UI control IDs
//--------------------------------------------------------------------------------------
enum {
    UI_TEXT,
    UI_CONTROLMODE,
    UI_TOGGLEFULLSCREEN,
    UI_CHANGEDEVICE,
    UI_ANIMATELIGHT,
    UI_VOLUMETRICMODEL,
    UI_AVSMAUTOBOUNDS,
    UI_SELECTEDSCENE,
    UI_AVSMSHADOWTEXTUREDIM,
    UI_AVSMSORTINGMETHOD,
    UI_AVSMNUMNODES,
    UI_VOLUMESHADOWINGMETHOD,
    UI_PARTICLESIZE,
    UI_PARTICLEOPACITY,
    UI_DSMERROR,
    UI_HAIR_SHADOW_THICKNESS, 
    UI_HAIR_THICKNESS, // world space thickness
    UI_HAIR_OPACITY,
    UI_HAIR_SORT,
    UI_HAIR_SORT_ENABLED,
    UI_HAIR_BLENDING,
    UI_PAUSEPARTICLEANIMATION,
    UI_LIGHTINGONLY,
    UI_VOLUMESHADOWCREATION,
    UI_VOLUMESHADOWLOOKUP,
    UI_DRAWTRANSMITTANCECURVE,
};

// List these top to bottom, since it is also the reverse draw order
enum {
    HUD_GENERIC = 0,
    HUD_AVSM,
    HUD_HAIR,
    HUD_EXPERT,
    HUD_NUM,
};

enum AVSMSortingMethod {
    AVSM_UNSORTED,
    AVSM_INSERTION_SORT,
};

enum ControlObject
{
    CONTROL_CAMERA = 0,
    CONTROL_LIGHT,
    CONTROL_NUM
};

enum VolumetricModel
{
    VOLUME_NONE = 0,
    VOLUME_PARTICLES,
    VOLUME_HAIR,
    VOLUME_PARTICLES_AND_HAIR,
};

class MainOptions {
  public:
    MainOptions() :
        gControlObject(CONTROL_CAMERA) {}

    CFirstPersonCamera gViewerCamera;
    CFirstPersonCamera gLightCamera;
    int gControlObject;
};

// State
App::Options gNewAppOptions;
UIConstants gNewUIConstants;
D3DXVECTOR3 gNewViewEye, gNewViewLookAt;
D3DXVECTOR3 gNewLightEye, gNewLightLookAt;
bool gRestoreAppOptions = false;
bool gRestoreUIConstants = false;
bool gRestoreViewState = false;

//--------------------------------------------------------------------------------------
// Global variables
//--------------------------------------------------------------------------------------
App* gApp = 0;
App::Options gAppOptions;
MainOptions gMainOptions;

CDXUTSDKMesh gMesh;
ParticleSystem *gParticleSystem = NULL;
D3DXMATRIXA16 gWorldMatrix;

// DXUT GUI stuff
CDXUTDialogResourceManager gDialogResourceManager;
CD3DSettingsDlg gD3DSettingsDlg;
CDXUTDialog gHUD[HUD_NUM];
CDXUTCheckBox* gAnimateLightCheck = 0;
CDXUTCheckBox* gAVSMAutoBoundsCheck = 0;
CDXUTComboBox* gSceneSelectCombo = 0;
CDXUTComboBox* gVolumeShadowingMethodCombo = 0;
CDXUTComboBox* gHairSortCombo = 0;
CDXUTComboBox* gAVSMShadowTextureDimCombo = 0;
CDXUTComboBox* gAVSMSortingMethodCombo = 0;
CDXUTComboBox* gNodeCountCombo = 0;
CDXUTComboBox* gVolumeModel = 0;
CDXUTTextHelper* gTextHelper = 0;

float gAspectRatio;
bool gDisplayUI = true;
bool gEnableShadowPicking = false;

// Any UI state passed directly to rendering shaders
UIConstants gUIConstants;

// Constants
static const float kLightRotationSpeed = 0.15f; // radians/sec
static const float kSliderFactorResolution = 10000.0f;
static const float kParticleSizeFactorUI = 20.0f; // Divisor on [0,100] range of particle size UI to particle size
static const float kDsmErrorFactorUI = 1000.0f;

//--------------------------------------------------------------------------------------
// Forward declarations 
//--------------------------------------------------------------------------------------
bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* deviceSettings, void* userContext);
void CALLBACK OnFrameMove(double time, FLOAT elapsedTime, void* userContext);
LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* noFurtherProcessing,
                         void* userContext);
void CALLBACK OnKeyboard(UINT character, bool keyDown, bool altDown, void* userContext);
void CALLBACK OnGUIEvent(UINT eventID, INT controlID, CDXUTControl* control, void* userContext);
HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* d3dDevice, const DXGI_SURFACE_DESC* backBufferSurfaceDesc,
                                     void* userContext);
HRESULT CALLBACK OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice, IDXGISwapChain* swapChain,
                                         const DXGI_SURFACE_DESC* backBufferSurfaceDesc, void* userContext);
void CALLBACK OnD3D11ReleasingSwapChain(void* userContext);
void CALLBACK OnD3D11DestroyDevice(void* userContext);
void CALLBACK OnD3D11FrameRender(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContext, double time,
                                 FLOAT elapsedTime, void* userContext);

void InitApp(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContext);
void DeinitApp();

void InitUI();
void RenderText();
void UpdateViewerCameraNearFar();
CFirstPersonCamera * GetCurrentUserCamera();

//--------------------------------------------------------------------------------------
// Entry point to the program. Initializes everything and goes into a message processing 
// loop. Idle time is used to render the scene.
//--------------------------------------------------------------------------------------
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPWSTR lpCmdLine, INT nCmdShow)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    // XXX: 
    // Insert proper alloc number reported by CRT at app exit time to
    // track down the allocation that is leaking.
    // XXX: 
    // _CrtSetBreakAlloc(933);
#endif

    // Set DXUT callbacks
    DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
    DXUTSetCallbackMsgProc(MsgProc);
    DXUTSetCallbackKeyboard(OnKeyboard);
    DXUTSetCallbackFrameMove(OnFrameMove);

    DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
    DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
    DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender);
    DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
    DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);
    
    DXUTInit(true, true, 0);
    InitUI();

    DXUTSetCursorSettings(true, true);
    DXUTSetHotkeyHandling(true, true, false);
    DXUTCreateWindow(L"Adaptive Volumetric Shadow Maps");
    DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, 1680, 1050);
    DXUTMainLoop();

    return DXUTGetExitCode();
}


void InitUI()
{
    // Setup default UI state
    // NOTE: All of these are made directly available in the shader constant buffer
    // This is convenient for development purposes.
    if (gRestoreUIConstants) {
        gUIConstants = gNewUIConstants;
        gRestoreUIConstants = false;
    } else {
        gUIConstants.lightingOnly = 0;
        gUIConstants.faceNormals = 0;
        gUIConstants.avsmSortingMethod = 0;
        gUIConstants.volumeShadowMethod = VOL_SHADOW_AVSM;
        gUIConstants.enableVolumeShadowLookup = 1;
        gUIConstants.pauseParticleAnimaton = 0;
        gUIConstants.particleSize = 1.0f;
        gUIConstants.particleOpacity = 33;
        gUIConstants.dsmError = 0.0075f;
        gUIConstants.hairThickness = 34;       // Actually, 3.40
        gUIConstants.hairShadowThickness = 10; // Actually, 0.10
        gUIConstants.hairOpacity = 18;         // Actually, 0.18
    }
    
    gD3DSettingsDlg.Init(&gDialogResourceManager);

    for (int i = 0; i < HUD_NUM; ++i) {
        gHUD[i].RemoveAllControls();
        gHUD[i].Init(&gDialogResourceManager);
        gHUD[i].SetCallback(OnGUIEvent);
    }

    int width = 200;
    const int yDelta = 20;

    // Generic HUD
    {
        CDXUTDialog* HUD = &gHUD[HUD_GENERIC];
        int y = 0;

        HUD->AddButton(UI_TOGGLEFULLSCREEN, L"Toggle full screen", 0, y, width, 23);
        y += yDelta;

        HUD->AddButton(UI_CHANGEDEVICE, L"Change device (F2)", 0, y, width, 23, VK_F2);
        y += yDelta;

        HUD->AddComboBox(UI_SELECTEDSCENE, 0, y, width, 23, 0, false, &gSceneSelectCombo);
        const int numScenes = 5;
        gSceneSelectCombo->SetDropHeight(numScenes * 40);
        y += yDelta;
        gSceneSelectCombo->AddItem(L"Power Plant", ULongToPtr(POWER_PLANT_SCENE));
        gSceneSelectCombo->AddItem(L"Ground Plane",
                                   ULongToPtr(GROUND_PLANE_SCENE));
        gSceneSelectCombo->AddItem(L"Hair and Particles", ULongToPtr(HAIR_AND_PARTICLES_SCENE));
        gSceneSelectCombo->SetSelectedByIndex(gAppOptions.scene);

        CDXUTComboBox* ControlMode;
        HUD->AddComboBox(UI_CONTROLMODE, 0, y, width, 23, 0, false, &ControlMode);
        y += yDelta;
        ControlMode->AddItem(L"Camera", IntToPtr(CONTROL_CAMERA));
        ControlMode->AddItem(L"Light", IntToPtr(CONTROL_LIGHT));
        ControlMode->SetSelectedByData(IntToPtr(gMainOptions.gControlObject));

        HUD->AddComboBox(UI_VOLUMETRICMODEL, 0, y, width, 23, 0, false, &gVolumeModel);
        y += yDelta;
        gVolumeModel->AddItem(L"None", IntToPtr(VOLUME_NONE));
        gVolumeModel->AddItem(L"Particles", IntToPtr(VOLUME_PARTICLES));
        gVolumeModel->AddItem(L"Hair", IntToPtr(VOLUME_HAIR));
        gVolumeModel->AddItem(L"Particles and hair", IntToPtr(VOLUME_PARTICLES_AND_HAIR));
        gVolumeModel->SetSelectedByData(IntToPtr(VOLUME_PARTICLES));

        HUD->AddCheckBox(UI_AVSMAUTOBOUNDS, L"AVSM Auto Bounds", 0, y, width, 23, 
                         gAppOptions.enableAutoBoundsAVSM != 0, 0, false,
                         &gAVSMAutoBoundsCheck);
        y += yDelta;

        HUD->AddCheckBox(UI_LIGHTINGONLY, L"Lighting Only", 0, y, width, 23, gUIConstants.lightingOnly != 0);
        y += yDelta;

        HUD->AddCheckBox(UI_ANIMATELIGHT, L"Animate Light", 0, y, width, 23, false, VK_SPACE, false, &gAnimateLightCheck);
        y += yDelta;

        HUD->SetSize(width, y);
    }

    // AVSM HUD
    {
        CDXUTDialog* HUD = &gHUD[HUD_AVSM];
        int y = -yDelta;
    
        HUD->AddStatic(UI_TEXT, L"Volume Shadowing Method:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddComboBox(UI_VOLUMESHADOWINGMETHOD, 0, y, width, 23, 0, false, &gVolumeShadowingMethodCombo);
        y += yDelta;
        gVolumeShadowingMethodCombo->AddItem(L"NO SHADOW", ULongToPtr(VOL_SHADOW_NO_SHADOW));
        gVolumeShadowingMethodCombo->AddItem(L"UNCOMPRESSED", ULongToPtr(VOL_SHADOW_UNCOMPRESSED));
        gVolumeShadowingMethodCombo->AddItem(L"DEEP SHADOW MAPS", ULongToPtr(VOL_SHADOW_DSM));
        gVolumeShadowingMethodCombo->AddItem(L"AVSM", ULongToPtr(VOL_SHADOW_AVSM));
        gVolumeShadowingMethodCombo->AddItem(L"AVSM BOX 4",
                                             ULongToPtr(VOL_SHADOW_AVSM_BOX_4));
        gVolumeShadowingMethodCombo->AddItem(L"AVSM GAUSS 7",
                                             ULongToPtr(VOL_SHADOW_AVSM_GAUSS_7));

        gVolumeShadowingMethodCombo->SetSelectedByData(ULongToPtr(VOL_SHADOW_AVSM));

        HUD->AddStatic(UI_TEXT, L"Texture Dimension:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddComboBox(UI_AVSMSHADOWTEXTUREDIM, 0, y, width, 23, 0, false, &gAVSMShadowTextureDimCombo);
        y += yDelta;
        gAVSMShadowTextureDimCombo->AddItem(L"256 x 256", ULongToPtr(256));
        gAVSMShadowTextureDimCombo->AddItem(L"512 x 512", ULongToPtr(512));
        gAVSMShadowTextureDimCombo->AddItem(L"1024 x 1024", ULongToPtr(1024));       
        gAVSMShadowTextureDimCombo->SetSelectedByData(ULongToPtr(256));

        HUD->AddStatic(UI_TEXT, L"AVSM Node Count:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddComboBox(UI_AVSMNUMNODES, 0, y, width, 23, 0, false, &gNodeCountCombo);
        y += yDelta;
        gNodeCountCombo->AddItem(L"4", ULongToPtr(4));
        gNodeCountCombo->AddItem(L"8", ULongToPtr(8));
        gNodeCountCombo->AddItem(L"12", ULongToPtr(12));
        gNodeCountCombo->AddItem(L"16", ULongToPtr(16));
        gNodeCountCombo->SetSelectedByData(ULongToPtr(gAppOptions.NodeCount));

        HUD->AddStatic(UI_TEXT, L"AVSM Per Pixel Sorting:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddComboBox(UI_AVSMSORTINGMETHOD, 0, y, width, 23, 0, false, &gAVSMSortingMethodCombo);
        y += yDelta;
        gAVSMSortingMethodCombo->AddItem(L"UNSORTED", ULongToPtr(AVSM_UNSORTED));
        gAVSMSortingMethodCombo->AddItem(L"SORTED", ULongToPtr(AVSM_INSERTION_SORT));
        gAVSMSortingMethodCombo->SetSelectedByData(ULongToPtr(AVSM_UNSORTED));

        HUD->AddStatic(UI_TEXT, L"Particle Size:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddSlider(UI_PARTICLESIZE, 5, y, 170, 22, 0, 100, 
                       static_cast<UINT>(gUIConstants.particleSize * kParticleSizeFactorUI));
        y += yDelta;

        HUD->AddStatic(UI_TEXT, L"Particle Opacity:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddSlider(UI_PARTICLEOPACITY, 5, y, 170, 22, 0, 100, gUIConstants.particleOpacity);
        y += yDelta;

        HUD->AddStatic(UI_TEXT, L"DSM Error:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddSlider(UI_DSMERROR, 5, y, 170, 22, 0, 100, 
                       static_cast<UINT>(gUIConstants.dsmError * kDsmErrorFactorUI));
        y += yDelta;

        HUD->AddCheckBox(UI_PAUSEPARTICLEANIMATION, L"Pause Particle Animation", 0, y, width, 23, gUIConstants.pauseParticleAnimaton != 0);
        y += yDelta;

        HUD->SetSize(width, y);
    }

    // Hair HUD
    {
        CDXUTDialog* HUD = &gHUD[HUD_HAIR];
        int y = -yDelta;

        HUD->AddStatic(UI_TEXT, L"Hair thickness:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddSlider(UI_HAIR_THICKNESS, 5, y, 170, 22, 0, 100,
                       gUIConstants.hairThickness);
        y += yDelta;

        HUD->AddStatic(UI_TEXT, L"Hair (shadow) thickness:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddSlider(UI_HAIR_SHADOW_THICKNESS, 5, y, 170, 22, 0, 100,
                       gUIConstants.hairShadowThickness);
        y += yDelta;

        HUD->AddStatic(UI_TEXT, L"Hair opacity:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddSlider(UI_HAIR_OPACITY, 5, y, 170, 22, 0, 100,
                       gUIConstants.hairOpacity);
        y += yDelta;

        HUD->AddStatic(UI_TEXT, L"Hair Sort Method:",  5, y, 140, 18);
        y += yDelta;
        HUD->AddComboBox(UI_HAIR_SORT, 0, y, width, 23, 0, false, 
                         &gHairSortCombo);
        y += yDelta;
        gHairSortCombo->AddItem(L"None", ULongToPtr(HAIR_SORT_NONE));
        gHairSortCombo->AddItem(L"Per Line", ULongToPtr(HAIR_SORT_PER_LINE));
        gHairSortCombo->AddItem(L"Per Pixel", ULongToPtr(HAIR_SORT_PER_PIXEL));
        gHairSortCombo->SetSelectedByData(ULongToPtr(HAIR_SORT_NONE));

        HUD->AddCheckBox(UI_HAIR_SORT_ENABLED, L"Enable hair sorting",  0, y, width, 23, gAppOptions.hairSortEnabled);
        y += yDelta;

        HUD->AddCheckBox(UI_HAIR_BLENDING, L"Enable hair blending", 0, y, width, 23, gAppOptions.hairBlending);
        y += yDelta;

        HUD->SetSize(width, y);

        // Initially hidden
        HUD->SetVisible(false);
    }

    // Expert HUD
    {
        CDXUTDialog* HUD = &gHUD[HUD_EXPERT];
        int y = 0;

        HUD->AddCheckBox(UI_VOLUMESHADOWCREATION, L"Do Volume Shadow Creation", 0,
                         y, width, 23, gAppOptions.enableVolumeShadowCreation != 0);
        y += 26;

        HUD->AddCheckBox(UI_VOLUMESHADOWLOOKUP, L"Do Volume Shadow Lookup", 0,
                         y, width, 23, gUIConstants.enableVolumeShadowLookup != 0);
        y += 26;

        HUD->AddCheckBox(UI_DRAWTRANSMITTANCECURVE, L"Draw Transmittance", 0,
                         y, width, 23, gAppOptions.enableTransmittanceCurve != 0);
        y += 26;

        HUD->SetSize(width, y);

        // Initially hidden
        HUD->SetVisible(false);
    }
}

void InitApp(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContext)
{
    DeinitApp();

    // Grab parameters from UI
    unsigned int shadowTextureDim     = 2048;

    unsigned int avsmShadowTextureDim = static_cast<unsigned int>(PtrToUlong(gAVSMShadowTextureDimCombo->GetSelectedData()));

    if (gRestoreAppOptions) {
        gAppOptions = gNewAppOptions;
        gRestoreAppOptions = false;
    }

    gApp = new App(d3dDevice, d3dDeviceContext, gAppOptions.NodeCount, shadowTextureDim, avsmShadowTextureDim);

    // Initialize with the current surface description
    gApp->OnD3D11ResizedSwapChain(d3dDevice, DXUTGetDXGIBackBufferSurfaceDesc());
}

void DeinitApp()
{
    SAFE_DELETE(gApp);
}


bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* deviceSettings, void* userContext)
{
    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool firstTime = true;
    if (firstTime) {
        firstTime = false;
        if (deviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE) {
            DXUTDisplaySwitchingToREFWarning(deviceSettings->ver);
        }
    }

    // We don't currently support framebuffer MSAA
    // Requires multi-frequency shading wrt. the GBuffer that is not yet implemented
    deviceSettings->d3d11.sd.SampleDesc.Count = 1;
    deviceSettings->d3d11.sd.SampleDesc.Quality = 0;

    // Also don't need a depth/stencil buffer... we'll manage that ourselves
    deviceSettings->d3d11.AutoCreateDepthStencil = false;

    return true;
}


void CALLBACK OnFrameMove(double time, FLOAT elapsedTime, void* userContext)
{
    // If requested, orbit the light
    if (gAnimateLightCheck->GetChecked()) {
        D3DXVECTOR3 eyePt = *gMainOptions.gLightCamera.GetEyePt();
        D3DXVECTOR3 lookAtPt = *gMainOptions.gLightCamera.GetLookAtPt();

        // Note: Could be replaced by gMesh.GetMeshBBoxCenter()
        D3DXVECTOR3 sceneCenter = D3DXVECTOR3(0.0f, 0.0f, 0.0f);
        
        // rotate around center of the scene
        float angle = kLightRotationSpeed * elapsedTime;
        eyePt -= sceneCenter;
        lookAtPt -= sceneCenter;
        D3DXMATRIX rotate;
        D3DXMatrixRotationY(&rotate, angle);
        D3DXVECTOR4 newEyePt;
        D3DXVECTOR4 newLookAtPt;
        D3DXVec3Transform(&newEyePt, &eyePt, &rotate);
        D3DXVec3Transform(&newLookAtPt, &lookAtPt, &rotate);
        eyePt = D3DXVECTOR3(newEyePt);
        lookAtPt = D3DXVECTOR3(newLookAtPt);
        eyePt += sceneCenter;
        lookAtPt += sceneCenter;

        gMainOptions.gLightCamera.SetViewParams(&eyePt, &lookAtPt);
    }
    
    // Update the camera's position based on user input 
    gMainOptions.gLightCamera.FrameMove(elapsedTime);
    gMainOptions.gViewerCamera.FrameMove(elapsedTime);
}

#define MAX_FRAMES_COUNT 1
float frameRate[MAX_FRAMES_COUNT];
int   frameRateCount = 0;

void RenderText()
{
    gTextHelper->Begin();

    gTextHelper->SetInsertionPos(2, 0 );
    gTextHelper->SetForegroundColor(D3DXCOLOR(0.0f, 0.0f, 1.0f, 1.0f));
    gTextHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));
    gTextHelper->DrawTextLine(DXUTGetDeviceStats());
    gTextHelper->DrawTextLine(L"F7: Hair Hud. F8: Expert HUD. F9: Turn HUDs on/off. F5: AVSM visualization");

    // Output DSM error
    std::wostringstream oss0;
    oss0 << "Deep Shadow Map Error = " << gUIConstants.dsmError << "\n";
    gTextHelper->DrawTextLine(oss0.str().c_str());

    // Output frame time
    std::wostringstream oss1;
    frameRate[frameRateCount % MAX_FRAMES_COUNT] = DXUTGetFPS();
    ++frameRateCount;

    float avg = 0;
    for (int i = 0; i < MAX_FRAMES_COUNT; ++i) {
        avg += frameRate[i];
    }

    avg /= (float)MAX_FRAMES_COUNT;

    oss1 << 1000.0f / avg << " ms / frame" << std::endl;
    oss1 << "AVSM pick (x, y) = (" << gAppOptions.pickedX << "," << gAppOptions.pickedY << ")" << std::endl;

    gTextHelper->DrawTextLine(oss1.str().c_str());

    gTextHelper->End();
}


LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* noFurtherProcessing,
                          void* userContext)
{
    // Pass messages to dialog resource manager calls so GUI state is updated correctly
    *noFurtherProcessing = gDialogResourceManager.MsgProc(hWnd, uMsg, wParam, lParam );
    if (*noFurtherProcessing) {
        return 0;
    }

    // Pass messages to settings dialog if its active
    if (gD3DSettingsDlg.IsActive()) {
        gD3DSettingsDlg.MsgProc(hWnd, uMsg, wParam, lParam);
        return 0;
    }

    // Give the dialogs a chance to handle the message first
    for (int i = 0; i < HUD_NUM; ++i) {
        *noFurtherProcessing = gHUD[i].MsgProc(hWnd, uMsg, wParam, lParam);
        if(*noFurtherProcessing) {
            return 0;
        }
    }

    // If shadowPicking is enabled, store the picked position
    if (gAppOptions.enableShadowPicking)  {
        switch(uMsg) {
            case WM_LBUTTONUP: {
                POINT ptCursor = {(short)LOWORD(lParam), (short)HIWORD(lParam)};
                int avsmShadowTextureDim = 
                    static_cast<int>(PtrToUlong(gAVSMShadowTextureDimCombo->GetSelectedData()));
                
                if (ptCursor.x < avsmShadowTextureDim &&
                       ptCursor.y < avsmShadowTextureDim) {
                    gAppOptions.pickedX = ptCursor.x;
                    gAppOptions.pickedY = ptCursor.y;
                }
                break;
            }
        }
    }

    // Pass all remaining windows messages to camera so it can respond to user input
    GetCurrentUserCamera()->HandleMessages(hWnd, uMsg, wParam, lParam);

    return 0;
}

static void
SetHairEnabled(bool enabled)
{
    gAppOptions.enableHair = enabled;

    // Turn off bounds check with hair on, and back on when hair is off.
    bool avsmBoundCheck = !enabled;
    gAppOptions.enableAutoBoundsAVSM = avsmBoundCheck;
    gAVSMAutoBoundsCheck->SetChecked(avsmBoundCheck);

    if (enabled) {
        // Up render targets
        gNodeCountCombo->SetSelectedByData(ULongToPtr(16));

        // Use a better filtering mode
        gVolumeShadowingMethodCombo->SetSelectedByData(ULongToPtr(VOL_SHADOW_AVSM_BOX_4));
    }
}

static void
DestroyScene()
{
    gMesh.Destroy();
    SAFE_DELETE(gParticleSystem);
}

static void 
CreateParticles(ID3D11Device *d3dDevice, SCENE_SELECTION scene)
{
    SAFE_DELETE(gParticleSystem);

    // Set up shader define macros
    std::string shadowAAStr; {
        std::ostringstream oss;
        unsigned int avsmShadowTextureDim = 
            static_cast<unsigned int>(PtrToUlong(gAVSMShadowTextureDimCombo->GetSelectedData()));
        oss << avsmShadowTextureDim;
        shadowAAStr = oss.str();
    }
    std::string avsmNodeCountStr; {
        std::ostringstream oss;
        unsigned int avsmNodeCount = 
            static_cast<unsigned int>(PtrToUlong(gNodeCountCombo->GetSelectedData()));
        oss << avsmNodeCount;
        avsmNodeCountStr = oss.str();
    }

    D3D10_SHADER_MACRO shaderDefines[] = {
        {"SHADOWAA_SAMPLES", shadowAAStr.c_str()},
        {"AVSM_NODE_COUNT", avsmNodeCountStr.c_str()},
        {0, 0}
    };

    switch (scene) {
        case HAIR_AND_PARTICLES_SCENE: {
            ParticleEmitter emitter0;
            {
                emitter0.mDrag           = 0.2f;
                emitter0.mGravity        = 0.0f;
                emitter0.mRandScaleX     = 2.5f;
                emitter0.mRandScaleY     = 2.0f;
                emitter0.mRandScaleZ     = 2.5f;
                emitter0.mLifetime       = 8.0f;
                emitter0.mStartSize      = 13.0f;//3.0f;//
                emitter0.mSizeRate       = 0.07f;
                emitter0.mpPos[0]        = -40.0f;
                emitter0.mpPos[1]        = -2.0f; 
                emitter0.mpPos[2]        = -55.0;
                emitter0.mpVelocity[0]   = 0.3f;
                emitter0.mpVelocity[1]   = 10.0f;
                emitter0.mpVelocity[2]   = 0.5f;
            }

            
            ParticleEmitter largeCoolingTower;
            {
                largeCoolingTower.mDrag           = 0.2f;
                largeCoolingTower.mGravity        = 0.0f;
                largeCoolingTower.mRandScaleX     = 5.5f;
                largeCoolingTower.mRandScaleY     = 4.0f;
                largeCoolingTower.mRandScaleZ     = 5.5f;
                largeCoolingTower.mLifetime       = 17.0f;
                largeCoolingTower.mStartSize      = 10.0f;//5.0f;//
                largeCoolingTower.mSizeRate       = 0.05f;
                largeCoolingTower.mpPos[0]        = -40.0f;
                largeCoolingTower.mpPos[1]        = -2.0f; 
                largeCoolingTower.mpPos[2]        = -40.0;
                largeCoolingTower.mpVelocity[0]   = 0.6f;
                largeCoolingTower.mpVelocity[1]   = 15.0f;
                largeCoolingTower.mpVelocity[2]   = 0.6f;
            }           

            const ParticleEmitter* emitters[3] = {&largeCoolingTower}; //{&emitter0, &largeCoolingTower};
            const UINT maxNumPartices = 5000;
            gParticleSystem = new ParticleSystem(maxNumPartices);
            gParticleSystem->InitializeParticles(emitters, 1, d3dDevice, 8, 8, shaderDefines);

            break;
        }
        case POWER_PLANT_SCENE: {
            ParticleEmitter emitter0;
            {
                emitter0.mDrag           = 0.2f;
                emitter0.mGravity        = 0.0f;
                emitter0.mRandScaleX     = 2.5f;
                emitter0.mRandScaleY     = 2.0f;
                emitter0.mRandScaleZ     = 2.5f;
                emitter0.mLifetime       = 7.0f;
                emitter0.mStartSize      = 6.0f;//3.0f;//
                emitter0.mSizeRate       = 0.05f;
                emitter0.mpPos[0]        = 40.0f;
                emitter0.mpPos[1]        = -2.0f; 
                emitter0.mpPos[2]        = 0.0;
                emitter0.mpVelocity[0]   = 0.3f;
                emitter0.mpVelocity[1]   = 10.0f;
                emitter0.mpVelocity[2]   = 0.5f;
            }

            ParticleEmitter emitter1;
            {
                emitter1.mDrag           = 0.2f;
                emitter1.mGravity        = 0.0f;
                emitter1.mRandScaleX     = 3.0f;
                emitter1.mRandScaleY     = 2.0f;
                emitter1.mRandScaleZ     = 3.0f;
                emitter1.mLifetime       = 5.0f;
                emitter1.mStartSize      = 6.0f;//3.0f;//
                emitter1.mSizeRate       = 0.005f;
                emitter1.mpPos[0]        = 60.0f;
                emitter1.mpPos[1]        = -2.0f; 
                emitter1.mpPos[2]        = 14.0;
                emitter1.mpVelocity[0]   = 0.0f;
                emitter1.mpVelocity[1]   = 6.0f;
                emitter1.mpVelocity[2]   = 0.0f;
            }
            
            ParticleEmitter largeCoolingTower;
            {
                largeCoolingTower.mDrag           = 0.2f;
                largeCoolingTower.mGravity        = 0.0f;
                largeCoolingTower.mRandScaleX     = 5.5f;
                largeCoolingTower.mRandScaleY     = 4.0f;
                largeCoolingTower.mRandScaleZ     = 5.5f;
                largeCoolingTower.mLifetime       = 7.0f;
                largeCoolingTower.mStartSize      = 10.0f;//5.0f;//
                largeCoolingTower.mSizeRate       = 0.05f;
                largeCoolingTower.mpPos[0]        = 0.0f;
                largeCoolingTower.mpPos[1]        = -2.0f; 
                largeCoolingTower.mpPos[2]        = 0.0;
                largeCoolingTower.mpVelocity[0]   = 0.6f;
                largeCoolingTower.mpVelocity[1]   = 15.0f;
                largeCoolingTower.mpVelocity[2]   = 0.6f;
            }           

            const ParticleEmitter* emitters[3] = {&emitter0, &emitter1, &largeCoolingTower};
            const UINT maxNumPartices = 5000;
            gParticleSystem = new ParticleSystem(maxNumPartices);
            gParticleSystem->InitializeParticles(emitters, 3, d3dDevice, 8, 8, shaderDefines);

            break;
        }

        case GROUND_PLANE_SCENE:{
            ParticleEmitter emitter1;
            {
                emitter1.mDrag           = 0.2f;
                emitter1.mGravity        = 0.0f;
                emitter1.mRandScaleX     = 3.0f;
                emitter1.mRandScaleY     = 2.0f;
                emitter1.mRandScaleZ     = 3.0f;
                emitter1.mLifetime       = 5.0f;
                emitter1.mStartSize      = 6.0f;//3.0f;//
                emitter1.mSizeRate       = 0.005f;
                emitter1.mpPos[0]        = 0.0f;
                emitter1.mpPos[1]        = -2.0f; 
                emitter1.mpPos[2]        = -40.0;
                emitter1.mpVelocity[0]   = 0.0f;
                emitter1.mpVelocity[1]   = 6.0f;
                emitter1.mpVelocity[2]   = 0.0f;
            }
            
            ParticleEmitter emitter0;
            {
                emitter0.mDrag           = 0.2f;
                emitter0.mGravity        = 0.0f;
                emitter0.mRandScaleX     = 2.5f;
                emitter0.mRandScaleY     = 2.0f;
                emitter0.mRandScaleZ     = 2.5f;
                emitter0.mLifetime       = 7.0f;
                emitter0.mStartSize      = 6.0f;//3.0f;//
                emitter0.mSizeRate       = 0.05f;
                emitter0.mpPos[0]        = -20.0f;
                emitter0.mpPos[1]        = -2.0f; 
                emitter0.mpPos[2]        = -50.0;
                emitter0.mpVelocity[0]   = 0.3f;
                emitter0.mpVelocity[1]   = 10.0f;
                emitter0.mpVelocity[2]   = 0.5f;
            }


            ParticleEmitter largeCoolingTower;
            {
                largeCoolingTower.mDrag           = 0.2f;
                largeCoolingTower.mGravity        = 0.0f;
                largeCoolingTower.mRandScaleX     = 5.5f;
                largeCoolingTower.mRandScaleY     = 4.0f;
                largeCoolingTower.mRandScaleZ     = 5.5f;
                largeCoolingTower.mLifetime       = 7.0f;
                largeCoolingTower.mStartSize      = 10.0f;//5.0f;//
                largeCoolingTower.mSizeRate       = 0.05f;
                largeCoolingTower.mpPos[0]        = -60.0f;
                largeCoolingTower.mpPos[1]        = -2.0f; 
                largeCoolingTower.mpPos[2]        = -70.0;
                largeCoolingTower.mpVelocity[0]   = 0.6f;
                largeCoolingTower.mpVelocity[1]   = 15.0f;
                largeCoolingTower.mpVelocity[2]   = 0.6f;
            }           

            const ParticleEmitter* emitters[3] = {&emitter0, &emitter1, &largeCoolingTower};
            const UINT maxNumPartices = 5000;
            gParticleSystem = new ParticleSystem(maxNumPartices);
            gParticleSystem->InitializeParticles(emitters, 3, d3dDevice, 8, 8, shaderDefines);

            break;
        }

        default: {
            assert(false);
        }
    }
}



static void
InitScene(ID3D11Device *d3dDevice)
{
    DestroyScene();

    D3DXVECTOR3 cameraEye(0.0f, 0.0f, 0.0f);
    D3DXVECTOR3 cameraAt(0.0f, 0.0f, 0.0f);
    float sceneScaling = 1.0f;
    float rotationScaling = 0.01f;
    float moveScaling = 10.0f;
    bool zAxisUp = false;

    switch (gAppOptions.scene) {
        case HAIR_AND_PARTICLES_SCENE: {
            gMesh.Create(d3dDevice, L"..\\media\\GroundPlane\\GroundPlane.sdkmesh");
            sceneScaling = 5.0f;
            moveScaling = 100.0f;
            cameraEye = D3DXVECTOR3(-0.473993, 140.063812, -238.968491);
            cameraAt = D3DXVECTOR3(-0.465148, 39.597244, 38.084045);

            gAppOptions.hairX = 65.0f;
            gAppOptions.hairY = 85.0f;
            gApp->UpdateHairMesh();

            SetHairEnabled(true);
            gVolumeModel->SetSelectedByData(IntToPtr(VOLUME_PARTICLES_AND_HAIR));

            break;
        }
        case POWER_PLANT_SCENE: {
            gMesh.Create(d3dDevice, L"..\\media\\powerplant\\powerplant.sdkmesh");
            sceneScaling = 1.0f;
            cameraEye = sceneScaling * D3DXVECTOR3(100.0f, 5.0f, 5.0f);
            cameraAt = sceneScaling * D3DXVECTOR3(0.0f, 0.0f, 0.0f);

            SetHairEnabled(false);
            gVolumeModel->SetSelectedByData(IntToPtr(VOLUME_PARTICLES));

            break;
        }
        case GROUND_PLANE_SCENE: {
            gMesh.Create(d3dDevice, L"..\\media\\GroundPlane\\GroundPlane.sdkmesh");
            sceneScaling = 5.0f;
            moveScaling = 100.0f;
            cameraEye = D3DXVECTOR3(-0.473993, 140.063812, 238.968491);
            cameraAt = D3DXVECTOR3(-0.465148, 139.597244, 238.084045);
            gAppOptions.hairX = 50.0f;
            gAppOptions.hairY = 80.0f;
            gApp->UpdateHairMesh();

            SetHairEnabled(false);
            gVolumeModel->SetSelectedByData(IntToPtr(VOLUME_PARTICLES));

            break;
        }
        default: {
            assert(false);
        }
    }

    
    D3DXMatrixScaling(&gWorldMatrix, sceneScaling, sceneScaling, sceneScaling);
    if (zAxisUp) {
        D3DXMATRIXA16 m;
        D3DXMatrixRotationX(&m, -D3DX_PI / 2.0f);
        gWorldMatrix *= m;
    }

    gMainOptions.gViewerCamera.SetViewParams(&cameraEye, &cameraAt);
    gMainOptions.gViewerCamera.SetScalers(rotationScaling, moveScaling);
    gMainOptions.gViewerCamera.FrameMove(0.0f);

    // Create a particle system for this specific scene
    CreateParticles(d3dDevice, gAppOptions.scene);

    if (gRestoreViewState) {
        gMainOptions.gViewerCamera.SetViewParams(&gNewViewEye, &gNewViewLookAt);
        gMainOptions.gLightCamera.SetViewParams(&gNewLightEye, &gNewLightLookAt);
        gRestoreViewState = false;
    }
}

void CALLBACK OnKeyboard(UINT character, bool keyDown, bool altDown, void* userContext)
{
    if(keyDown) {
        switch (character) {
        case VK_F5:
            // Toggle display of AVSM-first-node and picking state
            gAppOptions.enableShadowPicking = !gAppOptions.enableShadowPicking;
            break;
        case VK_F7:
            // Toggle visibility of expert HUD
            gHUD[HUD_HAIR].SetVisible(!gHUD[HUD_HAIR].GetVisible());
	        break;
        case VK_F8:
            // Toggle visibility of expert HUD
            gHUD[HUD_EXPERT].SetVisible(!gHUD[HUD_EXPERT].GetVisible());
	        break;
        case VK_F9:
            // Toggle display of UI on/off
            gDisplayUI = !gDisplayUI;
            break;
        }
    }
}

void CALLBACK OnGUIEvent(UINT eventID, INT controlID, CDXUTControl* control, void* userContext)
{
    CDXUTSlider* Slider;
    switch (controlID)
    {
        case UI_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen(); break;
        case UI_CONTROLMODE: {
                CDXUTComboBox* Combo = dynamic_cast<CDXUTComboBox*>(control);
                gMainOptions.gControlObject = PtrToInt(Combo->GetSelectedData());
                break;
        }
        case UI_CHANGEDEVICE:
            gD3DSettingsDlg.SetActive(!gD3DSettingsDlg.IsActive()); 
            break;
        case UI_VOLUMETRICMODEL: {
            CDXUTComboBox* Combo = dynamic_cast<CDXUTComboBox*>(control);
            VolumetricModel vol = static_cast<VolumetricModel>(PtrToInt(Combo->GetSelectedData()));
            switch(vol) {
            case VOLUME_NONE:
                gAppOptions.enableParticles = false;
                SetHairEnabled(false);
                break;
            case VOLUME_PARTICLES:
                gAppOptions.enableParticles = true;
                SetHairEnabled(false);
                break;
            case VOLUME_HAIR:
                gAppOptions.enableParticles = false;
                SetHairEnabled(true);
                break;
            case VOLUME_PARTICLES_AND_HAIR:
                gAppOptions.enableParticles = true;
                SetHairEnabled(true);
                break;
            }
            break;
        }
        case UI_AVSMAUTOBOUNDS:
            gAppOptions.enableAutoBoundsAVSM =
                dynamic_cast<CDXUTCheckBox*>(control)->GetChecked();
            break;
        case UI_SELECTEDSCENE: {
            gAppOptions.scene = 
                (SCENE_SELECTION) PtrToUlong(
                    gSceneSelectCombo->GetSelectedData());
            DestroyScene();
        }
        break;

        // These controls all imply changing parameters to the App constructor
        // (i.e. recreating resources and such), so we'll just clean up the app here and let it be
        // lazily recreated next render.
        case UI_AVSMSHADOWTEXTUREDIM:
            DeinitApp(); break;
        case UI_AVSMSORTINGMETHOD:
            gUIConstants.avsmSortingMethod = static_cast<unsigned int>(PtrToUlong(gAVSMSortingMethodCombo->GetSelectedData())); 
            break;
        case UI_AVSMNUMNODES: {
            unsigned int newNodeCount =
                static_cast<unsigned int>(PtrToUlong(gNodeCountCombo->GetSelectedData()));

            if (newNodeCount != gAppOptions.NodeCount) {
                gAppOptions.NodeCount = newNodeCount;
                gApp->SetNodeCount(newNodeCount);
            }
            break;
        }
        case UI_PARTICLESIZE: {
            // Map the slider's [0,100] integer range to floating point [0, 5.0f]
            Slider = gHUD[HUD_AVSM].GetSlider(UI_PARTICLESIZE);
            gUIConstants.particleSize = static_cast<float>(Slider->GetValue()) / kParticleSizeFactorUI;
            break;
        }
        case UI_PARTICLEOPACITY: {
            Slider = gHUD[HUD_AVSM].GetSlider(UI_PARTICLEOPACITY);
            gUIConstants.particleOpacity = Slider->GetValue();
            break;
        }
        case UI_DSMERROR: {
            Slider = gHUD[HUD_AVSM].GetSlider(UI_DSMERROR);
            gUIConstants.dsmError = static_cast<float>(Slider->GetValue()) / kDsmErrorFactorUI;
            break;
        }
        case UI_HAIR_THICKNESS: {
            Slider = gHUD[HUD_HAIR].GetSlider(UI_HAIR_THICKNESS);
            gUIConstants.hairThickness = Slider->GetValue();
            break;
        }
        case UI_HAIR_SHADOW_THICKNESS: {
            Slider = gHUD[HUD_HAIR].GetSlider(UI_HAIR_SHADOW_THICKNESS);
            gUIConstants.hairShadowThickness = Slider->GetValue();
            break;
        }
        case UI_HAIR_OPACITY: {
            Slider = gHUD[HUD_HAIR].GetSlider(UI_HAIR_OPACITY);
            gUIConstants.hairOpacity = Slider->GetValue();
            break;
        }
        case UI_HAIR_SORT: {
            gAppOptions.hairSort = 
                (unsigned int) PtrToUlong(gHairSortCombo->GetSelectedData());
            break;
        }
        case UI_HAIR_BLENDING: {
            gAppOptions.hairBlending = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); 
            break;
        }
        case UI_HAIR_SORT_ENABLED: {
            gAppOptions.hairSortEnabled = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); 
            break;
        }
        case UI_VOLUMESHADOWINGMETHOD:
            gUIConstants.volumeShadowMethod = (unsigned int)PtrToUlong(gVolumeShadowingMethodCombo->GetSelectedData()); 
            break;
        case UI_PAUSEPARTICLEANIMATION:
            gUIConstants.pauseParticleAnimaton = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); 
            break;
        case UI_LIGHTINGONLY:
            gUIConstants.lightingOnly = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); 
            break;
        case UI_VOLUMESHADOWCREATION:
            gAppOptions.enableVolumeShadowCreation = !gAppOptions.enableVolumeShadowCreation;
            break;
        case UI_VOLUMESHADOWLOOKUP:
            gUIConstants.enableVolumeShadowLookup = !gUIConstants.enableVolumeShadowLookup;
            break;
        case UI_DRAWTRANSMITTANCECURVE:
            gAppOptions.enableTransmittanceCurve =
                !gAppOptions.enableTransmittanceCurve;
            break;
        default:
            return;
    }
}


void CALLBACK OnD3D11DestroyDevice(void* userContext)
{
    DeinitApp();
    DestroyScene();
    
    gDialogResourceManager.OnD3D11DestroyDevice();
    gD3DSettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE(gTextHelper);
}


HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* d3dDevice, const DXGI_SURFACE_DESC* backBufferSurfaceDesc,
                                     void* userContext)
{
    ID3D11DeviceContext* d3dDeviceContext = DXUTGetD3D11DeviceContext();
    gDialogResourceManager.OnD3D11CreateDevice(d3dDevice, d3dDeviceContext);
    gD3DSettingsDlg.OnD3D11CreateDevice(d3dDevice);
    gTextHelper = new CDXUTTextHelper(d3dDevice, d3dDeviceContext, &gDialogResourceManager, 15);
    
    D3DXVECTOR3 vecEye(120.0f, 80.0f, -5.0f);
    D3DXVECTOR3 vecAt(50.0f,0.0f,30.0f);

    gMainOptions.gViewerCamera.SetViewParams(&vecEye, &vecAt);
    gMainOptions.gViewerCamera.SetRotateButtons(true, false, false);
    gMainOptions.gViewerCamera.SetScalers(0.01f, 50.0f);
    gMainOptions.gViewerCamera.SetDrag(true);
    gMainOptions.gViewerCamera.SetEnableYAxisMovement(true);
    gMainOptions.gViewerCamera.FrameMove(0);

    vecEye = D3DXVECTOR3(-103, 220, -56);
    vecAt =  D3DXVECTOR3(0.0f, 0.0f, 0.0f);

    gMainOptions.gLightCamera.SetViewParams(&vecEye, &vecAt);
    gMainOptions.gLightCamera.SetRotateButtons(true, false, false);
    gMainOptions.gLightCamera.SetScalers(0.01f, 50.0f);
    gMainOptions.gLightCamera.SetDrag(true);
    gMainOptions.gLightCamera.SetEnableYAxisMovement(true);
    gMainOptions.gLightCamera.SetProjParams(D3DX_PI / 4, 1.0f, 0.1f , 1000.0f);
    gMainOptions.gLightCamera.FrameMove(0);

    return S_OK;
}


void UpdateViewerCameraNearFar()
{
    // TODO: Set near/far based on scene...?
    gMainOptions.gViewerCamera.SetProjParams(D3DX_PI / 4, gAspectRatio, 0.05f, 500.0f);
}


HRESULT CALLBACK OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice, IDXGISwapChain* swapChain,
                                          const DXGI_SURFACE_DESC* backBufferSurfaceDesc, void* userContext)
{
    HRESULT hr;

    V_RETURN(gDialogResourceManager.OnD3D11ResizedSwapChain(d3dDevice, backBufferSurfaceDesc));
    V_RETURN(gD3DSettingsDlg.OnD3D11ResizedSwapChain(d3dDevice, backBufferSurfaceDesc));

    gAspectRatio = backBufferSurfaceDesc->Width / (FLOAT)backBufferSurfaceDesc->Height;

    UpdateViewerCameraNearFar();
    
    // Standard HUDs
    const int border = 20;
    int y = border;
    for (int i = 0; i < HUD_HAIR; ++i) {
        gHUD[i].SetLocation(backBufferSurfaceDesc->Width - gHUD[i].GetWidth() - border, y);
        y += gHUD[i].GetHeight() + border;
    }

    // Hair HUD
    gHUD[HUD_HAIR].SetLocation(border, backBufferSurfaceDesc->Height -
                               gHUD[HUD_EXPERT].GetHeight() -
                               gHUD[HUD_HAIR].GetHeight() - border - 50);

    // Expert HUD
    gHUD[HUD_EXPERT].SetLocation(border, backBufferSurfaceDesc->Height - gHUD[HUD_EXPERT].GetHeight() - border);

    // If there's no app, it'll pick this up when it gets lazily created so just ignore it
    if (gApp) {
        gApp->OnD3D11ResizedSwapChain(d3dDevice, backBufferSurfaceDesc);
    }

    return S_OK;
}


void CALLBACK OnD3D11ReleasingSwapChain(void* userContext)
{
    gDialogResourceManager.OnD3D11ReleasingSwapChain();
}


void CALLBACK OnD3D11FrameRender(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContext, double time,
                                 FLOAT elapsedTime, void* userContext)
{
    if (gD3DSettingsDlg.IsActive()) {
        gD3DSettingsDlg.OnRender(elapsedTime);
        return;
    }

    // Lazily create the application if need be
    if (!gApp) {
        InitApp(d3dDevice, d3dDeviceContext);
    }

    // Lazily load scene
    if (!gMesh.IsLoaded()) {
        InitScene(d3dDevice);
    }

    ID3D11RenderTargetView* pRTV = DXUTGetD3D11RenderTargetView();
    
    D3D11_VIEWPORT viewport;
    viewport.Width    = static_cast<float>(DXUTGetDXGIBackBufferSurfaceDesc()->Width);
    viewport.Height   = static_cast<float>(DXUTGetDXGIBackBufferSurfaceDesc()->Height);
    viewport.MinDepth = 0.0f;
    viewport.MaxDepth = 1.0f;
    viewport.TopLeftX = 0.0f;
    viewport.TopLeftY = 0.0f;
   
    gApp->Render(gAppOptions, 
                 d3dDeviceContext, pRTV, &gMesh, gParticleSystem, 
                 gWorldMatrix, GetCurrentUserCamera(), &gMainOptions.gLightCamera, &viewport, 
                 &gUIConstants);

    if (gDisplayUI) {
        d3dDeviceContext->RSSetViewports(1, &viewport);

        // Render HUDs in reverse order
        d3dDeviceContext->OMSetRenderTargets(1, &pRTV, 0);
        for (int i = HUD_NUM - 1; i >= 0; --i) {
            gHUD[i].OnRender(elapsedTime);
        }

        RenderText();
    }
}

CFirstPersonCamera * GetCurrentUserCamera()
{
    switch (gMainOptions.gControlObject) {
        case CONTROL_CAMERA: return &gMainOptions.gViewerCamera;
        case CONTROL_LIGHT:  return &gMainOptions.gLightCamera;
        default:             throw std::runtime_error("Unknown user control object!");
    }
}
