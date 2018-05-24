/////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");// you may not use this file except in compliance with the License.// You may obtain a copy of the License at//// http://www.apache.org/licenses/LICENSE-2.0//// Unless required by applicable law or agreed to in writing, software// distributed under the License is distributed on an "AS IS" BASIS,// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.// See the License for the specific language governing permissions and// limitations under the License.
//
// Modified by StephanieB5 to remove dependencies on DirectX SDK in 2017
//
/////////////////////////////////////////////////////////////////////////////////////////////

#include "DXUT.h"
#include "DXUTcamera.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "DDSTextureLoader.h"
#include "SDKmisc.h"
#include "SDKMesh.h"
#include "App.h"
#include "ShaderDefines.h"
#include <sstream>

using namespace DirectX;

// Constants
static const float kLightRotationSpeed = 0.05f;
static const float kSliderFactorResolution = 10000.0f;


enum SCENE_SELECTION {
    POWER_PLANT_SCENE,
    SPONZA_SCENE,
};

enum {
    UI_TOGGLEFULLSCREEN,
    UI_TOGGLEWARP,
    UI_CHANGEDEVICE,
    UI_ANIMATELIGHT,
    UI_FACENORMALS,
    UI_SELECTEDSCENE,
    UI_VISUALIZELIGHTCOUNT,
    UI_VISUALIZEPERSAMPLESHADING,
    UI_LIGHTINGONLY,
    UI_PERPIXEL,
    UI_LIGHTS,
    UI_LIGHTSTEXT,
    UI_LIGHTSPERPASS,
    UI_LIGHTSPERPASSTEXT,
    UI_CULLTECHNIQUE,
    UI_MSAA,
};

// List these top to bottom, since it is also the reverse draw order
enum {
    HUD_GENERIC = 0,
    HUD_PARTITIONS,
    HUD_FILTERING,
    HUD_EXPERT,
    HUD_NUM,
};


App* gApp = 0;

CFirstPersonCamera gViewerCamera;

CDXUTSDKMesh gMeshOpaque;
CDXUTSDKMesh gMeshAlpha;
XMMATRIX gWorldMatrix;
ID3D11ShaderResourceView* gSkyboxSRV = 0;

// DXUT GUI stuff
CDXUTDialogResourceManager gDialogResourceManager;
CD3DSettingsDlg gD3DSettingsDlg;
CDXUTDialog gHUD[HUD_NUM];
CDXUTCheckBox* gAnimateLightCheck = 0;
CDXUTComboBox* gMSAACombo = 0;
CDXUTComboBox* gSceneSelectCombo = 0;
CDXUTComboBox* gCullTechniqueCombo = 0;
CDXUTSlider* gLightsSlider = 0;
CDXUTTextHelper* gTextHelper = 0;

float gAspectRatio;
bool gDisplayUI = true;
bool gZeroNextFrameTime = true;

// Any UI state passed directly to rendering shaders
UIConstants gUIConstants;


bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* deviceSettings, void* userContext);
void CALLBACK OnFrameMove(double time, float elapsedTime, void* userContext);
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
                                 float elapsedTime, void* userContext);

void LoadSkybox(LPCWSTR fileName);

void InitApp(ID3D11Device* d3dDevice);
void DestroyApp();
void InitScene(ID3D11Device* d3dDevice);
void DestroyScene();

void InitUI();
void UpdateUIState();


int WINAPI wWinMain(HINSTANCE /*hInstance*/, HINSTANCE /*hPrevInstance*/, LPWSTR /*lpCmdLine*/, INT /*nCmdShow*/)
{
    // Enable run-time memory check for debug builds.
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    DXUTSetCallbackDeviceChanging(ModifyDeviceSettings);
    DXUTSetCallbackMsgProc(MsgProc);
    DXUTSetCallbackKeyboard(OnKeyboard);
    DXUTSetCallbackFrameMove(OnFrameMove);

    DXUTSetCallbackD3D11DeviceCreated(OnD3D11CreateDevice);
    DXUTSetCallbackD3D11SwapChainResized(OnD3D11ResizedSwapChain);
    DXUTSetCallbackD3D11FrameRender(OnD3D11FrameRender);
    DXUTSetCallbackD3D11SwapChainReleasing(OnD3D11ReleasingSwapChain);
    DXUTSetCallbackD3D11DeviceDestroyed(OnD3D11DestroyDevice);
    
    DXUTSetIsInGammaCorrectMode(true);

    DXUTInit(true, true, 0);
    InitUI();

    DXUTSetCursorSettings(true, true);
    DXUTSetHotkeyHandling(true, true, false);
    DXUTCreateWindow(L"Deferred Shading");
    DXUTCreateDevice(D3D_FEATURE_LEVEL_11_0, true, 1920, 1080);
    DXUTMainLoop();

    return DXUTGetExitCode();
}


void InitUI()
{
    // Setup default UI state
    // NOTE: All of these are made directly available in the shader constant buffer
    // This is convenient for development purposes.
    gUIConstants.forcePerPixel = 0;
    gUIConstants.lightingOnly = 0;
    gUIConstants.faceNormals = 0;
    gUIConstants.visualizeLightCount = 0;
    gUIConstants.visualizePerSampleShading = 0;
    gUIConstants.lightCullTechnique = CULL_COMPUTE_SHADER_TILE;
    
    gD3DSettingsDlg.Init(&gDialogResourceManager);

    for (int i = 0; i < HUD_NUM; ++i) {
        gHUD[i].Init(&gDialogResourceManager);
        gHUD[i].SetCallback(OnGUIEvent);
    }

    int width = 200;

    // Generic HUD
    {
        CDXUTDialog* HUD = &gHUD[HUD_GENERIC];
        int y = 0;

        HUD->AddButton(UI_TOGGLEFULLSCREEN, L"Toggle full screen", 0, y, width, 23);
        y += 26;

        // Warp doesn't support DX11 yet
        //HUD->AddButton(UI_TOGGLEWARP, L"Toggle WARP (F3)", 0, y, width, 23, VK_F3);
        //y += 26;

        HUD->AddButton(UI_CHANGEDEVICE, L"Change device (F2)", 0, y, width, 23, VK_F2);
        y += 26;

        HUD->AddComboBox(UI_MSAA, 0, y, width, 23, 0, false, &gMSAACombo);
        y += 26;
        gMSAACombo->AddItem(L"No MSAA", ULongToPtr(1));
        gMSAACombo->AddItem(L"2x MSAA", ULongToPtr(2));
        gMSAACombo->AddItem(L"4x MSAA", ULongToPtr(4));
        gMSAACombo->AddItem(L"8x MSAA", ULongToPtr(8));

        HUD->AddComboBox(UI_SELECTEDSCENE, 0, y, width, 23, 0, false, &gSceneSelectCombo);
        y += 26;
        gSceneSelectCombo->AddItem(L"Power Plant", ULongToPtr(POWER_PLANT_SCENE));
        gSceneSelectCombo->AddItem(L"Sponza", ULongToPtr(SPONZA_SCENE));

        HUD->AddCheckBox(UI_ANIMATELIGHT, L"Animate Lights", 0, y, width, 23, false, VK_SPACE, false, &gAnimateLightCheck);
        y += 26;

        HUD->AddCheckBox(UI_LIGHTINGONLY, L"Lighting Only", 0, y, width, 23, gUIConstants.lightingOnly != 0);
        y += 26;

        HUD->AddCheckBox(UI_PERPIXEL, L"Force Per Pixel Shading", 0, y, width, 23, gUIConstants.forcePerPixel != 0);
        y += 26;

        HUD->AddCheckBox(UI_FACENORMALS, L"Biased Sampler", 0, y, width, 23, gUIConstants.faceNormals != 0);
        y += 26;

        HUD->AddCheckBox(UI_VISUALIZELIGHTCOUNT, L"Visualize Light Count", 0, y, width, 23, gUIConstants.visualizeLightCount != 0);
        y += 26;

        HUD->AddCheckBox(UI_VISUALIZEPERSAMPLESHADING, L"Visualize Shading Freq.", 0, y, width, 23, gUIConstants.visualizePerSampleShading != 0);
        y += 26;

        HUD->AddStatic(UI_LIGHTSTEXT, L"Lights:", 0, y, width, 23);
        y += 26;
        HUD->AddSlider(UI_LIGHTS, 0, y, width, 23, 0, MAX_LIGHTS_POWER, MAX_LIGHTS_POWER, false, &gLightsSlider);
        y += 26;


        HUD->AddComboBox(UI_CULLTECHNIQUE, 0, y, width, 23, 0, false, &gCullTechniqueCombo);
        y += 26;
        gCullTechniqueCombo->AddItem(L"No Cull Forward", ULongToPtr(CULL_FORWARD_NONE));
        gCullTechniqueCombo->AddItem(L"No Cull Pre-Z", ULongToPtr(CULL_FORWARD_PREZ_NONE));
        gCullTechniqueCombo->AddItem(L"No Cull Deferred", ULongToPtr(CULL_DEFERRED_NONE));
        gCullTechniqueCombo->AddItem(L"Quad", ULongToPtr(CULL_QUAD));
        gCullTechniqueCombo->AddItem(L"Quad Deferred Light", ULongToPtr(CULL_QUAD_DEFERRED_LIGHTING));
        gCullTechniqueCombo->AddItem(L"Compute Shader Tile", ULongToPtr(CULL_COMPUTE_SHADER_TILE));
        gCullTechniqueCombo->SetSelectedByData(ULongToPtr(gUIConstants.lightCullTechnique));

        HUD->SetSize(width, y);
    }

    // Expert HUD
    {
        CDXUTDialog* HUD = &gHUD[HUD_EXPERT];
        int y = 0;
    
        HUD->SetSize(width, y);

        // Initially hidden
        HUD->SetVisible(false);
    }
    
    UpdateUIState();
}


void UpdateUIState()
{
    //int technique = PtrToUint(gCullTechniqueCombo->GetSelectedData());
}


void InitApp(ID3D11Device* d3dDevice)
{
    DestroyApp();
    
    // Get current UI settings
    unsigned int msaaSamples = PtrToUint(gMSAACombo->GetSelectedData());
    gApp = new App(d3dDevice, 1 << gLightsSlider->GetValue(), msaaSamples);

    // Initialize with the current surface description
    gApp->OnD3D11ResizedSwapChain(d3dDevice, DXUTGetDXGIBackBufferSurfaceDesc());

    // Zero out the elapsed time for the next frame
    gZeroNextFrameTime = true;
}


void DestroyApp()
{
    SAFE_DELETE(gApp);
}

void LoadSkybox(ID3D11Device* d3dDevice, LPCWSTR fileName)
{
    ID3D11Resource* resource = nullptr;
    HRESULT hr;
    // StephanieB5: All the texture files provide with sample files are DDS files
    hr = CreateDDSTextureFromFile(d3dDevice, fileName, &resource, nullptr, 0, nullptr);
    assert(SUCCEEDED(hr));

    d3dDevice->CreateShaderResourceView(resource, 0, &gSkyboxSRV);
    resource->Release();
}

void InitScene(ID3D11Device* d3dDevice)
{
    DestroyScene();

    XMVECTOR cameraEye = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    XMVECTOR cameraAt = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    float sceneScaling = 1.0f;
    XMVECTOR sceneTranslation = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    bool zAxisUp = false;

    SCENE_SELECTION scene = static_cast<SCENE_SELECTION>(PtrToUlong(gSceneSelectCombo->GetSelectedData()));
    switch (scene) {
        case POWER_PLANT_SCENE: {
            gMeshOpaque.Create(d3dDevice, L"media\\powerplant\\powerplant.sdkmesh");
            LoadSkybox(d3dDevice, L"media\\Skybox\\Clouds.dds");
            cameraEye = XMVectorSet(100.0f, 5.0f, 5.0f, 0.0f);
            sceneScaling = 1.0f;
            // sceneScaling == 1.0f so no need to rescale cameraEye nor cameraAt
        } break;

        case SPONZA_SCENE: {
            gMeshOpaque.Create(d3dDevice, L"media\\Sponza\\sponza_dds.sdkmesh");
            LoadSkybox(d3dDevice, L"media\\Skybox\\Clouds.dds");
            cameraEye = XMVectorSet(1200.0f, 200.0f, 100.0f, 0.0f);
            sceneScaling = 0.05f;
            cameraEye *= sceneScaling;
            // cameraAt is a zero vector and is unaffected by scaling
        } break;
    };
    
    gWorldMatrix = XMMatrixScaling(sceneScaling, sceneScaling, sceneScaling);
    if (zAxisUp) {
        XMMATRIX m = XMMatrixRotationX(-XM_PI / 2.0f);
        gWorldMatrix *= m;
    }
    {
        XMMATRIX t = XMMatrixTranslationFromVector(sceneTranslation);
        gWorldMatrix *= t;
    }

    gViewerCamera.SetViewParams(cameraEye, cameraAt);
    gViewerCamera.SetScalers(0.01f, 10.0f);
    gViewerCamera.FrameMove(0.0f);
    
    // Zero out the elapsed time for the next frame
    gZeroNextFrameTime = true;
}


void DestroyScene()
{
    gMeshOpaque.Destroy();
    gMeshAlpha.Destroy();
    SAFE_RELEASE(gSkyboxSRV);
}


bool CALLBACK ModifyDeviceSettings(DXUTDeviceSettings* deviceSettings, void* /*userContext*/)
{
    // For the first device created if its a REF device, optionally display a warning dialog box
    static bool firstTime = true;
    if (firstTime) {
        firstTime = false;
        if (deviceSettings->d3d11.DriverType == D3D_DRIVER_TYPE_REFERENCE) {
            DXUTDisplaySwitchingToREFWarning();
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


void CALLBACK OnFrameMove(double /*time*/, float elapsedTime, void* /*userContext*/)
{
    if (gZeroNextFrameTime) {
        elapsedTime = 0.0f;
    }
    
    // Update the camera's position based on user input
    gViewerCamera.FrameMove(elapsedTime);

    // If requested, animate scene
    if (gApp && gAnimateLightCheck->GetChecked()) {
        gApp->Move(elapsedTime);
    }
}


LRESULT CALLBACK MsgProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, bool* noFurtherProcessing,
                          void* /*userContext*/)
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

    // Pass all remaining windows messages to camera so it can respond to user input
    gViewerCamera.HandleMessages(hWnd, uMsg, wParam, lParam);

    return 0;
}


void CALLBACK OnKeyboard(UINT character, bool keyDown, bool /*altDown*/, void* /*userContext*/)
{
    if(keyDown) {
        switch (character) {
        case VK_F8:
            // Toggle visibility of expert HUD
            gHUD[HUD_EXPERT].SetVisible(!gHUD[HUD_EXPERT].GetVisible());
            break;
        case VK_F4:
            // Toggle display of UI on/off
            gDisplayUI = !gDisplayUI;
            break;
        }
    }
}


void CALLBACK OnGUIEvent(UINT /*eventID*/, INT controlID, CDXUTControl* control, void* /*userContext*/)
{
    switch (controlID) {
        case UI_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen(); break;
        case UI_TOGGLEWARP:
            DXUTToggleWARP(); break;
        case UI_CHANGEDEVICE:
            gD3DSettingsDlg.SetActive(!gD3DSettingsDlg.IsActive()); break;
        case UI_LIGHTINGONLY:
            gUIConstants.lightingOnly = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); break;
        case UI_PERPIXEL:
            gUIConstants.forcePerPixel = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); break;
        case UI_FACENORMALS:
            gUIConstants.faceNormals = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); break;        
        case UI_VISUALIZELIGHTCOUNT:
            gUIConstants.visualizeLightCount = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); break;            
        case UI_VISUALIZEPERSAMPLESHADING:
            gUIConstants.visualizePerSampleShading = dynamic_cast<CDXUTCheckBox*>(control)->GetChecked(); break;            
        case UI_SELECTEDSCENE:
            DestroyScene(); break;
        case UI_LIGHTS:
            gApp->SetActiveLights(DXUTGetD3D11Device(), 1 << gLightsSlider->GetValue()); break;
        case UI_CULLTECHNIQUE:
            gUIConstants.lightCullTechnique = static_cast<unsigned int>(PtrToUlong(gCullTechniqueCombo->GetSelectedData())); break;

        // These controls all imply changing parameters to the App constructor
        // (i.e. recreating resources and such), so we'll just clean up the app here and let it be
        // lazily recreated next render.
        case UI_MSAA:
            DestroyApp(); break;

        default:
            break;
    }

    UpdateUIState();
}


void CALLBACK OnD3D11DestroyDevice(void* /*userContext*/)
{
    DestroyApp();
    DestroyScene();
    
    gDialogResourceManager.OnD3D11DestroyDevice();
    gD3DSettingsDlg.OnD3D11DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE(gTextHelper);
}


HRESULT CALLBACK OnD3D11CreateDevice(ID3D11Device* d3dDevice, const DXGI_SURFACE_DESC* /*backBufferSurfaceDesc*/,
                                     void* /*userContext*/)
{    
    ID3D11DeviceContext* d3dDeviceContext = DXUTGetD3D11DeviceContext();
    gDialogResourceManager.OnD3D11CreateDevice(d3dDevice, d3dDeviceContext);
    gD3DSettingsDlg.OnD3D11CreateDevice(d3dDevice);
    gTextHelper = new CDXUTTextHelper(d3dDevice, d3dDeviceContext, &gDialogResourceManager, 15);
    
    gViewerCamera.SetRotateButtons(true, false, false);
    gViewerCamera.SetDrag(true);
    gViewerCamera.SetEnableYAxisMovement(true);

    return S_OK;
}


HRESULT CALLBACK OnD3D11ResizedSwapChain(ID3D11Device* d3dDevice, IDXGISwapChain* /*swapChain*/,
                                          const DXGI_SURFACE_DESC* backBufferSurfaceDesc, void* /*userContext*/)
{
    HRESULT hr;

    V_RETURN(gDialogResourceManager.OnD3D11ResizedSwapChain(d3dDevice, backBufferSurfaceDesc));
    V_RETURN(gD3DSettingsDlg.OnD3D11ResizedSwapChain(d3dDevice, backBufferSurfaceDesc));

    gAspectRatio = backBufferSurfaceDesc->Width / (float)backBufferSurfaceDesc->Height;

    // NOTE: Complementary Z (1-z) buffer used here, so swap near/far!
    gViewerCamera.SetProjParams(XM_PI / 4.0f, gAspectRatio, 300.0f, 0.05f);

    // Standard HUDs
    const int border = 20;
    int y = border;
    for (int i = 0; i < HUD_EXPERT; ++i) {
        gHUD[i].SetLocation(backBufferSurfaceDesc->Width - gHUD[i].GetWidth() - border, y);
        y += gHUD[i].GetHeight() + border;
    }

    // Expert HUD
    gHUD[HUD_EXPERT].SetLocation(border, 80);

    // If there's no app, it'll pick this up when it gets lazily created so just ignore it
    if (gApp) {
        gApp->OnD3D11ResizedSwapChain(d3dDevice, backBufferSurfaceDesc);
    }

    return S_OK;
}


void CALLBACK OnD3D11ReleasingSwapChain(void* /*userContext*/)
{
    gDialogResourceManager.OnD3D11ReleasingSwapChain();
}

#ifdef _PERF
template <class T>
void PrintStringToUI(const char* strMsg, T val, 
    size_t precision = 0,
    const char *suffix = "")
{
    std::wostringstream uiString;
    if (precision != 0) {
        uiString.precision(precision);
        uiString.setf(std::ios::fixed, std::ios::floatfield);
    }
    uiString << strMsg << val << suffix;
    gTextHelper->DrawTextLine(uiString.str().c_str());
}
#endif // _PERF

void CALLBACK OnD3D11FrameRender(ID3D11Device* d3dDevice, ID3D11DeviceContext* d3dDeviceContext, double /*time*/,
                                 float elapsedTime, void* /*userContext*/)
{
    if (gZeroNextFrameTime) {
        elapsedTime = 0.0f;
    }
    gZeroNextFrameTime = false;

    if (gD3DSettingsDlg.IsActive()) {
        gD3DSettingsDlg.OnRender(elapsedTime);
        return;
    }

    // Lazily create the application if need be
    if (!gApp) {
        InitApp(d3dDevice);
    }

    // Lazily load scene
    if (!gMeshOpaque.IsLoaded() && !gMeshAlpha.IsLoaded()) {
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

    gApp->Render(d3dDeviceContext, pRTV, gMeshOpaque, gMeshAlpha, gSkyboxSRV,
        gWorldMatrix, &gViewerCamera, &viewport, &gUIConstants);

#ifdef _PERF
    FrameStats fStats = gApp->GetFrameStats();
#endif // _PERF
    if (gDisplayUI) {
        d3dDeviceContext->RSSetViewports(1, &viewport);

        // Render HUDs in reverse order
        d3dDeviceContext->OMSetRenderTargets(1, &pRTV, 0);
        for (int i = HUD_NUM - 1; i >= 0; --i) {
            gHUD[i].OnRender(elapsedTime);
        }

        // Render text
        gTextHelper->Begin();

        gTextHelper->SetInsertionPos(2, 0);
        gTextHelper->SetForegroundColor(XMVectorSet(1.0f, 1.0f, 0.0f, 1.0f));
        gTextHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));
        //gTextHelper->DrawTextLine(DXUTGetDeviceStats());

        // Output frame time
        {
            std::wostringstream oss;
            oss << 1000.0f / DXUTGetFPS() << " ms / frame";
            gTextHelper->DrawTextLine(oss.str().c_str());
        }

        // Output light info
        {
            std::wostringstream oss;
            oss << "Lights: " << gApp->GetActiveLights();
            gTextHelper->DrawTextLine(oss.str().c_str());
        }

#ifdef _PERF
        static const UINT numFrames = 64;
        static UINT frame = numFrames - 1;
        static FrameStats accStats[numFrames];
        static bool statsReady = false;

        if (!statsReady) {
            for (UINT i = 0; i < numFrames - frame; ++i) {
                accStats[i].Accumulate(fStats);
            }
            if (frame == 0) {
                statsReady = true;
            }
        } else {
            for (UINT i = 0; i < numFrames; ++i) {
                accStats[i].Accumulate(fStats);
            }
        }

        if (statsReady) {
            accStats[frame].Normalize(numFrames);
            // Output frame stats
            {
                std::wostringstream oss;
                PrintStringToUI("G Buff Gen time:", 
                    accStats[frame].m_totalGBuffGen, 6, " ms");
                PrintStringToUI("Skybox time    :", 
                    accStats[frame].m_totalSkyBox, 6, " ms");
                PrintStringToUI("DPS time       :", 
                    accStats[frame].m_totalShadingTime, 6, " ms");
                gTextHelper->DrawTextLine(oss.str().c_str());
            }
            memset(&accStats[frame], 0, sizeof(accStats[frame]));
        }
        frame = (frame + 1) % numFrames;
#endif // _PERF


        gTextHelper->End();
    }
}
