/*
 * Copyright (C) 2009 Jorge Jimenez (jim@unizar.es)
 * Copyright (C) 2009 Diego Gutierrez (diegog@unizar.es)
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, 
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must display the names 'Jorge Jimenez'
 *    and 'Diego Gutierrez' as 'Real-Time Rendering R&D' in the credits of the
 *    application, if such credits exist. The authors of this work must be
 *    notified via email (jim@unizar.es) in this case of redistribution.
 * 
 * 3. Neither the name of copyright holders nor the names of its contributors 
 *    may be used to endorse or promote products derived from this software 
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS 
 * IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS 
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "DXUT.h"
#include "DXUTgui.h"
#include "DXUTsettingsDlg.h"
#include "SDKmisc.h"
#include "SDKmesh.h"

#include <sstream>
#include <iomanip>
#include <limits>

#include "Timer.h"
#include "Camera.h"
#include "RenderTarget.h"
#include "ShadowMap.h"
#include "Animation.h"
#include "Fade.h"
#include "SplashScreen.h"
#include "Intro.h"
#include "SubsurfaceScatteringPass.h"
#include "BloomPass.h"

using namespace std;

#define DEG2RAD(a) (a * D3DX_PI / 180.f)


Camera camera;
CDXUTDialogResourceManager dialogResourceManager;
CD3DSettingsDlg settingsDialog;
CDXUTDialog hud;

Timer *timer = NULL;
ID3DX10Font *font = NULL;
ID3DX10Sprite *sprite = NULL;
CDXUTTextHelper *txtHelper = NULL;
ID3D10Effect *mainEffect = NULL;
CDXUTSDKMesh mesh;
ID3D10InputLayout *vertexLayout = NULL;
ID3D10ShaderResourceView *beckmannMapView = NULL;
SplashScreen *splashScreen = NULL;
Intro *intro = NULL;

RenderTarget *mainRenderTarget = NULL;
RenderTarget *depthRenderTarget1X = NULL;
RenderTarget *depthRenderTarget = NULL;
RenderTarget *stencilRenderTarget = NULL;
DepthStencil *depthStencil = NULL;
DepthStencil *depthStencil1X = NULL;
RenderTarget *resolveRenderTarget = NULL;
ID3D10Texture2D *backbufferTexture = NULL;
SubsurfaceScatteringPass *subsurfaceScatteringPass = NULL;
BloomPass *bloomPass = NULL;


vector<Gaussian> skin6Gaussians;
const vector<Gaussian> *currentGaussians = &Gaussian::SKIN;


struct MsaaMode {
    wstring name;
    DXGI_SAMPLE_DESC desc;
};
const int NUM_MSAA_MODES = 7;
MsaaMode msaaModes[NUM_MSAA_MODES] = {
    {L"1x MSAA",   {1,  0}},
    {L"2x MSAA",   {2,  0}},
    {L"4x MSAA",   {4,  0}},
    {L"8x CSAA",   {4,  8}},
    {L"8xQ CSAA",  {8,  8}},
    {L"16x CSAA",  {4, 16}},
    {L"16xQ CSAA", {8, 16}},
};
int msaaIndex = 0;

double t0 = numeric_limits<double>::min();
double tFade = numeric_limits<double>::max();
bool skipIntro = false;

const int MAX_N_LIGHTS = 6;
const int N_HEADS = 1;
const int SHADOW_MAP_SIZE = 2048;
const int HUD_WIDTH = 130;
const float CAMERA_FOV = 20.0f;

struct Light {
    Camera camera;
    float fov;
    float spotExponent;
    D3DXVECTOR3 color;
    float attenuation;
    float range;
    float bias;
    ShadowMap *shadowMap;
};
Light lights[MAX_N_LIGHTS];
int nLights = 2;

enum Object { OBJECT_CAMERA, OBJECT_LIGHT1, OBJECT_LIGHT2, OBJECT_LIGHT3, OBJECT_LIGHT4, OBJECT_LIGHT5, OBJECT_LIGHT6 };
Object object = OBJECT_CAMERA;

enum State { STATE_SPLASH, STATE_INTRO, STATE_RUNNING };
#ifdef XYZRGB_BUILD
State state = STATE_SPLASH;
#else
State state = STATE_RUNNING;
#endif


#define IDC_TOGGLEFULLSCREEN    1
#define IDC_CHANGEDEVICE        2
#define IDC_TOGGLEREF           3
#define IDC_TOGGLEWARP          4
#define IDC_NLIGHTS             5
#define IDC_OBJECT              6
#define IDC_MESH                7
#define IDC_MATERIAL            8
#define IDC_PROFILE             9
#define IDC_MSAA               10
#define IDC_SSSLEVEL_LABEL     11
#define IDC_SSSLEVEL           12
#define IDC_CORRECTION_LABEL   13
#define IDC_CORRECTION         14
#define IDC_MAXDD_LABEL        15
#define IDC_MAXDD              16
#define IDC_BUMPINESS_LABEL    17
#define IDC_BUMPINESS          18
#define IDC_ROUGHNESS_LABEL    19
#define IDC_ROUGHNESS          20
#define IDC_SSS                21
#define IDC_BLOOM              22


bool CALLBACK isDeviceAcceptable(UINT adapter, UINT output, D3D10_DRIVER_TYPE deviceType, DXGI_FORMAT format, bool windowed, void *context);
HRESULT CALLBACK onCreateDevice(ID3D10Device *device, const DXGI_SURFACE_DESC *desc, void *context);
HRESULT CALLBACK onResizedSwapChain(ID3D10Device *device, IDXGISwapChain *pSwapChain, const DXGI_SURFACE_DESC *desc, void *context);
void CALLBACK onReleasingSwapChain(void *context);
void CALLBACK onDestroyDevice(void *context);
void CALLBACK onFrameRender(ID3D10Device *device, double time, float elapsedTime, void *context);
LRESULT CALLBACK msgProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam, bool *pbNoFurtherProcessing, void *context);
void CALLBACK keyboardProc(UINT nChar, bool bKeyDown, bool bAltDown, void *context);
void CALLBACK onFrameMove(double time, float elapsedTime, void *context);
bool CALLBACK modifyDeviceSettings(DXUTDeviceSettings *settings, void *context);
void CALLBACK onGUIEvent(UINT event, int id, CDXUTControl *control, void *context);
void initApp();


INT WINAPI wWinMain(HINSTANCE, HINSTANCE, LPWSTR, int) {
    // Enable run-time memory check for debug builds.
    #if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    #endif

    DXUTSetCallbackD3D10DeviceAcceptable(isDeviceAcceptable);
    DXUTSetCallbackD3D10DeviceCreated(onCreateDevice);
    DXUTSetCallbackD3D10SwapChainResized(onResizedSwapChain);
    DXUTSetCallbackD3D10SwapChainReleasing(onReleasingSwapChain);
    DXUTSetCallbackD3D10DeviceDestroyed(onDestroyDevice);
    DXUTSetCallbackD3D10FrameRender(onFrameRender);

    DXUTSetCallbackMsgProc(msgProc);
    DXUTSetCallbackKeyboard(keyboardProc);
    DXUTSetCallbackFrameMove(onFrameMove);
    DXUTSetCallbackDeviceChanging(modifyDeviceSettings);

    initApp();
    #ifdef XYZRGB_BUILD
    DXUTInit(true, true, NULL);
    #else
    DXUTInit(true, true, L"-forcevsync:0");
    #endif
    DXUTSetCursorSettings(true, true);
    DXUTCreateWindow(L"Screen-Space Subsurface Scattering Demo");
    DXUTCreateDevice(true, 1280, 720);
    DXUTMainLoop();

    return DXUTGetExitCode();
}


void buildObjectComboBox() {
    hud.GetComboBox(IDC_OBJECT)->RemoveAllItems();
    hud.GetComboBox(IDC_OBJECT)->AddItem(L"Camera", (LPVOID) 0);
    for (int i = 0; i < nLights; i++) {
        wstringstream s;
        s << "Light " << i + 1;
        hud.GetComboBox(IDC_OBJECT)->AddItem(s.str().c_str(), (LPVOID) (i + 1));
    }
}


void initApp() {
    settingsDialog.Init(&dialogResourceManager);
    hud.Init(&dialogResourceManager);
    hud.SetCallback(onGUIEvent);

    int iY = 10;

    hud.AddButton(IDC_TOGGLEFULLSCREEN, L"Toggle full screen", 35, iY, HUD_WIDTH, 22);
    hud.AddButton(IDC_CHANGEDEVICE, L"Change device (F2)", 35, iY += 24, HUD_WIDTH, 22, VK_F2);
    hud.AddButton(IDC_TOGGLEREF, L"Toggle REF (F3)", 35, iY += 24, HUD_WIDTH, 22, VK_F3);
    hud.AddButton(IDC_TOGGLEWARP, L"Toggle WARP (F4)", 35, iY += 24, HUD_WIDTH, 22, VK_F4);

    hud.AddComboBox(IDC_MSAA, 35, iY += 24, HUD_WIDTH, 22);
    for(char i = 0; i < NUM_MSAA_MODES; i++){
        hud.GetComboBox(IDC_MSAA)->AddItem(msaaModes[i].name.c_str(), (void*) i);
    }

    hud.AddComboBox(IDC_NLIGHTS, 35, iY += 24, HUD_WIDTH, 22, 0, false);
    hud.GetComboBox(IDC_NLIGHTS)->AddItem(L"1 Light", (LPVOID) 0);
    hud.GetComboBox(IDC_NLIGHTS)->AddItem(L"2 Lights", (LPVOID) 1);
    hud.GetComboBox(IDC_NLIGHTS)->AddItem(L"3 Lights", (LPVOID) 2);
    hud.GetComboBox(IDC_NLIGHTS)->AddItem(L"4 Lights", (LPVOID) 3);
    hud.GetComboBox(IDC_NLIGHTS)->AddItem(L"5 Lights", (LPVOID) 4);
    hud.GetComboBox(IDC_NLIGHTS)->AddItem(L"6 Lights", (LPVOID) 5);
    hud.GetComboBox(IDC_NLIGHTS)->SetSelectedByIndex(nLights - 1);

    hud.AddComboBox(IDC_OBJECT, 35, iY += 24, HUD_WIDTH, 22, 0, false);
    buildObjectComboBox();

    hud.AddComboBox(IDC_MESH, 35, iY += 24, HUD_WIDTH, 22, 0, false);

    #ifdef XYZRGB_BUILD
    hud.GetComboBox(IDC_MESH)->AddItem(L"Roberto", (LPVOID) 0);
    hud.GetComboBox(IDC_MESH)->AddItem(L"Apollo", (LPVOID) 1);
    #endif
    hud.GetComboBox(IDC_MESH)->AddItem(L"Teapot", (LPVOID) 2);

    hud.AddComboBox(IDC_MATERIAL, 35, iY += 24, HUD_WIDTH, 22, 0, false);
    hud.GetComboBox(IDC_MATERIAL)->AddItem(L"Skin-4G", (LPVOID) 0);
    hud.GetComboBox(IDC_MATERIAL)->AddItem(L"Skin-6G", (LPVOID) 1);
    hud.GetComboBox(IDC_MATERIAL)->AddItem(L"Marble", (LPVOID) 2);

    hud.AddStatic(IDC_SSSLEVEL_LABEL, L"SSS Level", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_SSSLEVEL, 35, iY += 24, HUD_WIDTH, 22);
    hud.AddStatic(IDC_CORRECTION_LABEL, L"Correction", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_CORRECTION, 35, iY += 24, HUD_WIDTH, 22);
    hud.AddStatic(IDC_MAXDD_LABEL, L"Max Derivative", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_MAXDD, 35, iY += 24, HUD_WIDTH, 22);

    iY += 24;
    hud.AddStatic(IDC_BUMPINESS_LABEL, L"Bumpiness", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_BUMPINESS, 35, iY += 24, HUD_WIDTH, 22);
    hud.AddStatic(IDC_ROUGHNESS_LABEL, L"Specular Roughness", 35, iY += 24, HUD_WIDTH, 22);
    hud.AddSlider(IDC_ROUGHNESS, 35, iY += 24, HUD_WIDTH, 22);

    hud.AddCheckBox(IDC_SSS, L"SSS Pass", 35, iY += 24, HUD_WIDTH, 22, true);
    hud.AddCheckBox(IDC_BLOOM, L"Bloom Pass", 35, iY += 24, HUD_WIDTH, 22, true);
    hud.AddCheckBox(IDC_PROFILE, L"Profile", 35, iY += 24, HUD_WIDTH, 22, false);

    // This 6-gaussian sum is included for comparison purposes
    float variances[] = {0.0484f, 0.187f, 0.567f, 1.99f, 7.41f};
    D3DXVECTOR3 weights[] = {
        D3DXVECTOR3(0.233f, 0.455f, 0.649f),
        D3DXVECTOR3(0.100f, 0.336f, 0.344f),
        D3DXVECTOR3(0.118f, 0.198f, 0.0f),
        D3DXVECTOR3(0.113f, 0.007f, 0.007f),
        D3DXVECTOR3(0.358f, 0.004f, 0.0f),
        D3DXVECTOR3(0.078f, 0.0f, 0.0f)
    };
    skin6Gaussians = Gaussian::gaussianSum(variances, weights, 5);
}


bool CALLBACK isDeviceAcceptable(UINT adapter, UINT output, D3D10_DRIVER_TYPE deviceType, DXGI_FORMAT format, bool windowed, void *context) {
    return true;
}


void setBumpiness(float bumpiness) {
    HRESULT hr;

    int min, max;
    hud.GetSlider(IDC_BUMPINESS)->GetRange(min, max);
    hud.GetSlider(IDC_BUMPINESS)->SetValue(min + int(bumpiness * (max - min)));
    
    wstringstream s;
    s << L"Bumpiness: " << bumpiness;
    hud.GetStatic(IDC_BUMPINESS_LABEL)->SetText(s.str().c_str());

    V(mainEffect->GetVariableByName("bumpiness")->AsScalar()->SetFloat(bumpiness));
}


void setRoughness(float roughness) {
    HRESULT hr;

    int min, max;
    hud.GetSlider(IDC_ROUGHNESS)->GetRange(min, max);
    hud.GetSlider(IDC_ROUGHNESS)->SetValue(min + int(roughness * (max - min)));
    
    wstringstream s;
    s << L"Specular Roughness: " << roughness;
    hud.GetStatic(IDC_ROUGHNESS_LABEL)->SetText(s.str().c_str());

    V(mainEffect->GetVariableByName("roughness")->AsScalar()->SetFloat(roughness));
}


void loadMainEffect() {
    HRESULT hr;

    SAFE_RELEASE(mainEffect);

    wstringstream s;
    s << L"Main" << nLights << L".fxo";
    V(D3DX10CreateEffectFromResource(GetModuleHandle(NULL), s.str().c_str(), NULL, NULL, NULL, NULL, D3DXFX_NOT_CLONEABLE, 0, DXUTGetD3D10Device(), NULL, NULL, &mainEffect, NULL, NULL));

    V(mainEffect->GetVariableByName("beckmannTex")->AsShaderResource()->SetResource(beckmannMapView));
    setBumpiness(1.0f);
}


void CALLBACK createTextureFromFile(ID3D10Device* device, char *filename, ID3D10ShaderResourceView **shaderResourceView, void *context, bool srgb) {
    HRESULT hr;

    wstringstream s;
    s << ((wchar_t *) context) << "\\" << filename;

    D3DX10_IMAGE_LOAD_INFO loadInfo;
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;

    if (srgb) {
        loadInfo.Filter = D3DX10_FILTER_POINT | D3DX10_FILTER_SRGB_IN;
        
        D3DX10_IMAGE_INFO info;
        V(D3DX10GetImageInfoFromResource(GetModuleHandle(NULL), s.str().c_str(), NULL, &info, NULL));
        loadInfo.Format = MAKE_SRGB(info.Format);
    }
    V(D3DX10CreateShaderResourceViewFromResource(device, GetModuleHandle(NULL), s.str().c_str(), &loadInfo, NULL, shaderResourceView, NULL));
}


void loadModel(ID3D10Device *device, const wstring &name, const wstring &path) {
    HRESULT hr;

    HRSRC src = FindResource(GetModuleHandle(NULL), name.c_str(), RT_RCDATA);
    HGLOBAL res = LoadResource(GetModuleHandle(NULL), src);
    UINT size = SizeofResource(GetModuleHandle(NULL), src);
    LPBYTE data = (LPBYTE) LockResource(res);

    SDKMESH_CALLBACKS10 callbacks;
    ZeroMemory(&callbacks, sizeof(SDKMESH_CALLBACKS10));
    callbacks.pCreateTextureFromFile = &createTextureFromFile;
    callbacks.pContext = (void *) path.c_str();

    V(mesh.Create(device, data, size, true, true, &callbacks));
}


HRESULT CALLBACK onCreateDevice(ID3D10Device *device, const DXGI_SURFACE_DESC *desc, void *context) {
    HRESULT hr;

    V_RETURN(dialogResourceManager.OnD3D10CreateDevice(device));
    V_RETURN(settingsDialog.OnD3D10CreateDevice(device));

    timer = new Timer(device);
    timer->setEnabled(hud.GetCheckBox(IDC_PROFILE)->GetChecked());

    V_RETURN(D3DX10CreateFont(device, 15, 0, FW_BOLD, 1, FALSE, DEFAULT_CHARSET,
                              OUT_DEFAULT_PRECIS, DEFAULT_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                              L"Arial", &font));
    V_RETURN(D3DX10CreateSprite(device, 512, &sprite));
    txtHelper = new CDXUTTextHelper(NULL, NULL, font, sprite, 15);

    D3DX10_IMAGE_LOAD_INFO loadInfo;
    ZeroMemory(&loadInfo, sizeof(D3DX10_IMAGE_LOAD_INFO));
    loadInfo.BindFlags = D3D10_BIND_SHADER_RESOURCE;
    loadInfo.MipLevels = 1;
    loadInfo.Format = DXGI_FORMAT_R8_UNORM;
    V_RETURN(D3DX10CreateShaderResourceViewFromResource(device, GetModuleHandle(NULL), L"BeckmannMap.png", &loadInfo, NULL, &beckmannMapView, NULL));

    loadMainEffect();
    
    const D3D10_INPUT_ELEMENT_DESC layout[] = {
        { "POSITION",  0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "NORMAL",    0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD",  0, DXGI_FORMAT_R32G32_FLOAT,    0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0 },
        { "TANGENT",   0, DXGI_FORMAT_R32G32B32_FLOAT, 0, D3D10_APPEND_ALIGNED_ELEMENT, D3D10_INPUT_PER_VERTEX_DATA, 0 }
    };
    UINT numElements = sizeof(layout) / sizeof(D3D10_INPUT_ELEMENT_DESC);

    D3D10_PASS_DESC passDesc;
    V_RETURN(mainEffect->GetTechniqueByName("RenderMSAA")->GetPassByIndex(0)->GetDesc(&passDesc));
    V_RETURN(device->CreateInputLayout(layout, numElements, passDesc.pIAInputSignature, passDesc.IAInputSignatureSize, &vertexLayout));

    device->IASetInputLayout(vertexLayout);

    #ifdef XYZRGB_BUILD
    loadModel(device, L"Roberto\\Roberto.sdkmesh", L"Roberto");
    V(mainEffect->GetVariableByName("fade")->AsScalar()->SetBool(true));
    setRoughness(0.3f);
    currentGaussians = &Gaussian::SKIN;
    #else
    loadModel(device, L"Teapot\\Teapot.sdkmesh", L"Teapot");
    V(mainEffect->GetVariableByName("fade")->AsScalar()->SetBool(false));
    setRoughness(0.5f);
    currentGaussians = &Gaussian::MARBLE;
    #endif

    ShadowMap::init(device);
    Fade::init(device);

    camera.setDistance(3.0f);

    for (int i = 0; i < MAX_N_LIGHTS; i++) {
        lights[i].fov = DEG2RAD(45.0f);
        lights[i].spotExponent = 0.0f;
        lights[i].attenuation = 1.0f / 128.0f;
        lights[i].range = 10.0f;
        lights[i].bias = -0.01f;

        lights[i].camera.setDistance(2.0);
        lights[i].camera.setProjection(lights[i].fov, 1.0f, 0.1f, lights[i].range);
    }
    for (int i = 0; i < nLights; i++) {
        lights[i].color = 1.6f * D3DXVECTOR3(1.0f, 1.0f, 1.0f) / float(nLights);
        lights[i].shadowMap = new ShadowMap(device, SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    }
    lights[0].camera.setAngle(D3DXVECTOR2(-0.52359879f, -0.52359879f));
    lights[1].camera.setAngle(D3DXVECTOR2(0.52359879f, -0.78539819f));
    lights[0].camera.setDistance(1.5f);
    lights[1].camera.setDistance(3.24f);

    UINT quality;
    device->CheckMultisampleQualityLevels(DXGI_FORMAT_R8G8B8A8_UNORM, 8, &quality);
    msaaIndex = quality >= 8? 3 : 2;
    hud.GetComboBox(IDC_MSAA)->SetSelectedByIndex(msaaIndex);

    splashScreen = new SplashScreen(device, sprite);
    intro = new Intro(camera, lights[0].camera, lights[1].camera);

    return S_OK;
}


HRESULT CALLBACK onResizedSwapChain(ID3D10Device *device, IDXGISwapChain *swapChain, const DXGI_SURFACE_DESC *desc, void *context) {
    HRESULT hr;

    V_RETURN(dialogResourceManager.OnD3D10ResizedSwapChain(device, desc));
    V_RETURN(settingsDialog.OnD3D10ResizedSwapChain(device, desc));

    mainRenderTarget = new RenderTarget(device, desc->Width, desc->Height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, msaaModes[msaaIndex].desc);
    depthRenderTarget1X = new RenderTarget(device, desc->Width, desc->Height, DXGI_FORMAT_R32_FLOAT);
    depthStencil = new DepthStencil(device, desc->Width, desc->Height,  DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS, msaaModes[msaaIndex].desc);
    if (msaaModes[msaaIndex].desc.Count > 1) {
        depthStencil1X = new DepthStencil(device, desc->Width, desc->Height,  DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24_UNORM_X8_TYPELESS);
        depthRenderTarget = new RenderTarget(device, desc->Width, desc->Height, DXGI_FORMAT_R32_FLOAT, msaaModes[msaaIndex].desc);
        stencilRenderTarget = new RenderTarget(device, desc->Width, desc->Height, DXGI_FORMAT_R32_FLOAT, msaaModes[msaaIndex].desc);
        resolveRenderTarget = new RenderTarget(device, desc->Width, desc->Height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB); 
    }

    DXUTGetDXGISwapChain()->GetBuffer(0, __uuidof(backbufferTexture), reinterpret_cast<void**>(&backbufferTexture));

    float aspect = (float) desc->Width / desc->Height;
    camera.setProjection(DEG2RAD(CAMERA_FOV), aspect, 0.1f, 100.0f);

    subsurfaceScatteringPass = new SubsurfaceScatteringPass(device, desc->Width, desc->Height, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, msaaModes[msaaIndex].desc.Count, camera.getProjectionMatrix(), 40.0f, 800.0f, 0.001f);
    
    bloomPass = new BloomPass(device, desc->Width / 2, desc->Height / 2, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, 2.2f, 0.192f);

    hud.SetLocation(desc->Width - (45 + HUD_WIDTH), 0);

    int min, max;
    hud.GetSlider(IDC_SSSLEVEL)->GetRange(min, max);
    hud.GetSlider(IDC_SSSLEVEL)->SetValue((int) (subsurfaceScatteringPass->getSssLevel() * (max - min) / 80.0f));

    hud.GetSlider(IDC_CORRECTION)->GetRange(min, max);
    hud.GetSlider(IDC_CORRECTION)->SetValue((int) (subsurfaceScatteringPass->getCorrection() * (max - min) / 4000.0f));

    hud.GetSlider(IDC_MAXDD)->GetRange(min, max);
    hud.GetSlider(IDC_MAXDD)->SetValue((int) (subsurfaceScatteringPass->getMaxdd() * (max - min) * 100.0f));

    wstringstream s;
    s << L"SSS Level: " << subsurfaceScatteringPass->getSssLevel();
    hud.GetStatic(IDC_SSSLEVEL_LABEL)->SetText(s.str().c_str());

    wstringstream s2;
    s2 << L"Correction: " << subsurfaceScatteringPass->getCorrection();
    hud.GetStatic(IDC_CORRECTION_LABEL)->SetText(s2.str().c_str());

    wstringstream s3;
    s3 << L"Max Derivative: " << subsurfaceScatteringPass->getMaxdd();
    hud.GetStatic(IDC_MAXDD_LABEL)->SetText(s3.str().c_str());

    return S_OK;
}


void shadowPass(ID3D10Device *device) {
    for (int i = 0; i < nLights; i++) {
        lights[i].shadowMap->begin(lights[i].camera.getViewMatrix(), lights[i].camera.getProjectionMatrix());
        for (int j = 0; j < N_HEADS; j++) {
            D3DXMATRIX world;
            D3DXMatrixTranslation(&world, j - (N_HEADS - 1) / 2.0f, 0.0f, 0.0f);

            lights[i].shadowMap->setWorldMatrix((float*) world);

            mesh.Render(device, lights[i].shadowMap->getTechnique());
        }
        lights[i].shadowMap->end();
    }
}


void mainPass(ID3D10Device *device, RenderTarget **renderTarget, DepthStencil **depth) {
    HRESULT hr;

    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device->ClearRenderTargetView(*mainRenderTarget, clearColor);

    float depthColor[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    device->ClearDepthStencilView(*depthStencil, D3D10_CLEAR_DEPTH | D3D10_CLEAR_STENCIL, 1.0, 0);

    device->IASetInputLayout(vertexLayout);

    V(mainEffect->GetVariableByName("view")->AsMatrix()->SetMatrix((float *) (D3DXMATRIX) camera.getViewMatrix()));
    V(mainEffect->GetVariableByName("projection")->AsMatrix()->SetMatrix((float *) (D3DXMATRIX) camera.getProjectionMatrix()));
    V(mainEffect->GetVariableByName("cameraPosition")->AsVector()->SetFloatVector((float *) (D3DXVECTOR3) camera.getEyePosition()));

    for (int i = 0; i < nLights; i++) {
        D3DXVECTOR3 t = lights[i].camera.getLookAtPosition() - lights[i].camera.getEyePosition();
        D3DXVECTOR3 dir;
        D3DXVec3Normalize(&dir, &t);

        D3DXVECTOR3 pos = lights[i].camera.getEyePosition();

        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("position")->AsVector()->SetFloatVector((float *) pos));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("direction")->AsVector()->SetFloatVector((float *) dir));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("falloffAngle")->AsScalar()->SetFloat(cos(0.5f  *lights[i].fov)));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("spotExponent")->AsScalar()->SetFloat(lights[i].spotExponent));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("color")->AsVector()->SetFloatVector((float *) lights[i].color));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("attenuation")->AsScalar()->SetFloat(lights[i].attenuation));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("range")->AsScalar()->SetFloat(lights[i].range));
        V(mainEffect->GetVariableByName("lights")->GetElement(i)->GetMemberByName("viewProjection")->AsMatrix()->SetMatrix(ShadowMap::getViewProjectionTextureMatrix(lights[i].camera.getViewMatrix(), lights[i].camera.getProjectionMatrix(), lights[i].bias))); 
        V(mainEffect->GetVariableByName("shadowMaps")->GetElement(i)->AsShaderResource()->SetResource(*lights[i].shadowMap));
    }

    if (msaaModes[msaaIndex].desc.Count > 1) {
        device->ClearRenderTargetView(*depthRenderTarget, clearColor);
        device->ClearRenderTargetView(*stencilRenderTarget, clearColor);
        ID3D10RenderTargetView *rt[] = { *mainRenderTarget, *depthRenderTarget, *stencilRenderTarget };
        device->OMSetRenderTargets(3, rt, *depthStencil);
    } else {
        device->ClearRenderTargetView(*depthRenderTarget1X, clearColor);
        ID3D10RenderTargetView *rt[] = { *mainRenderTarget, *depthRenderTarget1X };
        device->OMSetRenderTargets(2, rt, *depthStencil);
    }
    for (int i = 0; i < N_HEADS; i++) {
        D3DXMATRIX world;
        D3DXMatrixTranslation(&world, i - (N_HEADS - 1) / 2.0f, 0.0f, 0.0f);
        
        V(mainEffect->GetVariableByName("world")->AsMatrix()->SetMatrix((float *) world));
        V(mainEffect->GetVariableByName("worldInverseTranspose")->AsMatrix()->SetMatrixTranspose((float *) world));
        V(mainEffect->GetVariableByName("material")->AsScalar()->SetInt(1));

        if (msaaModes[msaaIndex].desc.Count > 1) {
            mesh.Render(device, mainEffect->GetTechniqueByName("RenderMSAA"), mainEffect->GetVariableByName("diffuseTex")->AsShaderResource(), mainEffect->GetVariableByName("normalTex")->AsShaderResource());
        } else {
            mesh.Render(device, mainEffect->GetTechniqueByName("RenderNoMSAA"), mainEffect->GetVariableByName("diffuseTex")->AsShaderResource(), mainEffect->GetVariableByName("normalTex")->AsShaderResource());
        }
    }
    device->OMSetRenderTargets(0, NULL, NULL);

    if (msaaModes[msaaIndex].desc.Count > 1) {
        device->ResolveSubresource(*resolveRenderTarget, 0, *mainRenderTarget, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB);
        subsurfaceScatteringPass->downsample(*depthRenderTarget1X, *depthRenderTarget, *stencilRenderTarget, *depthStencil1X);
        *renderTarget = resolveRenderTarget;
        *depth = depthStencil1X;
    } else {
        *renderTarget = mainRenderTarget;
        *depth = depthStencil;
    }
}


void renderText() {
    txtHelper->Begin();
    txtHelper->SetInsertionPos(2, 0);
    txtHelper->SetForegroundColor(D3DXCOLOR(1.0f, 1.0f, 0.0f, 1.0f));
    txtHelper->DrawTextLine(DXUTGetFrameStats(DXUTIsVsyncEnabled()));
    txtHelper->DrawTextLine(DXUTGetDeviceStats());

    if (timer->isEnabled()) {
        wstringstream s;
        s << setprecision(5) << std::fixed;
        s << *timer;
        txtHelper->DrawTextLine(s.str().c_str());
    }

    txtHelper->End();
}


void renderScene(ID3D10Device *device, double time, float elapsedTime) {
    if (settingsDialog.IsActive()) {
        settingsDialog.OnRender(elapsedTime);
        return;
    }

    // S h a d o w   P a s s
    timer->start();    
    shadowPass(device);
    timer->clock(L"SHW");

    // M a i n   P a s s
    RenderTarget *renderTarget;
    DepthStencil *depth;
    mainPass(device, &renderTarget, &depth);
    timer->clock(L"MAIN");

    // S u b s u r f a c e   S c a t t e r i n g   P a s s
    if (hud.GetCheckBox(IDC_SSS)->GetChecked()) {
        subsurfaceScatteringPass->render(*renderTarget, *renderTarget, *depthRenderTarget1X, *depth, *currentGaussians, 1);
    }
    timer->clock(L"SSS");

    // B l o o m   P a s s
    ID3D10RenderTargetView *backbufferView = DXUTGetD3D10RenderTargetView();
    if (hud.GetCheckBox(IDC_BLOOM)->GetChecked()) {
        bloomPass->render(*renderTarget, backbufferView);
    } else {
        device->CopyResource(backbufferTexture, *renderTarget);
    }
    timer->clock(L"BLOOM");
}


void resetAll() {
    camera.setAngle(D3DXVECTOR2(0.0f, 0.0f));
    camera.setAngleVelocity(D3DXVECTOR2(0.0f, 0.0f));
    camera.setDistance(3.0f);
    lights[0].camera.setDistance(1.5f);
    lights[0].camera.setAngle(D3DXVECTOR2(-0.52359879f, -0.52359879f));
    lights[0].camera.setAngleVelocity(D3DXVECTOR2(0.0f, 0.0f));
    lights[1].camera.setDistance(3.24f);
    lights[1].camera.setAngle(D3DXVECTOR2(0.52359879f, -0.78539819f));
    lights[1].camera.setAngleVelocity(D3DXVECTOR2(0.0f, 0.0f));
}


void CALLBACK onFrameRender(ID3D10Device *device, double time, float elapsedTime, void *context) {
    if (t0 == numeric_limits<double>::min()) {
        t0 = time;
    }
    if (tFade == numeric_limits<double>::max() && skipIntro) {
        tFade = time;
    }
    switch (state) {
        case STATE_SPLASH:
            splashScreen->render(float(time - t0));
            Fade::render(Animation::linear(float(time - tFade), 0.0f, 1.0f, 1.0f, 0.0f));

            if (splashScreen->hasFinished(float(time - t0)) || float(time - tFade) > 1.0f) {
               state = STATE_INTRO;
               t0 = time;
            }
            break;
        case STATE_INTRO: {
            renderScene(device, time, elapsedTime);
            intro->render(float(time - t0), (GetKeyState(VK_LBUTTON) & 128) == 0 || object != OBJECT_CAMERA);
            Fade::render(Animation::linear(float(time - tFade), 0.0f, 1.0f, 1.0f, 0.0f));

            if (intro->hasFinished(float(time - t0)) || float(time - tFade) > 1.0f) {
               state = STATE_RUNNING;
               t0 = time;
               resetAll();
               break;
            }
            break;
        }
        case STATE_RUNNING: {
            renderScene(device, time, elapsedTime);

            ID3D10RenderTargetView *backbufferView = DXUTGetD3D10RenderTargetView();
            device->OMSetRenderTargets(1, &backbufferView, NULL);
            hud.OnRender(elapsedTime);
            renderText();
            
            #ifdef XYZRGB_BUILD
            Fade::render(Animation::linear(float(time - t0), 1.0f, 4.0f, 0.0f, 1.0f));
            #endif
            break;
        }
    }
}


void CALLBACK onReleasingSwapChain(void *context) {
    dialogResourceManager.OnD3D10ReleasingSwapChain();
    SAFE_DELETE(mainRenderTarget);
    SAFE_DELETE(depthRenderTarget1X);
    SAFE_DELETE(depthRenderTarget);
    SAFE_DELETE(stencilRenderTarget);
    SAFE_DELETE(depthStencil);
    SAFE_DELETE(depthStencil1X);
    SAFE_DELETE(resolveRenderTarget);
    SAFE_RELEASE(backbufferTexture);
    SAFE_DELETE(subsurfaceScatteringPass);
    SAFE_DELETE(bloomPass);
}


void CALLBACK onDestroyDevice(void *context) {
    dialogResourceManager.OnD3D10DestroyDevice();
    settingsDialog.OnD3D10DestroyDevice();
    DXUTGetGlobalResourceCache().OnDestroyDevice();
    SAFE_DELETE(timer);
    SAFE_RELEASE(font);
    SAFE_RELEASE(sprite);
    SAFE_DELETE(txtHelper);
    SAFE_RELEASE(mainEffect);
    mesh.Destroy();
    SAFE_RELEASE(vertexLayout);
    SAFE_RELEASE(beckmannMapView);
    SAFE_DELETE(splashScreen);
    SAFE_DELETE(intro);
    for (int i = 0; i < nLights; i++) {
        SAFE_DELETE(lights[i].shadowMap);
    }
    ShadowMap::release();
    Fade::release();
}


bool CALLBACK modifyDeviceSettings(DXUTDeviceSettings *settings, void *context) {
    settingsDialog.GetDialogControl()->GetComboBox(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_COUNT)->SetEnabled(false);
    settingsDialog.GetDialogControl()->GetComboBox(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_QUALITY)->SetEnabled(false);
    settingsDialog.GetDialogControl()->GetStatic(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_COUNT_LABEL)->SetEnabled(false);
    settingsDialog.GetDialogControl()->GetStatic(DXUTSETTINGSDLG_D3D10_MULTISAMPLE_QUALITY_LABEL)->SetEnabled(false);
    settings->d3d10.AutoCreateDepthStencil = false;
    return true;
}


Camera *currentObject() { 
    switch (object) {
        case OBJECT_CAMERA:
            return &camera;
        default:
            return &lights[object - OBJECT_LIGHT1].camera;
    }
}


void CALLBACK onFrameMove(double time, float elapsedTime, void *context) {
    camera.frameMove(elapsedTime);
    for (int i = 0; i < nLights; i++) {
        lights[i].camera.frameMove(elapsedTime);
    }
}


LRESULT CALLBACK msgProc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam, bool *pbNoFurtherProcessing, void *context) {
    switch (state) {
        case STATE_INTRO:
            switch (msg) {
                case WM_LBUTTONUP:
                    intro->cameraReleased(float(DXUTGetTime() - t0));
                    break;
            }
            break;
        case STATE_RUNNING:
            *pbNoFurtherProcessing = dialogResourceManager.MsgProc(hwnd, msg, wparam, lparam);
            if (*pbNoFurtherProcessing) {
                return 0;
            }

            if (settingsDialog.IsActive()) {
                settingsDialog.MsgProc(hwnd, msg, wparam, lparam);
                return 0;
            }

            switch (msg) {
                case WM_MOUSEWHEEL: {
                    if (state == STATE_RUNNING) {
                        short value = (short) HIWORD(wparam);
                        int nObjects = nLights + 1; // + 1 Camera
                        if (value < 0) {
                            object = Object((int(object) + 1) % nObjects);
                        } else if (value > 0) {
                            object = Object((int(object) - 1) >= 0 ? (int(object) - 1) : (nObjects - 1));
                        }
                        hud.GetComboBox(IDC_OBJECT)->SetSelectedByIndex(object);
                    }
                    return 0;
                }
            }

            *pbNoFurtherProcessing = hud.MsgProc(hwnd, msg, wparam, lparam);
            if (*pbNoFurtherProcessing) {
                return 0;
            }

            break;
    }

    currentObject()->handleMessages(hwnd, msg, wparam, lparam);

    return 0;
}


void CALLBACK keyboardProc(UINT nChar, bool bKeyDown, bool bAltDown, void *context) {
    switch (nChar) {
        case '1': { // /near/ in the article
            camera.setDistance(3.0);
            break;
        }
        case '2': { // /medium/ in the article
            camera.setDistance(8.0);
            break;
        }
        case '3': { // /far/ in the article
            camera.setDistance(22.0);
            break;
        }
        case ' ':
            if (state == STATE_SPLASH || state == STATE_INTRO) {
                skipIntro = true;
            }
            break;
    }
}


void CALLBACK onGUIEvent(UINT event, int id, CDXUTControl *control, void *context) {
    HRESULT hr;

    switch (id) {
        case IDC_TOGGLEFULLSCREEN:
            DXUTToggleFullScreen();
            break;
        case IDC_CHANGEDEVICE:
            settingsDialog.SetActive(!settingsDialog.IsActive());
            break;
        case IDC_TOGGLEREF:
            DXUTToggleREF();
            break;
        case IDC_TOGGLEWARP:
            DXUTToggleWARP();
            break;
        case IDC_MSAA: {
            CDXUTComboBox *box = (CDXUTComboBox *) control;
            msaaIndex = box->GetSelectedIndex();
            onReleasingSwapChain(NULL);
            onResizedSwapChain(DXUTGetD3D10Device(), DXUTGetDXGISwapChain(), DXUTGetDXGIBackBufferSurfaceDesc(), NULL);
            break;
        }
        case IDC_NLIGHTS: {
            for (int i = 0; i < nLights; i++) {
                SAFE_DELETE(lights[i].shadowMap);
            }

            CDXUTComboBox *box = (CDXUTComboBox *) control;
            nLights = box->GetSelectedIndex() + 1;

            loadMainEffect();
            buildObjectComboBox();

            for (int i = 0; i < nLights; i++) {
                lights[i].color = 1.6f * D3DXVECTOR3(1.0f, 1.0f, 1.0f) / float(nLights);
                lights[i].shadowMap = new ShadowMap(DXUTGetD3D10Device(), SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
            }

            object = (Object) (min(int(object), nLights));
            hud.GetComboBox(IDC_OBJECT)->SetSelectedByIndex(object);
            break;
        }
        case IDC_OBJECT:
            if (event == EVENT_COMBOBOX_SELECTION_CHANGED) {
                object = Object(size_t(hud.GetComboBox(IDC_OBJECT)->GetSelectedData()));
            }
            break;
        case IDC_MESH:
            switch ((int) hud.GetComboBox(IDC_MESH)->GetSelectedData()) {
                case 0: {
                    mesh.Destroy();
                    loadModel(DXUTGetD3D10Device(), L"Roberto\\Roberto.sdkmesh", L"Roberto");
                    V(mainEffect->GetVariableByName("fade")->AsScalar()->SetBool(true));
                    currentGaussians = &Gaussian::SKIN;
                    hud.GetComboBox(IDC_MATERIAL)->SetSelectedByIndex(0);
                    setRoughness(0.3f);
                    break;
                }
                case 1: {
                    mesh.Destroy();
                    loadModel(DXUTGetD3D10Device(), L"Apollo\\Apollo.sdkmesh", L"Apollo");
                    V(mainEffect->GetVariableByName("fade")->AsScalar()->SetBool(false));
                    currentGaussians = &Gaussian::MARBLE;
                    hud.GetComboBox(IDC_MATERIAL)->SetSelectedByIndex(2);
                    setRoughness(0.5f);
                    break;
                }
                case 2: {
                    mesh.Destroy();
                    loadModel(DXUTGetD3D10Device(), L"Teapot\\Teapot.sdkmesh", L"Teapot");
                    V(mainEffect->GetVariableByName("fade")->AsScalar()->SetBool(false));
                    currentGaussians = &Gaussian::MARBLE;
                    hud.GetComboBox(IDC_MATERIAL)->SetSelectedByIndex(2);
                    setRoughness(0.5f);
                    break;
                }
            }
            break;
        case IDC_MATERIAL: 
            switch ((int) hud.GetComboBox(IDC_MATERIAL)->GetSelectedData()) {
                case 0:
                    currentGaussians = &Gaussian::SKIN;
                    break;
                case 1:
                    currentGaussians = &skin6Gaussians;
                    break;    
                case 2:
                    currentGaussians = &Gaussian::MARBLE;
                    break;
            }
            break;
        case IDC_SSSLEVEL: {
            CDXUTSlider *slider = (CDXUTSlider *) control;
            int min, max;
            slider->GetRange(min, max);
            float scale = float(slider->GetValue()) / (max - min);
            subsurfaceScatteringPass->setSssLevel(scale * 80.0f);

            wstringstream s;
            s << L"SSS Level: " << subsurfaceScatteringPass->getSssLevel();
            hud.GetStatic(IDC_SSSLEVEL_LABEL)->SetText(s.str().c_str());
            break;
        }
        case IDC_CORRECTION: {
            CDXUTSlider *slider = (CDXUTSlider *) control;
            int min, max;
            slider->GetRange(min, max);
            float scale = float(slider->GetValue()) / (max - min);
            subsurfaceScatteringPass->setCorrection(scale * 4000.0f);
            
            wstringstream s;
            s << L"Correction: " << subsurfaceScatteringPass->getCorrection();
            hud.GetStatic(IDC_CORRECTION_LABEL)->SetText(s.str().c_str());
            break;
        }
        case IDC_MAXDD: {
            CDXUTSlider *slider = (CDXUTSlider *) control;
            int min, max;
            slider->GetRange(min, max);
            float scale = float(slider->GetValue()) / (max - min);
            subsurfaceScatteringPass->setMaxdd(scale / 100.0f);
            
            wstringstream s;
            s << L"Max Derivative: " << subsurfaceScatteringPass->getMaxdd();
            hud.GetStatic(IDC_MAXDD_LABEL)->SetText(s.str().c_str());
            break;
        }
        case IDC_BUMPINESS: {
            CDXUTSlider *slider = (CDXUTSlider *) control;
            int min, max;
            slider->GetRange(min, max);
            float scale = float(slider->GetValue()) / (max - min);
            V(mainEffect->GetVariableByName("bumpiness")->AsScalar()->SetFloat(scale));
            
            wstringstream s;
            s << L"Bumpiness: " << scale;
            hud.GetStatic(IDC_BUMPINESS_LABEL)->SetText(s.str().c_str());
            break;
        }
        case IDC_ROUGHNESS: {
            CDXUTSlider *slider = (CDXUTSlider *) control;
            int min, max;
            slider->GetRange(min, max);
            float scale = float(slider->GetValue()) / (max - min);
            V(mainEffect->GetVariableByName("roughness")->AsScalar()->SetFloat(scale));
            
            wstringstream s;
            s << L"Specular Roughness: " << scale;
            hud.GetStatic(IDC_ROUGHNESS_LABEL)->SetText(s.str().c_str());
            break;
        }
        case IDC_PROFILE:
            if (event == EVENT_CHECKBOX_CHANGED) {
                timer->setEnabled(hud.GetCheckBox(IDC_PROFILE)->GetChecked());
            }
            break;
    }
}
